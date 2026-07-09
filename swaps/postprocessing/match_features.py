import logging
import os
import re
from dataclasses import dataclass, field, replace
from typing import Any, Callable
import numpy as np
import pandas as pd
import tqdm
from skimage.feature import match_template
import cv2
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.spatial.distance import cosine
from .helper import (
    load_peptide_batch_df_from_partquet,
    get_pept_act_from_parquet,
)
from .image_processing import (
    get_roi_descriptor,
    get_sift_descriptor,
    smooth_and_denoise_image,
    detect_2d_peak_with_watershed,
    calculate_peak_property_from_labels_and_image,
)
import duckdb
from matplotlib.colors import ListedColormap

Logger = logging.getLogger(__name__)
_WORKER_CONTEXT: dict[str, Any] = {}


@dataclass
class ConsensusAlignmentState:
    """Aligned image state for consensus generation and consensus decoys."""

    reference_idx: int
    target_shape: tuple[int, int]
    anchor_row: int
    anchor_col: int
    template_bounds: tuple[int, int, int, int]
    template: np.ndarray
    resized_images: list[np.ndarray]
    aligned_images: list[np.ndarray]
    matched_boxes: list[tuple[int, int, int, int]]
    aligned_anchors: list[tuple[float, float] | None]
    scaled_anchors: list[tuple[float, float] | None]
    shifts: list[tuple[int, int]]
    max_scores: list[float]
    match_score_maps: list[np.ndarray] = field(default_factory=list)
    match_score_peaks: list[tuple[int, int]] = field(default_factory=list)
    match_score_label_indices: list[int] = field(default_factory=list)


@dataclass
class ConsensusSegmentationState:
    """Consensus segmentation, snap decisions, and tracked target labels."""

    consensus: np.ndarray
    consensus_denoised: np.ndarray
    snapped_per_anchor: list[tuple[int, int] | None]
    watershed_labels: np.ndarray
    snap_log: dict[str, Any]
    target_label_ids: list[int]
    label_to_snap: dict[int, tuple[int, int]]
    non_none_indices: list[int]
    apply_seg: bool
    # Watershed peak coordinates (Nx2, row/col), reused by coSWA to snap
    # other confounder-group members' own anchors against this same
    # segmentation without recomputing it. Empty when bbox fallback was used.
    all_peaks: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=int))


@dataclass
class ConsensusFeatureBundle:
    """Full consensus state used for scoring targets and generating decoys."""

    alignment: ConsensusAlignmentState
    segmentation: ConsensusSegmentationState
    consensus_pp: pd.DataFrame | None
    individual_pps: list[pd.DataFrame | None]
    raw_aligned_images: list[np.ndarray] = field(default_factory=list)
    raw_aligned_denoised_images: list[np.ndarray] = field(default_factory=list)
    raw_consensus: np.ndarray | None = None
    raw_consensus_denoised: np.ndarray | None = None


def match_features_batches_parallel(
    dict_ref,
    raw_file_list,
    result_dir,
    peptide_indicies: np.ndarray | None = None,
    batch_size_max: int = 1500,
    max_workers: int = 4,
    processing_kwargs: dict | None = None,
    match_decoy: bool = True,
):
    if peptide_indicies is None:
        peptide_indicies = dict_ref["mz_rank"].values
        Logger.info("No peptide indices provided, using all mz_rank from dict_ref.")
    else:
        Logger.info(
            "Using provided peptide indices. Total count: %d", len(peptide_indicies)
        )

    # Sort mz_ranks so each batch is a contiguous range — this lets DuckDB skip
    # row groups in the mz-sorted parquet (produced by build_mz_sorted_activation).
    sorted_mz = np.sort(
        peptide_indicies
    )  # pyright: ignore[reportArgumentType, reportCallIssue]
    n_total = len(sorted_mz)

    # Number of batches: enough so every batch ≤ batch_size_max, AND enough for
    # good load balancing (≥ 2× max_workers so no worker idles at the tail).
    n_batches = max(
        max_workers * 2,
        int(np.ceil(n_total / batch_size_max)),
    )
    Logger.info(
        "Total peptides: %d, Batch size max: %d, Max workers: %d → Using %d batches",
        n_total,
        batch_size_max,
        max_workers,
        n_batches,
    )
    peptide_batches = np.array_split(sorted_mz, n_batches)
    actual_batch_size = len(peptide_batches[0])
    Logger.info(
        "Batching: %d peptides → %d batches of ≤%d (batch_size_max=%d, max_workers=%d)",
        n_total,
        len(peptide_batches),
        actual_batch_size,
        batch_size_max,
        max_workers,
    )
    results_target, results_decoy = [], []
    pp_reference_list, pp_match_target_list = [], []
    pp_match_decoy_list = []
    no_quant_log = []
    no_match_log = []
    snap_log_collection: dict[int, dict] = {}

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_match_features_worker,
        initargs=(
            dict_ref,
            raw_file_list,
            result_dir,
            processing_kwargs,
            match_decoy,
        ),
    ) as executor:
        futures = [
            executor.submit(_match_features_batch_worker, batch)
            for batch in peptide_batches
        ]

        for future in tqdm.tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Processing batches",
            unit="batch",
        ):
            (
                res_target,
                res_decoy,
                pp_reference_target,
                pp_match_target,
                pp_match_decoy,
                no_quant,
                no_match,
                batch_snap_log,
            ) = future.result()
            results_target.extend(res_target)
            results_decoy.extend(res_decoy)
            pp_reference_list.extend(pp_reference_target)
            pp_match_target_list.extend(pp_match_target)
            pp_match_decoy_list.extend(pp_match_decoy)
            no_quant_log.extend(no_quant)
            no_match_log.extend(no_match)
            snap_log_collection.update(batch_snap_log)
    # Final Data Assembly
    matches_target = pd.DataFrame(results_target)
    matches_decoy = pd.DataFrame(results_decoy)
    pp_reference_target = (
        pd.concat(pp_reference_list, ignore_index=True)
        if pp_reference_list
        else pd.DataFrame()
    )
    pp_match_target = (
        pd.concat(pp_match_target_list, ignore_index=True)
        if pp_match_target_list
        else pd.DataFrame()
    )
    pp_match_decoy = (
        pd.concat(pp_match_decoy_list, ignore_index=True)
        if pp_match_decoy_list
        else pd.DataFrame()
    )
    # Drop descriptor columns — only needed during comparison, not in output
    _desc_cols = ["sift_des", "zernike"]
    for _df in (pp_reference_target, pp_match_target, pp_match_decoy):
        _drop = [c for c in _desc_cols if c in _df.columns]
        if _drop:
            _df.drop(columns=_drop, inplace=True)
    df_no_quant = pd.DataFrame(no_quant_log)
    df_no_match = pd.DataFrame(no_match_log)
    return (
        matches_target,
        matches_decoy,
        pp_reference_target,
        pp_match_target,
        pp_match_decoy,
        df_no_quant,
        df_no_match,
        snap_log_collection,
    )


def _init_match_features_worker(
    dict_ref, raw_file_list, result_dir, processing_kwargs, match_decoy: bool = True
):
    """Store immutable batch context once per worker process."""

    _WORKER_CONTEXT["dict_ref"] = dict_ref
    _WORKER_CONTEXT["raw_file_list"] = raw_file_list
    _WORKER_CONTEXT["result_dir"] = result_dir
    _WORKER_CONTEXT["processing_kwargs"] = processing_kwargs
    _WORKER_CONTEXT["match_decoy"] = match_decoy
    _WORKER_CONTEXT["dict_ref_by_mz"] = (
        dict_ref.set_index("mz_rank")
        if dict_ref["mz_rank"].is_unique
        else dict_ref.drop_duplicates("mz_rank").set_index("mz_rank")
    )


def _match_features_batch_worker(batch):
    return match_features_batch(
        dict_ref=_WORKER_CONTEXT["dict_ref"],
        raw_file_list=_WORKER_CONTEXT["raw_file_list"],
        result_dir=_WORKER_CONTEXT["result_dir"],
        batch=batch,
        processing_kwargs=_WORKER_CONTEXT["processing_kwargs"],
        match_decoy=_WORKER_CONTEXT.get("match_decoy", True),
    )


def _feature_instance_id(mz_rank: int, anchor_id: int) -> str:
    """Build the peptide-plus-anchor identifier used as feature-level identity."""

    return f"{mz_rank}_{anchor_id}"


def _confounder_pool(
    pept_idx: int, batch_np: np.ndarray, dict_ref_by_mz: pd.DataFrame
) -> np.ndarray:
    """Return mz_ranks of in-batch confounders for pept_idx, empty if unavailable.

    Confounders are near-isobaric co-eluting candidates stored in dict_ref.
    We restrict to the current batch because activation data is only loaded
    for in-batch mz_ranks.  Cross-batch splits are rare (~1% of candidates)
    since confounder groups span at most ~28 mz_ranks vs batch sizes of ~1500.
    """
    if "confounders" not in dict_ref_by_mz.columns:
        return np.empty(0, dtype=batch_np.dtype)
    conf = dict_ref_by_mz.at[pept_idx, "confounders"]
    if not isinstance(conf, np.ndarray) or conf.size == 0:
        return np.empty(0, dtype=batch_np.dtype)
    in_batch = np.intersect1d(conf, batch_np)
    return in_batch[in_batch != pept_idx].astype(batch_np.dtype)


def _group_members_in_batch(
    dict_ref_by_mz: pd.DataFrame, batch_np: np.ndarray
) -> dict[int, list[int]]:
    """Map each confounder_group_id with >=1 member present in `batch_np` to
    ITS OWN batch-present real mz_ranks. Deliberately batch-scoped, not a
    filtered view of a run-wide mapping -- computing a run-wide grouping once
    per batch/worker would redo an O(N) groupby per batch, reintroducing a
    smaller version of the exact blow-up removed from the SWA write path
    (see helper.load_peptide_batch_df_from_partquet). A group split across
    batches needs no cross-batch coordination: each batch independently sees
    only its own present member(s), fetches the group's one stored parquet
    row via IN(group_id), and expands it only to those members.
    """
    if "confounder_group_id" not in dict_ref_by_mz.columns:
        return {}
    members_by_group: dict[int, list[int]] = {}
    for p in batch_np:
        p = int(p)
        gid = int(dict_ref_by_mz.at[p, "confounder_group_id"])
        if gid != -1:
            members_by_group.setdefault(gid, []).append(p)
    return members_by_group


def _parse_seg_mask_thres(val, default: tuple[int, int] = (3, 3)) -> tuple[int, int]:
    if isinstance(val, dict):
        return (int(val.get("rt", default[0])), int(val.get("im", default[1])))
    if isinstance(val, (list, tuple)) and len(val) == 2:
        return (int(val[0]), int(val[1]))
    if val is None:
        return default
    # legacy scalar: treat as minimum total area, derive square side
    side = max(1, int(val) // 3)  # pyright: ignore[reportArgumentType]
    return (side, side)


def _parse_jump_dist_thres(val, default: tuple[int, int] = (0, 0)) -> tuple[int, int]:
    if isinstance(val, dict):
        return (int(val.get("rt", default[0])), int(val.get("im", default[1])))
    if isinstance(val, (list, tuple)) and len(val) == 2:
        return (int(val[0]), int(val[1]))
    if val is None:
        return default
    side = max(0, int(val))  # pyright: ignore[reportArgumentType]
    return (side, side)


def _denoise_kwargs_for_stage(denoise_cfg: dict, stage: str) -> dict:
    """Build smooth_and_denoise_image kwargs for ops whose ``at`` field == stage."""
    kwargs: dict = {}
    smooth = dict(denoise_cfg.get("smooth") or {})
    clean = dict(denoise_cfg.get("clean") or {})
    log_tf = dict(denoise_cfg.get("log_transform") or {})
    if smooth.get("at") == stage:
        kwargs["smooth"] = {k: v for k, v in smooth.items() if k != "at"}
    if clean.get("at") == stage:
        kwargs["clean"] = {k: v for k, v in clean.items() if k != "at"}
    if log_tf.get("at") == stage:
        kwargs["log_transform"] = bool(log_tf.get("enabled", True))
    return kwargs


def _denoise_kwargs_all(denoise_cfg: dict) -> dict:
    """Combine raw + consensus stage kwargs (for full-pipeline denoising of raw_aligned)."""
    return {
        **_denoise_kwargs_for_stage(denoise_cfg, "raw"),
        **_denoise_kwargs_for_stage(denoise_cfg, "consensus"),
    }


def _annotate_peak_properties(
    peak_properties: pd.DataFrame | None,
    *,
    mz_rank: int,
    run_name: str,
    own_anchor_id: int,
    assimilated_to_anchor_id: int,
    feature_instance_id: str,
    own_feature_instance_id: str,
    source_run: str,
    source_type: str,
    decoy_mz_rank: int | None = None,
    undistinguishable_group_id: str | int = -1,
) -> pd.DataFrame | None:
    """Add anchor-aware metadata columns to a quantified peak-properties row.

    undistinguishable_group_id flags coSWA confounder-group members whose own
    independently-computed assigned segments spatially overlap (see
    _mark_overlapping_group_members in match_features_batch) -- -1 (the
    default) means not part of such an overlap. Always -1 at the point this
    function is called; patched in afterward once every member of the
    member's group has been processed.
    """

    if peak_properties is None:
        return None
    peak_properties = peak_properties.copy()
    peak_properties["Run_name"] = run_name
    peak_properties["mz_rank"] = mz_rank
    peak_properties["own_anchor_id"] = own_anchor_id
    peak_properties["assimilated_to_anchor_id"] = assimilated_to_anchor_id
    peak_properties["feature_instance_id"] = feature_instance_id
    peak_properties["own_feature_instance_id"] = own_feature_instance_id
    peak_properties["source_run"] = source_run
    peak_properties["source_type"] = source_type
    peak_properties["undistinguishable_group_id"] = undistinguishable_group_id
    if decoy_mz_rank is not None:
        peak_properties["decoy_mz_rank"] = decoy_mz_rank
    return peak_properties


def match_features_batch(
    dict_ref,
    raw_file_list,
    result_dir,
    batch,
    processing_kwargs: dict | None = None,
    visualize_dir: str | None = None,
    match_decoy: bool = True,
    illustration_dir: str | None = None,
):
    """Process one peptide batch using the consensus image path."""
    results_target, results_decoy = [], []
    pp_reference_list, pp_match_target_list, pp_match_decoy_list = [], [], []
    no_quant_log, no_match_log = [], []
    snap_log_collection: dict[int, dict] = {}
    batch_np = np.asarray(batch)

    _cached = _WORKER_CONTEXT.get("dict_ref_by_mz")
    dict_ref_by_mz = (
        _cached
        if _cached is not None
        else (
            dict_ref.set_index("mz_rank")
            if dict_ref["mz_rank"].is_unique
            else dict_ref.drop_duplicates("mz_rank").set_index("mz_rank")
        )
    )
    denoise_cfg = dict((processing_kwargs or {}).get("denoise", {}))
    raw_denoise_kwargs = _denoise_kwargs_for_stage(denoise_cfg, "raw")
    full_denoise_kwargs = _denoise_kwargs_all(denoise_cfg)
    _align_images = bool((processing_kwargs or {}).get("align_images", True))
    _jump_dist_thres = _parse_jump_dist_thres(
        (processing_kwargs or {}).get("jump_dist_thres")
    )

    # coSWA groups are stored on disk as a single row-set keyed by their
    # confounder_group_id (never duplicated to every member's mz_rank -- see
    # helper.load_peptide_batch_df_from_partquet). Compute this batch's own
    # group membership once: used to fetch+expand each group's row below, and
    # to drive the post-hoc segment-overlap tagging pass after the main loop
    # (see _mark_overlapping_group_members).
    _group_to_members = _group_members_in_batch(dict_ref_by_mz, batch_np)
    _members_by_group = {g: m for g, m in _group_to_members.items() if len(m) >= 2}

    # Load activation data for this mz_rank batch from the pre-built sorted parquet.
    # DuckDB skips row groups outside [min(batch_np), max(batch_np)], so I/O scales
    # with batch_size / N_total.  Requires build_mz_sorted_activation() to have been
    # run for each raw_file activation directory beforehand.
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    # expand_to_members=False: keep each in-batch group's row-set keyed by its
    # own confounder_group_id rather than duplicated out to every member's own
    # mz_rank. Group members' activation lookups are instead redirected to the
    # group id at query time (see _act_lookup_key below) -- same data, without
    # the O(members) duplication cost or the correspondingly larger, more
    # duplicate-heavy mz_rank index that per-candidate lookups would otherwise
    # run against for every candidate sharing this batch, group or solo.
    _act_dfs_raw = {
        raw_file: load_peptide_batch_df_from_partquet(
            os.path.join(result_dir, raw_file, "activation"),
            batch_np,
            group_to_members=_group_to_members or None,
            con=con,
            expand_to_members=False,
        )
        for raw_file in raw_file_list
    }
    con.close()

    # Per-raw-file {mz_rank: sub-dataframe} map, built once so each candidate's
    # activation fetch is an O(1) dict lookup instead of pandas' non-unique-
    # index .loc[[key]] resolution (get_indexer_non_unique), which does not
    # amortize to O(1) on repeated calls against the same index the way a
    # unique index's hash lookup does.
    act_dfs: dict[str, dict[int, pd.DataFrame]] = {
        raw_file: {int(mz): sub for mz, sub in df.groupby("mz_rank", sort=False)}
        for raw_file, df in _act_dfs_raw.items()
    }
    _empty_act_df = {raw_file: df.iloc[0:0] for raw_file, df in _act_dfs_raw.items()}

    def _select_mz(raw_file: str, mz_rank: int) -> pd.DataFrame:
        return act_dfs[raw_file].get(mz_rank, _empty_act_df[raw_file])

    _has_group_col = "confounder_group_id" in dict_ref_by_mz.columns

    def _act_lookup_key(pept_idx: int) -> int:
        """Map a candidate's own mz_rank to the key its activation is stored
        under in act_dfs: its confounder_group_id when it belongs to an
        in-batch group (act_dfs keeps one un-duplicated row-set per group --
        see expand_to_members=False above), else its own mz_rank unchanged."""
        if not _has_group_col:
            return pept_idx
        gid = int(dict_ref_by_mz.at[pept_idx, "confounder_group_id"])
        return gid if gid != -1 else pept_idx

    def _positional_anchors(stack, ref_rf, quant_set, loader):
        """Per-run anchor list over `stack`: this candidate's own reference /
        quant_only runs get its (frame, scan) apex; all other runs get None."""
        anchors: list[tuple[int, int] | None] = []
        for rf in stack:
            if rf == ref_rf or rf in quant_set:
                t = loader(rf)
                anchors.append((int(t[1]), int(t[2])))
            else:
                anchors.append(None)
        return anchors

    def _reference_match_quant_files(pept_idx: int):
        row_series = dict_ref_by_mz.loc[pept_idx, :]
        str_values = row_series[row_series.map(lambda x: isinstance(x, str))]
        reference_raw_file = str(str_values.index[(str_values == "Reference")][0])
        quant_only_raw_files = str_values.index[str_values == "Quant_Only"].tolist()
        match_raw_files = str_values.index[
            (str_values.str.contains("Match", regex=False))
            | (str_values == "Quant_Only")
        ].tolist()
        return reference_raw_file, quant_only_raw_files, match_raw_files

    # coSWA: every candidate -- group member or solo -- gets its own
    # independent alignment + watershed segmentation below (own roles, own
    # anchors, own window). Group members' assigned segments are compared for
    # spatial overlap AFTER the main loop (_mark_overlapping_group_members),
    # once every member of every in-batch group has been processed; pairs
    # (or larger connected sets) whose own segments overlap are tagged with a
    # shared undistinguishable_group_id, patched into the rows built below.
    #
    # Deliberately out of scope here: decoy generation (_confounder_pool /
    # peptide_swap sampling) is left completely untouched -- grouped
    # candidates go through the exact same per-candidate decoy code as solo
    # candidates, operating on whichever ConsensusFeatureBundle this loop
    # built for them.
    _member_overlap_cache: dict[int, dict] = {}

    for pept_idx in batch_np:
        pept_act_cache: dict[str, tuple[np.ndarray, int, int, tuple[int, int]]] = {}
        pept_act_raw_denoised_cache: dict[str, np.ndarray] = {}

        # coSWA: every candidate -- group member or solo -- is processed
        # fully independently here (own roles, own anchors, own window, own
        # alignment + watershed). Group members' assigned segments are
        # compared for spatial overlap only AFTER this loop finishes (see
        # _mark_overlapping_group_members below); the shared activation
        # trace is still fetched by confounder_group_id (_act_key), since
        # that reflects an upstream SWA-solve fact unrelated to how each
        # member is subsequently aligned/segmented here.
        _group_id = (
            int(dict_ref_by_mz.at[pept_idx, "confounder_group_id"])
            if "confounder_group_id" in dict_ref_by_mz.columns
            else -1
        )
        _act_key = _group_id if _group_id != -1 else int(pept_idx)

        def _get_pept_act_tuple(
            raw_file: str,
        ) -> tuple[np.ndarray, int, int, tuple[int, int]]:
            if raw_file not in pept_act_cache:
                pept_act_cache[raw_file] = (
                    get_pept_act_from_parquet(  # pyright: ignore[reportArgumentType]
                        _select_mz(raw_file, _act_key),
                        int(pept_idx),
                        dict_ref_by_mz,
                        raw_file,
                        return_offset=True,
                    )
                )
            return pept_act_cache[raw_file]

        def _get_raw_denoised_pept_act(raw_file: str) -> np.ndarray:
            if raw_file not in pept_act_raw_denoised_cache:
                pept_act_raw_denoised_cache[raw_file] = smooth_and_denoise_image(
                    _get_pept_act_tuple(raw_file)[0], **raw_denoise_kwargs
                )
            return pept_act_raw_denoised_cache[raw_file]

        # Roles are ALWAYS this candidate's OWN (fixes the coSWA bug where
        # group members used to reuse a representative's per-run role
        # assignment).
        reference_raw_file, quant_only_raw_files, match_raw_files = (
            _reference_match_quant_files(pept_idx)
        )
        _quant_only_set = set(quant_only_raw_files)

        _consensus_raw_files = [reference_raw_file] + match_raw_files
        _consensus_anchors = _positional_anchors(
            _consensus_raw_files,
            reference_raw_file,
            _quant_only_set,
            _get_pept_act_tuple,
        )

        own_anchor_id = 0
        feature_instance_id = _feature_instance_id(pept_idx, own_anchor_id)

        # Only files with known anchors contribute to the consensus average;
        # files without anchors are still aligned and quantified from the labels.
        _anchor_image_indices = [
            i for i, a in enumerate(_consensus_anchors) if a is not None
        ]
        _consensus_bundle = build_consensus_feature_bundle(
            images=[_get_raw_denoised_pept_act(rf) for rf in _consensus_raw_files],
            reference_idx=0,
            template_frac=0.3,
            anchors=_consensus_anchors,
            denoise_cfg=denoise_cfg,
            watershed_kwargs=dict(
                (processing_kwargs or {}).get("peak_consensus_kwargs", {})
            ),
            raw_images=[_get_pept_act_tuple(rf)[0] for rf in _consensus_raw_files],
            labels=_consensus_raw_files,
            apply_seg=bool((processing_kwargs or {}).get("apply_seg", True)),
            seg_mask_thres=_parse_seg_mask_thres(
                (processing_kwargs or {}).get("seg_mask_thres")
            ),
            jump_dist_thres=_jump_dist_thres,
            consensus_image_indices=_anchor_image_indices,
            align_images=_align_images,
        )
        if _group_id in _members_by_group:
            # Stash what the post-hoc overlap pass needs -- this member's own
            # alignment/segmentation state, run stack, and per-run absolute
            # window origins (to project its assigned segment mask into a
            # common run's real frame_idx/mobility_index coordinates).
            _member_overlap_cache[int(pept_idx)] = {
                "alignment": _consensus_bundle.alignment,
                "segmentation": _consensus_bundle.segmentation,
                "consensus_raw_files": _consensus_raw_files,
                "reference_raw_file": reference_raw_file,
                "window_origin_by_run": {
                    rf: _get_pept_act_tuple(rf)[3] for rf in _consensus_raw_files
                },
            }
        if visualize_dir is not None:
            _visualize_consensus_bundle(
                _consensus_bundle.alignment,
                _consensus_bundle.segmentation,
                fig_dir=visualize_dir,
                filename=f"mz{pept_idx}_consensus.png",
                labels=_consensus_raw_files,
            )
        _batch_svg_dir = (
            os.path.join(
                illustration_dir,
                f"batch_mz{int(batch_np.min())}-{int(batch_np.max())}",
            )
            if illustration_dir is not None
            else None
        )
        if _batch_svg_dir is not None:
            _save_illustration_svgs(
                int(pept_idx),
                _consensus_bundle,
                _consensus_raw_files,
                _batch_svg_dir,
                raw_images=[_get_pept_act_tuple(rf)[0] for rf in _consensus_raw_files],
            )
        consensus_pp = _consensus_bundle.consensus_pp
        individual_pps = _consensus_bundle.individual_pps
        snap_log_collection[int(pept_idx)] = _consensus_bundle.segmentation.snap_log
        _consensus_decoy_kwargs = dict(
            (processing_kwargs or {}).get("consensus_decoy_kwargs", {})
        )
        _consensus_decoy_strategies = {
            str(s)
            for s in _consensus_decoy_kwargs.get(
                "strategies", ["peptide_swap", "off_target_shift"]
            )
        }
        _n_peptide_swap_decoys = max(
            int(_consensus_decoy_kwargs.get("n_peptide_swap_decoys", 1)), 0
        )
        _n_off_target_decoys = max(
            int(_consensus_decoy_kwargs.get("n_off_target_shift_decoys", 1)), 0
        )
        _off_target_min_offset_frac = float(
            _consensus_decoy_kwargs.get("off_target_min_offset_frac", 0.35)
        )
        _off_target_max_overlap_fraction = float(
            _consensus_decoy_kwargs.get("off_target_max_overlap_fraction", 0.05)
        )
        _batch_exclude = (
            batch_np[batch_np != pept_idx]
            if match_decoy and batch_np.size > 1
            else np.array([], dtype=batch_np.dtype)
        )
        _use_confounder_sampling = bool(
            _consensus_decoy_kwargs.get("use_confounder_sampling", True)
        )
        _confounder_in_batch = (
            _confounder_pool(int(pept_idx), batch_np, dict_ref_by_mz)
            if _use_confounder_sampling
            else np.array([], dtype=batch_np.dtype)
        )
        _peptide_swap_decoys_by_rep: list[dict[str, dict[str, Any]]] = []
        if (
            match_decoy
            and "peptide_swap" in _consensus_decoy_strategies
            and _n_peptide_swap_decoys > 0
            and _batch_exclude.size > 0
        ):
            for _rep in range(_n_peptide_swap_decoys):
                _rep_specs: dict[str, dict[str, Any]] = {}
                _plot_raw_images: list[np.ndarray] = []
                _plot_raw_denoised_images: list[np.ndarray] = []
                _plot_labels: list[str] = []
                for _plot_i, _plot_rf in enumerate(_consensus_raw_files):
                    if _plot_rf == reference_raw_file:
                        _ref_raw = _get_pept_act_tuple(_plot_rf)[0]
                        _plot_raw_images.append(_ref_raw)
                        _plot_raw_denoised_images.append(
                            _get_raw_denoised_pept_act(_plot_rf)
                        )
                        _plot_labels.append(_plot_rf)
                        continue
                    _decoy_pool = (
                        _confounder_in_batch
                        if _confounder_in_batch.size > 0
                        else _batch_exclude
                    )
                    _decoy_mz = int(np.random.choice(_decoy_pool))
                    _decoy_act_df = _select_mz(_plot_rf, _act_lookup_key(_decoy_mz))
                    _decoy_raw, _, _ = get_pept_act_from_parquet(
                        _decoy_act_df,
                        _decoy_mz,
                        dict_ref_by_mz,
                        _plot_rf,
                        shape=_get_pept_act_tuple(_plot_rf)[0].shape,
                    )
                    _decoy_raw_denoised = smooth_and_denoise_image(
                        _decoy_raw, **raw_denoise_kwargs
                    )
                    _rep_specs[_plot_rf] = {
                        "decoy_mz_rank": _decoy_mz,
                        "decoy_raw_image": _decoy_raw,
                        "decoy_raw_denoised_image": _decoy_raw_denoised,
                    }
                    _plot_raw_images.append(_decoy_raw)
                    _plot_raw_denoised_images.append(_decoy_raw_denoised)
                    _plot_labels.append(f"{_plot_rf}\n(decoy mz{_decoy_mz})")
                _peptide_swap_decoys_by_rep.append(_rep_specs)
                if visualize_dir is not None or _batch_svg_dir is not None:
                    _plot_anchors = [_consensus_anchors[0]] + [None] * (
                        len(_consensus_raw_files) - 1
                    )
                    _decoy_bundle = build_consensus_feature_bundle(
                        images=_plot_raw_denoised_images,
                        reference_idx=0,
                        template_frac=0.3,
                        anchors=_plot_anchors,
                        denoise_cfg=denoise_cfg,
                        watershed_kwargs=dict(
                            (processing_kwargs or {}).get("peak_consensus_kwargs", {})
                        ),
                        raw_images=_plot_raw_images,
                        labels=_plot_labels,
                        apply_seg=bool(
                            (processing_kwargs or {}).get("apply_seg", True)
                        ),
                        seg_mask_thres=_parse_seg_mask_thres(
                            (processing_kwargs or {}).get("seg_mask_thres")
                        ),
                        jump_dist_thres=_parse_jump_dist_thres(
                            (processing_kwargs or {}).get("jump_dist_thres")
                        ),
                        align_images=_align_images,
                    )
                    if visualize_dir is not None:
                        _visualize_consensus_bundle(
                            _decoy_bundle.alignment,
                            _decoy_bundle.segmentation,
                            fig_dir=visualize_dir,
                            filename=(
                                f"mz{pept_idx}_consensus_decoy_peptide_swap_rep{_rep}.png"
                            ),
                            labels=_plot_labels,
                        )
                    if _batch_svg_dir is not None:
                        _save_illustration_svgs(
                            int(pept_idx),
                            _decoy_bundle,
                            _plot_labels,
                            _batch_svg_dir,
                            raw_images=_plot_raw_images,
                            filename_prefix=f"decoy_peptide_swap_rep{_rep}_",
                        )

        _off_target_label_shifts: list[tuple[int, int] | None] = []
        if (
            match_decoy
            and "off_target_shift" in _consensus_decoy_strategies
            and _n_off_target_decoys > 0
        ):
            for _rep in range(_n_off_target_decoys):
                _shift = _choose_off_target_shift(
                    _consensus_bundle.segmentation.watershed_labels,
                    _consensus_bundle.segmentation.target_label_ids,
                    rep=_rep,
                    min_offset_frac=_off_target_min_offset_frac,
                    max_overlap_fraction=_off_target_max_overlap_fraction,
                )
                _off_target_label_shifts.append(_shift)
                if _shift is not None and (
                    visualize_dir is not None or _batch_svg_dir is not None
                ):
                    _shifted_seg = _make_shifted_consensus_segmentation_state(
                        _consensus_bundle.segmentation,
                        _shift,
                    )
                    if visualize_dir is not None:
                        _visualize_consensus_bundle(
                            _consensus_bundle.alignment,
                            _shifted_seg,
                            fig_dir=visualize_dir,
                            filename=(
                                f"mz{pept_idx}_consensus_decoy_off_target_shift_rep{_rep}.png"
                            ),
                            labels=_consensus_raw_files,
                        )
                    if _batch_svg_dir is not None:
                        _save_illustration_svgs(
                            int(pept_idx),
                            _consensus_bundle,
                            _consensus_raw_files,
                            _batch_svg_dir,
                            segmentation_override=_shifted_seg,
                            filename_prefix=f"decoy_off_target_shift_rep{_rep}_",
                            skip_per_run=True,
                        )
        if consensus_pp is not None:
            for _ci, (_rf, _ind_pp) in enumerate(
                zip(_consensus_raw_files, individual_pps)
            ):
                _run_type = (
                    "reference"
                    if _rf == reference_raw_file
                    else ("quant_only" if _rf in _quant_only_set else "match_target")
                )
                if _ind_pp is None:
                    no_quant_log.append(
                        {
                            "mz_rank": pept_idx,
                            "run_name": _rf,
                            "type": _run_type,
                            "feature_instance_id": feature_instance_id,
                        }
                    )
                    if _rf != reference_raw_file:
                        no_match_log.append(
                            {
                                "mz_rank": pept_idx,
                                "run_name": _rf,
                                "type": _run_type,
                                "feature_instance_id": feature_instance_id,
                            }
                        )
                    continue
                _annotated_pp = _annotate_peak_properties(
                    _ind_pp,
                    mz_rank=pept_idx,
                    run_name=_rf,
                    own_anchor_id=own_anchor_id,
                    assimilated_to_anchor_id=own_anchor_id,
                    feature_instance_id=feature_instance_id,
                    own_feature_instance_id=feature_instance_id,
                    source_run="consensus",
                    source_type="Consensus",
                    undistinguishable_group_id=-1,  # patched post-loop if overlapping
                )
                if _annotated_pp is None:
                    continue
                if _rf == reference_raw_file:
                    pp_reference_list.append(_annotated_pp)
                    continue
                _match_t = compare_peak_properties(consensus_pp, _annotated_pp)
                _match_t["mz_rank"] = pept_idx
                _match_t["feature_instance_id"] = feature_instance_id
                _match_t["own_anchor_id"] = own_anchor_id
                _match_t["assimilated_to_anchor_id"] = own_anchor_id
                _match_t["source_run"] = "consensus"
                _match_t["source_type"] = "Consensus"
                _match_t["undistinguishable_group_id"] = -1  # patched post-loop
                results_target.append(_match_t)
                pp_match_target_list.append(_annotated_pp)

                if not match_decoy:
                    continue

                if (
                    "peptide_swap" in _consensus_decoy_strategies
                    and _n_peptide_swap_decoys > 0
                    and _peptide_swap_decoys_by_rep
                ):
                    for _rep in range(_n_peptide_swap_decoys):
                        _rep_spec = _peptide_swap_decoys_by_rep[_rep].get(_rf)
                        if _rep_spec is None:
                            continue
                        decoy_pept_idx = int(_rep_spec["decoy_mz_rank"])
                        decoy_act = _rep_spec["decoy_raw_image"]
                        decoy_pp_raw, _, _ = _build_consensus_peptide_swap_decoy(
                            _consensus_bundle,
                            decoy_act,
                            _rf,
                            raw_denoise_kwargs=raw_denoise_kwargs,
                        )
                        if decoy_pp_raw is None:
                            no_quant_log.append(
                                {
                                    "mz_rank": pept_idx,
                                    "run_name": _rf,
                                    "type": "match_decoy",
                                    "feature_instance_id": feature_instance_id,
                                    "decoy_strategy": "peptide_swap_consensus",
                                    "decoy_rep": _rep,
                                    "decoy_mz_rank": decoy_pept_idx,
                                }
                            )
                            no_match_log.append(
                                {
                                    "mz_rank": pept_idx,
                                    "run_name": _rf,
                                    "type": "match_decoy",
                                    "feature_instance_id": feature_instance_id,
                                    "decoy_strategy": "peptide_swap_consensus",
                                    "decoy_rep": _rep,
                                    "decoy_mz_rank": decoy_pept_idx,
                                }
                            )
                            continue
                        _prop_d = _annotate_peak_properties(
                            decoy_pp_raw,
                            mz_rank=pept_idx,
                            run_name=_rf,
                            own_anchor_id=own_anchor_id,
                            assimilated_to_anchor_id=own_anchor_id,
                            feature_instance_id=feature_instance_id,
                            own_feature_instance_id=feature_instance_id,
                            source_run="consensus",
                            source_type="Consensus",
                            decoy_mz_rank=decoy_pept_idx,
                        )
                        if _prop_d is None:
                            continue
                        _prop_d["decoy_strategy"] = "peptide_swap_consensus"
                        _prop_d["decoy_rep"] = _rep
                        pp_match_decoy_list.append(_prop_d)
                        _match_d = compare_peak_properties(consensus_pp, _prop_d)
                        _match_d["mz_rank"] = pept_idx
                        _match_d["decoy_mz_rank"] = decoy_pept_idx
                        _match_d["feature_instance_id"] = feature_instance_id
                        _match_d["own_anchor_id"] = own_anchor_id
                        _match_d["assimilated_to_anchor_id"] = own_anchor_id
                        _match_d["source_run"] = "consensus"
                        _match_d["source_type"] = "Consensus"
                        _match_d["decoy_strategy"] = "peptide_swap_consensus"
                        _match_d["decoy_rep"] = _rep
                        results_decoy.append(_match_d)

                if (
                    "off_target_shift" in _consensus_decoy_strategies
                    and _n_off_target_decoys > 0
                ):
                    for _rep in range(_n_off_target_decoys):
                        _precomputed_shift = (
                            _off_target_label_shifts[_rep]
                            if _rep < len(_off_target_label_shifts)
                            else None
                        )
                        decoy_pp_raw, label_shift = _build_consensus_off_target_decoy(
                            _consensus_bundle,
                            run_index=_ci,
                            run_name=_rf,
                            rep=_rep,
                            min_offset_frac=_off_target_min_offset_frac,
                            max_overlap_fraction=_off_target_max_overlap_fraction,
                            label_shift=_precomputed_shift,
                        )
                        if decoy_pp_raw is None or label_shift is None:
                            no_quant_log.append(
                                {
                                    "mz_rank": pept_idx,
                                    "run_name": _rf,
                                    "type": "match_decoy",
                                    "feature_instance_id": feature_instance_id,
                                    "decoy_strategy": "off_target_shift_consensus",
                                    "decoy_rep": _rep,
                                }
                            )
                            no_match_log.append(
                                {
                                    "mz_rank": pept_idx,
                                    "run_name": _rf,
                                    "type": "match_decoy",
                                    "feature_instance_id": feature_instance_id,
                                    "decoy_strategy": "off_target_shift_consensus",
                                    "decoy_rep": _rep,
                                }
                            )
                            continue
                        _prop_d = _annotate_peak_properties(
                            decoy_pp_raw,
                            mz_rank=pept_idx,
                            run_name=_rf,
                            own_anchor_id=own_anchor_id,
                            assimilated_to_anchor_id=own_anchor_id,
                            feature_instance_id=feature_instance_id,
                            own_feature_instance_id=feature_instance_id,
                            source_run="consensus",
                            source_type="Consensus",
                            decoy_mz_rank=-1,
                        )
                        if _prop_d is None:
                            continue
                        _prop_d["decoy_strategy"] = "off_target_shift_consensus"
                        _prop_d["decoy_rep"] = _rep
                        _prop_d["label_shift_rt"] = int(label_shift[0])
                        _prop_d["label_shift_im"] = int(label_shift[1])
                        pp_match_decoy_list.append(_prop_d)
                        _match_d = compare_peak_properties(consensus_pp, _prop_d)
                        _match_d["mz_rank"] = pept_idx
                        _match_d["decoy_mz_rank"] = -1
                        _match_d["feature_instance_id"] = feature_instance_id
                        _match_d["own_anchor_id"] = own_anchor_id
                        _match_d["assimilated_to_anchor_id"] = own_anchor_id
                        _match_d["source_run"] = "consensus"
                        _match_d["source_type"] = "Consensus"
                        _match_d["decoy_strategy"] = "off_target_shift_consensus"
                        _match_d["decoy_rep"] = _rep
                        _match_d["label_shift_rt"] = int(label_shift[0])
                        _match_d["label_shift_im"] = int(label_shift[1])
                        results_decoy.append(_match_d)
        else:
            # consensus_pp is None: consensus generation failed — log all runs
            for _rf in _consensus_raw_files:
                no_quant_log.append(
                    {
                        "mz_rank": pept_idx,
                        "run_name": _rf,
                        "type": (
                            "reference"
                            if _rf == reference_raw_file
                            else (
                                "quant_only"
                                if _rf in _quant_only_set
                                else "match_target"
                            )
                        ),
                        "feature_instance_id": feature_instance_id,
                    }
                )

    # coSWA: now that every member of every in-batch group has been
    # independently aligned + segmented above, check whether their own
    # assigned segments spatially overlap and tag the overlapping ones.
    # undistinguishable_group_id was written as -1 everywhere above (the tag
    # isn't knowable until this point), so patch it into the already-built
    # rows for the subset of mz_ranks flagged below.
    _undistinguishable_tag = _mark_overlapping_group_members(
        _members_by_group, _member_overlap_cache
    )
    if _undistinguishable_tag:
        for _row in results_target:
            _tag = _undistinguishable_tag.get(int(_row["mz_rank"]))
            if _tag is not None:
                _row["undistinguishable_group_id"] = _tag
        for _pp_list in (pp_reference_list, pp_match_target_list):
            for _df in _pp_list:
                _tag = _undistinguishable_tag.get(int(_df["mz_rank"].iat[0]))
                if _tag is not None:
                    _df["undistinguishable_group_id"] = _tag

    return (
        results_target,
        results_decoy,
        pp_reference_list,
        pp_match_target_list,
        pp_match_decoy_list,
        no_quant_log,
        no_match_log,
        snap_log_collection,
    )


def compare_peak_properties(peak_properties_a, peak_properties_b):
    return {
        "template_matching_score": peak_properties_b["template_matching_score"].values[
            0
        ],
        "sift_similarities": compare_sift_descriptors_similarities(
            peak_properties_a["sift_des"].values[0],
            peak_properties_b["sift_des"].values[0],
        ),
        "zernike_similarities": compare_image_descriptors_cosine(
            peak_properties_a["zernike"].values[0],
            peak_properties_b["zernike"].values[0],
        ),
        "sift_distance": compare_image_descriptors_euclidean(
            peak_properties_a["sift_des"].values[0],
            peak_properties_b["sift_des"].values[0],
            l2_norm=False,
        ),
        "zernike_distance": compare_image_descriptors_euclidean(
            peak_properties_a["zernike"].values[0],
            peak_properties_b["zernike"].values[0],
            l2_norm=True,
        ),
        "rt_shift": abs(
            peak_properties_a["shift_rt"].values[0]
            - peak_properties_b["shift_rt"].values[0]
        ),
        "im_shift": abs(
            peak_properties_a["shift_im"].values[0]
            - peak_properties_b["shift_im"].values[0]
        ),
        "rt_length_diff": abs(
            peak_properties_a["rt_length"].values[0]
            - peak_properties_b["rt_length"].values[0]
        ),
        "im_length_diff": abs(
            peak_properties_a["im_length"].values[0]
            - peak_properties_b["im_length"].values[0]
        ),
        "rt_length_diff_rel": abs(
            peak_properties_a["rt_length"].values[0]
            - peak_properties_b["rt_length"].values[0]
        )
        / peak_properties_a["rt_length"].values[0],
        "im_length_diff_rel": abs(
            peak_properties_a["im_length"].values[0]
            - peak_properties_b["im_length"].values[0]
        )
        / peak_properties_a["im_length"].values[0],
        "int_max_diff_rel": abs(
            peak_properties_a["intensity_max"].values[0]
            - peak_properties_b["intensity_max"].values[0]
        )
        / peak_properties_a["intensity_max"].values[0],
        "int_sum_diff_rel": abs(
            peak_properties_a["intensity_sum"].values[0]
            - peak_properties_b["intensity_sum"].values[0]
        )
        / peak_properties_a["intensity_sum"].values[0],
        "area_diff_rel": abs(
            peak_properties_a["area"].values[0] - peak_properties_b["area"].values[0]
        )
        / peak_properties_a["area"].values[0],
        "reference_run": peak_properties_a["Run_name"].values[0],
        "matched_run": peak_properties_b["Run_name"].values[0],
    }


def compare_image_descriptors_cosine(des1, des2, log_transform: bool = True):
    if des1 is None or des2 is None:
        return 0.0

    # SIFT descriptors must be float32 for NORM_L2
    # This line prevents the "Assertion failed" error
    d1 = des1.astype(np.float32).flatten()
    d2 = des2.astype(np.float32).flatten()

    if np.all(d1 == 0) or np.all(d2 == 0):
        return 0.0

    # Option 2: rescale from [-1, 1] to [0, 1] (preserves information)
    similarity = (1 - cosine(d1, d2) + 1) / 2
    if log_transform:
        similarity = -np.log(1 - similarity + 1e-8)
    return similarity


def compare_image_descriptors_euclidean(des1, des2, l2_norm: bool = False):
    if des1 is None or des2 is None:
        return 0.0

    # SIFT descriptors must be float32 for NORM_L2
    # This line prevents the "Assertion failed" error
    d1 = des1.astype(np.float32).flatten()
    d2 = des2.astype(np.float32).flatten()
    if l2_norm:
        # L2 normalize (unit vectors) — critical for fair comparison
        d1 = d1 / (np.linalg.norm(d1) + 1e-8)
        d2 = d2 / (np.linalg.norm(d2) + 1e-8)

    dist = np.linalg.norm(d1 - d2)

    # # Convert distance to similarity (example using exponential decay)
    # similarity = np.exp(-dist / 100.0)  # Adjust the denominator as needed
    return dist


def compare_sift_descriptors_similarities(des1, des2):
    if des1 is None or des2 is None:
        return 0.0

    # SIFT descriptors must be float32 for NORM_L2
    # This line prevents the "Assertion failed" error
    d1 = des1.astype(np.float32)
    d2 = des2.astype(np.float32)

    # Use L2 (Euclidean) distance for SIFT
    # NORM_HAMMING is only for binary descriptors like ORB
    dist = cv2.norm(d1, d2, cv2.NORM_L2)

    # Convert distance to a 0-1 similarity score
    # SIFT distances for a match are usually < 200
    similarity = np.exp(-dist / 362.0)  # mid-point of range
    return similarity


def _draw_rect(ax, rt_start, im_start, rt_end, im_end, color, linestyle):
    """Draw a bbox rectangle on *ax*; axes use (x=IM col, y=RT row, origin=lower)."""
    import matplotlib.patches as mpatches

    rect = mpatches.Rectangle(
        (im_start, rt_start),
        im_end - im_start,
        rt_end - rt_start,
        edgecolor=color,
        facecolor="none",
        linestyle=linestyle,
        linewidth=1.5,
    )
    ax.add_patch(rect)


def _make_grid_fig(n: int, max_cols: int, extra_rows: int = 0):
    """Create a (n_rows + extra_rows) × max_cols subplot grid.

    Returns ``(fig, axes)`` where *axes* is always a 2-D ndarray of shape
    ``(n_rows + extra_rows, max_cols)``.
    """
    import math
    import matplotlib.pyplot as plt

    n_rows = math.ceil(n / max_cols) + extra_rows
    fig, axes = plt.subplots(n_rows, max_cols, figsize=(4 * max_cols, 4 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if max_cols == 1:
        axes = axes[:, np.newaxis]
    return fig, axes


def _save_or_show(fig, fig_dir, filename: str) -> None:
    """Save *fig* to *fig_dir*/*filename*, or show it interactively."""
    import matplotlib.pyplot as plt

    if fig_dir is not None:
        fig.savefig(os.path.join(fig_dir, filename), dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def _resize_image_to_shape(
    image: np.ndarray, target_shape: tuple[int, int]
) -> np.ndarray:
    rows, cols = int(target_shape[0]), int(target_shape[1])
    image_f32 = image.astype(np.float32)
    resized = cv2.resize(image_f32, (cols, rows), interpolation=cv2.INTER_LINEAR)
    return resized.astype(np.float64)


def _scale_anchor_to_target_shape(
    anchor: tuple[int, int] | None,
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
) -> tuple[float, float] | None:
    if anchor is None:
        return None
    scale_r = int(target_shape[0]) / int(source_shape[0])
    scale_c = int(target_shape[1]) / int(source_shape[1])
    return (float(anchor[0]) * scale_r, float(anchor[1]) * scale_c)


def _build_reference_template(
    reference_image: np.ndarray,
    template_anchor: tuple[int, int] | None,
    template_frac: float,
) -> tuple[int, int, tuple[int, int, int, int], np.ndarray]:
    rows, cols = reference_image.shape
    if template_anchor is None:
        anchor_row, anchor_col = np.unravel_index(
            np.argmax(reference_image), reference_image.shape
        )
    else:
        anchor_row, anchor_col = int(template_anchor[0]), int(template_anchor[1])
    template_rt_start = max(int(anchor_row - template_frac * rows), 0)
    template_rt_end = min(int(anchor_row + template_frac * rows), rows)
    template_im_start = max(int(anchor_col - template_frac * cols), 0)
    template_im_end = min(int(anchor_col + template_frac * cols), cols)
    template_bounds = (
        template_rt_start,
        template_im_start,
        template_rt_end,
        template_im_end,
    )
    template = reference_image[
        template_rt_start:template_rt_end, template_im_start:template_im_end
    ]
    return anchor_row, anchor_col, template_bounds, template


def _align_resized_image_to_template(
    resized_image: np.ndarray,
    template: np.ndarray,
    template_bounds: tuple[int, int, int, int],
    scaled_anchor: tuple[float, float] | None = None,
) -> tuple[
    np.ndarray,
    tuple[int, int, int, int],
    tuple[float, float] | None,
    tuple[int, int],
    float,
    np.ndarray,
    tuple[int, int],
]:
    from scipy.ndimage import shift as nd_shift

    template_rt_start, template_im_start, template_rt_end, template_im_end = (
        template_bounds
    )
    match_score = match_template(resized_image, template)
    max_score_index = np.unravel_index(np.argmax(match_score), match_score.shape)
    match_rt_topleft, match_im_topleft = max_score_index
    shift = (
        int(template_rt_start - match_rt_topleft),
        int(template_im_start - match_im_topleft),
    )
    aligned_image = nd_shift(resized_image, shift=shift, mode="constant", cval=0.0)
    aligned_anchor = (
        (float(scaled_anchor[0] + shift[0]), float(scaled_anchor[1] + shift[1]))
        if scaled_anchor is not None
        else None
    )
    return (
        aligned_image,
        (template_rt_start, template_im_start, template_rt_end, template_im_end),
        aligned_anchor,
        shift,
        float(match_score.max()),
        match_score,
        (int(match_rt_topleft), int(match_im_topleft)),
    )


def align_images_to_reference(
    images: list[np.ndarray],
    reference_idx: int = 0,
    target_shape: tuple[int, int] | None = None,
    template_anchor: tuple[int, int] | None = None,
    template_frac: float = 0.3,
    anchors: list[tuple[int, int] | None] | None = None,
    additional_anchors: list[list[tuple[int, int] | None]] | None = None,
    align_images: bool = True,
) -> ConsensusAlignmentState:
    """Resize and align images to a reference template for consensus scoring.

    If `template_anchor` is not given, it defaults to the centroid of all
    anchor points -- `anchors` plus every list in `additional_anchors` (e.g.
    one per confounder-group member, scaled into the reference frame) -- so
    the template is centred on the whole group rather than pinned to a
    single candidate's own anchor; falls back to the reference image's peak
    if no anchors are provided. `template_frac` is likewise widened (never
    narrowed) to the smallest fraction that still covers every anchor point
    around the resolved template anchor, capped at 0.5.
    """

    if not images:
        raise ValueError("images must contain at least one image.")
    if reference_idx < 0 or reference_idx >= len(images):
        raise ValueError(
            f"reference_idx {reference_idx} is out of range for {len(images)} images."
        )
    if anchors is not None and len(anchors) != len(images):
        raise ValueError(
            "anchors must have the same length as images "
            f"(got {len(anchors)}, expected {len(images)})."
        )
    if additional_anchors is not None:
        for _extra in additional_anchors:
            if len(_extra) != len(images):
                raise ValueError(
                    "each list in additional_anchors must have the same length "
                    f"as images (got {len(_extra)}, expected {len(images)})."
                )
    if not (0 < template_frac <= 0.5):
        raise ValueError(f"template_frac must be in (0, 0.5], got {template_frac}.")

    ref_image = images[reference_idx]
    resolved_target_shape = (
        ref_image.shape if target_shape is None else tuple(map(int, target_shape))
    )
    resized_images = [
        _resize_image_to_shape(image, resolved_target_shape) for image in images
    ]
    scaled_anchors = [
        (
            _scale_anchor_to_target_shape(
                anchors[i], images[i].shape, resolved_target_shape
            )
            if anchors is not None
            else None
        )
        for i in range(len(images))
    ]
    _template_anchor_pool = list(scaled_anchors)
    for _extra in additional_anchors or []:
        _template_anchor_pool.extend(
            _scale_anchor_to_target_shape(
                _extra[i], images[i].shape, resolved_target_shape
            )
            for i in range(len(images))
        )
    _valid_template_anchors = [a for a in _template_anchor_pool if a is not None]
    reference_resized = resized_images[reference_idx]
    resolved_template_anchor = template_anchor
    if resolved_template_anchor is None and _valid_template_anchors:
        resolved_template_anchor = (
            float(np.mean([a[0] for a in _valid_template_anchors])),
            float(np.mean([a[1] for a in _valid_template_anchors])),
        )
    resolved_template_frac = template_frac
    if resolved_template_anchor is not None and _valid_template_anchors:
        _rows, _cols = reference_resized.shape
        _needed_frac = max(
            (
                max(
                    abs(a[0] - resolved_template_anchor[0]) / _rows,
                    abs(a[1] - resolved_template_anchor[1]) / _cols,
                )
                for a in _valid_template_anchors
            ),
            default=0.0,
        )
        resolved_template_frac = min(max(template_frac, _needed_frac), 0.5)
    anchor_row, anchor_col, template_bounds, template = _build_reference_template(
        reference_resized,
        resolved_template_anchor,
        resolved_template_frac,
    )

    aligned_images: list[np.ndarray] = []
    matched_boxes: list[tuple[int, int, int, int]] = []
    aligned_anchors: list[tuple[float, float] | None] = []
    shifts: list[tuple[int, int]] = []
    max_scores: list[float] = []
    match_score_maps: list[np.ndarray] = []
    match_score_peaks: list[tuple[int, int]] = []
    match_score_label_indices: list[int] = []

    for i, resized_image in enumerate(resized_images):
        if i == reference_idx:
            aligned_images.append(resized_image.copy())
            matched_boxes.append(template_bounds)
            aligned_anchors.append(scaled_anchors[i])
            shifts.append((0, 0))
            max_scores.append(1.0)
            continue
        if not align_images:
            aligned_images.append(resized_image.copy())
            matched_boxes.append(template_bounds)
            aligned_anchors.append(scaled_anchors[i])
            shifts.append((0, 0))
            max_scores.append(0.0)
            continue
        (
            aligned_image,
            matched_box,
            aligned_anchor,
            shift,
            max_score,
            match_score_map,
            match_score_peak,
        ) = _align_resized_image_to_template(
            resized_image,
            template,
            template_bounds,
            scaled_anchors[i],
        )
        aligned_images.append(aligned_image)
        matched_boxes.append(matched_box)
        aligned_anchors.append(aligned_anchor)
        shifts.append(shift)
        max_scores.append(max_score)
        match_score_maps.append(match_score_map)
        match_score_peaks.append(match_score_peak)
        match_score_label_indices.append(i)

    return ConsensusAlignmentState(
        reference_idx=reference_idx,
        target_shape=resolved_target_shape,
        anchor_row=anchor_row,
        anchor_col=anchor_col,
        template_bounds=template_bounds,
        template=template,
        resized_images=resized_images,
        aligned_images=aligned_images,
        matched_boxes=matched_boxes,
        aligned_anchors=aligned_anchors,
        scaled_anchors=scaled_anchors,
        shifts=shifts,
        max_scores=max_scores,
        match_score_maps=match_score_maps,
        match_score_peaks=match_score_peaks,
        match_score_label_indices=match_score_label_indices,
    )


def _snap_anchor_to_watershed_label(
    r: int,
    c: int,
    watershed_labels: np.ndarray,
    all_peaks: np.ndarray,
    labeled_coords: np.ndarray,
    jump_dist_thres: tuple[int, int],
) -> tuple[tuple[int, int] | None, int | None, dict[str, Any] | None]:
    """
    Snap a single (r, c) anchor onto the nearest peak within its watershed
    label. Pure decision logic shared by the main per-run anchor loop in
    segment_consensus_from_aligned and by coSWA's per-confounder-group-member
    snapping (which reuses one group's already-computed watershed_labels/
    all_peaks instead of resegmenting per candidate).

    Returns (snapped_rc, label_id, jump_info):
      - Anchor inside a labeled region: snaps to the nearest peak within that
        label. jump_info is None.
      - Anchor in background: jumps to the nearest labeled pixel, then snaps
        to that label's nearest peak. jump_info carries the jump details
        (nearest_labeled_pixel, rt_dist, im_dist, dist_to_label[,
        jumped_label, snapped_peak]) for the caller to log.
      - Discarded (background jump distance exceeds jump_dist_thres, or no
        labeled pixels exist at all): snapped_rc and label_id are None.
    """
    anchor_ws = int(watershed_labels[r, c])
    if anchor_ws > 0:
        # Anchor is inside a labeled region — snap to nearest peak in that label.
        # The watershed invariant guarantees every label has at least one peak.
        same_ws_peaks = all_peaks[
            watershed_labels[all_peaks[:, 0], all_peaks[:, 1]] == anchor_ws
        ]
        dists = np.hypot(same_ws_peaks[:, 0] - r, same_ws_peaks[:, 1] - c)
        nearest = same_ws_peaks[int(np.argmin(dists))]
        snapped_rc = (int(nearest[0]), int(nearest[1]))
        return snapped_rc, anchor_ws, None

    if labeled_coords.shape[0] == 0:
        return None, None, None

    # Anchor is in background — jump to the nearest labeled region.
    dists = np.hypot(labeled_coords[:, 0] - r, labeled_coords[:, 1] - c)
    nearest_idx = int(np.argmin(dists))
    nearest_labeled_rc = labeled_coords[nearest_idx]
    rt_dist = abs(r - int(nearest_labeled_rc[0]))
    im_dist = abs(c - int(nearest_labeled_rc[1]))
    jump_info: dict[str, Any] = {
        "nearest_labeled_pixel": (
            int(nearest_labeled_rc[0]),
            int(nearest_labeled_rc[1]),
        ),
        "rt_dist": rt_dist,
        "im_dist": im_dist,
        "dist_to_label": float(dists[nearest_idx]),
    }
    if (jump_dist_thres[0] > 0 and rt_dist > jump_dist_thres[0]) or (
        jump_dist_thres[1] > 0 and im_dist > jump_dist_thres[1]
    ):
        return None, None, jump_info

    jump_ws = int(watershed_labels[nearest_labeled_rc[0], nearest_labeled_rc[1]])
    same_ws_peaks = all_peaks[
        watershed_labels[all_peaks[:, 0], all_peaks[:, 1]] == jump_ws
    ]
    dists_peak = np.hypot(same_ws_peaks[:, 0] - r, same_ws_peaks[:, 1] - c)
    nearest_peak = same_ws_peaks[int(np.argmin(dists_peak))]
    snapped_rc = (int(nearest_peak[0]), int(nearest_peak[1]))
    jump_info["jumped_label"] = jump_ws
    jump_info["snapped_peak"] = snapped_rc
    return snapped_rc, jump_ws, jump_info


def _project_anchor_into_aligned_space(
    anchor: tuple[int, int] | None,
    source_shape: tuple[int, int],
    alignment_state: "ConsensusAlignmentState",
    run_position_i: int,
) -> tuple[float, float] | None:
    """
    Project a raw (row, col) anchor for run `run_position_i` into the same
    aligned pixel space as alignment_state.aligned_anchors, by applying the
    identical scale-then-shift transform align_images_to_reference already
    computed for that run (alignment_state.shifts[run_position_i] is (0, 0)
    for the reference run and whenever align_images=False, so this formula
    is valid uniformly).

    Used by coSWA to re-project OTHER confounder-group members' own anchors
    into a representative member's already-computed alignment, without
    recomputing image resize/registration for every member.
    """
    scaled = _scale_anchor_to_target_shape(
        anchor, source_shape, alignment_state.target_shape
    )
    if scaled is None:
        return None
    shift = alignment_state.shifts[run_position_i]
    return (scaled[0] + shift[0], scaled[1] + shift[1])


def _snap_all_anchors_to_watershed(
    alignment_state: ConsensusAlignmentState,
    consensus: np.ndarray,
    consensus_denoised: np.ndarray,
    watershed_labels: np.ndarray,
    all_peaks: np.ndarray,
    apply_seg: bool,
    seg_mask_thres: tuple[int, int],
    jump_dist_thres: tuple[int, int],
    collapse_to_single_label: bool = False,
    priority_anchor_index: int | None = None,
) -> ConsensusSegmentationState:
    """
    Snap every non-None anchor in alignment_state onto the given watershed
    segmentation (watershed_labels/all_peaks, already computed over
    consensus_denoised), with bbox-fallback if segmentation is unusable or
    yields too small a target-label span.

    collapse_to_single_label (coSWA per-member assignment): when True and the
    candidate's anchors snap to more than one watershed label, keep only ONE
    label -- the majority vote over the per-anchor snapped labels, tie-broken
    by the label of priority_anchor_index (the candidate's reference-run
    anchor). Anchors on the losing labels are moved to discard_record so the
    snap overlay still renders them (as discarded 'x'). This is the sole quant
    differentiator between confounder-group members, which share one activation
    trace and hence one intensity per segment.

    Factored out of segment_consensus_from_aligned so coSWA can reuse one
    confounder group's already-computed watershed_labels/all_peaks/consensus/
    consensus_denoised to cheaply snap OTHER group members' own anchors
    (fresh alignment_state, same underlying merged-activation image and
    segmentation) without recomputing the expensive watershed detection.

    watershed_labels may be a shared/cached array reused across multiple
    calls (one per confounder-group member) -- the bbox-fallback branch
    below mutates it via slice assignment, so copy defensively rather than
    risk corrupting a caller's cached segmentation.
    """
    watershed_labels = watershed_labels.copy()
    rows, cols = alignment_state.target_shape
    non_none_indices = [
        i for i, aa in enumerate(alignment_state.aligned_anchors) if aa is not None
    ]
    snapped_per_anchor: list[tuple[int, int] | None] = [None] * len(
        alignment_state.aligned_anchors
    )
    snap_log: dict[str, Any] = {
        "snap_record": {},
        "discard_record": {},
        "no_seg_log": None,
        "jump_anchor_log": {},
    }
    target_label_ids: list[int] = []
    seen_label_ids: set[int] = set()
    label_to_snap: dict[int, tuple[int, int]] = {}
    anchor_to_label: dict[int, int] = {}
    use_bbox_fallback = not apply_seg
    if non_none_indices:
        use_bbox_fallback = (
            (not apply_seg) or all_peaks.shape[0] == 0 or watershed_labels.max() == 0
        )

        if not use_bbox_fallback:
            # Normal case: peaks and watershed labels detected successfully.
            # Hoist once: used by any anchor that falls in background (anchor_ws == 0)
            labeled_coords = np.argwhere(watershed_labels > 0)
            for i in non_none_indices:
                aa = alignment_state.aligned_anchors[i]
                assert aa is not None
                r = int(np.clip(round(aa[0]), 0, rows - 1))
                c = int(np.clip(round(aa[1]), 0, cols - 1))
                snapped_rc, label_id, jump_info = _snap_anchor_to_watershed_label(
                    r, c, watershed_labels, all_peaks, labeled_coords, jump_dist_thres
                )
                if label_id is None:
                    if jump_info is not None:
                        snap_log["discard_record"][i] = {"anchor": (r, c), **jump_info}
                    continue
                snapped_per_anchor[i] = snapped_rc
                anchor_to_label[i] = label_id
                if jump_info is None:
                    snap_log["snap_record"][i] = ((r, c), snapped_rc)
                else:
                    snap_log["jump_anchor_log"][i] = {"anchor": (r, c), **jump_info}
                    snap_log["snap_record"][i] = ((r, c), snapped_rc)
                if label_id not in seen_label_ids:
                    target_label_ids.append(label_id)
                    seen_label_ids.add(label_id)
                    label_to_snap[label_id] = snapped_rc

            # coSWA: collapse a member's multi-label snap to ONE segment
            # (majority vote, tie-broken by the reference-run anchor).
            if collapse_to_single_label and len(target_label_ids) > 1:
                _counts: dict[int, int] = {}
                for _lid in anchor_to_label.values():
                    _counts[_lid] = _counts.get(_lid, 0) + 1
                _max_count = max(_counts.values())
                _top = [_lid for _lid, _c in _counts.items() if _c == _max_count]
                _winner = None
                if priority_anchor_index is not None:
                    _prio = anchor_to_label.get(priority_anchor_index)
                    if _prio in _top:
                        _winner = _prio
                if _winner is None:
                    _winner = min(_top)
                for _i, _lid in list(anchor_to_label.items()):
                    if _lid != _winner:
                        snapped_per_anchor[_i] = None
                        _rec = snap_log["snap_record"].pop(_i, None)
                        snap_log["jump_anchor_log"].pop(_i, None)
                        snap_log["discard_record"][_i] = {
                            "anchor": _rec[0] if _rec is not None else None,
                            "collapsed_to_label": _winner,
                        }
                target_label_ids = [_winner]
                seen_label_ids = {_winner}
                label_to_snap = {_winner: label_to_snap[_winner]}

            # Roll back to bbox if target-label span is below (rt, im) thresholds.
            if any(t > 0 for t in seg_mask_thres) and target_label_ids:
                _target_mask = np.isin(watershed_labels, target_label_ids)
                _rt_span = int(np.any(_target_mask, axis=1).sum())
                _im_span = int(np.any(_target_mask, axis=0).sum())
                if _rt_span < seg_mask_thres[0] or _im_span < seg_mask_thres[1]:
                    use_bbox_fallback = True
                    watershed_labels = np.zeros(consensus_denoised.shape, dtype=int)
                    all_peaks = np.empty((0, 2), dtype=int)
                    snapped_per_anchor = [None] * len(alignment_state.aligned_anchors)
                    snap_log = {
                        "snap_record": {},
                        "discard_record": snap_log["discard_record"],
                        "no_seg_log": None,
                        "jump_anchor_log": {},
                    }
                    target_label_ids.clear()
                    seen_label_ids.clear()
                    label_to_snap.clear()

        if use_bbox_fallback:
            # No usable watershed — fallback to snapping anchors to a bbox around their mean position.
            _anchor_rs = [
                int(np.clip(round(_aa[0]), 0, rows - 1))
                for i in non_none_indices
                for _aa in (alignment_state.aligned_anchors[i],)
                if _aa is not None
            ]
            _anchor_cs = [
                int(np.clip(round(_aa[1]), 0, cols - 1))
                for i in non_none_indices
                for _aa in (alignment_state.aligned_anchors[i],)
                if _aa is not None
            ]
            _ctr_r = int(round(float(np.mean(_anchor_rs))))
            _ctr_c = int(round(float(np.mean(_anchor_cs))))
            _rt_start = max(int(_ctr_r - 0.3 * rows), 0)
            _rt_end = min(int(_ctr_r + 0.3 * rows), rows)
            _im_start = max(int(_ctr_c - 0.3 * cols), 0)
            _im_end = min(int(_ctr_c + 0.3 * cols), cols)
            watershed_labels[_rt_start:_rt_end, _im_start:_im_end] = (
                consensus[_rt_start:_rt_end, _im_start:_im_end] > 0
            ).astype(int)
            snap_log["no_seg_log"] = {
                "anchor_positions": list(zip(_anchor_rs, _anchor_cs)),
                "rect": (_rt_start, _im_start, _rt_end, _im_end),
            }
            for i, (r_i, c_i) in zip(non_none_indices, zip(_anchor_rs, _anchor_cs)):
                snapped_per_anchor[i] = (r_i, c_i)
                snap_log["snap_record"][i] = ((r_i, c_i), (r_i, c_i))
            if watershed_labels.max() > 0:
                target_label_ids.append(1)
                seen_label_ids.add(1)
                label_to_snap[1] = (_anchor_rs[0], _anchor_cs[0])
            # if we're using bbox fallback, skip the denoising that was tuned for watershed-based peaks
    return ConsensusSegmentationState(
        consensus=consensus,  # stacked mean of aligned_images
        consensus_denoised=consensus_denoised,  # denoised with consensus_denoise_kwar
        snapped_per_anchor=snapped_per_anchor,
        watershed_labels=watershed_labels,
        snap_log=snap_log,
        target_label_ids=target_label_ids,
        label_to_snap=label_to_snap,
        non_none_indices=non_none_indices,
        apply_seg=not use_bbox_fallback,
        all_peaks=all_peaks,
    )


def _project_member_mask_to_common_run(
    member: dict, common_run: str, common_run_position: int
) -> set[tuple[int, int]]:
    """Project one coSWA group member's own assigned-segment mask into
    `common_run`'s absolute (frame_idx, mobility_index) coordinates.

    The member's own aligned/consensus space differs only from
    `common_run`'s own raw crop window by that run's per-run alignment shift
    (`member["alignment"].shifts[common_run_position]`); adding back the
    window's own absolute origin (`member["window_origin_by_run"][common_run]`)
    lands the mask in `common_run`'s real coordinate grid, directly
    comparable across members regardless of which run each one used as its
    own alignment reference.
    """
    seg = member["segmentation"]
    if not seg.target_label_ids:
        return set()
    mask = np.isin(seg.watershed_labels, seg.target_label_ids)
    rows, cols = np.where(mask)
    if rows.size == 0:
        return set()
    shift = member["alignment"].shifts[common_run_position]
    origin = member["window_origin_by_run"][common_run]
    abs_rows = rows - shift[0] + origin[0]
    abs_cols = cols - shift[1] + origin[1]
    return set(zip(abs_rows.tolist(), abs_cols.tolist()))


def _mark_overlapping_group_members(
    members_by_group: dict[int, list[int]],
    member_cache: dict[int, dict],
) -> dict[int, str]:
    """Flag coSWA group members whose own independently-computed assigned
    segments spatially overlap.

    Each member in `member_cache` was aligned + segmented fully
    independently (own roles, own anchors, own window). This projects every
    member's own assigned-segment mask into one common run's absolute
    coordinates (the group representative's -- i.e. min(mz_rank) --
    reference run, fixed once per group for consistency across all pairwise
    comparisons) and flags pairs whose projected pixel sets intersect.
    Overlapping members within a group are connected-component-grouped and
    given a shared tag, mirroring the old `undistinguishable_group_id`
    convention.
    """
    tags: dict[int, str] = {}
    for gid, members in members_by_group.items():
        present = [m for m in members if m in member_cache]
        if len(present) < 2:
            continue
        common_run = member_cache[min(present)]["reference_raw_file"]
        projected: dict[int, set[tuple[int, int]]] = {}
        for m in present:
            stack = member_cache[m]["consensus_raw_files"]
            if common_run not in stack:
                continue  # shouldn't happen: every member's own stack spans all runs
            projected[m] = _project_member_mask_to_common_run(
                member_cache[m], common_run, stack.index(common_run)
            )
        adj: dict[int, set[int]] = {m: set() for m in projected}
        keys = list(projected)
        for i, m1 in enumerate(keys):
            for m2 in keys[i + 1 :]:
                if projected[m1] & projected[m2]:
                    adj[m1].add(m2)
                    adj[m2].add(m1)
        visited: set[int] = set()
        counter = 0
        for m in keys:
            if m in visited:
                continue
            comp: list[int] = []
            stack_: list[int] = [m]
            visited.add(m)
            while stack_:
                cur = stack_.pop()
                comp.append(cur)
                for nb in adj[cur]:
                    if nb not in visited:
                        visited.add(nb)
                        stack_.append(nb)
            if len(comp) >= 2:
                tag = f"{gid}_{counter}"
                counter += 1
                for mm in comp:
                    tags[mm] = tag
    return tags


def segment_consensus_from_aligned(
    alignment_state: ConsensusAlignmentState,
    denoise_kwargs: dict | None = None,
    watershed_kwargs: dict | None = None,
    apply_seg: bool = True,
    seg_mask_thres: tuple[int, int] = (2, 5),
    jump_dist_thres: tuple[int, int] = (0, 0),
    consensus_image_indices: list[int] | None = None,
) -> ConsensusSegmentationState:
    """Segment a consensus image and track which labels belong to target anchors."""

    seg_mask_thres = _parse_seg_mask_thres(seg_mask_thres)
    jump_dist_thres = _parse_jump_dist_thres(jump_dist_thres)
    _imgs_for_consensus = (
        [alignment_state.aligned_images[i] for i in consensus_image_indices]
        if consensus_image_indices is not None
        else alignment_state.aligned_images
    )
    consensus = np.stack(_imgs_for_consensus, axis=0).mean(axis=0)
    consensus_denoised = smooth_and_denoise_image(consensus, **(denoise_kwargs or {}))
    watershed_labels: np.ndarray = np.zeros(consensus_denoised.shape, dtype=int)
    all_peaks: np.ndarray = np.empty((0, 2), dtype=int)
    non_none_indices = [
        i for i, aa in enumerate(alignment_state.aligned_anchors) if aa is not None
    ]
    if non_none_indices and apply_seg:
        _wkwargs = dict(watershed_kwargs or {})
        all_peaks, _unused_labels, _, watershed_labels, _ = detect_2d_peak_with_watershed(
            consensus_denoised,
            int_threshold=_wkwargs.get("int_threshold", 0.5),
            h_rel=_wkwargs.get("h_rel", 0.15),
            norm_percentile=_wkwargs.get("norm_percentile", 95),
            compactness=_wkwargs.get("compactness", 0.001),
        )
    return _snap_all_anchors_to_watershed(
        alignment_state,
        consensus,
        consensus_denoised,
        watershed_labels,
        all_peaks,
        apply_seg,
        seg_mask_thres,
        jump_dist_thres,
    )


def _align_raw_images_with_shifts(
    raw_images: list[np.ndarray],
    target_shape: tuple[int, int],
    shifts: list[tuple[int, int]],
) -> list[np.ndarray]:
    from scipy.ndimage import shift as nd_shift

    raw_aligned: list[np.ndarray] = []
    for i, raw_image in enumerate(raw_images):
        raw_resized = _resize_image_to_shape(raw_image, target_shape)
        if shifts[i] == (0, 0):
            raw_aligned.append(raw_resized)
        else:
            raw_aligned.append(
                nd_shift(raw_resized, shift=shifts[i], mode="constant", cval=0.0)
            )
    return raw_aligned


def _extract_feature_rows_for_label_ids(
    label_ids: list[int],
    label_image: np.ndarray,
    raw_image: np.ndarray,
    denoised_image: np.ndarray,
    *,
    run_name: str,
    shift: tuple[int, int],
    template_matching_score: float,
    snap_resolver: Callable[[int], tuple[int, int] | None],
) -> pd.DataFrame | None:
    # Pick the dominant label by area in label_image (deterministic across runs
    # sharing the same segmentation).
    dominant_label_id = max(label_ids, key=lambda lid: int((label_image == lid).sum()))
    snap_rc = snap_resolver(dominant_label_id)
    if snap_rc is None:
        return None

    merged_mask = np.isin(label_image, label_ids).astype(np.int32)
    peak_properties = calculate_peak_property_from_labels_and_image(
        merged_mask,
        raw_image,
        min_peak_area=0,
        min_peak_sum_intensity=0,
    )
    if peak_properties is None:
        return None

    peak_properties = peak_properties.reset_index(drop=True)
    peak_properties["snap_rt"] = int(snap_rc[0])
    peak_properties["snap_im"] = int(snap_rc[1])
    peak_properties["shift_rt"] = int(shift[0])
    peak_properties["shift_im"] = int(shift[1])
    peak_properties["template_matching_score"] = float(template_matching_score)
    peak_properties["sift_des"] = None
    peak_properties.at[0, "sift_des"] = get_sift_descriptor(
        denoised_image,
        (int(snap_rc[0]), int(snap_rc[1])),
        patch_size=int(0.6 * min(denoised_image.shape)),
    )
    _r0 = int(peak_properties["bbox-0"].values[0])
    _r1 = int(peak_properties["bbox-2"].values[0])
    _c0 = int(peak_properties["bbox-1"].values[0])
    _c1 = int(peak_properties["bbox-3"].values[0])
    _min_side = 18
    _H, _W = denoised_image.shape
    if _r1 - _r0 < _min_side:
        _pad = (_min_side - (_r1 - _r0) + 1) // 2
        _r0, _r1 = max(0, _r0 - _pad), min(_H, _r1 + _pad)
    if _c1 - _c0 < _min_side:
        _pad = (_min_side - (_c1 - _c0) + 1) // 2
        _c0, _c1 = max(0, _c0 - _pad), min(_W, _c1 + _pad)
    seg_bbox = denoised_image[_r0:_r1, _c0:_c1]
    peak_properties["zernike"] = None
    peak_properties.at[0, "zernike"] = get_roi_descriptor(seg_bbox)
    peak_properties["Run_name"] = run_name
    return peak_properties


def extract_peak_properties_from_consensus_labels(
    alignment_state: ConsensusAlignmentState,
    segmentation_state: ConsensusSegmentationState,
    *,
    raw_images: list[np.ndarray] | None = None,
    labels: list[str] | None = None,
) -> tuple[
    pd.DataFrame | None,
    list[pd.DataFrame | None],
    list[np.ndarray],
    list[np.ndarray],
    np.ndarray | None,
    np.ndarray | None,
]:
    """Extract per-run and consensus peak properties from consensus labels."""

    individual_pps: list[pd.DataFrame | None] = [None] * len(
        alignment_state.aligned_images
    )
    if (
        raw_images is None
        or not segmentation_state.non_none_indices
        or segmentation_state.watershed_labels.max() == 0
    ):
        return None, individual_pps, [], [], None, None

    raw_aligned = _align_raw_images_with_shifts(
        raw_images,
        alignment_state.target_shape,
        alignment_state.shifts,
    )  # no raw denoise kwargs

    raw_consensus = segmentation_state.consensus  # with raw denoise kwargs
    raw_aligned_logged = alignment_state.aligned_images
    raw_consensus_logged_mean = raw_consensus
    consensus_pp: pd.DataFrame | None = None
    if segmentation_state.target_label_ids:
        for i in range(len(raw_aligned)):
            run_name = labels[i] if (labels is not None and i < len(labels)) else str(i)
            individual_pps[i] = _extract_feature_rows_for_label_ids(
                segmentation_state.target_label_ids,
                segmentation_state.watershed_labels,
                raw_aligned[i],
                raw_aligned_logged[i],
                run_name=run_name,
                shift=alignment_state.shifts[i],
                template_matching_score=alignment_state.max_scores[i],
                snap_resolver=lambda label_id, i=i: (
                    segmentation_state.snapped_per_anchor[i]
                    if segmentation_state.snapped_per_anchor[i] is not None
                    else segmentation_state.label_to_snap.get(label_id)
                ),
            )
        consensus_pp = _extract_feature_rows_for_label_ids(
            segmentation_state.target_label_ids,
            segmentation_state.watershed_labels,
            raw_consensus,
            raw_consensus_logged_mean,
            run_name="consensus",
            shift=(0, 0),
            template_matching_score=1.0,
            snap_resolver=lambda label_id: segmentation_state.label_to_snap.get(
                label_id
            ),
        )

    return (
        consensus_pp,
        individual_pps,
        raw_aligned,
        raw_aligned_logged,
        raw_consensus,
        raw_consensus_logged_mean,
    )


def _reuse_alignment_with_new_anchors(
    cached: ConsensusAlignmentState,
    anchors: list[tuple[int, int] | None],
    source_shapes: list[tuple[int, int]],
) -> ConsensusAlignmentState:
    """
    Build a new ConsensusAlignmentState that reuses a cached alignment's
    expensive fields (resized/aligned images, per-run shifts, template match
    scores) verbatim, only recomputing aligned_anchors/scaled_anchors for a
    NEW set of raw anchors.

    Valid whenever the underlying raw images are the same as the ones the
    cached alignment was built from (true for coSWA confounder-group members,
    whose per-run activation images are identical by construction -- see
    inference.collapse_candidates_by_confounder_group /
    helper.expand_group_ids_to_members) -- the optimal image-to-image
    registration (shifts) doesn't depend on which candidate's anchor is
    being projected, only on the images themselves.
    """
    n = len(cached.aligned_images)
    assert len(anchors) == n and len(source_shapes) == n
    scaled_anchors = [
        _scale_anchor_to_target_shape(anchors[i], source_shapes[i], cached.target_shape)
        for i in range(n)
    ]
    aligned_anchors = [
        (
            (scaled_anchors[i][0] + cached.shifts[i][0], scaled_anchors[i][1] + cached.shifts[i][1])
            if scaled_anchors[i] is not None
            else None
        )
        for i in range(n)
    ]
    return replace(cached, aligned_anchors=aligned_anchors, scaled_anchors=scaled_anchors)


def build_consensus_feature_bundle(
    images: list[np.ndarray],
    reference_idx: int = 0,
    target_shape: tuple[int, int] | None = None,
    template_anchor: tuple[int, int] | None = None,
    template_frac: float = 0.3,
    anchors: list[tuple[int, int] | None] | None = None,
    additional_anchors: list[list[tuple[int, int] | None]] | None = None,
    denoise_cfg: dict | None = None,
    watershed_kwargs: dict | None = None,
    labels: list[str] | None = None,
    raw_images: list[np.ndarray] | None = None,
    apply_seg: bool = True,
    seg_mask_thres: tuple[int, int] = (3, 3),
    jump_dist_thres: tuple[int, int] = (0, 0),
    consensus_image_indices: list[int] | None = None,
    align_images: bool = True,
    reuse_from: ConsensusFeatureBundle | None = None,
    collapse_to_single_label: bool = False,
    priority_anchor_index: int | None = None,
    precomputed_states: (
        tuple[ConsensusAlignmentState, ConsensusSegmentationState] | None
    ) = None,
) -> ConsensusFeatureBundle:
    """Build alignment, segmentation, and feature tables for consensus scoring.

    If reuse_from is given (a bundle already built for another candidate that
    shares identical per-run raw images -- a coSWA confounder-group
    representative), skip the expensive image alignment (SIFT/template
    matching) and watershed segmentation entirely and reuse them, only
    recomputing the cheap per-anchor snap + peak-property extraction for this
    candidate's own `anchors`. `images` is ignored in that case (the
    representative's own aligned_images are reused); `raw_images` is still
    required as usual (identical content to the representative's, but still
    passed explicitly to keep this function's contract uniform).

    `additional_anchors` (e.g. every other confounder-group member's own
    per-run anchor list) only widens the template placement -- its centroid
    and required coverage radius -- and is otherwise ignored; it does not
    affect `anchors`, which remains this call's own per-run anchor points.

    If precomputed_states is given instead (a coSWA group member's own
    alignment + segmentation, already produced by an earlier reuse_from-style
    call for this exact candidate -- see match_features_batch's confounder-
    group pre-pass), skip straight to peak-property extraction: `images`,
    `anchors`, `reuse_from`, and every alignment/segmentation kwarg are
    ignored. `raw_images` is still required as usual.
    """

    _n_runs = (
        len(images)
        if reuse_from is None and precomputed_states is None
        else len(raw_images or [])
    )
    if labels is not None and len(labels) != _n_runs:
        raise ValueError(
            "labels must have the same length as images "
            f"(got {len(labels)}, expected {_n_runs})."
        )
    _denoise_cfg = denoise_cfg or {}
    _consensus_denoise_kwargs = _denoise_kwargs_for_stage(_denoise_cfg, "consensus")
    _full_denoise_kwargs = _denoise_kwargs_all(_denoise_cfg)
    if precomputed_states is not None:
        alignment_state, segmentation_state = precomputed_states
    elif reuse_from is None:
        alignment_state = align_images_to_reference(
            images=images,
            reference_idx=reference_idx,
            target_shape=target_shape,
            template_anchor=template_anchor,
            template_frac=template_frac,
            anchors=anchors,
            additional_anchors=additional_anchors,
            align_images=align_images,
        )
        segmentation_state = segment_consensus_from_aligned(
            alignment_state,
            denoise_kwargs=_consensus_denoise_kwargs,
            watershed_kwargs=watershed_kwargs,
            apply_seg=apply_seg,
            seg_mask_thres=seg_mask_thres,
            jump_dist_thres=jump_dist_thres,
            consensus_image_indices=consensus_image_indices,
        )
    else:
        source_shapes = [img.shape for img in (raw_images or [])]
        alignment_state = _reuse_alignment_with_new_anchors(
            reuse_from.alignment, anchors or [], source_shapes
        )
        segmentation_state = _snap_all_anchors_to_watershed(
            alignment_state,
            reuse_from.segmentation.consensus,
            reuse_from.segmentation.consensus_denoised,
            reuse_from.segmentation.watershed_labels,
            reuse_from.segmentation.all_peaks,
            apply_seg,
            _parse_seg_mask_thres(seg_mask_thres),
            _parse_jump_dist_thres(jump_dist_thres),
            collapse_to_single_label=collapse_to_single_label,
            priority_anchor_index=priority_anchor_index,
        )
    (
        consensus_pp,
        individual_pps,
        raw_aligned,
        raw_aligned_denoised,
        raw_consensus,
        raw_consensus_denoised,
    ) = extract_peak_properties_from_consensus_labels(
        alignment_state,
        segmentation_state,
        raw_images=raw_images,
        labels=labels,
    )
    return ConsensusFeatureBundle(
        alignment=alignment_state,
        segmentation=segmentation_state,
        consensus_pp=consensus_pp,
        individual_pps=individual_pps,
        raw_aligned_images=raw_aligned,
        raw_aligned_denoised_images=raw_aligned_denoised,
        raw_consensus=raw_consensus,
        raw_consensus_denoised=raw_consensus_denoised,
    )


def _make_shifted_consensus_segmentation_state(
    segmentation_state: ConsensusSegmentationState,
    label_shift: tuple[int, int],
) -> ConsensusSegmentationState:
    """Build a display-ready consensus segmentation state with shifted labels."""

    shifted_labels = _shift_integer_label_image(
        segmentation_state.watershed_labels, label_shift
    )
    shifted_label_to_snap: dict[int, tuple[int, int]] = {}
    for label_id, snap_rc in segmentation_state.label_to_snap.items():
        shifted_snap = _resolve_shifted_label_snap(
            shifted_labels,
            label_id,
            (int(snap_rc[0] + label_shift[0]), int(snap_rc[1] + label_shift[1])),
        )
        if shifted_snap is not None:
            shifted_label_to_snap[int(label_id)] = shifted_snap

    shifted_snapped_per_anchor: list[tuple[int, int] | None] = []
    shifted_snap_record: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {}
    shifted_discard_record = dict(segmentation_state.snap_log.get("discard_record", {}))
    for idx, old_snap in enumerate(segmentation_state.snapped_per_anchor):
        if old_snap is None:
            shifted_snapped_per_anchor.append(None)
            continue
        label_id = int(segmentation_state.watershed_labels[old_snap[0], old_snap[1]])
        shifted_snap = (
            _resolve_shifted_label_snap(
                shifted_labels,
                label_id,
                (
                    int(old_snap[0] + label_shift[0]),
                    int(old_snap[1] + label_shift[1]),
                ),
            )
            if label_id > 0
            else None
        )
        shifted_snapped_per_anchor.append(shifted_snap)
        if shifted_snap is not None and idx in segmentation_state.snap_log.get(
            "snap_record", {}
        ):
            orig_rc, _old_snap = segmentation_state.snap_log["snap_record"][idx]
            shifted_snap_record[idx] = (orig_rc, shifted_snap)

    shifted_snap_log = {
        "snap_record": shifted_snap_record,
        "discard_record": shifted_discard_record,
        "no_seg_log": segmentation_state.snap_log.get("no_seg_log"),
        "jump_anchor_log": segmentation_state.snap_log.get("jump_anchor_log"),
        "label_shift": (int(label_shift[0]), int(label_shift[1])),
    }
    return ConsensusSegmentationState(
        consensus=segmentation_state.consensus,
        consensus_denoised=segmentation_state.consensus_denoised,
        snapped_per_anchor=shifted_snapped_per_anchor,
        watershed_labels=shifted_labels,
        snap_log=shifted_snap_log,
        target_label_ids=list(segmentation_state.target_label_ids),
        label_to_snap=shifted_label_to_snap,
        non_none_indices=list(segmentation_state.non_none_indices),
        apply_seg=segmentation_state.apply_seg,
    )


def _visualize_consensus_bundle(
    alignment_state: ConsensusAlignmentState,
    segmentation_state: ConsensusSegmentationState,
    *,
    fig_dir: str | None,
    filename: str,
    labels: list[str] | None = None,
    aligned_images: list[np.ndarray] | None = None,
    consensus: np.ndarray | None = None,
    consensus_denoised: np.ndarray | None = None,
) -> None:
    """Visualize aligned images plus consensus panels for targets or decoys."""

    import math
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    display_aligned = (
        alignment_state.aligned_images if aligned_images is None else aligned_images
    )
    display_consensus = segmentation_state.consensus if consensus is None else consensus
    display_consensus_denoised = (
        segmentation_state.consensus_denoised
        if consensus_denoised is None
        else consensus_denoised
    )

    max_cols = 5
    n = len(display_aligned)
    vmin = min(img.min() for img in display_aligned)
    vmax = max(img.max() for img in display_aligned)
    snap_record = segmentation_state.snap_log["snap_record"]
    discard_record = segmentation_state.snap_log["discard_record"]
    non_none_indices = segmentation_state.non_none_indices
    _anchor_color_map: dict[int, Any] = {
        i_idx: plt.cm.tab10(k % 10) for k, i_idx in enumerate(non_none_indices)
    }

    fig, axes = _make_grid_fig(n, max_cols, extra_rows=1)
    _watershed_overlay: np.ma.MaskedArray | None = None
    if segmentation_state.watershed_labels.max() > 0:
        _target_label_mask = np.zeros_like(
            segmentation_state.watershed_labels, dtype=float
        )
        for _snap in segmentation_state.snapped_per_anchor:
            if _snap is None:
                continue
            _lid = int(segmentation_state.watershed_labels[_snap[0], _snap[1]])
            if _lid > 0:
                _target_label_mask[segmentation_state.watershed_labels == _lid] = float(
                    _lid
                )
        if _target_label_mask.max() > 0:
            _watershed_overlay = np.ma.masked_where(
                _target_label_mask == 0, _target_label_mask
            )

    for i, img in enumerate(display_aligned):
        row, col = divmod(i, max_cols)
        ax = axes[row, col]
        title = labels[i] if labels is not None else f"Image {i}"
        if i == alignment_state.reference_idx:
            title += "\n(reference)"
        ax.imshow(img, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
        if _watershed_overlay is not None:
            white_cmap = ListedColormap(["white"])
            ax.imshow(
                _watershed_overlay,
                aspect="auto",
                origin="lower",
                cmap=white_cmap,
                alpha=0.3,
                interpolation="nearest",
            )
        ax.set_title(title, fontsize=6)
        ax.set_xlabel("IM axis")
        ax.set_ylabel("RT axis")
        rt0, im0, rt1, im1 = alignment_state.matched_boxes[i]
        if i == alignment_state.reference_idx:
            ax.plot(
                alignment_state.anchor_col,
                alignment_state.anchor_row,
                "*",
                color="white",
                markersize=10,
                markeredgewidth=1,
                zorder=5,
            )
            _draw_rect(ax, rt0, im0, rt1, im1, color="red", linestyle="solid")
        else:
            _draw_rect(ax, rt0, im0, rt1, im1, color="red", linestyle="dashed")
        aa = alignment_state.aligned_anchors[i]
        if aa is not None and i in _anchor_color_map:
            ax.plot(
                aa[1],
                aa[0],
                "o",
                color=_anchor_color_map[i],
                markersize=7,
                markeredgecolor="black",
                markeredgewidth=0.8,
                zorder=5,
            )

    n_img_rows = math.ceil(n / max_cols)
    for empty in range(n, n_img_rows * max_cols):
        row, col = divmod(empty, max_cols)
        axes[row, col].set_visible(False)

    for col in range(max_cols):
        axes[-1, col].set_visible(False)
    _last_cols = [0, max_cols // 2, max_cols - 1]
    ax_craw, ax_csm, ax_cws = [axes[-1, c] for c in _last_cols]
    for _ax in (ax_craw, ax_csm, ax_cws):
        _ax.set_visible(True)

    im_craw = ax_craw.imshow(
        display_consensus, aspect="auto", origin="lower", vmin=vmin, vmax=vmax
    )
    ax_craw.set_title("Consensus (mean)", fontsize=9)
    ax_craw.set_xlabel("IM axis")
    ax_craw.set_ylabel("RT axis")
    fig.colorbar(im_craw, ax=ax_craw, shrink=0.8, label="Intensity")

    ax_csm.imshow(display_consensus_denoised, aspect="auto", origin="lower")
    ax_csm.set_title("Consensus (smoothed)", fontsize=9)
    ax_csm.set_xlabel("IM axis")
    ax_csm.set_ylabel("RT axis")

    if segmentation_state.watershed_labels.max() > 0:
        ax_cws.imshow(
            segmentation_state.watershed_labels,
            aspect="auto",
            origin="lower",
            cmap="tab20",
        )
        for label_val in range(1, segmentation_state.watershed_labels.max() + 1):
            mask = segmentation_state.watershed_labels == label_val
            if mask.any():
                rows, cols = np.where(mask)
                cy, cx = rows.mean(), cols.mean()
                ax_cws.text(
                    cx,
                    cy,
                    str(label_val),
                    ha="center",
                    va="center",
                    fontsize=7,
                    fontweight="bold",
                    color="white",
                )
    else:
        ax_cws.text(
            0.5,
            0.5,
            "No watershed\n(no anchors or no signal)",
            ha="center",
            va="center",
            transform=ax_cws.transAxes,
            fontsize=9,
        )
    ax_cws.set_title("Watershed segmentation", fontsize=9)
    ax_cws.set_xlabel("IM axis")
    ax_cws.set_ylabel("RT axis")

    for i_idx in non_none_indices:
        _c = _anchor_color_map[i_idx]
        for _ax in (ax_craw, ax_csm, ax_cws):
            if i_idx in snap_record:
                _orig_rc, _snap_rc = snap_record[i_idx]
                _ax.plot(
                    _orig_rc[1],
                    _orig_rc[0],
                    "*",
                    color=_c,
                    markersize=10,
                    markeredgecolor="black",
                    markeredgewidth=0.5,
                    zorder=6,
                )
                _ax.plot(
                    _snap_rc[1],
                    _snap_rc[0],
                    "o",
                    color=_c,
                    markersize=7,
                    markeredgecolor="black",
                    markeredgewidth=0.8,
                    zorder=7,
                )
            elif i_idx in discard_record:
                _orig_rc = discard_record[i_idx]["anchor"]
                _ax.plot(
                    _orig_rc[1],
                    _orig_rc[0],
                    "x",
                    color=_c,
                    markersize=9,
                    markeredgewidth=1.5,
                    zorder=6,
                )

    legend_handles = [
        mlines.Line2D(
            [],
            [],
            marker="*",
            color="white",
            markerfacecolor="white",
            markersize=10,
            linestyle="None",
            label="Anchor (template centre)",
        ),
        mpatches.Patch(
            edgecolor="red",
            facecolor="none",
            linestyle="solid",
            linewidth=1.5,
            label="Template region (reference)",
        ),
        mpatches.Patch(
            edgecolor="red",
            facecolor="none",
            linestyle="dashed",
            linewidth=1.5,
            label="Matched template region",
        ),
    ]
    if non_none_indices:
        legend_handles += [
            mlines.Line2D(
                [],
                [],
                marker="o",
                color=plt.cm.tab10(0),
                markerfacecolor=plt.cm.tab10(0),
                markeredgecolor="black",
                markeredgewidth=0.8,
                markersize=7,
                linestyle="None",
                label="Per-image anchor (aligned, ●)",
            ),
            mlines.Line2D(
                [],
                [],
                marker="*",
                color=plt.cm.tab10(0),
                markerfacecolor=plt.cm.tab10(0),
                markeredgecolor="black",
                markeredgewidth=0.5,
                markersize=10,
                linestyle="None",
                label="Anchor on consensus — original (★, same colour = same anchor)",
            ),
            mlines.Line2D(
                [],
                [],
                marker="o",
                color=plt.cm.tab10(0),
                markerfacecolor=plt.cm.tab10(0),
                markeredgecolor="black",
                markeredgewidth=0.8,
                markersize=7,
                linestyle="None",
                label="Anchor on consensus — snapped (●)",
            ),
        ]
    if discard_record:
        legend_handles.append(
            mlines.Line2D(
                [],
                [],
                marker="x",
                color=plt.cm.tab10(0),
                markersize=9,
                markeredgewidth=1.5,
                linestyle="None",
                label="Anchor on consensus — discarded (✕, no connected peak)",
            )
        )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(4, len(legend_handles)),
        fontsize=8,
        framealpha=0.8,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.suptitle("Resized, aligned images and mean consensus", fontsize=11)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    _save_or_show(fig, fig_dir, filename)


def _save_illustration_svgs(
    pept_idx: int,
    bundle: "ConsensusFeatureBundle",
    labels: list[str],
    svg_dir: str,
    raw_images: list[np.ndarray] | None = None,
    filename_prefix: str = "",
    segmentation_override: "ConsensusSegmentationState | None" = None,
    skip_per_run: bool = False,
) -> None:
    """Save individual clean SVG images for one peptide: raw, aligned, consensus, watershed.

    raw_images: original (pre-alignment) raw images, one per run in labels order.
                When provided, raw SVGs are always generated even if watershed failed.
    filename_prefix: prepended to every output filename (e.g. "decoy_peptide_swap_rep0_").
    segmentation_override: if provided, use instead of bundle.segmentation for consensus panels.
    skip_per_run: if True, skip per-run raw/aligned panels (useful for off-target decoys where
                  per-run images are identical to the target).
    """
    import matplotlib.pyplot as plt

    os.makedirs(svg_dir, exist_ok=True)

    seg = (
        segmentation_override
        if segmentation_override is not None
        else bundle.segmentation
    )

    def _sanitize(s: str) -> str:
        return re.sub(r"[^\w\-]", "_", os.path.basename(s))[:60]

    def _make_ax(
        img: np.ndarray,
        cmap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> "tuple[plt.Figure, plt.Axes]":
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(img, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.axis("off")
        return fig, ax

    def _save_fig(fig: "plt.Figure", fname: str) -> None:
        fig.savefig(
            os.path.join(svg_dir, f"{filename_prefix}{fname}"),
            format="svg",
            bbox_inches="tight",
            pad_inches=0.02,
        )
        plt.close(fig)

    # bundle.raw_aligned_images is only populated when watershed succeeds; fall
    # back to aligning the raw images ourselves using the same shifts.
    raw_aligned = bundle.raw_aligned_images
    if not raw_aligned and raw_images is not None:
        raw_aligned = _align_raw_images_with_shifts(
            raw_images,
            bundle.alignment.target_shape,
            bundle.alignment.shifts,
        )
    aligned_imgs = bundle.alignment.aligned_images

    def _range(imgs: list[np.ndarray]) -> tuple[float | None, float | None]:
        if not imgs:
            return None, None
        return float(min(img.min() for img in imgs)), float(
            max(img.max() for img in imgs)
        )

    # raw and denoised-aligned images live on different intensity scales — keep
    # them separate so neither set looks empty next to the other
    raw_vmin, raw_vmax = _range([np.log2(1 + img) for img in raw_aligned])
    aligned_vmin, aligned_vmax = _range(aligned_imgs)

    # anchor colour map — same indexing as _visualize_consensus_bundle
    non_none_indices = seg.non_none_indices
    anchor_color_map: dict[int, Any] = {
        i_idx: plt.cm.tab10(k % 10) for k, i_idx in enumerate(non_none_indices)
    }
    snap_record: dict = seg.snap_log.get("snap_record", {})
    discard_record: dict = seg.snap_log.get("discard_record", {})

    def _overlay_run_anchor(ax: "plt.Axes", i: int, *, aligned: bool) -> None:
        """Star at the template centre (reference only) + coloured dot for anchor.

        aligned=False → use pre-alignment (scaled) anchor; aligned=True → use post-alignment anchor.
        """
        if i == bundle.alignment.reference_idx:
            ax.plot(
                bundle.alignment.anchor_col,
                bundle.alignment.anchor_row,
                "*",
                color="white",
                markersize=10,
                markeredgewidth=1,
                zorder=5,
            )
        anchors = (
            bundle.alignment.aligned_anchors
            if aligned
            else bundle.alignment.scaled_anchors
        )
        aa = anchors[i]
        if aa is not None and i in anchor_color_map:
            ax.plot(
                aa[1],
                aa[0],
                "o",
                color=anchor_color_map[i],
                markersize=7,
                markeredgecolor="black",
                markeredgewidth=0.8,
                zorder=5,
            )

    def _overlay_consensus_anchors(ax: "plt.Axes") -> None:
        """Star (original) + dot (snapped) or × (discarded) for each anchor."""
        for i_idx in non_none_indices:
            c = anchor_color_map[i_idx]
            if i_idx in snap_record:
                orig_rc, snap_rc = snap_record[i_idx]
                ax.plot(
                    orig_rc[1],
                    orig_rc[0],
                    "*",
                    color=c,
                    markersize=10,
                    markeredgecolor="black",
                    markeredgewidth=0.5,
                    zorder=6,
                )
                ax.plot(
                    snap_rc[1],
                    snap_rc[0],
                    "o",
                    color=c,
                    markersize=7,
                    markeredgecolor="black",
                    markeredgewidth=0.8,
                    zorder=7,
                )
            elif i_idx in discard_record:
                orig_rc = discard_record[i_idx]["anchor"]
                ax.plot(
                    orig_rc[1],
                    orig_rc[0],
                    "x",
                    color=c,
                    markersize=9,
                    markeredgewidth=1.5,
                    zorder=6,
                )

    if not skip_per_run:
        for i, label in enumerate(labels):
            safe = _sanitize(label)
            if i < len(raw_aligned):
                fig, ax = _make_ax(
                    np.log2(1 + raw_aligned[i]), vmin=raw_vmin, vmax=raw_vmax
                )
                _overlay_run_anchor(ax, i, aligned=False)
                _save_fig(fig, f"mz{pept_idx}_raw_{i:02d}_{safe}.svg")
            if i < len(aligned_imgs):
                fig, ax = _make_ax(
                    aligned_imgs[i], vmin=aligned_vmin, vmax=aligned_vmax
                )
                _overlay_run_anchor(ax, i, aligned=True)
                _save_fig(fig, f"mz{pept_idx}_aligned_{i:02d}_{safe}.svg")

    fig, ax = _make_ax(seg.consensus, vmin=aligned_vmin, vmax=aligned_vmax)
    _overlay_consensus_anchors(ax)
    _save_fig(fig, f"mz{pept_idx}_consensus.svg")

    fig, ax = _make_ax(seg.consensus_denoised)
    _overlay_consensus_anchors(ax)
    _save_fig(fig, f"mz{pept_idx}_consensus_denoised.svg")

    if seg.watershed_labels.max() > 0:
        fig, ax = _make_ax(seg.watershed_labels, cmap="tab20")
        _overlay_consensus_anchors(ax)
        _save_fig(fig, f"mz{pept_idx}_watershed.svg")


def _shift_integer_label_image(
    label_image: np.ndarray, shift: tuple[int, int]
) -> np.ndarray:
    """Shift an integer label image with zero padding and no wrap-around."""

    dr, dc = int(shift[0]), int(shift[1])
    shifted = np.zeros_like(label_image)
    src_r0 = max(-dr, 0)
    src_r1 = min(label_image.shape[0] - dr, label_image.shape[0])
    src_c0 = max(-dc, 0)
    src_c1 = min(label_image.shape[1] - dc, label_image.shape[1])
    dst_r0 = max(dr, 0)
    dst_r1 = dst_r0 + max(src_r1 - src_r0, 0)
    dst_c0 = max(dc, 0)
    dst_c1 = dst_c0 + max(src_c1 - src_c0, 0)
    if src_r1 <= src_r0 or src_c1 <= src_c0:
        return shifted
    shifted[dst_r0:dst_r1, dst_c0:dst_c1] = label_image[src_r0:src_r1, src_c0:src_c1]
    return shifted


def _resolve_shifted_label_snap(
    shifted_labels: np.ndarray,
    label_id: int,
    desired_rc: tuple[int, int],
) -> tuple[int, int] | None:
    mask = shifted_labels == int(label_id)
    if not mask.any():
        return None
    rows, cols = shifted_labels.shape
    desired_r = int(np.clip(desired_rc[0], 0, rows - 1))
    desired_c = int(np.clip(desired_rc[1], 0, cols - 1))
    if mask[desired_r, desired_c]:
        return (desired_r, desired_c)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    dists = np.hypot(coords[:, 0] - desired_r, coords[:, 1] - desired_c)
    nearest = coords[int(np.argmin(dists))]
    return (int(nearest[0]), int(nearest[1]))


def _choose_off_target_shift(
    watershed_labels: np.ndarray,
    target_label_ids: list[int],
    rep: int = 0,
    min_offset_frac: float = 0.35,
    max_overlap_fraction: float = 0.05,
) -> tuple[int, int] | None:
    """Choose a large integer shift that moves target labels away from themselves."""

    if not target_label_ids:
        return None
    base_mask = np.isin(watershed_labels, target_label_ids)
    base_area = int(base_mask.sum())
    if base_area == 0:
        return None
    rows, cols = watershed_labels.shape
    frac_values = [min_offset_frac, min(0.5, min_offset_frac + 0.15)]
    candidates: list[tuple[int, int]] = []
    for frac in frac_values:
        dr = max(1, int(round(rows * frac)))
        dc = max(1, int(round(cols * frac)))
        candidates.extend(
            [
                (dr, 0),
                (-dr, 0),
                (0, dc),
                (0, -dc),
                (dr, dc),
                (dr, -dc),
                (-dr, dc),
                (-dr, -dc),
            ]
        )
    seen: set[tuple[int, int]] = set()
    unique_candidates = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique_candidates.append(candidate)
    evaluated: list[tuple[float, float, tuple[int, int]]] = []
    valid: list[tuple[float, float, tuple[int, int]]] = []
    for candidate in unique_candidates:
        shifted_labels = _shift_integer_label_image(watershed_labels, candidate)
        shifted_mask = np.isin(shifted_labels, target_label_ids)
        shifted_area = int(shifted_mask.sum())
        if shifted_area == 0:
            continue
        overlap = np.logical_and(base_mask, shifted_mask).sum() / float(
            max(base_area, 1)
        )
        dist = float(np.hypot(candidate[0], candidate[1]))
        record = (float(overlap), -dist, candidate)
        evaluated.append(record)
        if overlap <= max_overlap_fraction:
            valid.append(record)
    if valid:
        valid_sorted = sorted(valid, key=lambda item: (item[0], item[1]))
        return valid_sorted[int(rep) % len(valid_sorted)][2]
    if evaluated:
        evaluated_sorted = sorted(evaluated, key=lambda item: (item[0], item[1]))
        return evaluated_sorted[int(rep) % len(evaluated_sorted)][2]
    return None


def _build_consensus_peptide_swap_decoy(
    bundle: ConsensusFeatureBundle,
    decoy_raw_image: np.ndarray,
    run_name: str,
    raw_denoise_kwargs: dict | None = None,
) -> tuple[pd.DataFrame | None, tuple[int, int], float]:
    """Align a wrong same-run peptide image and score it under target consensus labels."""

    decoy_raw_logged = smooth_and_denoise_image(
        decoy_raw_image, **(raw_denoise_kwargs or {})
    )
    decoy_raw_logged_resized = _resize_image_to_shape(
        decoy_raw_logged, bundle.alignment.target_shape
    )
    (
        _aligned_denoised,
        _matched_box,
        _aligned_anchor,
        shift,
        max_score,
        _match_score_map,
        _match_score_peak,
    ) = _align_resized_image_to_template(
        decoy_raw_logged_resized,
        bundle.alignment.template,
        bundle.alignment.template_bounds,
        None,
    )
    decoy_raw_resized = _resize_image_to_shape(
        decoy_raw_image, bundle.alignment.target_shape
    )
    from scipy.ndimage import shift as nd_shift

    decoy_raw_aligned = nd_shift(
        decoy_raw_resized, shift=shift, mode="constant", cval=0.0
    )

    decoy_pp = _extract_feature_rows_for_label_ids(
        bundle.segmentation.target_label_ids,
        bundle.segmentation.watershed_labels,
        decoy_raw_aligned,
        decoy_raw_logged_resized,
        run_name=run_name,
        shift=shift,
        template_matching_score=max_score,
        snap_resolver=lambda label_id: bundle.segmentation.label_to_snap.get(label_id),
    )
    return decoy_pp, shift, max_score


def _build_consensus_off_target_decoy(
    bundle: ConsensusFeatureBundle,
    run_index: int,
    run_name: str,
    rep: int = 0,
    min_offset_frac: float = 0.35,
    max_overlap_fraction: float = 0.05,
    label_shift: tuple[int, int] | None = None,
) -> tuple[pd.DataFrame | None, tuple[int, int] | None]:
    """Quantify the target run against a deliberately shifted consensus label mask."""

    if not bundle.raw_aligned_images or not bundle.raw_aligned_denoised_images:
        return None, None
    resolved_label_shift = (
        _choose_off_target_shift(
            bundle.segmentation.watershed_labels,
            bundle.segmentation.target_label_ids,
            rep=rep,
            min_offset_frac=min_offset_frac,
            max_overlap_fraction=max_overlap_fraction,
        )
        if label_shift is None
        else (int(label_shift[0]), int(label_shift[1]))
    )
    if resolved_label_shift is None:
        return None, None
    shifted_labels = _shift_integer_label_image(
        bundle.segmentation.watershed_labels, resolved_label_shift
    )
    decoy_pp = _extract_feature_rows_for_label_ids(
        bundle.segmentation.target_label_ids,
        shifted_labels,
        bundle.raw_aligned_images[run_index],
        bundle.raw_aligned_denoised_images[run_index],
        run_name=run_name,
        shift=bundle.alignment.shifts[run_index],
        template_matching_score=bundle.alignment.max_scores[run_index],
        snap_resolver=lambda label_id: _resolve_shifted_label_snap(
            shifted_labels,
            label_id,
            (
                int(
                    bundle.segmentation.label_to_snap[label_id][0]
                    + resolved_label_shift[0]
                ),
                int(
                    bundle.segmentation.label_to_snap[label_id][1]
                    + resolved_label_shift[1]
                ),
            ),
        ),
    )
    return decoy_pp, resolved_label_shift


def generate_consensus_image(
    images: list[np.ndarray],
    reference_idx: int = 0,
    target_shape: tuple[int, int] | None = None,
    template_anchor: tuple[int, int] | None = None,
    template_frac: float = 0.3,
    anchors: list[tuple[int, int] | None] | None = None,
    denoise_cfg: dict | None = None,
    watershed_kwargs: dict | None = None,
    visualize: bool = False,
    labels: list[str] | None = None,
    raw_images: list[np.ndarray] | None = None,
    fig_dir: str | None = None,
    filename: str = "consensus_image.png",
    apply_seg: bool = True,
    seg_mask_thres: tuple[int, int] = (3, 3),
) -> tuple[
    np.ndarray,
    list[np.ndarray],
    list[tuple[int, int] | None],
    np.ndarray,
    dict[str, Any],
    pd.DataFrame | None,
    list[pd.DataFrame | None],
]:
    """Resize, align, and average a collection of denoised images into a consensus.

    A patch of size ``±template_frac * dim`` is extracted from the (resized)
    reference image centred on *template_anchor*, then
    :func:`skimage.feature.match_template` locates that patch in every other
    image and the resulting integer shift is applied with
    :func:`scipy.ndimage.shift`.

    Then a consensus image is generated and smoothed and watershed segmented.

    Then peak properties of the consensus image and individual images are calculated.

    Parameters
    ----------
    images : list of 2-D arrays
        The n images to aggregate.  They may differ in shape.
    reference_idx : int, optional
        Index into *images* that is used as the alignment target.
        Default is 0 (first image).
    target_shape : (rows, cols) or None, optional
        Shape to which every image is resized before alignment.  When *None*
        the shape of the reference image (at *reference_idx*) is used.
    template_anchor : (row, col) or None, optional
        Pixel coordinate **in the resized reference image** that centres the
        template patch.  When *None* the peak (argmax) of the resized reference
        image is used.
    template_frac : float, optional
        Fraction of each image dimension used as the half-extent of the
        template patch (``±template_frac * dim``).  Must be in ``(0, 0.5)``.
        Default is ``0.3``.
    visualize : bool, optional
        When *True* a figure is produced showing all n resized+aligned images
        in a grid of at most 5 columns, with the consensus on its own final row.
    labels : list of str or None, optional
        Panel titles for the visualisation.  Must have the same length as
        *images* when provided.
    fig_dir : str or None, optional
        Directory in which the figure is saved.  When *None* the figure is
        only shown interactively.
    filename : str, optional
        File-name used when saving the figure.  Default is
        ``"consensus_image.png"``.

    Returns
    -------
    consensus : 2-D ndarray
        Mean image computed over all aligned images.
    aligned_images : list of 2-D ndarray
        The n resized and aligned images (same order as *images*).
    """
    bundle = build_consensus_feature_bundle(
        images=images,
        reference_idx=reference_idx,
        target_shape=target_shape,
        template_anchor=template_anchor,
        template_frac=template_frac,
        anchors=anchors,
        denoise_cfg=denoise_cfg,
        watershed_kwargs=watershed_kwargs,
        labels=labels,
        raw_images=raw_images,
        apply_seg=apply_seg,
        seg_mask_thres=seg_mask_thres,
    )
    alignment = bundle.alignment
    segmentation = bundle.segmentation
    rows, cols = alignment.target_shape
    anchor_row, anchor_col = alignment.anchor_row, alignment.anchor_col
    aligned = alignment.aligned_images
    matched_boxes = alignment.matched_boxes
    aligned_anchors = alignment.aligned_anchors
    match_score = alignment.match_score_maps
    match_score_peaks = alignment.match_score_peaks
    match_score_label_indices = alignment.match_score_label_indices
    consensus = segmentation.consensus
    consensus_denoised = segmentation.consensus_denoised
    non_none_indices = segmentation.non_none_indices
    snapped_per_anchor = segmentation.snapped_per_anchor
    watershed_labels = segmentation.watershed_labels
    snap_log = segmentation.snap_log
    consensus_pp = bundle.consensus_pp
    individual_pps = bundle.individual_pps

    # --- 7. Optional visualisation -------------------------------------------
    if visualize:
        _visualize_consensus_bundle(
            bundle.alignment,
            bundle.segmentation,
            fig_dir=fig_dir,
            filename=filename,
            labels=labels,
        )

    return (
        consensus,
        aligned,
        snapped_per_anchor,
        watershed_labels,
        snap_log,
        consensus_pp,
        individual_pps,
    )
