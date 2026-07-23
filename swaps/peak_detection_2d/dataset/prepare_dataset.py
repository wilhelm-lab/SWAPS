"""Ground-truth dataset preparation for CNN-based consensus segmentation.

Ground truth comes from FragPipe's own `combined_ion.tsv` (per-run RT/IM
elution boundaries from direct MS/MS identification, independent of SWAPS'
own MS1-only peak picking) -- not from SWAPS' dict_ref, which only retains the
identification apex, not the start/end window.

The consensus image + hint channel are built via the exact same alignment
machinery match_features_batch uses in production (same per-candidate role
assignment, same anchor computation, same denoise stages, same
align_images_to_reference call) so training-time inputs match inference-time
inputs. Only the *segmentation* step (watershed) is skipped here -- ground
truth for that step comes from combined_ion.tsv instead.
"""

import logging
import os
from dataclasses import dataclass

import cv2
import h5py
import numpy as np
import pandas as pd
import yaml

from prepare_dict.search_engine_output_parser import maxquant_mods_to_fragpipe_modseq
from postprocessing.helper import get_pept_act_from_parquet
from postprocessing.image_processing import smooth_and_denoise_image
from postprocessing.match_features import (
    ConsensusAlignmentState,
    _denoise_kwargs_for_stage,
    _load_batch_activation_dfs,
    _positional_anchors,
    _project_anchor_into_aligned_space,
    _reference_match_quant_files,
    align_images_to_reference,
)

Logger = logging.getLogger(__name__)

COMBINED_ION_CACHE_RELPATH = os.path.join(
    "peak_detection_2d", "combined_ion_frame_idx.parquet"
)


def _load_effective_config(swaps_dir: str) -> dict:
    cfg_path = os.path.join(swaps_dir, "effective_config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _raw_run_dirs(swaps_dir: str) -> list[str]:
    """SWAPS raw-file result subdirectories (contain ms1scans.csv/mobility_values.csv)."""
    return sorted(
        d
        for d in os.listdir(swaps_dir)
        if os.path.isdir(os.path.join(swaps_dir, d))
        and os.path.exists(os.path.join(swaps_dir, d, "ms1scans.csv"))
    )


def _load_fragpipe_manifest_run_labels(search_output_path: str) -> dict[str, str]:
    """Parse FragPipe's own `fragpipe-files.fp-manifest` (if present): maps
    each run's FragPipe-assigned "<experiment>_<replicate>" label to the raw
    .d file's own basename (without extension) -- the SWAPS raw-dir name.

    Authoritative for custom experiment/replicate naming (set via FragPipe's
    experiment_annotation.tsv) where combined_ion.tsv's per-run columns are
    named after that short label (e.g. "HYE124_A_1") and bear no
    string-prefix relationship at all to the raw file's own (much longer)
    name -- unlike the hyphen-sanitized-suffix convention
    _match_combined_ion_run_prefix's fallback heuristic assumes, which only
    holds when FragPipe's default (raw-filename-derived) experiment naming
    was used.

    Returns {} if the manifest doesn't exist (e.g. non-FragPipe search
    engines, or FragPipe runs that didn't retain their manifest).
    """
    manifest_path = os.path.join(search_output_path, "fragpipe-files.fp-manifest")
    if not os.path.exists(manifest_path):
        return {}
    manifest = pd.read_csv(
        manifest_path,
        sep="\t",
        header=None,
        names=["path", "experiment", "replicate", "data_type"],
    )
    run_labels = manifest["experiment"] + "_" + manifest["replicate"].astype(str)
    raw_dirs = manifest["path"].apply(
        lambda p: os.path.basename(p).rsplit(".", 1)[0]
    )
    return dict(zip(run_labels, raw_dirs))


def _match_combined_ion_run_prefix(
    run_col_prefixes: list[str],
    raw_dirs: list[str],
    manifest_run_labels: dict[str, str] | None = None,
) -> dict[str, str]:
    """Map each combined_ion.tsv per-run column prefix to its SWAPS raw-dir name.

    Tries `manifest_run_labels` first (see _load_fragpipe_manifest_run_labels)
    -- an exact, authoritative lookup, needed when FragPipe was run with
    custom experiment/replicate names. Falls back to prefix-matching for any
    column not resolved that way (or when no manifest was given/found):
    combined_ion.tsv's per-run columns are named after FragPipe's own
    experiment folder, which -- under FragPipe's *default* naming -- is the
    SWAPS raw-dir name plus a FragPipe-added suffix (e.g. "..._5133" ->
    "..._5133_1") and with "-" sanitized to "_" (e.g. raw dir
    "...Slot2-4_1_12685" -> column prefix "...Slot2_4_1_12685_1") -- match by
    "raw dir name (hyphens normalized) is a prefix of the column prefix"
    instead of hardcoding the suffix, so this holds regardless of what
    FragPipe appends or renames.
    """
    mapping: dict[str, str] = {}
    raw_dirs_set = set(raw_dirs)
    for col_prefix in run_col_prefixes:
        manifest_dir = (manifest_run_labels or {}).get(col_prefix)
        if manifest_dir is not None and manifest_dir in raw_dirs_set:
            mapping[col_prefix] = manifest_dir
            continue
        candidates = [
            d for d in raw_dirs if col_prefix.startswith(d.replace("-", "_"))
        ]
        if len(candidates) != 1:
            Logger.warning(
                "Could not uniquely match combined_ion run column %r to a raw dir "
                "(candidates=%s); skipping this run.",
                col_prefix,
                candidates,
            )
            continue
        mapping[col_prefix] = candidates[0]
    return mapping


def _build_fp_modseq(dict_ref: pd.DataFrame) -> pd.DataFrame:
    """Add an `fp_modseq` column bridging dict_ref onto FragPipe's own
    "Modified Sequence" key space. Rows that fail to parse are dropped --
    they can't be matched to combined_ion.tsv and so can't be used as ground
    truth anyway.
    """
    out = dict_ref[["mz_rank", "Sequence", "Modifications", "Charge"]].copy()
    out["fp_modseq"] = [
        maxquant_mods_to_fragpipe_modseq(seq, mod)
        for seq, mod in zip(out["Sequence"], out["Modifications"])
    ]
    n_failed = int(out["fp_modseq"].isnull().sum())
    if n_failed:
        Logger.warning(
            "%d/%d dict_ref rows failed fp_modseq parsing; excluded from ground truth",
            n_failed,
            len(out),
        )
    return out.dropna(subset=["fp_modseq"])


def _rt_im_to_idx(
    rt_start_min: np.ndarray,
    rt_end_min: np.ndarray,
    im_start: np.ndarray,
    im_end: np.ndarray,
    ms1scans: pd.DataFrame,
    mobility_values_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized real-unit -> frame_idx/mobility_idx conversion, same
    searchsorted convention as `ims_3d.get_ref_rt_im_range`.
    """
    rt_array = ms1scans["Time_minute"].to_numpy()
    im_array = mobility_values_df["mobility_values"].to_numpy()
    frame_idx_left = np.clip(np.searchsorted(rt_array, rt_start_min, side="left") - 1, 0, None)
    frame_idx_right = np.searchsorted(rt_array, rt_end_min, side="right")
    mobility_idx_left = np.clip(np.searchsorted(im_array, im_start, side="left") - 1, 0, None)
    mobility_idx_right = np.searchsorted(im_array, im_end, side="right")
    return frame_idx_left, frame_idx_right, mobility_idx_left, mobility_idx_right


def get_or_build_combined_ion_frame_idx(
    swaps_dir: str, force: bool = False
) -> pd.DataFrame:
    """Get-or-build the per-(mz_rank, run) ground-truth boundary table for one
    experiment, in frame_idx/mobility_idx units. Cached to
    `<swaps_dir>/peak_detection_2d/combined_ion_frame_idx.parquet` so the
    join+conversion only runs once per experiment.

    Only rows with `<run> Match Type == "MS/MS"` (a direct identification in
    that run) are kept -- "MBR" rows are FragPipe/IonQuant's own transferred
    quantification, not independent evidence, and "unmatched" rows have no
    boundary at all.

    Returns columns: mz_rank, run, rt_start_min, rt_end_min, im_start, im_end,
    frame_idx_left, frame_idx_right, mobility_idx_left, mobility_idx_right.
    """
    cache_path = os.path.join(swaps_dir, COMBINED_ION_CACHE_RELPATH)
    if not force and os.path.exists(cache_path):
        Logger.info("Loading cached combined_ion frame-idx table from %s", cache_path)
        return pd.read_parquet(cache_path)

    Logger.info("Building combined_ion frame-idx table for %s", swaps_dir)
    cfg = _load_effective_config(swaps_dir)
    combined_ion_path = os.path.join(cfg["SEARCH_OUTPUT_PATH"], "combined_ion.tsv")
    combined_ion = pd.read_csv(combined_ion_path, sep="\t", low_memory=False)
    combined_ion = combined_ion.rename(columns={"Modified Sequence": "fp_modseq"})

    dict_ref = pd.read_pickle(os.path.join(swaps_dir, "dict_ref.pkl"))
    dict_ref = _build_fp_modseq(dict_ref)

    run_col_prefixes = sorted(
        {
            c[: -len(" Retention Time Start")]
            for c in combined_ion.columns
            if c.endswith(" Retention Time Start")
        }
    )
    raw_dirs = _raw_run_dirs(swaps_dir)
    manifest_run_labels = _load_fragpipe_manifest_run_labels(cfg["SEARCH_OUTPUT_PATH"])
    run_prefix_to_dir = _match_combined_ion_run_prefix(
        run_col_prefixes, raw_dirs, manifest_run_labels
    )
    Logger.info(
        "Matched %d/%d combined_ion run columns to SWAPS raw dirs",
        len(run_prefix_to_dir),
        len(run_col_prefixes),
    )

    matched = dict_ref.merge(
        combined_ion, on=["fp_modseq", "Charge"], how="inner", suffixes=("", "_ci")
    )
    Logger.info(
        "Matched %d/%d dict_ref rows to combined_ion by (fp_modseq, Charge)",
        len(matched),
        len(dict_ref),
    )

    per_run_tables = []
    for run_prefix, run_dir in run_prefix_to_dir.items():
        match_type_col = f"{run_prefix} Match Type"
        rt_start_col = f"{run_prefix} Retention Time Start"
        rt_end_col = f"{run_prefix} Retention Time End"
        im_start_col = f"{run_prefix} Ion Mobility Start"
        im_end_col = f"{run_prefix} Ion Mobility End"

        run_hits = matched.loc[
            matched[match_type_col] == "MS/MS",
            ["mz_rank", rt_start_col, rt_end_col, im_start_col, im_end_col],
        ].copy()
        if run_hits.empty:
            continue

        ms1scans = pd.read_csv(os.path.join(swaps_dir, run_dir, "ms1scans.csv"))
        mobility_values_df = pd.read_csv(
            os.path.join(swaps_dir, run_dir, "mobility_values.csv")
        )

        rt_start_min = run_hits[rt_start_col].to_numpy() / 60.0
        rt_end_min = run_hits[rt_end_col].to_numpy() / 60.0
        im_start = run_hits[im_start_col].to_numpy()
        im_end = run_hits[im_end_col].to_numpy()

        frame_idx_left, frame_idx_right, mobility_idx_left, mobility_idx_right = (
            _rt_im_to_idx(
                rt_start_min, rt_end_min, im_start, im_end, ms1scans, mobility_values_df
            )
        )

        per_run_tables.append(
            pd.DataFrame(
                {
                    "mz_rank": run_hits["mz_rank"].to_numpy(),
                    "run": run_dir,
                    "rt_start_min": rt_start_min,
                    "rt_end_min": rt_end_min,
                    "im_start": im_start,
                    "im_end": im_end,
                    "frame_idx_left": frame_idx_left,
                    "frame_idx_right": frame_idx_right,
                    "mobility_idx_left": mobility_idx_left,
                    "mobility_idx_right": mobility_idx_right,
                }
            )
        )

    if not per_run_tables:
        result = pd.DataFrame(
            columns=[
                "mz_rank",
                "run",
                "rt_start_min",
                "rt_end_min",
                "im_start",
                "im_end",
                "frame_idx_left",
                "frame_idx_right",
                "mobility_idx_left",
                "mobility_idx_right",
            ]
        )
    else:
        result = pd.concat(per_run_tables, ignore_index=True)

    Logger.info(
        "Built ground-truth boundary table: %d (mz_rank, run) rows across %d runs, "
        "%d distinct mz_ranks",
        len(result),
        result["run"].nunique() if len(result) else 0,
        result["mz_rank"].nunique() if len(result) else 0,
    )
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    result.to_parquet(cache_path, index=False)
    return result


@dataclass
class GroundTruthDatapoint:
    """One training example, in this peptide's own (un-padded) aligned shape --
    fixed-shape pad/crop to the dataset-wide target shape happens separately."""

    mz_rank: int
    image: np.ndarray  # consensus image (aligned, averaged, consensus-denoised)
    hint_channel: np.ndarray  # binary, 1 at every known anchor position
    mask: np.ndarray  # binary ground-truth peak mask (union across runs)
    contributing_runs: list[str]  # runs whose combined_ion boundary contributed
    reference_raw_file: str


def _project_run_boundary_to_bbox(
    frame_idx_left: int,
    frame_idx_right: int,
    mobility_idx_left: int,
    mobility_idx_right: int,
    window_origin: tuple[int, int],
    raw_crop_shape: tuple[int, int],
    alignment_state: ConsensusAlignmentState,
    run_position_i: int,
) -> tuple[int, int, int, int] | None:
    """Project one run's absolute-frame/mobility boundary box into the shared
    aligned/consensus coordinate space, reusing the exact same scale+shift
    transform align_images_to_reference computed for that run
    (_project_anchor_into_aligned_space, applied to both box corners).

    Returns (row0, col0, row1, col1) in the alignment's target_shape, clipped
    to bounds, or None if the box falls entirely outside the aligned image or
    projection fails.
    """
    rt0_rel = frame_idx_left - window_origin[0]
    rt1_rel = frame_idx_right - window_origin[0]
    im0_rel = mobility_idx_left - window_origin[1]
    im1_rel = mobility_idx_right - window_origin[1]
    corner0 = _project_anchor_into_aligned_space(
        (rt0_rel, im0_rel), raw_crop_shape, alignment_state, run_position_i
    )
    corner1 = _project_anchor_into_aligned_space(
        (rt1_rel, im1_rel), raw_crop_shape, alignment_state, run_position_i
    )
    if corner0 is None or corner1 is None:
        return None
    row0, row1 = sorted((corner0[0], corner1[0]))
    col0, col1 = sorted((corner0[1], corner1[1]))
    rows, cols = alignment_state.target_shape
    row0 = int(np.clip(round(row0), 0, rows))
    row1 = int(np.clip(round(row1), 0, rows))
    col0 = int(np.clip(round(col0), 0, cols))
    col1 = int(np.clip(round(col1), 0, cols))
    if row1 <= row0 or col1 <= col0:
        return None
    return row0, col0, row1, col1


def prepare_ground_truth_batch(
    dict_ref: pd.DataFrame,
    raw_file_list: list[str],
    result_dir: str,
    batch: list[int] | np.ndarray,
    boundary_table: pd.DataFrame,
    processing_kwargs: dict | None = None,
    merge_confounders_enabled: bool = True,
) -> dict[int, GroundTruthDatapoint]:
    """Build ground-truth (image, hint_channel, mask) for a batch of mz_ranks.

    Mirrors match_features_batch's per-candidate setup (role assignment,
    per-run raw crop + denoise, align_images_to_reference) so the CNN sees
    the same consensus image at training time as it will at inference time --
    only the watershed segmentation step is skipped; the mask comes from
    `boundary_table` (see get_or_build_combined_ion_frame_idx) instead.

    `processing_kwargs` should be that experiment's own
    `effective_config.yaml`'s `MATCH_FEATURES_KWARGS` (or an equivalent dict),
    so denoise stages / template_frac / align_images / use_shift_crop_pad
    match production exactly.
    """
    batch_np = np.asarray(batch)
    dict_ref_by_mz = (
        dict_ref.set_index("mz_rank")
        if dict_ref["mz_rank"].is_unique
        else dict_ref.drop_duplicates("mz_rank").set_index("mz_rank")
    )
    denoise_cfg = dict((processing_kwargs or {}).get("denoise", {}))
    raw_denoise_kwargs = _denoise_kwargs_for_stage(denoise_cfg, "raw")
    consensus_denoise_kwargs = _denoise_kwargs_for_stage(denoise_cfg, "consensus")
    template_frac = float((processing_kwargs or {}).get("template_frac", 0.3))
    align_images_flag = bool((processing_kwargs or {}).get("align_images", True))
    use_shift_crop_pad = bool((processing_kwargs or {}).get("use_shift_crop_pad", False))
    post_align_log_transform = bool(
        _denoise_kwargs_for_stage(denoise_cfg, "aligned").get("log_transform", False)
    )

    act_dfs, empty_act_df, _group_to_members, has_group_col = (
        _load_batch_activation_dfs(
            dict_ref_by_mz,
            raw_file_list,
            result_dir,
            batch_np,
            merge_confounders_enabled=merge_confounders_enabled,
        )
    )

    def _select_mz(raw_file: str, mz_rank: int) -> pd.DataFrame:
        return act_dfs[raw_file].get(mz_rank, empty_act_df[raw_file])

    boundary_by_mz = {
        int(mz): sub for mz, sub in boundary_table.groupby("mz_rank", sort=False)
    }

    results: dict[int, GroundTruthDatapoint] = {}
    for pept_idx in batch_np:
        pept_idx = int(pept_idx)
        pept_act_cache: dict[str, tuple[np.ndarray, int, int, tuple[int, int]]] = {}

        group_id = (
            int(dict_ref_by_mz.at[pept_idx, "confounder_group_id"])
            if has_group_col
            else -1
        )
        act_key = group_id if group_id != -1 else pept_idx

        def _get_pept_act_tuple(
            raw_file: str,
        ) -> tuple[np.ndarray, int, int, tuple[int, int]]:
            if raw_file not in pept_act_cache:
                pept_act_cache[raw_file] = get_pept_act_from_parquet(
                    _select_mz(raw_file, act_key),
                    pept_idx,
                    dict_ref_by_mz,
                    raw_file,
                    return_offset=True,
                )
            return pept_act_cache[raw_file]

        try:
            reference_raw_file, quant_only_raw_files, match_raw_files = (
                _reference_match_quant_files(dict_ref_by_mz, pept_idx)
            )
        except IndexError:
            Logger.warning("No reference run found for mz_rank %s; skipping.", pept_idx)
            continue
        quant_only_set = set(quant_only_raw_files)
        consensus_raw_files = [reference_raw_file] + match_raw_files
        consensus_anchors = _positional_anchors(
            consensus_raw_files, reference_raw_file, quant_only_set, _get_pept_act_tuple
        )
        anchor_image_indices = [
            i for i, a in enumerate(consensus_anchors) if a is not None
        ]
        if not anchor_image_indices:
            Logger.debug("mz_rank %s has no known anchor in any run; skipping.", pept_idx)
            continue

        raw_images = [_get_pept_act_tuple(rf)[0] for rf in consensus_raw_files]
        window_origins = [_get_pept_act_tuple(rf)[3] for rf in consensus_raw_files]
        denoised_images = [
            smooth_and_denoise_image(img, **raw_denoise_kwargs) for img in raw_images
        ]

        alignment_state = align_images_to_reference(
            images=denoised_images,
            reference_idx=0,
            template_frac=template_frac,
            anchors=consensus_anchors,
            align_images=align_images_flag,
            post_align_log_transform=post_align_log_transform,
            use_shift_crop_pad=use_shift_crop_pad,
        )

        consensus = np.stack(
            [alignment_state.aligned_images[i] for i in anchor_image_indices], axis=0
        ).mean(axis=0)
        consensus_denoised = smooth_and_denoise_image(
            consensus, **consensus_denoise_kwargs
        )

        rows, cols = alignment_state.target_shape
        hint_channel = np.zeros((rows, cols), dtype=np.float32)
        for aa in alignment_state.aligned_anchors:
            if aa is None:
                continue
            r = int(np.clip(round(aa[0]), 0, rows - 1))
            c = int(np.clip(round(aa[1]), 0, cols - 1))
            hint_channel[r, c] = 1.0

        mask = np.zeros((rows, cols), dtype=np.float32)
        contributing_runs: list[str] = []
        run_boundaries = boundary_by_mz.get(pept_idx)
        if run_boundaries is not None:
            for _, row in run_boundaries.iterrows():
                run = row["run"]
                if run not in consensus_raw_files:
                    continue
                run_pos = consensus_raw_files.index(run)
                bbox = _project_run_boundary_to_bbox(
                    int(row["frame_idx_left"]),
                    int(row["frame_idx_right"]),
                    int(row["mobility_idx_left"]),
                    int(row["mobility_idx_right"]),
                    window_origins[run_pos],
                    raw_images[run_pos].shape,
                    alignment_state,
                    run_pos,
                )
                if bbox is None:
                    continue
                row0, col0, row1, col1 = bbox
                mask[row0:row1, col0:col1] = 1.0
                contributing_runs.append(run)

        if not contributing_runs:
            Logger.debug(
                "mz_rank %s has no projectable ground-truth boundary; skipping.",
                pept_idx,
            )
            continue

        results[pept_idx] = GroundTruthDatapoint(
            mz_rank=pept_idx,
            image=consensus_denoised,
            hint_channel=hint_channel,
            mask=mask,
            contributing_runs=contributing_runs,
            reference_raw_file=reference_raw_file,
        )

    return results


def round_up_to_multiple(value: int, multiple: int = 16) -> int:
    return int(np.ceil(value / multiple) * multiple)


def compute_target_shape(
    image_shapes: list[tuple[int, int]], percentile: float = 99.0, multiple: int = 16
) -> tuple[int, int]:
    """Empirical fixed output shape: `percentile`-th pct of observed (H, W),
    rounded up to a multiple of `multiple` (so the UNET's repeated
    MaxPool2d(2,2) downsampling divides evenly). Logs the resulting
    crop/pad split so the choice isn't silent.
    """
    shapes = np.asarray(image_shapes)
    target_h = round_up_to_multiple(int(np.percentile(shapes[:, 0], percentile)), multiple)
    target_w = round_up_to_multiple(int(np.percentile(shapes[:, 1], percentile)), multiple)
    n_cropped = int(((shapes[:, 0] > target_h) | (shapes[:, 1] > target_w)).sum())
    Logger.info(
        "Target shape (%d, %d) from %d samples (p%s, rounded to multiple of %d); "
        "%d/%d (%.1f%%) would be center-cropped, rest zero-padded.",
        target_h,
        target_w,
        len(shapes),
        percentile,
        multiple,
        n_cropped,
        len(shapes),
        100 * n_cropped / len(shapes),
    )
    return target_h, target_w


def compute_target_shape_per_experiment(
    shapes_by_experiment: dict[str, list[tuple[int, int]]],
    percentile: float = 99.0,
    multiple: int = 16,
) -> tuple[int, int]:
    """Distribution-shift-aware target shape: compute `percentile`-th pct (H, W)
    *within each experiment* separately, then take the max across experiments,
    rounded up to a multiple of `multiple`.

    Pooling all experiments' shapes into one `compute_target_shape` call
    understates whichever experiment has the fattest tail on a given axis
    (e.g. gradient length differences shift the RT-extent distribution
    across runs) -- a shorter-gradient experiment's samples end up
    disproportionately cropped relative to their own distribution. Taking the
    per-experiment max instead means no single experiment's own tail drives
    more cropping in another experiment's samples.
    """
    per_exp_pct = {
        name: (
            np.percentile(np.asarray(shapes)[:, 0], percentile),
            np.percentile(np.asarray(shapes)[:, 1], percentile),
        )
        for name, shapes in shapes_by_experiment.items()
    }
    target_h = round_up_to_multiple(
        int(max(h for h, _ in per_exp_pct.values())), multiple
    )
    target_w = round_up_to_multiple(
        int(max(w for _, w in per_exp_pct.values())), multiple
    )
    Logger.info(
        "Per-experiment p%s (H, W): %s -> target shape (%d, %d)",
        percentile,
        {k: (round(h, 1), round(w, 1)) for k, (h, w) in per_exp_pct.items()},
        target_h,
        target_w,
    )
    return target_h, target_w


def pad_or_crop_to_shape(
    array: np.ndarray, target_shape: tuple[int, int]
) -> tuple[np.ndarray, bool]:
    """Center-pad (zeros) or center-crop `array` to `target_shape` -- never
    interpolate/resample, so pixel identity (one MS1 frame x one mobility
    bin) and intensity values stay exact; mask boundaries stay pixel-accurate.

    Returns (result, was_cropped) -- was_cropped flags samples where the
    original was larger than target_shape in either dimension, so callers can
    track how often that happens.
    """
    src_h, src_w = array.shape
    tgt_h, tgt_w = target_shape
    was_cropped = src_h > tgt_h or src_w > tgt_w

    if src_h > tgt_h:
        start = (src_h - tgt_h) // 2
        array = array[start : start + tgt_h, :]
        src_h = tgt_h
    if src_w > tgt_w:
        start = (src_w - tgt_w) // 2
        array = array[:, start : start + tgt_w]
        src_w = tgt_w

    if src_h < tgt_h or src_w < tgt_w:
        pad_h, pad_w = tgt_h - src_h, tgt_w - src_w
        array = np.pad(
            array,
            ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2)),
            mode="constant",
            constant_values=0,
        )
    return array, was_cropped


def pad_or_crop_datapoint(
    dp: GroundTruthDatapoint, target_shape: tuple[int, int]
) -> tuple[GroundTruthDatapoint, bool]:
    """Apply pad_or_crop_to_shape identically to image/hint/mask (same
    source shape, so the same offsets apply to all three)."""
    image, was_cropped = pad_or_crop_to_shape(dp.image, target_shape)
    hint_channel, _ = pad_or_crop_to_shape(dp.hint_channel, target_shape)
    mask, _ = pad_or_crop_to_shape(dp.mask, target_shape)
    return (
        GroundTruthDatapoint(
            mz_rank=dp.mz_rank,
            image=image,
            hint_channel=hint_channel,
            mask=mask,
            contributing_runs=dp.contributing_runs,
            reference_raw_file=dp.reference_raw_file,
        ),
        was_cropped,
    )


def _rescale_transform(
    src_shape: tuple[int, int], target_shape: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Shared geometry for pad_or_rescale_to_shape and _rescale_hint_points:
    the intermediate (rescaled-if-needed) shape and the centering pad offset
    applied afterwards, so a raster resize and a point-coordinate transform
    land in the exact same pixel registration.
    """
    src_h, src_w = src_shape
    tgt_h, tgt_w = target_shape
    inter_h = min(src_h, tgt_h)
    inter_w = min(src_w, tgt_w)
    pad_top = (tgt_h - inter_h) // 2
    pad_left = (tgt_w - inter_w) // 2
    return inter_h, inter_w, pad_top, pad_left


def pad_or_rescale_to_shape(
    array: np.ndarray,
    target_shape: tuple[int, int],
    interpolation: int = cv2.INTER_LINEAR,
) -> tuple[np.ndarray, bool]:
    """Zero-pad axes that already fit within target_shape (exact, no
    interpolation); *rescale* -- not crop -- only the axis/axes that exceed
    it, each independently, down to exactly target_shape on that axis.

    Reuses the same cv2.resize mechanism align_images_to_reference already
    applies to every non-reference run before template matching (see
    _resize_image_to_shape) -- this isn't new machinery, just the existing
    resize path applied at the dataset-fixed-shape step instead of the per-
    peptide alignment step, and only to the axes that actually need it.

    Doing this instead of pad_or_crop_to_shape means the rare oversized tail
    (e.g. a handful of unusually wide confounder-group windows) loses spatial
    resolution but keeps its full extent and complete ground-truth mask,
    rather than being truncated. Only one axis is rescaled when only one
    exceeds target_shape, so an image that already fits on IM but not RT
    doesn't get its IM axis stretched/zoomed along with it.

    Not used for the hint channel -- see _rescale_hint_points: resampling a
    sparse single-pixel raster (even with nearest-neighbor) can drop an
    anchor point entirely when downscaling, so hint points are rescaled as
    coordinates instead of as an image.

    Returns (result, was_rescaled).
    """
    src_h, src_w = array.shape
    inter_h, inter_w, pad_top, pad_left = _rescale_transform(
        array.shape, target_shape
    )
    was_rescaled = inter_h != src_h or inter_w != src_w
    if was_rescaled:
        # cv2.resize's dsize is (width, height); float32 through cv2 like
        # _resize_image_to_shape, cast back to the input's own dtype.
        out_dtype = array.dtype
        array = cv2.resize(
            array.astype(np.float32), (inter_w, inter_h), interpolation=interpolation
        ).astype(out_dtype if np.issubdtype(out_dtype, np.floating) else np.float64)

    tgt_h, tgt_w = target_shape
    if inter_h < tgt_h or inter_w < tgt_w:
        array = np.pad(
            array,
            (
                (pad_top, tgt_h - inter_h - pad_top),
                (pad_left, tgt_w - inter_w - pad_left),
            ),
            mode="constant",
            constant_values=0,
        )
    return array, was_rescaled


def invert_pad_or_rescale_to_shape(
    array: np.ndarray,
    original_shape: tuple[int, int],
    padded_shape: tuple[int, int],
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Invert pad_or_rescale_to_shape: given `array` already at `padded_shape`
    (whatever pad_or_rescale_to_shape produced from something originally
    shaped `original_shape`), recover an array back at `original_shape` --
    crop off the centering pad, then rescale back up any axis that had been
    rescaled down. Uses the same _rescale_transform geometry as the forward
    call, so the two stay in lockstep by construction.

    Used at CNN inference time (match_features._segment_consensus_with_cnn)
    to map a predicted mask at the model's fixed training shape back onto a
    real consensus image's own native per-peptide aligned shape.
    """
    inter_h, inter_w, pad_top, pad_left = _rescale_transform(
        original_shape, padded_shape
    )
    cropped = array[pad_top : pad_top + inter_h, pad_left : pad_left + inter_w]
    src_h, src_w = original_shape
    if inter_h == src_h and inter_w == src_w:
        return cropped
    out_dtype = cropped.dtype
    result = cv2.resize(
        cropped.astype(np.float32), (src_w, src_h), interpolation=interpolation
    )
    return result.astype(
        out_dtype if np.issubdtype(out_dtype, np.floating) else np.float64
    )


def _rescale_hint_points(
    hint_channel: np.ndarray, target_shape: tuple[int, int]
) -> np.ndarray:
    """Rescale the hint channel by transforming its (sparse) anchor point
    coordinates directly, rather than resampling the raster -- resampling
    (even nearest-neighbor) can skip an isolated 1-pixel point entirely when
    downscaling, silently dropping a known anchor. Scaling the coordinates
    the same way match_features.py's own anchor projection does
    (_scale_anchor_to_target_shape/_project_anchor_into_aligned_space) can't
    lose a point that way -- every input point maps to exactly one output
    pixel (with possible collisions when multiple anchors land on the same
    rescaled pixel, which is fine: the hint channel only needs to mark that
    a position is anchored, not count how many).
    """
    src_h, src_w = hint_channel.shape
    inter_h, inter_w, pad_top, pad_left = _rescale_transform(
        hint_channel.shape, target_shape
    )
    scale_h = inter_h / src_h
    scale_w = inter_w / src_w
    tgt_h, tgt_w = target_shape
    result = np.zeros(target_shape, dtype=hint_channel.dtype)
    rows, cols = np.nonzero(hint_channel)
    if rows.size == 0:
        return result
    new_rows = np.clip(
        np.round(rows * scale_h).astype(int) + pad_top, 0, tgt_h - 1
    )
    new_cols = np.clip(
        np.round(cols * scale_w).astype(int) + pad_left, 0, tgt_w - 1
    )
    result[new_rows, new_cols] = 1
    return result


def pad_or_rescale_datapoint(
    dp: GroundTruthDatapoint, target_shape: tuple[int, int]
) -> tuple[GroundTruthDatapoint, bool]:
    """Apply pad_or_rescale_to_shape to the continuous-intensity image
    (bilinear, matching _resize_image_to_shape's own choice) and the mask
    (bilinear then re-thresholded at 0.5, keeping it a valid binary target);
    rescale the hint channel by its point coordinates (_rescale_hint_points)
    rather than resampling its raster, so a known anchor is never dropped."""
    image, was_rescaled = pad_or_rescale_to_shape(
        dp.image, target_shape, interpolation=cv2.INTER_LINEAR
    )
    hint_channel = _rescale_hint_points(dp.hint_channel, target_shape)
    mask, _ = pad_or_rescale_to_shape(
        dp.mask, target_shape, interpolation=cv2.INTER_LINEAR
    )
    mask = (mask > 0.5).astype(np.float32)
    return (
        GroundTruthDatapoint(
            mz_rank=dp.mz_rank,
            image=image,
            hint_channel=hint_channel,
            mask=mask,
            contributing_runs=dp.contributing_runs,
            reference_raw_file=dp.reference_raw_file,
        ),
        was_rescaled,
    )


def load_experiment_context(swaps_dir: str) -> dict:
    """Load everything needed to build ground-truth batches for one experiment:
    the (cached) boundary table, dict_ref_with_activation.pkl (same dict
    match_features_batch uses in production -- carries the per-run role and
    anchor columns dict_ref.pkl alone lacks), that experiment's own
    MATCH_FEATURES_KWARGS, and its raw-file list.
    """
    boundary_table = get_or_build_combined_ion_frame_idx(swaps_dir)
    dict_ref = pd.read_pickle(os.path.join(swaps_dir, "dict_ref_with_activation.pkl"))
    cfg = _load_effective_config(swaps_dir)
    processing_kwargs = cfg["MATCH_FEATURES_KWARGS"]
    raw_file_list = sorted(
        {
            c[: -len("_MS1_frame_idx_exp")]
            for c in dict_ref.columns
            if c.endswith("_MS1_frame_idx_exp")
        }
    )
    return {
        "swaps_dir": swaps_dir,
        "boundary_table": boundary_table,
        "dict_ref": dict_ref,
        "processing_kwargs": processing_kwargs,
        "raw_file_list": raw_file_list,
    }


def _eligible_mz_ranks(
    dict_ref: pd.DataFrame, boundary_table: pd.DataFrame, include_decoys: bool
) -> np.ndarray:
    """mz_ranks with at least one usable ground-truth boundary, decoys
    excluded by default -- there's no real "true peak" to supervise on."""
    eligible = boundary_table["mz_rank"].drop_duplicates()
    if not include_decoys and "Decoy" in dict_ref.columns:
        decoy_by_mz = dict_ref.drop_duplicates("mz_rank").set_index("mz_rank")["Decoy"]
        eligible = eligible[~eligible.map(decoy_by_mz).fillna(False)]
    return eligible.to_numpy()


def build_experiment_ground_truth(
    swaps_dir: str,
    n_samples: int,
    batch_size: int = 500,
    random_seed: int = 42,
    include_decoys: bool = False,
) -> list[GroundTruthDatapoint]:
    """Sample n_samples eligible mz_ranks from one experiment and build their
    ground-truth datapoints, processed in batches -- reuses
    prepare_ground_truth_batch's batch activation loading instead of loading
    per mz_rank, the same way match_features_batch processes candidates in
    batches for the same I/O reason (one DuckDB scan per batch, not per
    peptide).
    """
    ctx = load_experiment_context(swaps_dir)
    eligible = _eligible_mz_ranks(ctx["dict_ref"], ctx["boundary_table"], include_decoys)
    rng = np.random.default_rng(random_seed)
    shuffled = rng.permutation(eligible)

    collected: list[GroundTruthDatapoint] = []
    for start in range(0, len(shuffled), batch_size):
        if len(collected) >= n_samples:
            break
        chunk = shuffled[start : start + batch_size]
        batch_results = prepare_ground_truth_batch(
            ctx["dict_ref"],
            ctx["raw_file_list"],
            swaps_dir,
            chunk,
            ctx["boundary_table"],
            ctx["processing_kwargs"],
        )
        collected.extend(batch_results.values())
        Logger.info(
            "%s: %d/%d collected after %d/%d candidates tried",
            os.path.basename(swaps_dir.rstrip("/")),
            len(collected),
            n_samples,
            min(start + batch_size, len(shuffled)),
            len(shuffled),
        )
    if len(collected) < n_samples:
        Logger.warning(
            "%s: only %d/%d requested samples produced (ran out of eligible "
            "candidates).",
            swaps_dir,
            len(collected),
            n_samples,
        )
    return collected[:n_samples]


def _write_split_hdf5(
    datapoints: list[GroundTruthDatapoint],
    sources: list[str],
    target_shape: tuple[int, int],
    h5_path: str,
) -> pd.DataFrame:
    """Write one split to a single chunked/compressed HDF5 file with 3
    contiguous [N, H, W] datasets (images/hints/masks) -- not one HDF5 group
    per sample (the old peak_detection_2d infra's approach): thousands of
    small groups with no chunking/compression is a known HDF5 anti-pattern
    (heavy per-object metadata overhead, poor random-access performance).
    Per-sample metadata goes to a parquet sidecar instead of HDF5 attrs.
    """
    n = len(datapoints)
    h, w = target_shape
    os.makedirs(os.path.dirname(h5_path), exist_ok=True)
    meta_rows = []
    with h5py.File(h5_path, "w") as f:
        images = f.create_dataset(
            "images", shape=(n, h, w), dtype="float32", chunks=(1, h, w),
            compression="lzf",
        )
        hints = f.create_dataset(
            "hints", shape=(n, h, w), dtype="float32", chunks=(1, h, w),
            compression="lzf",
        )
        masks = f.create_dataset(
            "masks", shape=(n, h, w), dtype="float32", chunks=(1, h, w),
            compression="lzf",
        )
        for i, (dp, source) in enumerate(zip(datapoints, sources)):
            padded, was_rescaled = pad_or_rescale_datapoint(dp, target_shape)
            images[i] = padded.image
            hints[i] = padded.hint_channel
            masks[i] = padded.mask
            meta_rows.append(
                {
                    "row": i,
                    "mz_rank": dp.mz_rank,
                    "source_experiment": source,
                    "contributing_runs": ";".join(dp.contributing_runs),
                    "reference_raw_file": dp.reference_raw_file,
                    "was_rescaled": was_rescaled,
                }
            )
    return pd.DataFrame(meta_rows)


def build_training_dataset(
    experiment_dirs: list[str],
    output_dir: str,
    n_per_experiment: int = 5000,
    target_shape: tuple[int, int] | None = None,
    target_shape_percentile: float = 90.0,
    shape_sample_size: int = 400,
    batch_size: int = 500,
    random_seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    include_decoys: bool = False,
) -> dict:
    """Bulk-build the CNN training dataset from multiple experiments: sample
    n_per_experiment eligible peptides per experiment, resize every sample to
    one common empirically-determined target_shape -- padding axes that
    already fit at `target_shape_percentile`, rescaling (not cropping) only
    the axes that exceed it (see pad_or_rescale_to_shape and
    compute_target_shape_per_experiment) -- and write one chunked/compressed
    HDF5 + parquet-metadata file per split (train/val/test) under
    `output_dir`.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_datapoints: list[GroundTruthDatapoint] = []
    all_sources: list[str] = []
    for swaps_dir in experiment_dirs:
        source_experiment = os.path.basename(swaps_dir.rstrip("/"))
        dps = build_experiment_ground_truth(
            swaps_dir,
            n_per_experiment,
            batch_size=batch_size,
            random_seed=random_seed,
            include_decoys=include_decoys,
        )
        all_datapoints.extend(dps)
        all_sources.extend([source_experiment] * len(dps))

    if target_shape is None:
        # Per-experiment (not pooled) percentiles -- see
        # compute_target_shape_per_experiment: pooling would understate
        # whichever experiment has the fattest tail on a given axis (e.g.
        # gradient-length differences shift the RT-extent distribution),
        # disproportionately favoring the other experiments' own tails.
        # Safe to use a lower percentile than a pad/crop-only scheme would
        # want, since the tail beyond it is rescaled (not truncated) --
        # see pad_or_rescale_to_shape.
        rng = np.random.default_rng(random_seed)
        shapes_by_experiment: dict[str, list[tuple[int, int]]] = {}
        for source in set(all_sources):
            idx = [i for i, s in enumerate(all_sources) if s == source]
            if len(idx) > shape_sample_size:
                idx = rng.choice(idx, shape_sample_size, replace=False)
            shapes_by_experiment[source] = [
                all_datapoints[i].image.shape for i in idx
            ]
        target_shape = compute_target_shape_per_experiment(
            shapes_by_experiment, percentile=target_shape_percentile
        )

    rng = np.random.default_rng(random_seed)
    order = rng.permutation(len(all_datapoints))
    n_train = int(len(order) * train_ratio)
    n_val = int(len(order) * val_ratio)
    split_indices = {
        "train": order[:n_train],
        "val": order[n_train : n_train + n_val],
        "test": order[n_train + n_val :],
    }

    manifest: dict = {"target_shape": list(target_shape), "splits": {}}
    for split, idx in split_indices.items():
        split_dps = [all_datapoints[i] for i in idx]
        split_sources = [all_sources[i] for i in idx]
        h5_path = os.path.join(output_dir, f"{split}.h5")
        meta = _write_split_hdf5(split_dps, split_sources, target_shape, h5_path)
        meta.to_parquet(os.path.join(output_dir, f"{split}_metadata.parquet"), index=False)
        manifest["splits"][split] = {"h5_path": h5_path, "n": len(split_dps)}
        Logger.info("Wrote %d samples to %s", len(split_dps), h5_path)

    manifest_path = os.path.join(output_dir, "manifest.yaml")
    with open(manifest_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f)
    return manifest
