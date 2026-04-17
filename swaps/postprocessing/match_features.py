import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional, Tuple
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
    fast_intensity_weighted_ncc,
)
import duckdb
from matplotlib.colors import ListedColormap

Logger = logging.getLogger(__name__)
_WORKER_CONTEXT: dict[str, Any] = {}


@dataclass
class QuantificationResult:
    """Container for one quantified peptide feature in one run.

    The result keeps both the raw inputs needed for later re-use as a matching
    template and the derived peak properties/snapped anchor that downstream
    anchor-family logic depends on.
    """

    run_name: str
    case: Literal["Reference", "Quant_Only", "Match"]
    image: np.ndarray
    smoothed_image: np.ndarray
    input_anchor: tuple[int, int]
    peak_properties: pd.DataFrame | None
    snapped_anchor: tuple[int, int] | None
    labels: np.ndarray | None = None
    labels_multi_markers: np.ndarray | None = None
    template_matching_score: float = np.nan
    shift: tuple[int, int] = (0, 0)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        """Return True when a valid quantified feature was produced."""

        return self.peak_properties is not None and self.snapped_anchor is not None


@dataclass
class ReferenceMatchState:
    """Mutable reference state used to update the reusable match bounding box.

    The source quantification remains the peptide's original reference feature.
    Quant-only runs can expand the bbox offsets used for future template-guided
    matching, but they do not create separate match identities.
    """

    reference_result: QuantificationResult
    match_props: pd.DataFrame
    bbox_offsets: dict[str, int]


def match_features_batches_parallel(
    dict_ref,
    raw_file_list,
    result_dir,
    peptide_indicies: np.ndarray | None = None,
    batch_size_max: int = 1500,
    max_workers: int = 4,
    processing_kwargs: dict | None = None,
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
    sorted_mz = np.sort(peptide_indicies)
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

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_match_features_worker,
        initargs=(dict_ref, raw_file_list, result_dir, processing_kwargs),
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
            ) = future.result()
            results_target.extend(res_target)
            results_decoy.extend(res_decoy)
            pp_reference_list.extend(pp_reference_target)
            pp_match_target_list.extend(pp_match_target)
            pp_match_decoy_list.extend(pp_match_decoy)
            no_quant_log.extend(no_quant)
            no_match_log.extend(no_match)
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
    )


def _init_match_features_worker(dict_ref, raw_file_list, result_dir, processing_kwargs):
    """Store immutable batch context once per worker process."""

    _WORKER_CONTEXT["dict_ref"] = dict_ref
    _WORKER_CONTEXT["raw_file_list"] = raw_file_list
    _WORKER_CONTEXT["result_dir"] = result_dir
    _WORKER_CONTEXT["processing_kwargs"] = processing_kwargs
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
    )


def _feature_instance_id(mz_rank: int, anchor_id: int) -> str:
    """Build the peptide-plus-anchor identifier used as feature-level identity."""

    return f"{mz_rank}_{anchor_id}"


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
) -> pd.DataFrame | None:
    """Add anchor-aware metadata columns to a quantified peak-properties row."""

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
    if decoy_mz_rank is not None:
        peak_properties["decoy_mz_rank"] = decoy_mz_rank
    return peak_properties


def _extract_single_peak_properties(
    peak_properties: pd.DataFrame | None,
    snapped_anchor: tuple[int, int] | None,
) -> pd.DataFrame | None:
    """Pick the row corresponding to the snapped anchor from a multi-label table."""

    if peak_properties is None or snapped_anchor is None:
        return None
    peak_properties = peak_properties.copy()
    mask = (peak_properties["snap_rt"].astype(int) == int(snapped_anchor[0])) & (
        peak_properties["snap_im"].astype(int) == int(snapped_anchor[1])
    )
    if mask.any():
        return peak_properties.loc[mask].iloc[[0]].reset_index(drop=True)
    return peak_properties.iloc[[0]].reset_index(drop=True)


def _bbox_offsets_from_prop(
    peak_properties: pd.DataFrame, anchor: tuple[int, int]
) -> dict[str, int]:
    """Express a peak bbox as offsets around a snapped anchor."""

    row = peak_properties.iloc[0]
    return {
        "top": max(int(anchor[0]) - int(row["bbox-0"]), 0),
        "bottom": max(int(row["bbox-2"]) - int(anchor[0]), 0),
        "left": max(int(anchor[1]) - int(row["bbox-1"]), 0),
        "right": max(int(row["bbox-3"]) - int(anchor[1]), 0),
    }


def _expand_offsets(
    base_offsets: dict[str, int], new_offsets: dict[str, int]
) -> dict[str, int]:
    """Expand bbox offsets to cover both the existing and the new segmentation."""

    return {
        key: max(int(base_offsets.get(key, 0)), int(new_offsets.get(key, 0)))
        for key in ["top", "bottom", "left", "right"]
    }


def _bbox_tuple_from_prop(peak_properties: pd.DataFrame) -> tuple[int, int, int, int]:
    """Return the first-row bbox as a plain integer tuple."""

    row = peak_properties.iloc[0]
    return tuple(int(row[col]) for col in ["bbox-0", "bbox-1", "bbox-2", "bbox-3"])


def _expansion_deltas_from_aligned_and_quant_bbox(
    ref_probe_new_bbox: tuple[int, int, int, int],
    ref_bbox: tuple[int, int, int, int],
) -> dict[str, int]:
    """Compute directional expansion needed to cover both aligned and quant bboxes.

    The returned values are *additional* expansion amounts beyond the current
    aligned bbox, expressed as top/bottom/left/right deltas.
    """

    return {
        "top": max(int(ref_probe_new_bbox[0]) - int(ref_bbox[0]), 0),
        "left": max(int(ref_probe_new_bbox[1]) - int(ref_bbox[1]), 0),
        "bottom": max(int(ref_probe_new_bbox[2]) - int(ref_bbox[2]), 0),
        "right": max(int(ref_probe_new_bbox[3]) - int(ref_bbox[3]), 0),
    }


def _add_offset_deltas(
    base_offsets: dict[str, int], deltas: dict[str, int]
) -> dict[str, int]:
    """Add expansion deltas onto an existing offset dictionary."""

    return {
        key: int(base_offsets.get(key, 0)) + int(deltas.get(key, 0))
        for key in ["top", "bottom", "left", "right"]
    }


def _clone_props_with_offsets(
    peak_properties: pd.DataFrame,
    anchor: tuple[int, int],
    offsets: dict[str, int],
    image_shape: tuple[int, int],
) -> pd.DataFrame:
    """Clone one-row peak properties and replace the bbox around the given anchor."""

    prop = peak_properties.copy()
    bbox0 = max(int(anchor[0]) - int(offsets["top"]), 0)
    bbox1 = max(int(anchor[1]) - int(offsets["left"]), 0)
    bbox2 = min(int(anchor[0]) + int(offsets["bottom"]), image_shape[0])
    bbox3 = min(int(anchor[1]) + int(offsets["right"]), image_shape[1])
    prop.loc[:, "bbox-0"] = bbox0
    prop.loc[:, "bbox-1"] = bbox1
    prop.loc[:, "bbox-2"] = bbox2
    prop.loc[:, "bbox-3"] = bbox3
    prop.loc[:, "rt_length"] = bbox2 - bbox0
    prop.loc[:, "im_length"] = bbox3 - bbox1
    return prop


def _anchor_inside_bbox(
    anchor: tuple[int, int] | None, peak_properties: pd.DataFrame | None
) -> bool:
    """Return True when the anchor lies inside the first-row bbox."""

    if anchor is None or peak_properties is None or peak_properties.empty:
        return False
    row = peak_properties.iloc[0]
    return int(row["bbox-0"]) <= int(anchor[0]) < int(row["bbox-2"]) and int(
        row["bbox-1"]
    ) <= int(anchor[1]) < int(row["bbox-3"])


def _quantify_peptide_run(
    act_df,
    pept_idx,
    dict_ref,
    run_name,
    case: Literal["Reference", "Quant_Only", "Match"],
    template_anchor_override: Optional[tuple[int, int]] = None,
    reference_image: np.ndarray | None = None,
    reference_props: pd.DataFrame | None = None,
    precomputed_pept_act: tuple[np.ndarray, int, int] | None = None,
    precomputed_smoothed_image: np.ndarray | None = None,
    processing_kwargs: dict | None = None,
    visualize_dir: str | None = None,
    visualize_name: str | None = None,
):
    """Quantify one peptide in one run and return the full quantification state.

    Parameters
    ----------
    template_anchor_override : tuple[int, int] | None
        Position (rt, im) in the *reference* image used to crop the template patch
        for cross-correlation matching (passed as ``anchor`` to
        ``quantify_from_coords``).  Represents where the peak lives in the reference
        run.  Defaults to ``(rt_msms_pos, im_msms_pos)`` when ``None``.
    """

    if precomputed_pept_act is None:
        pept_act, rt_msms_pos, im_msms_pos = get_pept_act_from_parquet(
            act_df, pept_idx, dict_ref, run_name
        )
    else:
        pept_act, rt_msms_pos, im_msms_pos = precomputed_pept_act
    template_anchor = (
        template_anchor_override
        if template_anchor_override
        else (rt_msms_pos, im_msms_pos)
    )
    if visualize_name is None:
        visualize_name = f"mz{pept_idx}_{run_name}_{case.lower()}.png"
    match case:
        case "Match":
            result = quantify_from_coords(
                pept_act,
                template_anchor=(int(template_anchor[0]), int(template_anchor[1])),
                reference_image=reference_image,
                propA=reference_props,
                pre_smoothed_image=precomputed_smoothed_image,
                patch_size=min(pept_act.shape),
                **(processing_kwargs or {}),
                visualize_dir=visualize_dir,
                visualize_filename=visualize_name,
            )
        case "Reference":
            result = quantify_from_coords(
                pept_act,
                template_anchor=(int(template_anchor[0]), int(template_anchor[1])),
                pre_smoothed_image=precomputed_smoothed_image,
                patch_size=min(pept_act.shape),
                **(processing_kwargs or {}),
                visualize_dir=visualize_dir,
                visualize_filename=visualize_name,
            )

        case _:
            raise ValueError(f"Unknown case: {case}")
    if isinstance(result, QuantificationResult):
        result.run_name = run_name
        result.case = case
        result.metadata["pept_idx"] = pept_idx
        return result
    else:
        return None


def _attach_quant_only_metadata(
    prop_t: pd.DataFrame,
    record: dict[str, Any],
    match_state: "ReferenceMatchState",
) -> None:
    """Attach quant-only diagnostic metadata columns to a matched peak-properties row."""
    quant_direct: QuantificationResult = record["direct"]
    if quant_direct.snapped_anchor is not None:
        prop_t["quant_direct_snap_rt"] = int(quant_direct.snapped_anchor[0])
        prop_t["quant_direct_snap_im"] = int(quant_direct.snapped_anchor[1])
    if quant_direct.peak_properties is not None:
        for i, col in enumerate(["bbox-0", "bbox-1", "bbox-2", "bbox-3"]):
            prop_t[f"quant_direct_bbox_{i}"] = int(
                quant_direct.peak_properties[col].values[0]
            )
    prop_t["quant_anchor_inside_match_bbox"] = bool(record["inside_match_bbox"])
    prop_t["quant_anchor_expanded_match_bbox"] = bool(record["expanded_match_bbox"])
    prop_t["bbox_updated_from_original"] = bool(record["bbox_updated_from_original"])
    for side in ("top", "bottom", "left", "right"):
        prop_t[f"reference_bbox_{side}"] = int(match_state.bbox_offsets[side])


def _visualize_probe_expansion(
    smoothed_image_raw_file: np.ndarray,
    smoothed_image_ref_file: np.ndarray,
    quant_direct_snapped_anchor: tuple[int, int],
    quant_direct_bbox: tuple[int, int, int, int],
    ref_snapped_anchor: tuple[int, int],
    ref_bbox: tuple[int, int, int, int],
    ref_probe_new_bbox: tuple[int, int, int, int],
    match_props_bbox_before: tuple[int, int, int, int],
    match_props_bbox_after: tuple[int, int, int, int],
    mz_rank: int,
    raw_file: str,
    bbox_version: int,
    visualization_dir: str,
) -> None:
    """Visualize a bbox expansion event triggered by a quant-only probe run.

    Two panels:
      1. Smoothed activation image of *raw_file*:
         - Yellow star : quant_direct.snapped_anchor
         - Blue solid  : quant_direct segmentation bbox
      2. Smoothed activation image of the reference run:
         - Yellow star : reference_result.snapped_anchor
         - White solid : original reference segmentation bbox
         - Blue dashed : ref_probe_new_bbox (aligned probe bbox on reference space)
         - Red solid   : match_state.match_props bbox *before* _clone_props_with_offsets
         - Red dashed  : match_state.match_props bbox *after* _clone_props_with_offsets

    The legend is placed to the right of both panels without overlapping them.
    """
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt

    def _draw_bbox(
        ax, bbox: tuple[int, int, int, int], color: str, linestyle: str
    ) -> None:
        rect = plt.Rectangle(
            (bbox[1], bbox[0]),
            bbox[3] - bbox[1],
            bbox[2] - bbox[0],
            edgecolor=color,
            facecolor="none",
            linestyle=linestyle,
            linewidth=1.5,
        )
        ax.add_patch(rect)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: quant-only run smoothed image
    ax0 = axes[0]
    ax0.set_title(f"{raw_file}\n(smoothed)", fontsize=8)
    ax0.imshow(smoothed_image_raw_file, aspect="auto", origin="lower")
    ax0.plot(
        quant_direct_snapped_anchor[1],
        quant_direct_snapped_anchor[0],
        "*",
        markersize=10,
        markeredgewidth=1.5,
        color="yellow",
    )
    _draw_bbox(ax0, quant_direct_bbox, color="blue", linestyle="solid")

    # Panel 2: reference run smoothed image
    ax1 = axes[1]
    ax1.set_title(f"reference\n(smoothed)", fontsize=8)
    ax1.imshow(smoothed_image_ref_file, aspect="auto", origin="lower")
    ax1.plot(
        ref_snapped_anchor[1],
        ref_snapped_anchor[0],
        "*",
        markersize=10,
        markeredgewidth=1.5,
        color="yellow",
    )
    _draw_bbox(ax1, ref_bbox, color="white", linestyle="solid")
    _draw_bbox(ax1, ref_probe_new_bbox, color="blue", linestyle="dashed")
    _draw_bbox(ax1, match_props_bbox_before, color="red", linestyle="solid")
    _draw_bbox(ax1, match_props_bbox_after, color="red", linestyle="dashed")

    legend_handles = [
        mlines.Line2D(
            [],
            [],
            marker="*",
            color="yellow",
            linestyle="None",
            markersize=8,
            label="snapped anchor",
        ),
        mpatches.Patch(
            edgecolor="blue",
            facecolor="none",
            linestyle="solid",
            label="quant_direct bbox (panel 1)",
        ),
        mpatches.Patch(
            edgecolor="white", facecolor="none", label="original ref bbox (panel 2)"
        ),
        mpatches.Patch(
            edgecolor="blue",
            facecolor="none",
            linestyle="dashed",
            label="ref_probe_new_bbox (panel 2)",
        ),
        mpatches.Patch(
            edgecolor="red",
            facecolor="none",
            linestyle="solid",
            label="match_props bbox before update (panel 2)",
        ),
        mpatches.Patch(
            edgecolor="red",
            facecolor="none",
            linestyle="dashed",
            label="match_props bbox after update (panel 2)",
        ),
    ]
    fig.subplots_adjust(right=0.72)
    fig.legend(
        handles=legend_handles,
        loc="center right",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=8,
        framealpha=0.85,
        frameon=True,
    )

    os.makedirs(visualization_dir, exist_ok=True)
    fname = f"probe_expansion_mz{mz_rank}_{raw_file}_bbox_v{bbox_version}.png"
    fig.savefig(os.path.join(visualization_dir, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


def match_features_batch(
    dict_ref,
    raw_file_list,
    result_dir,
    batch,
    processing_kwargs: dict | None = None,
    visualize_dir: str | None = None,
    match_decoy: bool = True,
):
    """Process one peptide batch with sequential quant-only bbox expansion.

    Workflow per peptide:
    1. Quantify the main reference run once.
    2. Use the reference to template-match each quant-only run.
    3. If a quant-only run's own snapped anchor falls outside the matched bbox,
       expand the reusable reference bbox.
    4. Record which quant-only runs were already inside the bbox and which ones
       caused an expansion.
    5. Match all regular `Match` and `Quant_only` runs using the final expanded bbox.
    """
    results_target, results_decoy = [], []
    pp_reference_list, pp_match_target_list, pp_match_decoy_list = [], [], []
    no_quant_log, no_match_log = [], []
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
    smooth_kwargs = dict((processing_kwargs or {}).get("smooth_kwargs", {}))

    # Load activation data for this mz_rank batch from the pre-built sorted parquet.
    # DuckDB skips row groups outside [min(batch_np), max(batch_np)], so I/O scales
    # with batch_size / N_total.  Requires build_mz_sorted_activation() to have been
    # run for each raw_file activation directory beforehand.
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    act_dfs = {
        raw_file: load_peptide_batch_df_from_partquet(
            os.path.join(result_dir, raw_file, "activation"),
            batch_np,
            con=con,
        ).set_index("mz_rank", drop=False)
        for raw_file in raw_file_list
    }
    con.close()

    def _select_mz(df_indexed: pd.DataFrame, mz_rank: int) -> pd.DataFrame:
        try:
            return df_indexed.loc[[mz_rank]]
        except KeyError:
            return df_indexed.iloc[0:0]

    for pept_idx in batch_np:
        pept_act_cache: dict[str, tuple[np.ndarray, int, int]] = {}
        pept_act_smoothed_cache: dict[str, np.ndarray] = {}

        def _get_pept_act_tuple(raw_file: str) -> tuple[np.ndarray, int, int]:
            if raw_file not in pept_act_cache:
                pept_act_cache[raw_file] = get_pept_act_from_parquet(
                    _select_mz(act_dfs[raw_file], int(pept_idx)),
                    int(pept_idx),
                    dict_ref_by_mz,
                    raw_file,
                )
            return pept_act_cache[raw_file]

        def _get_smoothed_pept_act(raw_file: str) -> np.ndarray:
            if raw_file not in pept_act_smoothed_cache:
                pept_act_smoothed_cache[raw_file] = smooth_and_denoise_image(
                    _get_pept_act_tuple(raw_file)[0], **smooth_kwargs
                )
            return pept_act_smoothed_cache[raw_file]

        row_series = dict_ref_by_mz.loc[pept_idx, :]
        str_values = row_series[row_series.map(lambda x: isinstance(x, str))]
        reference_raw_file = str(str_values.index[(str_values == "Reference")][0])
        quant_only_raw_files = str_values.index[str_values == "Quant_Only"].tolist()
        match_raw_files = str_values.index[
            (str_values.str.contains("Match", regex=False))
            | (str_values == "Quant_Only")
        ].tolist()

        own_anchor_id = 0
        feature_instance_id = _feature_instance_id(pept_idx, own_anchor_id)
        match_state: ReferenceMatchState | None = None

        # --- Step 1: Quantify reference run ---
        reference_result = _quantify_peptide_run(
            act_df=_select_mz(act_dfs[reference_raw_file], pept_idx),
            pept_idx=pept_idx,
            dict_ref=dict_ref,
            run_name=reference_raw_file,
            case="Reference",
            precomputed_pept_act=_get_pept_act_tuple(reference_raw_file),
            precomputed_smoothed_image=_get_smoothed_pept_act(reference_raw_file),
            processing_kwargs=processing_kwargs,
            visualize_dir=visualize_dir,
        )
        if reference_result.succeeded:
            prop_ref = _annotate_peak_properties(
                reference_result.peak_properties,
                mz_rank=pept_idx,
                run_name=reference_raw_file,
                own_anchor_id=own_anchor_id,
                assimilated_to_anchor_id=own_anchor_id,
                feature_instance_id=feature_instance_id,
                own_feature_instance_id=feature_instance_id,
                source_run=reference_raw_file,
                source_type="Reference",
            )
            if prop_ref is not None:
                reference_result.peak_properties = prop_ref
                pp_reference_list.append(prop_ref)
                match_state = ReferenceMatchState(
                    reference_result=reference_result,
                    match_props=prop_ref.copy(),
                    bbox_offsets=_bbox_offsets_from_prop(
                        prop_ref,
                        reference_result.snapped_anchor,  # type: ignore[arg-type]
                    ),
                )
        else:
            no_quant_log.append(
                {
                    "mz_rank": pept_idx,
                    "run_name": reference_raw_file,
                    "type": "reference",
                }
            )

        # --- Step 2: Probe quant-only runs and expand bbox ---
        bbox_version = 0
        quant_only_probe_records: dict[str, dict[str, Any]] = {}

        for raw_file in quant_only_raw_files:
            quant_direct = _quantify_peptide_run(
                act_df=_select_mz(act_dfs[raw_file], pept_idx),
                pept_idx=pept_idx,
                dict_ref=dict_ref,
                run_name=raw_file,
                case="Reference",
                precomputed_pept_act=_get_pept_act_tuple(raw_file),
                precomputed_smoothed_image=_get_smoothed_pept_act(raw_file),
                processing_kwargs=processing_kwargs,
                visualize_dir=visualize_dir,
            )
            record: dict[str, Any] = {
                "direct": quant_direct,
                "probe": None,
                "probe_bbox_version": None,
                "inside_match_bbox": False,
                "expanded_match_bbox": False,
                "bbox_updated_from_original": False,
                "reference_bbox_before_update": None,
                "reference_bbox_after_update": None,
                "aligned_bbox_before_update": None,
                "aligned_bbox_after_update": None,
            }
            quant_only_probe_records[raw_file] = record

            if match_state is None:
                if quant_direct.succeeded:
                    # Reference run failed — promote this quant-only run as the reference
                    prop_promoted = _annotate_peak_properties(
                        quant_direct.peak_properties,
                        mz_rank=pept_idx,
                        run_name=raw_file,
                        own_anchor_id=own_anchor_id,
                        assimilated_to_anchor_id=own_anchor_id,
                        feature_instance_id=feature_instance_id,
                        own_feature_instance_id=feature_instance_id,
                        source_run=raw_file,
                        source_type="Reference",
                    )
                    if prop_promoted is not None:
                        quant_direct.peak_properties = prop_promoted
                        pp_reference_list.append(prop_promoted)
                        match_state = ReferenceMatchState(
                            reference_result=quant_direct,
                            match_props=prop_promoted.copy(),
                            bbox_offsets=_bbox_offsets_from_prop(
                                prop_promoted,
                                quant_direct.snapped_anchor,  # type: ignore[arg-type]
                            ),
                        )
                        match_raw_files = [
                            f for f in match_raw_files if f != raw_file
                        ] + [reference_raw_file]
                        reference_raw_file = raw_file
                        Logger.info(
                            f"mz{pept_idx}: Promoted quant-only run as reference: {raw_file}"
                        )
                else:
                    no_quant_log.append(
                        {
                            "mz_rank": pept_idx,
                            "run_name": raw_file,
                            "type": "reference_promoted",
                        }
                    )
                continue  # promoted run skips probe; failed promotion also skips

            if not quant_direct.succeeded:
                continue  # cannot probe without a valid reference

            quant_probe = _quantify_peptide_run(  # match quant_only to reference, this one is on reference space
                act_df=_select_mz(act_dfs[reference_raw_file], pept_idx),
                pept_idx=pept_idx,
                dict_ref=dict_ref,
                run_name=reference_raw_file,
                case="Match",
                template_anchor_override=quant_direct.snapped_anchor,
                # anchor_override=quant_direct.snapped_anchor,
                reference_image=quant_direct.smoothed_image,
                reference_props=quant_direct.peak_properties,
                precomputed_pept_act=_get_pept_act_tuple(reference_raw_file),
                precomputed_smoothed_image=_get_smoothed_pept_act(reference_raw_file),
                processing_kwargs=processing_kwargs,
                visualize_dir=visualize_dir,
                visualize_name=f"mz{pept_idx}_{raw_file}_quant_only_probe.png",
            )
            record["probe"] = quant_probe
            record["probe_bbox_version"] = bbox_version
            if not quant_probe.succeeded:
                continue  # probe failed, cannot use for bbox expansion or future matching

            inside_match_bbox = _anchor_inside_bbox(
                quant_probe.snapped_anchor, match_state.reference_result.peak_properties
            )
            record["inside_match_bbox"] = bool(inside_match_bbox)

            if not inside_match_bbox:
                ref_probe_new_bbox = _bbox_tuple_from_prop(quant_probe.peak_properties)
                ref_bbox = _bbox_tuple_from_prop(
                    match_state.reference_result.peak_properties
                )
                record["reference_bbox_before_update"] = _bbox_tuple_from_prop(
                    match_state.match_props
                )
                # record["aligned_bbox_before_update"] = ref_probe_new_bbox
                expansion_deltas = _expansion_deltas_from_aligned_and_quant_bbox(
                    ref_probe_new_bbox, ref_bbox
                )
                expanded_offsets = _add_offset_deltas(
                    match_state.bbox_offsets, expansion_deltas
                )
                bbox_changed = expanded_offsets != match_state.bbox_offsets
                match_state.bbox_offsets = expanded_offsets
                match_state.match_props = _clone_props_with_offsets(
                    match_state.match_props,
                    match_state.reference_result.snapped_anchor,  # type: ignore[arg-type]
                    match_state.bbox_offsets,
                    match_state.reference_result.image.shape,  # type: ignore[arg-type]
                )
                record["expanded_match_bbox"] = True
                record["bbox_updated_from_original"] = bool(bbox_changed)
                record["reference_bbox_after_update"] = _bbox_tuple_from_prop(
                    match_state.match_props
                )
                if (
                    visualize_dir is not None
                    and quant_direct.peak_properties is not None
                ):
                    _visualize_probe_expansion(
                        smoothed_image_raw_file=_get_smoothed_pept_act(raw_file),
                        smoothed_image_ref_file=_get_smoothed_pept_act(
                            reference_raw_file
                        ),
                        quant_direct_snapped_anchor=quant_direct.snapped_anchor,  # type: ignore[arg-type]
                        quant_direct_bbox=_bbox_tuple_from_prop(
                            quant_direct.peak_properties
                        ),
                        ref_snapped_anchor=match_state.reference_result.snapped_anchor,  # type: ignore[arg-type]
                        ref_bbox=ref_bbox,
                        ref_probe_new_bbox=ref_probe_new_bbox,
                        match_props_bbox_before=record["reference_bbox_before_update"],
                        match_props_bbox_after=record["reference_bbox_after_update"],
                        mz_rank=pept_idx,
                        raw_file=raw_file,
                        bbox_version=bbox_version,
                        visualization_dir=visualize_dir,
                    )
                if bbox_changed:
                    bbox_version += 1

        # --- Step 3: Match all non-reference runs ---
        if not match_raw_files:
            continue

        if match_state is None:
            Logger.info(
                f"mz{pept_idx}: No valid reference for matching; {len(quant_only_raw_files)} quant-only runs;"
                f"logging all {len(match_raw_files)} non-reference runs as no-quant."
            )
            # Reference failed: log all non-reference runs as failures and move on
            for raw_file in match_raw_files:
                is_quant_only = raw_file in quant_only_probe_records
                no_quant_log.append(
                    {
                        "mz_rank": pept_idx,
                        "run_name": raw_file,
                        "type": "quant_only" if is_quant_only else "match_target",
                    }
                )
                if not is_quant_only:
                    if match_decoy:
                        no_quant_log.append(
                            {
                                "mz_rank": pept_idx,
                                "run_name": raw_file,
                                "type": "match_decoy",
                            }
                        )
                    no_match_log.append(
                        {
                            "mz_rank": pept_idx,
                            "run_name": raw_file,
                            "type": "match_target",
                        }
                    )
                    if match_decoy:
                        no_match_log.append(
                            {
                                "mz_rank": pept_idx,
                                "run_name": raw_file,
                                "type": "match_decoy",
                            }
                        )
            continue

        if match_decoy:
            batch_exclude = batch_np[batch_np != pept_idx]

        for raw_file in match_raw_files:
            target_act_df = _select_mz(act_dfs[raw_file], pept_idx)
            is_quant_only_run = raw_file in quant_only_probe_records

            target_result = _quantify_peptide_run(
                act_df=target_act_df,
                pept_idx=pept_idx,
                dict_ref=dict_ref,
                run_name=raw_file,
                case="Match",
                template_anchor_override=match_state.reference_result.snapped_anchor,
                # anchor_override=match_state.reference_result.snapped_anchor,
                reference_image=match_state.reference_result.smoothed_image,
                reference_props=match_state.match_props,
                precomputed_pept_act=_get_pept_act_tuple(raw_file),
                precomputed_smoothed_image=_get_smoothed_pept_act(raw_file),
                processing_kwargs=processing_kwargs,
                visualize_dir=visualize_dir,
            )

            if match_decoy:
                decoy_pept_idx = np.random.choice(batch_exclude)
                decoy_act_df = _select_mz(act_dfs[raw_file], int(decoy_pept_idx))
                decoy_act, _, _ = get_pept_act_from_parquet(
                    decoy_act_df,
                    decoy_pept_idx,
                    dict_ref,
                    raw_file,
                    shape=target_result.image.shape,  # here enforce the shape of target image
                )
                decoy_result = _quantify_peptide_run(
                    act_df=decoy_act_df,
                    pept_idx=decoy_pept_idx,
                    dict_ref=dict_ref,
                    run_name=raw_file,
                    case="Match",
                    template_anchor_override=match_state.reference_result.snapped_anchor,
                    # anchor_override=decoy_anchor,
                    reference_image=match_state.reference_result.smoothed_image,
                    reference_props=match_state.match_props,
                    precomputed_pept_act=(
                        decoy_act,
                        int(match_state.reference_result.snapped_anchor[0]),
                        int(match_state.reference_result.snapped_anchor[1]),
                    ),
                    processing_kwargs=processing_kwargs,
                    visualize_dir=visualize_dir,
                    visualize_name=f"mz{pept_idx}_{raw_file}_match_decoy{decoy_pept_idx}.png",
                )

            prop_t = _annotate_peak_properties(
                target_result.peak_properties,
                mz_rank=pept_idx,
                run_name=raw_file,
                own_anchor_id=own_anchor_id,
                assimilated_to_anchor_id=own_anchor_id,
                feature_instance_id=feature_instance_id,
                own_feature_instance_id=feature_instance_id,
                source_run=reference_raw_file,
                source_type="Quant_Only" if is_quant_only_run else "Reference",
            )
            prop_d = (
                _annotate_peak_properties(
                    decoy_result.peak_properties,
                    mz_rank=pept_idx,
                    run_name=raw_file,
                    own_anchor_id=own_anchor_id,
                    assimilated_to_anchor_id=own_anchor_id,
                    feature_instance_id=feature_instance_id,
                    own_feature_instance_id=feature_instance_id,
                    source_run=reference_raw_file,
                    source_type="Quant_Only" if is_quant_only_run else "Reference",
                    decoy_mz_rank=int(decoy_pept_idx),
                )
                if match_decoy
                else None
            )

            if is_quant_only_run and prop_t is not None:
                _attach_quant_only_metadata(
                    prop_t, quant_only_probe_records[raw_file], match_state
                )
                if (
                    visualize_dir is not None
                    and quant_only_probe_records[raw_file]["bbox_updated_from_original"]
                    and quant_only_probe_records[raw_file][
                        "reference_bbox_before_update"
                    ]
                    is not None
                    and quant_only_probe_records[raw_file][
                        "reference_bbox_after_update"
                    ]
                    is not None
                    and quant_only_probe_records[raw_file]["aligned_bbox_before_update"]
                    is not None
                    and quant_only_probe_records[raw_file]["aligned_bbox_after_update"]
                    is not None
                ):
                    _visualize_quant_only_bbox_expansion(
                        pept_idx=pept_idx,
                        raw_file=raw_file,
                        match_state=match_state,
                        target_result=target_result,
                        prop_t=prop_t,
                        record=quant_only_probe_records[raw_file],
                        visualize_dir=visualize_dir,
                    )

            if prop_t is not None:
                target_result.peak_properties = prop_t
                pp_match_target_list.append(prop_t)
                match_t = compare_peak_properties(
                    match_state.reference_result.peak_properties, prop_t
                )
                match_t["mz_rank"] = pept_idx
                match_t["feature_instance_id"] = feature_instance_id
                match_t["own_anchor_id"] = own_anchor_id
                match_t["assimilated_to_anchor_id"] = own_anchor_id
                match_t["source_run"] = reference_raw_file
                match_t["source_type"] = (
                    "Quant_Only" if is_quant_only_run else "Reference"
                )
                results_target.append(match_t)
            else:
                no_quant_log.append(
                    {
                        "mz_rank": pept_idx,
                        "run_name": raw_file,
                        "type": "quant_only" if is_quant_only_run else "match_target",
                        "feature_instance_id": feature_instance_id,
                    }
                )
                no_match_log.append(
                    {
                        "mz_rank": pept_idx,
                        "run_name": raw_file,
                        "type": "match_target",
                        "feature_instance_id": feature_instance_id,
                    }
                )

            if match_decoy:
                if prop_d is not None:
                    decoy_result.peak_properties = prop_d
                    pp_match_decoy_list.append(prop_d)
                    match_d = compare_peak_properties(
                        match_state.reference_result.peak_properties, prop_d
                    )
                    match_d["mz_rank"] = pept_idx
                    match_d["decoy_mz_rank"] = decoy_pept_idx
                    match_d["feature_instance_id"] = feature_instance_id
                    match_d["own_anchor_id"] = own_anchor_id
                    match_d["assimilated_to_anchor_id"] = own_anchor_id
                    match_d["source_run"] = reference_raw_file
                    match_d["source_type"] = "Reference"
                    results_decoy.append(match_d)
                else:
                    no_quant_log.append(
                        {
                            "mz_rank": pept_idx,
                            "run_name": raw_file,
                            "type": "match_decoy",
                            "feature_instance_id": feature_instance_id,
                        }
                    )
                    no_match_log.append(
                        {
                            "mz_rank": pept_idx,
                            "run_name": raw_file,
                            "type": "match_decoy",
                            "feature_instance_id": feature_instance_id,
                        }
                    )

    return (
        results_target,
        results_decoy,
        pp_reference_list,
        pp_match_target_list,
        pp_match_decoy_list,
        no_quant_log,
        no_match_log,
    )


def _visualize_quantify_from_coords(
    reference_image,
    pept_act_image,
    pept_act_image_smoothed,
    save_dir: str,
    bbox_center: Optional[Tuple[int, int]] = None,
    target_msms_pos: Optional[Tuple[int, int]] = None,
    target_snapped_msms_pos: Optional[Tuple[int, int]] = None,
    template_msms_pos: Optional[Tuple[int, int]] = None,
    template_snapped_msms_pos: Optional[Tuple[int, int]] = None,
    template_box: Optional[Tuple[int, int, int, int]] = None,
    seg_box: Optional[Tuple[int, int, int, int]] = None,
    matched_template_box: Optional[Tuple[int, int, int, int]] = None,
    filename: str = "quantify_from_coords.png",
    labels: np.ndarray | None = None,
):
    """Visualize feature quantification with a fixed 5-panel layout.

    Panels (left to right):
      1. Reference                     – N/A when reference_image is None
      2. pept_act_image (Raw)
      3. pept_act_image_smoothed
      4. pept_act_image_smoothed_aligned
      5. watershed_labels              – N/A when labels is None

    Each panel carries a legend explaining the overlaid markers.
    """
    import matplotlib.pyplot as plt

    LEGEND = (
        "+ red = center of label bbox\n"
        "\u2605 white = MS/MS pos\n"
        "\u2605 yellow = snapped\n"
        "\u2014 red = template bbox\n"
        "-- red = matched template bbox\n"
        "-- blue = segmentation bbox"
    )

    def _draw_panel(
        ax,
        img,
        title,
        draw_markers=True,
        labels_local=None,
        template_box=None,
        matched_template_box=None,
        bbox_center=None,
        target_msms_pos=None,
        target_snapped_msms_pos=None,
        template_msms_pos=None,
        template_snapped_msms_pos=None,
        seg_box=None,
    ):
        ax.set_title(title, fontsize=9)
        if img is None:
            ax.set_facecolor("#f0f0f0")
            ax.text(
                0.5,
                0.5,
                "N/A",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
                color="gray",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            return
        ax.imshow(img, aspect="auto", origin="lower")
        if labels_local is not None:
            masked = np.ma.masked_where(labels_local == 0, labels_local)
            ax.imshow(
                masked,
                aspect="auto",
                origin="lower",
                cmap="tab10",
                interpolation="nearest",
                alpha=0.35,
            )
        if draw_markers:
            if bbox_center is not None:
                bc = (
                    tuple(bbox_center[0])
                    if isinstance(bbox_center, np.ndarray)
                    else bbox_center
                )
                ax.plot(bc[1], bc[0], "r+", markersize=10, markeredgewidth=2)
            if target_msms_pos is not None:
                ax.plot(
                    target_msms_pos[1],
                    target_msms_pos[0],
                    "*",
                    markersize=10,
                    markeredgewidth=2,
                    color="white",
                )
            if (
                target_snapped_msms_pos is not None
                and len(target_snapped_msms_pos) == 2
            ):
                ax.plot(
                    target_snapped_msms_pos[1],
                    target_snapped_msms_pos[0],
                    "*",
                    markersize=10,
                    markeredgewidth=2,
                    color="yellow",
                )
            if template_msms_pos is not None:
                ax.plot(
                    template_msms_pos[1],
                    template_msms_pos[0],
                    "*",
                    markersize=10,
                    markeredgewidth=2,
                    color="white",
                )
            if (
                template_snapped_msms_pos is not None
                and len(template_snapped_msms_pos) == 2
            ):
                ax.plot(
                    template_snapped_msms_pos[1],
                    template_snapped_msms_pos[0],
                    "*",
                    markersize=10,
                    markeredgewidth=2,
                    color="yellow",
                )
        if template_box is not None:
            rect = plt.Rectangle(
                (template_box[1], template_box[0]),
                template_box[3] - template_box[1],
                template_box[2] - template_box[0],
                edgecolor="red",
                facecolor="none",
                linestyle="solid",
                linewidth=1.5,
            )
            ax.add_patch(rect)
        if matched_template_box is not None:
            rect = plt.Rectangle(
                (matched_template_box[1], matched_template_box[0]),
                matched_template_box[3] - matched_template_box[1],
                matched_template_box[2] - matched_template_box[0],
                edgecolor="red",
                facecolor="none",
                linestyle="dashed",
                linewidth=1.5,
            )
            ax.add_patch(rect)
        if seg_box is not None:
            rect = plt.Rectangle(
                (seg_box[1], seg_box[0]),
                seg_box[3] - seg_box[1],
                seg_box[2] - seg_box[0],
                edgecolor="blue",
                facecolor="none",
                linestyle="dashed",
                linewidth=1.5,
            )
            ax.add_patch(rect)

    fig, axes = plt.subplots(1, 4, figsize=(25, 5))

    # Panel 1: Reference (N/A for reference-only runs)
    _draw_panel(
        axes[0],
        reference_image,
        "Reference",
        draw_markers=True,
        template_box=template_box,
        template_msms_pos=template_msms_pos,
        template_snapped_msms_pos=template_snapped_msms_pos,
    )

    # Panel 2: Raw image
    _draw_panel(
        axes[1],
        pept_act_image,
        "pept_act_image (Raw)",
        matched_template_box=matched_template_box,
        target_msms_pos=target_msms_pos,
        target_snapped_msms_pos=target_snapped_msms_pos,
    )

    # Panel 3: Smoothed
    _draw_panel(
        axes[2],
        pept_act_image_smoothed,
        "pept_act_image_smoothed",
        matched_template_box=matched_template_box,
        seg_box=seg_box,
        target_msms_pos=target_msms_pos,
        target_snapped_msms_pos=target_snapped_msms_pos,
        bbox_center=bbox_center,
    )

    # Panel 4: Watershed labels standalone
    ax_lbl = axes[3]
    ax_lbl.set_title("labels", fontsize=9)
    if labels is None:
        ax_lbl.set_facecolor("#f0f0f0")
        ax_lbl.text(
            0.5,
            0.5,
            "N/A",
            ha="center",
            va="center",
            transform=ax_lbl.transAxes,
            fontsize=12,
            color="gray",
        )
        ax_lbl.set_xticks([])
        ax_lbl.set_yticks([])
    else:
        masked = np.ma.masked_where(labels == 0, labels)
        ax_lbl.imshow(np.zeros_like(labels), aspect="auto", origin="lower", cmap="gray")
        ax_lbl.imshow(
            masked,
            aspect="auto",
            origin="lower",
            cmap="tab10",
            interpolation="nearest",
        )
        for lbl_val in np.unique(labels):
            if lbl_val == 0:
                continue
            ys, xs = np.where(labels == lbl_val)
            ax_lbl.text(
                xs.mean(),
                ys.mean(),
                str(lbl_val),
                ha="center",
                va="center",
                fontsize=7,
                color="white",
                fontweight="bold",
            )
        if bbox_center is not None:
            bc = (
                tuple(bbox_center[0])
                if isinstance(bbox_center, np.ndarray)
                else bbox_center
            )
            ax_lbl.plot(bc[1], bc[0], "r+", markersize=10, markeredgewidth=2)
        if target_msms_pos is not None:
            ax_lbl.plot(
                target_msms_pos[1],
                target_msms_pos[0],
                "*",
                markersize=10,
                markeredgewidth=2,
                color="white",
            )
        if target_snapped_msms_pos is not None and len(target_snapped_msms_pos) == 2:
            ax_lbl.plot(
                target_snapped_msms_pos[1],
                target_snapped_msms_pos[0],
                "*",
                markersize=10,
                markeredgewidth=2,
                color="yellow",
            )
    fig.subplots_adjust(right=0.80)
    fig.text(
        0.815,
        0.5,
        LEGEND,
        fontsize=8,
        va="center",
        ha="left",
        transform=fig.transFigure,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="black", alpha=0.75),
        color="white",
        linespacing=1.6,
    )
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)


def quantify_from_coords(
    pept_act_image,
    template_anchor,
    reference_image: np.ndarray | None = None,
    propA: pd.DataFrame | None = None,
    pre_smoothed_image: np.ndarray | None = None,
    apply_seg: bool = True,
    smooth_kwargs: dict | None = None,
    peak_kwargs: dict | None = None,
    align_kwargs: dict | None = None,
    filter_kwargs: dict | None = None,
    patch_size: int | None = None,
    visualize_dir: str | None = None,
    visualize_filename: str = "quantify_from_coords.png",
):
    """
    Quantify features from a peptide activity image given anchor coordinates and optional reference information.
    Parameters
    ----------
    pept_act_image : np.ndarray
        The peptide activity image.
    template_anchor : tuple
        The template/source anchor coordinates (row, column).
        If no reference_image is given, then it's the Reference case
        and it will be used on the target image as well.
    reference_image : np.ndarray | None, optional
        The reference image for template matching.
    propA : pd.DataFrame | None, optional
        The properties dataframe for template matching.
    smooth_kwargs : dict | None, optional
        Keyword arguments for smoothing the image.
    peak_kwargs : dict | None, optional
        Keyword arguments for peak detection.
    align_kwargs : dict | None, optional
        Keyword arguments for alignment.
    patch_size : int | None, optional
        The size of the patch to extract.
    visualize_dir : str | None, optional
        The directory to save visualizations.
    visualize_filename : str, optional
        The filename for the visualization.

    Returns
    -------
    QuantificationResult
        Structured quantification output containing the smoothed image, peak
        properties, snapped anchor, watershed labels, and template-matching
        metadata used by downstream anchor-family logic.
    """
    if reference_image is not None and (
        template_anchor[0] >= reference_image.shape[0]
        or template_anchor[1] >= reference_image.shape[1]
        or template_anchor[0] < 0
        or template_anchor[1] < 0
    ):
        logging.warning(
            "Anchor coordinates %s are out of bounds of the reference image dimensions %s.",
            template_anchor,
            reference_image.shape,
        )
        return QuantificationResult(
            run_name="",
            case="Reference",
            image=pept_act_image,
            smoothed_image=pept_act_image,
            input_anchor=(int(template_anchor[0]), int(template_anchor[1])),
            peak_properties=None,
            snapped_anchor=None,
        )

    template_anchor = np.array([(int(template_anchor[0]), int(template_anchor[1]))])

    smooth_kwargs = {} if smooth_kwargs is None else dict(smooth_kwargs)
    peak_kwargs = {} if peak_kwargs is None else dict(peak_kwargs)
    align_kwargs = {} if align_kwargs is None else dict(align_kwargs)
    filter_kwargs = {} if filter_kwargs is None else dict(filter_kwargs)
    if "min_peak_area" not in filter_kwargs:
        filter_kwargs["min_peak_area"] = 10
    if "min_peak_sum_intensity" not in filter_kwargs:
        filter_kwargs["min_peak_sum_intensity"] = 500
    if "int_threshold" not in peak_kwargs:
        peak_kwargs["int_threshold"] = 1
    if "threshold_rel" not in peak_kwargs:
        peak_kwargs["threshold_rel"] = 0.2
    if "min_distance" not in peak_kwargs:
        peak_kwargs["min_distance"] = 10

    pept_act_image_smoothed = (
        pre_smoothed_image.copy()
        if pre_smoothed_image is not None
        else smooth_and_denoise_image(pept_act_image, **smooth_kwargs)
    )
    # Case "Match": perform template matching to find the best match for the reference peak
    # and then run watershed with the matched position as (updated) anchor
    target_anchor_shifted = None
    if reference_image is not None and propA is not None:
        # Getting template for "Match" case
        template_im_start = max(
            (template_anchor[0][1] - 0.3 * reference_image.shape[1]).astype(int), 0
        )
        template_im_end = min(
            (template_anchor[0][1] + 0.3 * reference_image.shape[1]).astype(int),
            reference_image.shape[1],
        )
        template_rt_start = max(
            (template_anchor[0][0] - 0.3 * reference_image.shape[0]).astype(int), 0
        )
        template_rt_end = min(
            (template_anchor[0][0] + 0.3 * reference_image.shape[0]).astype(int),
            reference_image.shape[0],
        )  # Use up to 36% of the image size as the template size to
        # make sure the template can cover the peak region even when the
        # anchor is not very accurate, which can be common for low abundance peptides with weak MS/MS signal
        template = reference_image[
            template_rt_start:template_rt_end,
            template_im_start:template_im_end,
        ]  # template is larger than the segementation to make template matching more robust

        template_match_result = match_template(pept_act_image_smoothed, template)
        max_score_index = np.unravel_index(
            np.argmax(template_match_result), template_match_result.shape
        )
        match_box_im_topleft, match_box_rt_topleft = max_score_index[
            ::-1
        ]  # template box top left, not the bounding box of segmentation
        shift = (
            match_box_rt_topleft - template_rt_start,
            match_box_im_topleft - template_im_start,
        )
        match_bbox_mask = np.zeros(pept_act_image_smoothed.shape, dtype=int)
        match_bbox = (
            np.clip(
                propA["bbox-0"].values[0].astype(int) + shift[0],
                0,
                match_bbox_mask.shape[0],
            ),
            np.clip(
                propA["bbox-1"].values[0].astype(int) + shift[1],
                0,
                match_bbox_mask.shape[1],
            ),
            np.clip(
                propA["bbox-2"].values[0].astype(int) + shift[0],
                0,
                match_bbox_mask.shape[0],
            ),
            np.clip(
                propA["bbox-3"].values[0].astype(int) + shift[1],
                0,
                match_bbox_mask.shape[1],
            ),
        )
        match_bbox_mask[
            match_bbox[0] : match_bbox[2],
            match_bbox[1] : match_bbox[3],
        ] = 1  # matched bounding box calculated as original bbox plus shift

        target_anchor_shifted = np.array(
            [
                (
                    np.clip(
                        template_anchor[0][0] + shift[0],
                        0,
                        pept_act_image_smoothed.shape[0] - 1,
                    ),
                    np.clip(
                        template_anchor[0][1] + shift[1],
                        0,
                        pept_act_image_smoothed.shape[1] - 1,
                    ),
                )
            ]
        )  # target anchor is updated: shifted anchor for the matched image
        labels = ((pept_act_image_smoothed != 0) & match_bbox_mask.astype(bool)).astype(
            int
        )
        labels_with_multi_marker = (
            labels  # Only one label is available in matched images
        )
        # alternatively, get match labels from watershed with the shifted anchor, which will be more robust to noise but may fail when the shift is large and there are multiple local maximum in the shifted region
        # _, labels, _, labels_with_multi_marker, snapped_anchor = (
        #     detect_2d_peak_with_watershed(
        #         pept_act_image_smoothed,
        #         **peak_kwargs,
        #         coordinates=anchor,
        #     )
        # )
        template_matching_score_max = np.max(template_match_result)

    # Case quantification without template matching, directly run watershed with the original anchor
    # Which will be snapped into the nearest connected local maximum if the anchor is not already a local maximum
    else:
        if apply_seg:
            _, labels, _, labels_with_multi_marker, snapped_anchor = (
                detect_2d_peak_with_watershed(
                    pept_act_image_smoothed,
                    **peak_kwargs,
                    coordinates=template_anchor,
                )
            )
        else:
            template_im_start = max(
                (template_anchor[0][1] - 0.3 * pept_act_image_smoothed.shape[1]).astype(
                    int
                ),
                0,
            )
            template_im_end = min(
                (template_anchor[0][1] + 0.3 * pept_act_image_smoothed.shape[1]).astype(
                    int
                ),
                pept_act_image_smoothed.shape[1],
            )
            template_rt_start = max(
                (template_anchor[0][0] - 0.3 * pept_act_image_smoothed.shape[0]).astype(
                    int
                ),
                0,
            )
            template_rt_end = min(
                (template_anchor[0][0] + 0.3 * pept_act_image_smoothed.shape[0]).astype(
                    int
                ),
                pept_act_image_smoothed.shape[0],
            )  # Use up to 36% of the image size as the template size to
            # make sure the template can cover the peak region even when the
            # anchor is not very accurate, which can be common for low abundance peptides with weak MS/MS signal

            labels = np.zeros(pept_act_image_smoothed.shape, dtype=int)
            labels[
                template_rt_start:template_rt_end, template_im_start:template_im_end
            ] = (
                pept_act_image_smoothed[
                    template_rt_start:template_rt_end, template_im_start:template_im_end
                ]
                > 0
            )
            labels_with_multi_marker = labels  # Only one label is available in this case as well, as watershed is not applied
        template_matching_score_max = np.nan
    peak_properties = calculate_peak_property_from_labels_and_image(
        labels, pept_act_image, **filter_kwargs
    )
    if peak_properties is None:
        if visualize_dir is not None:
            _visualize_quantify_from_coords(
                reference_image,
                pept_act_image,
                pept_act_image_smoothed,
                bbox_center=None,
                save_dir=visualize_dir,
                target_msms_pos=template_anchor[0],
                target_snapped_msms_pos=(
                    snapped_anchor
                    if reference_image is None and "snapped_anchor" in locals()
                    else (
                        target_anchor_shifted[0]
                        if target_anchor_shifted is not None
                        else template_anchor[0]
                    )
                ),
                template_msms_pos=None,
                template_snapped_msms_pos=(
                    template_anchor[0] if reference_image is not None else None
                ),
                filename=visualize_filename,
                labels=labels_with_multi_marker,
                template_box=(
                    (
                        template_rt_start,
                        template_im_start,
                        template_rt_end,
                        template_im_end,
                    )
                    if propA is not None
                    else None
                ),
                matched_template_box=(
                    (
                        template_rt_start + shift[0],
                        template_im_start + shift[1],
                        template_rt_end + shift[0],
                        template_im_end + shift[1],
                    )
                    if propA is not None
                    else None
                ),
                seg_box=(
                    (
                        (
                            propA["bbox-0"].values[0].astype(int) + shift[0],
                            propA["bbox-1"].values[0].astype(int) + shift[1],
                            propA["bbox-2"].values[0].astype(int) + shift[0],
                            propA["bbox-3"].values[0].astype(int) + shift[1],
                        )
                    )
                    if propA is not None
                    else None
                ),
            )
        return QuantificationResult(
            run_name="",
            case="Reference",
            image=pept_act_image,
            smoothed_image=pept_act_image_smoothed,
            input_anchor=(int(template_anchor[0][0]), int(template_anchor[0][1])),
            peak_properties=None,
            snapped_anchor=(
                tuple(int(x) for x in snapped_anchor)
                if reference_image is None and "snapped_anchor" in locals()
                else (
                    (
                        int(target_anchor_shifted[0][0]),
                        int(target_anchor_shifted[0][1]),
                    )
                    if target_anchor_shifted is not None
                    else (int(template_anchor[0][0]), int(template_anchor[0][1]))
                )
            ),
            labels=labels,
            labels_multi_markers=labels_with_multi_marker,
            template_matching_score=template_matching_score_max,
            shift=tuple(int(x) for x in shift) if "shift" in locals() else (0, 0),
        )
    else:
        # successful match
        seg_bbox = pept_act_image_smoothed[
            peak_properties["bbox-0"]
            .values[0]
            .astype(int) : peak_properties["bbox-2"]
            .values[0]
            .astype(int),
            peak_properties["bbox-1"]
            .values[0]
            .astype(int) : peak_properties["bbox-3"]
            .values[0]
            .astype(int),
        ]  # Centers around the updated anchor
        peak_properties["snap_rt"] = (
            snapped_anchor[0] if "snapped_anchor" in locals() else template_anchor[0][0]
        )
        peak_properties["snap_im"] = (
            snapped_anchor[1] if "snapped_anchor" in locals() else template_anchor[0][1]
        )
        peak_properties["template_matching_score"] = template_matching_score_max
        peak_properties["sift_des"] = None
        peak_properties.at[0, "sift_des"] = get_sift_descriptor(
            np.log1p(pept_act_image),
            (
                peak_properties["snap_rt"].values[0],
                peak_properties["snap_im"].values[0],
            ),
            patch_size=patch_size,
        )
        zernike = get_roi_descriptor(
            seg_bbox,
        )
        peak_properties["zernike"] = None
        peak_properties.at[0, "zernike"] = zernike

        if reference_image is not None:
            peak_properties["shift_rt"] = shift[0]
            peak_properties["shift_im"] = shift[1]
        else:
            peak_properties["shift_rt"] = 0
            peak_properties["shift_im"] = 0
        if visualize_dir is not None:
            _visualize_quantify_from_coords(
                reference_image,
                pept_act_image,
                pept_act_image_smoothed,
                bbox_center=np.array(
                    [
                        (
                            peak_properties["centroid-0"].values[0],
                            peak_properties["centroid-1"].values[0],
                        )
                    ]
                ),
                target_msms_pos=template_anchor[0],
                target_snapped_msms_pos=(
                    snapped_anchor
                    if reference_image is None and "snapped_anchor" in locals()
                    else (
                        target_anchor_shifted[0]
                        if target_anchor_shifted is not None
                        else template_anchor[0]
                    )
                ),
                template_msms_pos=None,
                template_snapped_msms_pos=(
                    template_anchor[0] if reference_image is not None else None
                ),
                save_dir=visualize_dir,
                filename=visualize_filename,
                labels=labels_with_multi_marker,
                template_box=(
                    (
                        template_rt_start,
                        template_im_start,
                        template_rt_end,
                        template_im_end,
                    )
                    if reference_image is not None
                    else None
                ),
                seg_box=(
                    (
                        peak_properties["bbox-0"].values[0].astype(int),
                        peak_properties["bbox-1"].values[0].astype(int),
                        peak_properties["bbox-2"].values[0].astype(int),
                        peak_properties["bbox-3"].values[0].astype(int),
                    )
                ),
                matched_template_box=(
                    (
                        template_rt_start + shift[0],
                        template_im_start + shift[1],
                        template_rt_end + shift[0],
                        template_im_end + shift[1],
                    )
                    if propA is not None
                    else None
                ),
            )
        peak_properties = _extract_single_peak_properties(
            peak_properties,
            (
                tuple(int(x) for x in snapped_anchor)
                if "snapped_anchor" in locals() and len(snapped_anchor) == 2
                else (int(template_anchor[0][0]), int(template_anchor[0][1]))
            ),
        )
        return QuantificationResult(
            run_name="",
            case="Reference",
            image=pept_act_image,
            smoothed_image=pept_act_image_smoothed,
            input_anchor=(int(template_anchor[0][0]), int(template_anchor[0][1])),
            peak_properties=peak_properties,
            snapped_anchor=(
                tuple(int(x) for x in snapped_anchor)
                if reference_image is None and "snapped_anchor" in locals()
                else (
                    (
                        int(target_anchor_shifted[0][0]),
                        int(target_anchor_shifted[0][1]),
                    )
                    if target_anchor_shifted is not None
                    else (int(template_anchor[0][0]), int(template_anchor[0][1]))
                )
            ),
            labels=labels,
            labels_multi_markers=labels_with_multi_marker,
            template_matching_score=template_matching_score_max,
            shift=tuple(int(x) for x in shift) if "shift" in locals() else (0, 0),
        )


def compare_peak_properties(peak_properties_a, peak_properties_b):
    return {
        "template_matching_score": peak_properties_b["template_matching_score"].values[
            0
        ],
        "sift_similarities": compare_sift_descriptors(
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
        ),
        "zernike_distance": compare_image_descriptors_euclidean(
            peak_properties_a["zernike"].values[0],
            peak_properties_b["zernike"].values[0],
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


def smooth_and_denoise_image(
    image,
    smooth_filter: Literal["gaussian", "uniform"] = "gaussian",
    log_transform: bool = True,
    threshold: float = 10,
    gaussian_kwargs: dict | None = None,
    uniform_kwargs: dict | None = None,
    remove_kwargs: dict | None = None,
):
    """Smooth image with filters and denoise by remove small objects

    Parameters
    ----------
    image : 2D array
        Input image to be smoothed.
    smooth_filter : str, optional
        Type of filter to use. Options are "gaussian" or "uniform". Default is "gaussian".
    threshold : float, optional
        Threshold used to create a mask before removing small objects.
    gaussian_kwargs : dict, optional
        Keyword arguments for scipy.ndimage.gaussian_filter.
    uniform_kwargs : dict, optional
        Keyword arguments for scipy.ndimage.uniform_filter.
    remove_kwargs : dict, optional
        Keyword arguments for skimage.morphology.remove_small_objects.
    """
    gaussian_kwargs = {} if gaussian_kwargs is None else dict(gaussian_kwargs)
    uniform_kwargs = {} if uniform_kwargs is None else dict(uniform_kwargs)
    remove_kwargs = {} if remove_kwargs is None else dict(remove_kwargs)

    if "sigma" not in gaussian_kwargs:
        gaussian_kwargs["sigma"] = 2  # (rt, im)
        gaussian_kwargs["mode"] = "nearest"
    if "size" not in uniform_kwargs:
        uniform_kwargs["size"] = (1, 5)
    if "min_size" not in remove_kwargs:
        remove_kwargs["min_size"] = 5

    match smooth_filter:
        case "gaussian":
            image_smoothed = gaussian_filter(image, **gaussian_kwargs)
        case "uniform":
            blurred = uniform_filter(image, **uniform_kwargs)
            image_smoothed = np.maximum(image, blurred)
    # remove small objects after smoothing
    cleaned_mask = remove_small_objects(image_smoothed >= threshold, **remove_kwargs)
    image_smoothed = image_smoothed * cleaned_mask

    # log transform smoothed and cleaned up
    if log_transform:
        image_smoothed = np.log10(1 + image_smoothed)
    return image_smoothed


def get_orb_peak_descriptor(
    img, peak_coords, patch_size=100
):  # This doesn't work well when image is noisy or only one smooth peak exists
    """
    Computes the ORB descriptor for a specific peak.
    Returns the descriptor (feature vector).

    Parameters
    ----------
    img : 2D array
        Input image (should be in uint8 format).
    peak_coords : tuple
        (y, x) coordinates of the peak for which to compute the descriptor.
    patch_size : int, optional
        Size of the patch around the peak to consider for descriptor computation. Default is 31.
    """
    # 1. Normalize and convert to 8-bit once
    img_8bit = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

    # 2. Initialize ORB
    orb = cv2.ORB_create()
    y, x = peak_coords

    # 3. Create the KeyPoint at the peak
    kp = [cv2.KeyPoint(x=float(x), y=float(y), size=patch_size)]

    # 4. Compute the descriptor
    _, des = orb.compute(img_8bit, kp)

    return des


def get_sift_descriptor(img, peak_coords, patch_size=31):
    """
    Computes a SIFT descriptor for a specific peak coordinate.
    """
    # 1. SIFT works best on 8-bit images.
    # Normalization ensures intensity differences don't break the gradient math.
    img_8bit = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

    # 2. Initialize SIFT
    sift = cv2.SIFT_create()

    y, x = peak_coords

    # 3. Create a KeyPoint.
    # 'size' determines the area the descriptor looks at.
    # 'angle=0' is used because your images are already aligned.
    kp = [cv2.KeyPoint(x=float(x), y=float(y), size=patch_size, angle=0)]

    # 4. Compute the descriptor
    _, des = sift.compute(img_8bit, kp)

    return des


def get_roi_descriptor(roi, radius=None):
    if roi.max() == roi.min():
        roi_norm = np.zeros_like(roi, dtype=np.float32)
    else:
        roi_norm = (roi - roi.min()) / (roi.max() - roi.min())

    # Zernike Moments
    roi_uint8 = (roi_norm * 255).astype(np.uint8)
    radius = radius if radius is not None else max(roi.shape) // 2
    zernike = zernike_moments(
        roi_uint8, radius, cm=(roi.shape[1] // 2, roi.shape[0] // 2), degree=8
    )
    return zernike


def compare_image_descriptors_cosine(des1, des2):
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
    return similarity


def compare_image_descriptors_euclidean(des1, des2):
    if des1 is None or des2 is None:
        return 0.0

    # SIFT descriptors must be float32 for NORM_L2
    # This line prevents the "Assertion failed" error
    d1 = des1.astype(np.float32).flatten()
    d2 = des2.astype(np.float32).flatten()

    dist = np.linalg.norm(d1 - d2)

    # # Convert distance to similarity (example using exponential decay)
    # similarity = np.exp(-dist / 100.0)  # Adjust the denominator as needed
    return dist


def compare_sift_descriptors(des1, des2):
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
    similarity = np.exp(-dist / 100.0)
    return similarity


def calc_quant_corr(pp_reference, pp_match_target, quant_dir):
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(quant_dir, exist_ok=True)
    if "source_type" in pp_match_target.columns:
        pp_quant_only = pp_match_target.loc[
            pp_match_target["source_type"] == "Quant_Only"
        ].copy()
        pp_match_target_only = pp_match_target.loc[
            pp_match_target["source_type"] != "Quant_Only"
        ].copy()
    else:
        pp_quant_only = pd.DataFrame(columns=pp_match_target.columns)
        pp_match_target_only = pp_match_target
    pp_quant_only_pivoted = pp_quant_only.pivot_table(
        index="mz_rank",
        columns="Run_name",
        values="intensity_sum",
        aggfunc="max",
    ).reset_index()
    pp_reference_pivoted = pp_reference.pivot_table(
        index="mz_rank",
        columns="Run_name",
        values="intensity_sum",
        aggfunc="max",
    ).reset_index()
    # Log-transform numeric columns and compute pairwise Pearson correlations (pairwise complete cases)
    pp_match_target_pivoted = pp_match_target_only.pivot_table(
        index="mz_rank",
        columns="Run_name",
        values="intensity_sum",
        aggfunc="max",
    ).reset_index()
    # Log-transform numeric columns and compute pairwise Pearson correlations (pairwise complete cases)
    num_cols = pp_match_target_pivoted.select_dtypes(
        include=[np.number]
    ).columns.difference(["mz_rank"])
    pp_log = pp_match_target_pivoted.copy()
    pp_log[num_cols] = np.log2(pp_log[num_cols] + 1)

    corr_matrix = pp_log[num_cols].corr(method="pearson", min_periods=1)
    corr_matrix.to_csv(
        os.path.join(
            quant_dir, "pp_match_target_filtered_log_intensity_correlation_matrix.csv"
        )
    )

    # 1. Concatenate with MultiIndex
    pp_all_pivoted = pd.concat(
        [
            pp_reference_pivoted.set_index("mz_rank"),
            pp_quant_only_pivoted.set_index("mz_rank"),
            pp_match_target_pivoted.set_index("mz_rank"),
        ],
        axis=1,
        keys=["reference", "quant_only", "match_target"],
    )

    # 2. Identify numeric columns (excluding the index 'mz_rank')
    # Since mz_rank is the index now, we just take all columns
    num_cols = pp_all_pivoted.select_dtypes(include=[np.number]).columns

    # 3. Log transformation (using log2(x+1) to handle zeros)
    pp_log = np.log2(pp_all_pivoted[num_cols] + 1)

    # 4. Correlation Matrix
    # min_periods=1 ensures you get a value even if there's only one overlapping point
    corr_matrix = pp_log.corr(method="pearson", min_periods=1)
    corr_matrix.to_csv(
        os.path.join(quant_dir, "pp_all_log_intensity_correlation_matrix.csv")
    )
    # Optional: Flatten the MultiIndex for easier viewing if it's too cluttered

    count_matrix = pp_log.notna().astype(int).T.dot(pp_log.notna().astype(int))
    sns.heatmap(corr_matrix)
    ax = plt.gca()
    for i in range(count_matrix.shape[0]):
        for j in range(count_matrix.shape[1]):
            _ = ax.text(
                j + 0.5,
                i + 0.5,
                str(int(count_matrix.iloc[i, j])),
                ha="center",
                va="center",
                fontsize=3,
                color="white",
            )

    plt.xticks(
        ticks=np.arange(len(corr_matrix.columns)),
        labels=[c[0] + c[1][-5:] for c in corr_matrix.columns.values],
        fontsize=5,
        # rotation=45,
    )
    plt.yticks(
        ticks=np.arange(len(corr_matrix.index)),
        labels=[c[0] + c[1][-5:] for c in corr_matrix.index.values],
        fontsize=5,
    )
    plt.savefig(
        os.path.join(
            quant_dir,
            "correlation_matrix_of_log_intensity_with_counts.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_match_type_from_combined(
    df, colors=None, labels=None, stack_order=None, fig_dir=None, fig_name_suffix=""
):
    import matplotlib.pyplot as plt

    if colors is None:
        colors = {
            "MS/MS": "#55A868",
            "MS/MS Quant": "#55A868",
            "MS/MS Ref": "#4C72B0",
            "MBR": "#C44E52",
            "unmatched": "#BBBBBB",
        }

    if stack_order is None:
        stack_order = ["MS/MS", "MS/MS Quant", "MS/MS Ref", "MBR", "unmatched"]

    match_type_cols = [col for col in df.columns if "Match Type" in col]

    counts_dict = {}
    for col in match_type_cols:
        counts_dict[col] = df[col].value_counts(dropna=True)

    counts = pd.DataFrame(counts_dict).T.fillna(0)

    # Reorder columns to control stack and legend order
    ordered_cols = [c for c in stack_order if c in counts.columns]
    counts = counts[ordered_cols]

    if labels is None:
        labels = [f"Run{i+1}" for i in range(len(counts.index))]

    plt.figure(figsize=(10, 8))
    ax = counts.plot(kind="bar", stacked=True, color=colors)

    plt.xlabel("Match Type Column")
    plt.ylabel("Count")
    plt.title("Entry Counts per Match Type Column")
    plt.xticks(rotation=45, ticks=range(len(counts.index)), labels=labels)

    for container in ax.containers:
        for bar in container:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + height / 2,
                    f"{int(height)}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    plt.legend(title="Entry", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    if fig_dir is not None:
        plt.savefig(
            os.path.join(fig_dir, f"match_type_counts{fig_name_suffix}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
    plt.show()
