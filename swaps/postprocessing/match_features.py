import logging
import os
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Tuple
import numpy as np
import pandas as pd
import tqdm
from scipy.ndimage import gaussian_filter, uniform_filter
from skimage.morphology import remove_small_objects
from skimage.feature import match_template
import cv2
from concurrent.futures import ProcessPoolExecutor
from mahotas.features import zernike_moments
from scipy.spatial.distance import cosine
from swaps.utils.ims_utils import (
    detect_2d_peak_with_watershed,
    calculate_peak_property_from_labels_and_image,
)
from .helper import (
    load_peptide_batch_df_from_partquet,
    get_pept_act_from_parquet,
)
import seaborn as sns
import matplotlib.pyplot as plt

Logger = logging.getLogger(__name__)


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
    batch_size: int = 100,
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
    peptide_batches = np.array_split(
        peptide_indicies, max(1, len(peptide_indicies) // batch_size)
    )
    results_target, results_decoy = [], []
    pp_reference_list, pp_match_target_list = [], []
    pp_quant_only_list = []
    pp_match_decoy_list = []
    no_quant_log = []
    no_match_log = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                match_features_batch,
                dict_ref,
                raw_file_list,
                result_dir,
                batch,
                processing_kwargs,
            )
            for batch in peptide_batches
        ]

        for future in tqdm.tqdm(futures, desc="Processing batches", unit="batch"):
            (
                res_target,
                res_decoy,
                pp_reference_target,
                pp_quant_only,
                pp_match_target,
                pp_match_decoy,
                no_quant,
                no_match,
            ) = future.result()
            results_target.extend(res_target)
            results_decoy.extend(res_decoy)
            pp_reference_list.extend(pp_reference_target)
            pp_quant_only_list.extend(pp_quant_only)
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
    pp_quant_only = (
        pd.concat(pp_quant_only_list, ignore_index=True)
        if pp_quant_only_list
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
    df_no_quant = pd.DataFrame(no_quant_log)
    df_no_match = pd.DataFrame(no_match_log)
    return (
        matches_target,
        matches_decoy,
        pp_reference_target,
        pp_quant_only,
        pp_match_target,
        pp_match_decoy,
        df_no_quant,
        df_no_match,
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
    aligned_bbox: tuple[int, int, int, int],
    quant_bbox: tuple[int, int, int, int],
) -> dict[str, int]:
    """Compute directional expansion needed to cover both aligned and quant bboxes.

    The returned values are *additional* expansion amounts beyond the current
    aligned bbox, expressed as top/bottom/left/right deltas.
    """

    return {
        "top": max(int(aligned_bbox[0]) - int(quant_bbox[0]), 0),
        "left": max(int(aligned_bbox[1]) - int(quant_bbox[1]), 0),
        "bottom": max(int(quant_bbox[2]) - int(aligned_bbox[2]), 0),
        "right": max(int(quant_bbox[3]) - int(aligned_bbox[3]), 0),
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
    anchor_override: Optional[tuple[int, int]] = None,
    reference_image: np.ndarray | None = None,
    reference_props: pd.DataFrame | None = None,
    processing_kwargs: dict | None = None,
    visualize_dir: str | None = None,
):
    """Quantify one peptide in one run and return the full quantification state."""

    pept_act, rt_msms_pos, im_msms_pos = get_pept_act_from_parquet(
        act_df.loc[act_df["mz_rank"] == pept_idx], pept_idx, dict_ref, run_name
    )
    target_anchor = (
        anchor_override if anchor_override is not None else (rt_msms_pos, im_msms_pos)
    )
    template_anchor = (
        template_anchor_override
        if template_anchor_override is not None
        else target_anchor
    )
    if (
        target_anchor[0] >= pept_act.shape[0]
        or target_anchor[1] >= pept_act.shape[1]
        or target_anchor[0] < 0
        or target_anchor[1] < 0
    ):
        logging.warning(
            "Skipping %s for mz_rank %s in %s due to anchor out of bounds",
            case,
            pept_idx,
            run_name,
        )
        return QuantificationResult(
            run_name=run_name,
            case=case,
            image=pept_act,
            smoothed_image=pept_act,
            input_anchor=(int(target_anchor[0]), int(target_anchor[1])),
            peak_properties=None,
            snapped_anchor=None,
            metadata={"pept_idx": pept_idx},
        )
    result = quantify_from_coords(
        pept_act,
        anchor=(int(template_anchor[0]), int(template_anchor[1])),
        target_anchor=(int(target_anchor[0]), int(target_anchor[1])),
        reference_image=reference_image,
        propA=reference_props,
        patch_size=min(pept_act.shape),
        **(processing_kwargs or {}),
        visualize_dir=visualize_dir,
        visualize_filename=f"mz{pept_idx}_{run_name}_{case.lower()}.png",
    )
    result.run_name = run_name
    result.case = case
    result.metadata["pept_idx"] = pept_idx
    return result


def match_features_batch(
    dict_ref,
    raw_file_list,
    result_dir,
    batch,
    processing_kwargs: dict | None = None,
    visualize_dir: str | None = None,
):
    """Process one peptide batch with sequential quant-only bbox expansion.

    Workflow per peptide:
    1. Quantify the main reference run once.
    2. Use the reference to template-match each quant-only run.
    3. If a quant-only run's own snapped anchor falls outside the matched bbox,
       expand the reusable reference bbox and redo the quant-only matching.
    4. Record which quant-only runs were already inside the bbox and which ones
       caused an expansion.
    5. Match all regular `Match` runs using the final expanded bbox.
    """

    results_target, results_decoy = [], []
    pp_reference_list, pp_match_target_list = [], []
    pp_quant_only_list = []
    pp_match_decoy_list = []
    no_quant_log = []
    no_match_log = []
    act_dfs = {}
    for raw_file in raw_file_list:
        parquet_path = os.path.join(result_dir, raw_file, "activation", "*.parquet")
        act_dfs[raw_file] = load_peptide_batch_df_from_partquet(parquet_path, batch)
    for pept_idx in batch:
        row_series = dict_ref.loc[dict_ref["mz_rank"] == pept_idx, :].iloc[0]
        is_ref_mask = row_series.map(
            lambda x: x == "Reference" if isinstance(x, str) else False
        )
        is_quant_only_mask = row_series.map(
            lambda x: x == "Quant_Only" if isinstance(x, str) else False
        )
        is_match_mask = row_series.map(
            lambda x: "Match" in x if isinstance(x, str) else False
        )

        reference_raw_file = str(is_ref_mask.idxmax())
        quant_only_raw_file = is_quant_only_mask.index[is_quant_only_mask].tolist()
        match_raw_file = is_match_mask.index[is_match_mask].tolist()

        own_anchor_id = 0
        feature_instance_id = _feature_instance_id(pept_idx, own_anchor_id)
        match_state: ReferenceMatchState | None = None

        reference_result = _quantify_peptide_run(
            act_df=act_dfs[reference_raw_file].loc[
                act_dfs[reference_raw_file]["mz_rank"] == pept_idx
            ],
            pept_idx=pept_idx,
            dict_ref=dict_ref,
            run_name=reference_raw_file,
            case="Reference",
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

        bbox_version = 0
        quant_only_probe_records: dict[str, dict[str, Any]] = {}

        if len(quant_only_raw_file) > 0:
            for raw_file in quant_only_raw_file:
                quant_direct = _quantify_peptide_run(
                    act_df=act_dfs[raw_file].loc[
                        act_dfs[raw_file]["mz_rank"] == pept_idx
                    ],
                    pept_idx=pept_idx,
                    dict_ref=dict_ref,
                    run_name=raw_file,
                    case="Quant_Only",
                    processing_kwargs=processing_kwargs,
                    visualize_dir=visualize_dir,
                )
                quant_only_probe_records[raw_file] = {
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
                if match_state is None or not quant_direct.succeeded:
                    continue

                probe_result = _quantify_peptide_run(
                    act_df=act_dfs[raw_file].loc[
                        act_dfs[raw_file]["mz_rank"] == pept_idx
                    ],
                    pept_idx=pept_idx,
                    dict_ref=dict_ref,
                    run_name=raw_file,
                    case="Match",
                    template_anchor_override=match_state.reference_result.snapped_anchor,
                    anchor_override=match_state.reference_result.snapped_anchor,
                    reference_image=match_state.reference_result.smoothed_image,
                    reference_props=match_state.match_props,
                    processing_kwargs=processing_kwargs,
                    visualize_dir=visualize_dir,
                )
                quant_only_probe_records[raw_file]["probe"] = probe_result
                quant_only_probe_records[raw_file]["probe_bbox_version"] = bbox_version

                inside_match_bbox = _anchor_inside_bbox(
                    quant_direct.snapped_anchor,
                    probe_result.peak_properties,
                )
                quant_only_probe_records[raw_file]["inside_match_bbox"] = bool(
                    inside_match_bbox
                )
                if (
                    not inside_match_bbox
                    and quant_direct.peak_properties is not None
                    and quant_direct.snapped_anchor is not None
                    and probe_result.peak_properties is not None
                ):
                    quant_only_probe_records[raw_file][
                        "reference_bbox_before_update"
                    ] = _bbox_tuple_from_prop(match_state.match_props)
                    quant_only_probe_records[raw_file]["aligned_bbox_before_update"] = (
                        _bbox_tuple_from_prop(probe_result.peak_properties)
                    )
                    quant_bbox = _bbox_tuple_from_prop(quant_direct.peak_properties)
                    aligned_bbox = _bbox_tuple_from_prop(probe_result.peak_properties)
                    expansion_deltas = _expansion_deltas_from_aligned_and_quant_bbox(
                        aligned_bbox,
                        quant_bbox,
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
                        match_state.reference_result.image.shape,
                    )
                    quant_only_probe_records[raw_file]["expanded_match_bbox"] = True
                    quant_only_probe_records[raw_file]["bbox_updated_from_original"] = (
                        bool(bbox_changed)
                    )
                    quant_only_probe_records[raw_file][
                        "reference_bbox_after_update"
                    ] = _bbox_tuple_from_prop(match_state.match_props)
                    aligned_after_box = (
                        max(aligned_bbox[0] - expansion_deltas["top"], 0),
                        max(aligned_bbox[1] - expansion_deltas["left"], 0),
                        min(
                            aligned_bbox[2] + expansion_deltas["bottom"],
                            probe_result.image.shape[0],
                        ),
                        min(
                            aligned_bbox[3] + expansion_deltas["right"],
                            probe_result.image.shape[1],
                        ),
                    )
                    quant_only_probe_records[raw_file][
                        "aligned_bbox_after_update"
                    ] = aligned_after_box
                    if bbox_changed:
                        bbox_version += 1

        non_reference_runs = quant_only_raw_file + match_raw_file
        if len(non_reference_runs) > 0:
            if match_state is not None:
                for raw_file in non_reference_runs:
                    batch_exclude = batch[batch != pept_idx]
                    decoy_pept_idx = np.random.choice(batch_exclude)
                    run_subset = act_dfs[raw_file].loc[
                        act_dfs[raw_file]["mz_rank"].isin([pept_idx, decoy_pept_idx])
                    ]
                    is_quant_only_run = raw_file in quant_only_probe_records
                    target_result = None
                    if is_quant_only_run:
                        record = quant_only_probe_records[raw_file]
                        if (
                            record["probe"] is not None
                            and record["probe_bbox_version"] == bbox_version
                        ):
                            target_result = record["probe"]
                        else:
                            target_result = _quantify_peptide_run(
                                act_df=run_subset.loc[
                                    run_subset["mz_rank"] == pept_idx
                                ],
                                pept_idx=pept_idx,
                                dict_ref=dict_ref,
                                run_name=raw_file,
                                case="Match",
                                template_anchor_override=match_state.reference_result.snapped_anchor,
                                anchor_override=match_state.reference_result.snapped_anchor,
                                reference_image=match_state.reference_result.smoothed_image,
                                reference_props=match_state.match_props,
                                processing_kwargs=processing_kwargs,
                                visualize_dir=visualize_dir,
                            )
                    else:
                        target_result = _quantify_peptide_run(
                            act_df=run_subset.loc[run_subset["mz_rank"] == pept_idx],
                            pept_idx=pept_idx,
                            dict_ref=dict_ref,
                            run_name=raw_file,
                            case="Match",
                            template_anchor_override=match_state.reference_result.snapped_anchor,
                            anchor_override=match_state.reference_result.snapped_anchor,
                            reference_image=match_state.reference_result.smoothed_image,
                            reference_props=match_state.match_props,
                            processing_kwargs=processing_kwargs,
                            visualize_dir=visualize_dir,
                        )
                    decoy_act, _, _ = get_pept_act_from_parquet(
                        run_subset.loc[run_subset["mz_rank"] == decoy_pept_idx],
                        decoy_pept_idx,
                        dict_ref,
                        raw_file,
                        shape=target_result.image.shape,
                    )
                    decoy_result = quantify_from_coords(
                        decoy_act,
                        anchor=match_state.reference_result.snapped_anchor,
                        target_anchor=match_state.reference_result.snapped_anchor,
                        reference_image=match_state.reference_result.smoothed_image,
                        propA=match_state.match_props,
                        patch_size=min(decoy_act.shape),
                        **(processing_kwargs or {}),
                        visualize_dir=None,
                        visualize_filename=(
                            f"mz{pept_idx}_decoy{decoy_pept_idx}_{raw_file}.png"
                        ),
                    )
                    decoy_result.run_name = raw_file
                    decoy_result.case = "Match"

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
                    prop_d = _annotate_peak_properties(
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
                    if is_quant_only_run:
                        record = quant_only_probe_records[raw_file]
                        quant_direct = record["direct"]
                        if prop_t is not None:
                            if quant_direct.snapped_anchor is not None:
                                prop_t["quant_direct_snap_rt"] = int(
                                    quant_direct.snapped_anchor[0]
                                )
                                prop_t["quant_direct_snap_im"] = int(
                                    quant_direct.snapped_anchor[1]
                                )
                            if quant_direct.peak_properties is not None:
                                prop_t["quant_direct_bbox_0"] = int(
                                    quant_direct.peak_properties["bbox-0"].values[0]
                                )
                                prop_t["quant_direct_bbox_1"] = int(
                                    quant_direct.peak_properties["bbox-1"].values[0]
                                )
                                prop_t["quant_direct_bbox_2"] = int(
                                    quant_direct.peak_properties["bbox-2"].values[0]
                                )
                                prop_t["quant_direct_bbox_3"] = int(
                                    quant_direct.peak_properties["bbox-3"].values[0]
                                )
                            prop_t["quant_anchor_inside_match_bbox"] = bool(
                                record["inside_match_bbox"]
                            )
                            prop_t["quant_anchor_expanded_match_bbox"] = bool(
                                record["expanded_match_bbox"]
                            )
                            prop_t["bbox_updated_from_original"] = bool(
                                record["bbox_updated_from_original"]
                            )
                            prop_t["reference_bbox_top"] = int(
                                match_state.bbox_offsets["top"]
                            )
                            prop_t["reference_bbox_bottom"] = int(
                                match_state.bbox_offsets["bottom"]
                            )
                            prop_t["reference_bbox_left"] = int(
                                match_state.bbox_offsets["left"]
                            )
                            prop_t["reference_bbox_right"] = int(
                                match_state.bbox_offsets["right"]
                            )
                            if (
                                visualize_dir is not None
                                and record["bbox_updated_from_original"]
                                and record["reference_bbox_before_update"] is not None
                                and record["reference_bbox_after_update"] is not None
                                and record["aligned_bbox_before_update"] is not None
                                and record["aligned_bbox_after_update"] is not None
                            ):
                                before_box = record["reference_bbox_before_update"]
                                after_box = record["reference_bbox_after_update"]
                                shifted_before_box = record[
                                    "aligned_bbox_before_update"
                                ]
                                shifted_after_box = record["aligned_bbox_after_update"]
                                _visualize_quantify_from_coords(
                                    match_state.reference_result.smoothed_image,
                                    target_result.image,
                                    target_result.smoothed_image,
                                    target_result.image,
                                    target_result.smoothed_image,
                                    bbox_center=np.array(
                                        [
                                            (
                                                prop_t["centroid-0"].values[0],
                                                prop_t["centroid-1"].values[0],
                                            )
                                        ]
                                    ),
                                    save_dir=visualize_dir,
                                    msms_pos=match_state.reference_result.snapped_anchor,
                                    snapped_msms_pos=target_result.snapped_anchor,
                                    filename=(
                                        f"mz{pept_idx}_{raw_file}_quant_only_bbox_update.png"
                                    ),
                                    labels=target_result.labels_multi_markers,
                                    template_box=shifted_after_box,
                                    context_panels=[
                                        {
                                            "image": match_state.reference_result.smoothed_image,
                                            "title": "reference_bbox_before_after",
                                            "msms_pos": match_state.reference_result.snapped_anchor,
                                            "snapped_msms_pos": match_state.reference_result.snapped_anchor,
                                            "template_boxes": [
                                                {
                                                    "box": before_box,
                                                    "edgecolor": "orange",
                                                    "linestyle": "--",
                                                },
                                                {
                                                    "box": after_box,
                                                    "edgecolor": "cyan",
                                                    "linestyle": "-",
                                                },
                                            ],
                                        },
                                        {
                                            "image": target_result.smoothed_image,
                                            "title": "quant_only_bbox_before_after",
                                            "msms_pos": match_state.reference_result.snapped_anchor,
                                            "snapped_msms_pos": target_result.snapped_anchor,
                                            "template_boxes": [
                                                {
                                                    "box": shifted_before_box,
                                                    "edgecolor": "orange",
                                                    "linestyle": "--",
                                                },
                                                {
                                                    "box": shifted_after_box,
                                                    "edgecolor": "cyan",
                                                    "linestyle": "-",
                                                },
                                            ],
                                        },
                                    ],
                                )

                    if prop_t is not None:
                        target_result.peak_properties = prop_t
                        if is_quant_only_run:
                            pp_quant_only_list.append(prop_t)
                        else:
                            pp_match_target_list.append(prop_t)
                            match_t = compare_peak_properties(
                                match_state.reference_result.peak_properties, prop_t
                            )
                            match_t["mz_rank"] = pept_idx
                            match_t["feature_instance_id"] = feature_instance_id
                            match_t["own_anchor_id"] = own_anchor_id
                            match_t["assimilated_to_anchor_id"] = own_anchor_id
                            match_t["source_run"] = reference_raw_file
                            match_t["source_type"] = "Reference"
                            results_target.append(match_t)
                    else:
                        no_quant_log.append(
                            {
                                "mz_rank": pept_idx,
                                "run_name": raw_file,
                                "type": (
                                    "quant_only"
                                    if is_quant_only_run
                                    else "match_target"
                                ),
                                "feature_instance_id": feature_instance_id,
                            }
                        )
                        if not is_quant_only_run:
                            no_match_log.append(
                                {
                                    "mz_rank": pept_idx,
                                    "run_name": raw_file,
                                    "type": "match_target",
                                    "feature_instance_id": feature_instance_id,
                                }
                            )
                    if (prop_d is not None) and (not is_quant_only_run):
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
                    elif not is_quant_only_run:
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
            else:
                for raw_file in non_reference_runs:
                    no_quant_log.append(
                        {
                            "mz_rank": pept_idx,
                            "run_name": raw_file,
                            "type": (
                                "quant_only"
                                if raw_file in quant_only_probe_records
                                else "match_target"
                            ),
                        }
                    )
                    if raw_file not in quant_only_probe_records:
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
                        no_match_log.append(
                            {
                                "mz_rank": pept_idx,
                                "run_name": raw_file,
                                "type": "match_decoy",
                            }
                        )

    return (
        results_target,
        results_decoy,
        pp_reference_list,
        pp_quant_only_list,
        pp_match_target_list,
        pp_match_decoy_list,
        no_quant_log,
        no_match_log,
    )


def _visualize_quantify_from_coords(
    reference_image,
    pept_act_image,
    pept_act_image_smoothed,
    pept_act_image_aligned,
    pept_act_image_smoothed_aligned,
    save_dir: str,
    bbox_center: Optional[Tuple[int, int]] = None,
    msms_pos: Optional[Tuple[int, int]] = None,
    snapped_msms_pos: Optional[Tuple[int, int]] = None,
    template_box: Optional[Tuple[int, int, int, int]] = None,
    filename: str = "quantify_from_coords.png",
    labels: np.ndarray | None = None,
    context_panels: list[dict[str, Any]] | None = None,
):
    """Visualize feature quantification and optional template-guided mapping context.

    The base visualization keeps the existing segmentation-oriented panels.
    Optional ``context_panels`` can be prepended to show extra states such as a
    quant-only source image and how it maps into the reference image.
    """

    def _draw_panel(
        ax,
        img,
        title: str,
        *,
        bbox_center_local: Optional[Tuple[int, int]] = None,
        msms_pos_local: Optional[Tuple[int, int]] = None,
        snapped_msms_pos_local: Optional[Tuple[int, int]] = None,
        template_box_local: Optional[Tuple[int, int, int, int]] = None,
        template_boxes_local: list[dict[str, Any]] | None = None,
        labels_local: np.ndarray | None = None,
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
            masked_labels = np.ma.masked_where(labels_local == 0, labels_local)
            ax.imshow(
                masked_labels,
                aspect="auto",
                origin="lower",
                cmap="tab10",
                interpolation="nearest",
                alpha=0.35,
            )
        if bbox_center_local is not None:
            ax.plot(
                bbox_center_local[1],
                bbox_center_local[0],
                "r+",
                markersize=10,
                markeredgewidth=2,
            )
        if msms_pos_local is not None:
            ax.plot(
                msms_pos_local[1],
                msms_pos_local[0],
                "*",
                markersize=10,
                markeredgewidth=2,
                color="white",
            )
        if snapped_msms_pos_local is not None and len(snapped_msms_pos_local) == 2:
            ax.plot(
                snapped_msms_pos_local[1],
                snapped_msms_pos_local[0],
                "*",
                markersize=10,
                markeredgewidth=2,
                color="yellow",
            )
        boxes_to_draw = []
        if template_box_local is not None:
            boxes_to_draw.append(
                {"box": template_box_local, "edgecolor": "red", "linewidth": 2}
            )
        if template_boxes_local:
            boxes_to_draw.extend(template_boxes_local)
        for box_info in boxes_to_draw:
            box = box_info["box"]
            ax.add_patch(
                plt.Rectangle(
                    (box[1], box[0]),
                    box[3] - box[1],
                    box[2] - box[0],
                    fill=False,
                    edgecolor=box_info.get("edgecolor", "red"),
                    linewidth=box_info.get("linewidth", 2),
                    linestyle=box_info.get("linestyle", "-"),
                )
            )

    images = []
    if context_panels:
        for panel in context_panels:
            images.append(
                (
                    panel.get("image"),
                    panel.get("title", "context"),
                    panel.get("labels"),
                    panel.get("bbox_center"),
                    panel.get("msms_pos"),
                    panel.get("snapped_msms_pos"),
                    panel.get("template_box"),
                    panel.get("template_boxes"),
                )
            )
    images.extend(
        [
            (
                reference_image,
                "reference_image",
                None,
                None,
                None,
                None,
                template_box,
                None,
            ),
            (
                pept_act_image,
                "pept_act_image",
                None,
                bbox_center,
                msms_pos,
                snapped_msms_pos,
                None,
                None,
            ),
            (
                pept_act_image_smoothed,
                "pept_act_image_smoothed",
                None,
                bbox_center,
                msms_pos,
                snapped_msms_pos,
                None,
                None,
            ),
            (
                pept_act_image_aligned,
                "pept_act_image_aligned",
                labels,
                bbox_center,
                msms_pos,
                snapped_msms_pos,
                template_box,
                None,
            ),
            (
                pept_act_image_smoothed_aligned,
                "pept_act_image_smoothed_aligned",
                labels,
                bbox_center,
                msms_pos,
                snapped_msms_pos,
                template_box,
                None,
            ),
        ]
    )
    n_cols = len(images) + (1 if labels is not None else 0)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]
    for ax, (
        img,
        title,
        lbl,
        bbox_center_local,
        msms_pos_local,
        snapped_msms_pos_local,
        template_box_local,
        template_boxes_local,
    ) in zip(axes, images):
        _draw_panel(
            ax,
            img,
            title,
            bbox_center_local=(
                tuple(bbox_center_local[0])
                if isinstance(bbox_center_local, np.ndarray)
                else bbox_center_local
            ),
            msms_pos_local=msms_pos_local,
            snapped_msms_pos_local=snapped_msms_pos_local,
            template_box_local=template_box_local,
            template_boxes_local=template_boxes_local,
            labels_local=lbl,
        )
    if labels is not None:
        ax_lbl = axes[-1]
        ax_lbl.set_title("watershed_labels", fontsize=9)
        masked_labels = np.ma.masked_where(labels == 0, labels)
        ax_lbl.imshow(np.zeros_like(labels), aspect="auto", origin="lower", cmap="gray")
        ax_lbl.imshow(
            masked_labels,
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
            ax_lbl.plot(
                bbox_center[0][1],
                bbox_center[0][0],
                "r+",
                markersize=10,
                markeredgewidth=2,
            )
        if msms_pos is not None:
            ax_lbl.plot(
                msms_pos[1],
                msms_pos[0],
                "*",
                markersize=10,
                markeredgewidth=2,
                color="white",
            )  # white * for MS/MS position
        if template_box is not None:
            axes[len(context_panels or [])].add_patch(
                plt.Rectangle(
                    (template_box[1], template_box[0]),
                    template_box[3] - template_box[1],
                    template_box[2] - template_box[0],
                    fill=False,
                    edgecolor="red",
                    linewidth=2,
                )
            )

    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)


def quantify_from_coords(
    pept_act_image,
    anchor,
    target_anchor: tuple[int, int] | None = None,
    reference_image: np.ndarray | None = None,
    propA: pd.DataFrame | None = None,
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
    anchor : tuple
        The template/source anchor coordinates (row, column).
    target_anchor : tuple | None, optional
        The target-image anchor coordinates. If omitted, the same anchor is used
        for both the template crop and the target placement.
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
    target_anchor = anchor if target_anchor is None else target_anchor
    if (
        target_anchor[0] >= pept_act_image.shape[0]
        or target_anchor[1] >= pept_act_image.shape[1]
        or target_anchor[0] < 0
        or target_anchor[1] < 0
    ):
        logging.warning(
            "Target anchor coordinates %s are out of bounds of the image dimensions %s.",
            target_anchor,
            pept_act_image.shape,
        )
        return QuantificationResult(
            run_name="",
            case="Reference",
            image=pept_act_image,
            smoothed_image=pept_act_image,
            input_anchor=(int(target_anchor[0]), int(target_anchor[1])),
            peak_properties=None,
            snapped_anchor=None,
        )
    if reference_image is not None and (
        anchor[0] >= reference_image.shape[0]
        or anchor[1] >= reference_image.shape[1]
        or anchor[0] < 0
        or anchor[1] < 0
    ):
        logging.warning(
            "Anchor coordinates %s are out of bounds of the reference image dimensions %s.",
            anchor,
            reference_image.shape,
        )
        return QuantificationResult(
            run_name="",
            case="Reference",
            image=pept_act_image,
            smoothed_image=pept_act_image,
            input_anchor=(int(target_anchor[0]), int(target_anchor[1])),
            peak_properties=None,
            snapped_anchor=None,
        )
    if reference_image is None and (
        anchor[0] >= pept_act_image.shape[0]
        or anchor[1] >= pept_act_image.shape[1]
        or anchor[0] < 0
        or anchor[1] < 0
    ):
        logging.warning(
            "Anchor coordinates %s are out of bounds of the image dimensions %s.",
            anchor,
            pept_act_image.shape,
        )
        return QuantificationResult(
            run_name="",
            case="Reference",
            image=pept_act_image,
            smoothed_image=pept_act_image,
            input_anchor=(int(target_anchor[0]), int(target_anchor[1])),
            peak_properties=None,
            snapped_anchor=None,
        )

    anchor = np.array([(int(anchor[0]), int(anchor[1]))])
    target_anchor_arr = np.array([(int(target_anchor[0]), int(target_anchor[1]))])
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

    pept_act_image_smoothed = smooth_and_denoise_image(pept_act_image, **smooth_kwargs)
    context_panels: list[dict[str, Any]] | None = None
    # Case "Match": perform template matching to find the best match for the reference peak
    # and then run watershed with the matched position as (updated) anchor
    if reference_image is not None and propA is not None:
        # Getting template for "Match" case
        template_im_start = max(
            (anchor[0][1] - 0.3 * reference_image.shape[1]).astype(int), 0
        )
        template_im_end = min(
            (anchor[0][1] + 0.3 * reference_image.shape[1]).astype(int),
            pept_act_image.shape[1],
        )
        template_rt_start = max(
            (anchor[0][0] - 0.3 * reference_image.shape[0]).astype(int), 0
        )
        template_rt_end = min(
            (anchor[0][0] + 0.3 * reference_image.shape[0]).astype(int),
            pept_act_image.shape[0],
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
        match_bbox_mask[
            propA["bbox-0"].values[0].astype(int)
            + shift[0] : propA["bbox-2"].values[0].astype(int)
            + shift[0],
            propA["bbox-1"].values[0].astype(int)
            + shift[1] : propA["bbox-3"].values[0].astype(int)
            + shift[1],
        ] = 1  # matched bounding box calculated as original bbox plus shift

        anchor = np.array(
            [
                (
                    np.clip(
                        target_anchor_arr[0][0] + shift[0],
                        0,
                        pept_act_image_smoothed.shape[0] - 1,
                    ),
                    np.clip(
                        target_anchor_arr[0][1] + shift[1],
                        0,
                        pept_act_image_smoothed.shape[1] - 1,
                    ),
                )
            ]
        )  # anchor is updated: shifted anchor for the matched image
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
        context_panels = [
            {
                "image": reference_image,
                "title": "source_template_smoothed",
                "msms_pos": tuple(anchor[0]),
                "snapped_msms_pos": tuple(anchor[0]),
                "template_box": (
                    template_rt_start,
                    template_im_start,
                    template_rt_end,
                    template_im_end,
                ),
            },
            {
                "image": pept_act_image_smoothed,
                "title": "target_smoothed_before_mapping",
                "msms_pos": tuple(target_anchor_arr[0]),
                "snapped_msms_pos": tuple(anchor[0]),
                "template_box": (
                    max(
                        propA["bbox-0"].values[0].astype(int) + shift[0],
                        0,
                    ),
                    max(
                        propA["bbox-1"].values[0].astype(int) + shift[1],
                        0,
                    ),
                    min(
                        propA["bbox-2"].values[0].astype(int) + shift[0],
                        pept_act_image_smoothed.shape[0],
                    ),
                    min(
                        propA["bbox-3"].values[0].astype(int) + shift[1],
                        pept_act_image_smoothed.shape[1],
                    ),
                ),
            },
        ]

    # Case quantification without template matching, directly run watershed with the original anchor
    # Which will be snapped into the nearest connected local maximum if the anchor is not already a local maximum
    else:
        if apply_seg:
            _, labels, _, labels_with_multi_marker, snapped_anchor = (
                detect_2d_peak_with_watershed(
                    pept_act_image_smoothed,
                    **peak_kwargs,
                    coordinates=anchor,
                )
            )
        else:
            # Getting template for "Match" case
            template_im_start = max(
                (anchor[0][1] - 0.3 * pept_act_image_smoothed.shape[1]).astype(int), 0
            )
            template_im_end = min(
                (anchor[0][1] + 0.3 * pept_act_image_smoothed.shape[1]).astype(int),
                pept_act_image_smoothed.shape[1],
            )
            template_rt_start = max(
                (anchor[0][0] - 0.3 * pept_act_image_smoothed.shape[0]).astype(int), 0
            )
            template_rt_end = min(
                (anchor[0][0] + 0.3 * pept_act_image_smoothed.shape[0]).astype(int),
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
                pept_act_image,
                pept_act_image_smoothed,
                bbox_center=None,
                save_dir=visualize_dir,
                msms_pos=target_anchor_arr[0] if reference_image is None else None,
                snapped_msms_pos=(
                    snapped_anchor if "snapped_anchor" in locals() else anchor[0]
                ),
                filename=visualize_filename,
                labels=labels_with_multi_marker,
                context_panels=context_panels,
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
            )
        return QuantificationResult(
            run_name="",
            case="Reference",
            image=pept_act_image,
            smoothed_image=pept_act_image_smoothed,
            input_anchor=(int(target_anchor_arr[0][0]), int(target_anchor_arr[0][1])),
            peak_properties=None,
            snapped_anchor=(
                tuple(int(x) for x in snapped_anchor)
                if "snapped_anchor" in locals() and len(snapped_anchor) == 2
                else (int(target_anchor_arr[0][0]), int(target_anchor_arr[0][1]))
            ),
            labels=labels,
            labels_multi_markers=labels_with_multi_marker,
            template_matching_score=template_matching_score_max,
            shift=tuple(int(x) for x in shift) if "shift" in locals() else (0, 0),
        )
    else:
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
            snapped_anchor[0] if "snapped_anchor" in locals() else anchor[0][0]
        )
        peak_properties["snap_im"] = (
            snapped_anchor[1] if "snapped_anchor" in locals() else anchor[0][1]
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
        hu, zernike = get_roi_descriptor(
            seg_bbox,
        )
        peak_properties["hu"] = None
        peak_properties["zernike"] = None
        peak_properties.at[0, "hu"] = hu
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
                msms_pos=target_anchor_arr[0] if reference_image is None else None,
                snapped_msms_pos=(
                    snapped_anchor if "snapped_anchor" in locals() else anchor[0]
                ),
                save_dir=visualize_dir,
                filename=visualize_filename,
                labels=labels_with_multi_marker,
                context_panels=context_panels,
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
            )
        peak_properties = _extract_single_peak_properties(
            peak_properties,
            (
                tuple(int(x) for x in snapped_anchor)
                if "snapped_anchor" in locals() and len(snapped_anchor) == 2
                else (int(target_anchor_arr[0][0]), int(target_anchor_arr[0][1]))
            ),
        )
        return QuantificationResult(
            run_name="",
            case="Reference",
            image=pept_act_image,
            smoothed_image=pept_act_image_smoothed,
            input_anchor=(int(target_anchor_arr[0][0]), int(target_anchor_arr[0][1])),
            peak_properties=peak_properties,
            snapped_anchor=(
                tuple(int(x) for x in snapped_anchor)
                if "snapped_anchor" in locals() and len(snapped_anchor) == 2
                else (int(target_anchor_arr[0][0]), int(target_anchor_arr[0][1]))
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
        "hu_similarities": compare_image_descriptors_cosine(
            peak_properties_a["hu"].values[0], peak_properties_b["hu"].values[0]
        ),
        "zernike_similarities": compare_image_descriptors_cosine(
            peak_properties_a["zernike"].values[0],
            peak_properties_b["zernike"].values[0],
        ),
        "sift_distance": compare_image_descriptors_euclidean(
            peak_properties_a["sift_des"].values[0],
            peak_properties_b["sift_des"].values[0],
        ),
        "hu_distance": compare_image_descriptors_euclidean(
            peak_properties_a["hu"].values[0], peak_properties_b["hu"].values[0]
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

    # Hu Moments
    moments = cv2.moments(roi_norm)
    hu = cv2.HuMoments(moments).flatten()
    # Log transform Hu moments (they span huge ranges)
    hu_abs = np.abs(hu)
    hu = np.zeros_like(hu)
    mask = hu_abs > 0
    hu[mask] = -np.sign(hu[mask]) * np.log10(hu_abs[mask])

    # Zernike Moments
    roi_uint8 = (roi_norm * 255).astype(np.uint8)
    radius = radius if radius is not None else max(roi.shape) // 2
    zernike = zernike_moments(
        roi_uint8, radius, cm=(roi.shape[1] // 2, roi.shape[0] // 2), degree=8
    )
    return hu, zernike


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


def calc_quant_corr(pp_quant_only, pp_reference, pp_match_target, quant_dir):
    os.makedirs(quant_dir, exist_ok=True)
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
    pp_match_target_pivoted = pp_match_target.pivot_table(
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
    # Logger.info("corr_matrix columns: %s", corr_matrix.columns)
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
