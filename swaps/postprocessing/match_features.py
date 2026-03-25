import logging
import os
from typing import Tuple, Literal, Optional
import numpy as np
import pandas as pd
import tqdm
from scipy.ndimage import gaussian_filter, distance_transform_edt, uniform_filter
from skimage.registration import phase_cross_correlation
from skimage.morphology import remove_small_objects
from skimage.metrics import structural_similarity as ssim
from skimage.feature import match_template
from scipy.ndimage import shift as ndi_shift
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


def process_pept_run(
    act_df,
    pept_idx,
    dict_ref,
    run_name,
    case: Literal["Reference", "Quant_Only", "Match"],
    decoy_pept_idx: Optional[int] = None,
    rt_ref: Optional[int] = None,
    im_ref: Optional[int] = None,
    smooth_ref: Optional[np.ndarray] = None,
    prop_ref: Optional[pd.DataFrame] = None,
    processing_kwargs: dict | None = None,
    visualize_dir: str | None = None,
):
    pept_act, rt_msms_pos, im_msms_pos = get_pept_act_from_parquet(
        act_df.loc[act_df["mz_rank"] == pept_idx], pept_idx, dict_ref, run_name
    )

    match case:
        case "Reference":
            # Boundary Check
            if rt_msms_pos >= pept_act.shape[0] or im_msms_pos >= pept_act.shape[1]:
                logging.info(
                    "Pept_act shape: %s, rt_center: %s, im_center: %s",
                    pept_act.shape,
                    rt_msms_pos,
                    im_msms_pos,
                )
                logging.warning(
                    "Skipping reference for mz_rank %s due to center out of bounds",
                    pept_idx,
                )
                return None, None
            smooth_a, prop_a = quantify_from_coords(
                pept_act,
                anchor=(rt_msms_pos, im_msms_pos),
                patch_size=min(pept_act.shape),
                **(processing_kwargs or {}),
                visualize_dir=visualize_dir,
                visualize_filename=f"mz{pept_idx}_{run_name}_reference.png",
            )
            if prop_a is not None:
                prop_a["Run_name"] = run_name
                prop_a["mz_rank"] = pept_idx

            return (
                smooth_a,
                prop_a,
            )
        case "Quant_Only":
            # Boundary Check
            if rt_msms_pos >= pept_act.shape[0] or im_msms_pos >= pept_act.shape[1]:
                logging.warning(
                    "Skipping quant only for mz_rank %s due to center out of bounds",
                    pept_idx,
                )
                return None
            prop_q = quantify_from_coords(
                pept_act,
                anchor=(rt_msms_pos, im_msms_pos),
                patch_size=min(pept_act.shape),
                **(processing_kwargs or {}),
                visualize_dir=visualize_dir,
                visualize_filename=f"mz{pept_idx}_{run_name}_quant_only.png",
            )[1]
            if prop_q is not None:
                prop_q["Run_name"] = run_name
                prop_q["mz_rank"] = pept_idx
            return prop_q
        case "Match":
            assert all(
                param is not None
                for param in [decoy_pept_idx, rt_ref, im_ref, smooth_ref, prop_ref]
            ), "All parameters must be provided for 'Match' case"
            if rt_ref >= pept_act.shape[0] or im_ref >= pept_act.shape[1]:
                logging.warning(
                    "Skipping matches for mz_rank %s due to reference center out of bounds",
                    pept_idx,
                )
                return None, None, None, None
            pept_act_decoy, _, _ = get_pept_act_from_parquet(
                act_df.loc[act_df["mz_rank"] == decoy_pept_idx],
                decoy_pept_idx,
                dict_ref,
                run_name,
                shape=pept_act.shape,
            )

            # Target quantification
            _, prop_t = quantify_from_coords(
                pept_act,
                anchor=(rt_ref, im_ref),
                reference_image=smooth_ref,
                propA=prop_ref,
                patch_size=min(pept_act.shape),
                **(processing_kwargs or {}),
                visualize_dir=None,  # Do not plot target quantification to save time, as the main focus is on the match comparison results rather than the individual quantification quality for each match, and we already have the reference quantification visualization to show the feature quality of the reference
                visualize_filename=f"mz{pept_idx}_{run_name}_match_target.png",
            )

            # Decoy quantification
            _, prop_d = quantify_from_coords(
                pept_act_decoy,
                anchor=(rt_ref, im_ref),
                reference_image=smooth_ref,
                propA=prop_ref,
                patch_size=min(pept_act.shape),
                **(processing_kwargs or {}),
                visualize_dir=None,  # Do not plot decoy quantification to save time, as the decoy is randomly selected and may not have meaningful features
                visualize_filename=f"mz{pept_idx}_decoy{decoy_pept_idx}_{run_name}_match_decoy.png",
            )
            # 5. Process Matches
            if prop_ref is not None and prop_t is not None:
                prop_t["Run_name"] = run_name
                prop_t["mz_rank"] = pept_idx
                match_t = compare_peak_properties(prop_ref, prop_t)
                match_t["mz_rank"] = pept_idx

            else:
                match_t = None
            if prop_ref is not None and prop_d is not None:
                prop_d["Run_name"] = run_name
                prop_d["mz_rank"] = pept_idx
                prop_d["decoy_mz_rank"] = decoy_pept_idx
                match_d = compare_peak_properties(prop_ref, prop_d)
                match_d["mz_rank"] = pept_idx
                match_d["decoy_mz_rank"] = decoy_pept_idx

            else:
                match_d = None
            return prop_t, prop_d, match_t, match_d


def match_features_batch(
    dict_ref,
    raw_file_list,
    result_dir,
    batch,
    processing_kwargs: dict | None = None,
    visualize_dir: str | None = None,
):
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
        # Extract the single row as a Series to make index filtering easier
        row_series = dict_ref.loc[dict_ref["mz_rank"] == pept_idx, :].iloc[0]

        # 1. Create masks using the Series
        is_ref_mask = row_series.map(
            lambda x: x == "Reference" if isinstance(x, str) else False
        )
        is_quant_only_mask = row_series.map(
            lambda x: x == "Quant_Only" if isinstance(x, str) else False
        )
        is_match_mask = row_series.map(
            lambda x: "Match" in x if isinstance(x, str) else False
        )

        # 2. Get the Reference (String)
        # idxmax on a Series returns the index label (column name) of the True value
        reference_raw_file = str(is_ref_mask.idxmax())
        # 3. Get the Quant_Only and Match (Flat Lists of strings)
        # We simply filter the index of the series by the boolean mask
        quant_only_raw_file = is_quant_only_mask.index[is_quant_only_mask].tolist()
        match_raw_file = is_match_mask.index[is_match_mask].tolist()

        # Get reference
        smooth_a, prop_a = process_pept_run(  # type: ignore
            act_dfs[reference_raw_file].loc[
                act_dfs[reference_raw_file]["mz_rank"] == pept_idx
            ],
            pept_idx,
            dict_ref,
            run_name=reference_raw_file,
            case="Reference",
            processing_kwargs=processing_kwargs,
            visualize_dir=visualize_dir,
        )  # type: ignore
        if prop_a is not None:
            pp_reference_list.append(prop_a)
        else:
            no_quant_log.append(
                {
                    "mz_rank": pept_idx,
                    "run_name": reference_raw_file,
                    "type": "reference",
                }
            )

        # Quant only TODO: multi-coordinates from quant_only runs
        if len(quant_only_raw_file) > 0:
            for raw_file in quant_only_raw_file:
                prop_q = process_pept_run(  # type: ignore
                    act_dfs[raw_file].loc[act_dfs[raw_file]["mz_rank"] == pept_idx],
                    pept_idx,
                    dict_ref,
                    run_name=raw_file,
                    case="Quant_Only",
                    processing_kwargs=processing_kwargs,
                    visualize_dir=visualize_dir,
                )
                if prop_q is not None:
                    pp_quant_only_list.append(prop_q)
                else:
                    no_quant_log.append(
                        {
                            "mz_rank": pept_idx,
                            "run_name": raw_file,
                            "type": "quant_only",
                        }
                    )

        # Matches
        if len(match_raw_file) > 0:
            if smooth_a is not None and prop_a is not None:
                for raw_file in match_raw_file:
                    batch_exclude = batch[batch != pept_idx]
                    decoy_pept_idx = np.random.choice(batch_exclude)
                    prop_t, prop_d, match_t, match_d = process_pept_run(
                        act_df=act_dfs[raw_file].loc[
                            act_dfs[raw_file]["mz_rank"].isin(
                                [pept_idx, decoy_pept_idx]
                            )
                        ],
                        pept_idx=pept_idx,
                        dict_ref=dict_ref,
                        run_name=raw_file,
                        case="Match",
                        decoy_pept_idx=decoy_pept_idx,
                        rt_ref=prop_a["snap_rt"].values[0].astype(int),
                        im_ref=prop_a["snap_im"].values[0].astype(int),
                        smooth_ref=smooth_a,
                        prop_ref=prop_a,
                        processing_kwargs=processing_kwargs,
                        visualize_dir=visualize_dir,
                    )
                    if prop_t is not None:
                        pp_match_target_list.append(prop_t)
                    else:
                        no_quant_log.append(
                            {
                                "mz_rank": pept_idx,
                                "run_name": raw_file,
                                "type": "match_target",
                            }
                        )
                    if prop_d is not None:
                        pp_match_decoy_list.append(prop_d)

                    else:
                        no_quant_log.append(
                            {
                                "mz_rank": pept_idx,
                                "run_name": raw_file,
                                "type": "match_decoy",
                            }
                        )
                    if match_t is not None:
                        results_target.append(match_t)
                    else:
                        no_match_log.append(
                            {
                                "mz_rank": pept_idx,
                                "run_name": raw_file,
                                "type": "match_target",
                            }
                        )
                    if match_d is not None:
                        results_decoy.append(match_d)
                    else:
                        no_match_log.append(
                            {
                                "mz_rank": pept_idx,
                                "run_name": raw_file,
                                "type": "match_decoy",
                            }
                        )
            else:
                # log no quantification for both match target and decoy if reference quantification is not available,
                # as the match processing relies on the reference quantification for alignment and property comparison
                for raw_file in match_raw_file:
                    no_quant_log.append(
                        {
                            "mz_rank": pept_idx,
                            "run_name": raw_file,
                            "type": "match_target",
                        }
                    )
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
):
    # labels apply to the aligned panels (watershed ran on smoothed_aligned)
    images = [
        (reference_image, "reference_image", None),
        (pept_act_image, "pept_act_image", None),
        (pept_act_image_smoothed, "pept_act_image_smoothed", None),
        (pept_act_image_aligned, "pept_act_image_aligned", labels),
        (pept_act_image_smoothed_aligned, "pept_act_image_smoothed_aligned", labels),
    ]
    n_cols = 6 if labels is not None else 5
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    for ax, (img, title, lbl) in zip(axes, images):
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
        else:
            ax.imshow(img, aspect="auto", origin="lower")
            if bbox_center is not None:
                ax.plot(
                    bbox_center[0][1],
                    bbox_center[0][0],
                    "r+",
                    markersize=10,
                    markeredgewidth=2,
                )
            if msms_pos is not None:
                ax.plot(
                    msms_pos[1],
                    msms_pos[0],
                    "*",
                    markersize=10,
                    markeredgewidth=2,
                    color="white",
                )  # white * for MS/MS position
            if snapped_msms_pos is not None and len(snapped_msms_pos) == 2:
                # Logger.info("snapped_msms_pos: %s", snapped_msms_pos)
                ax.plot(
                    snapped_msms_pos[1],
                    snapped_msms_pos[0],
                    "*",
                    markersize=10,
                    markeredgewidth=2,
                    color="yellow",
                )  # yellow * for snapped MS/MS position
            if template_box is not None:
                ax.add_patch(
                    plt.Rectangle(
                        (template_box[1], template_box[0]),
                        template_box[3] - template_box[1],
                        template_box[2] - template_box[0],
                        fill=False,
                        edgecolor="red",
                        linewidth=2,
                    )
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
            axes[0].add_patch(
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
        The anchor coordinates (row, column).
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
    pd.DataFrame | None
        The quantified properties dataframe or None if no features are found.
    """
    assert (
        anchor[0] < pept_act_image.shape[0] and anchor[1] < pept_act_image.shape[1]
    ), "Anchor coordinates are out of bounds of the image dimensions."
    anchor = np.array([(anchor[0].astype(int), anchor[1].astype(int))])
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
                        anchor[0][0] + shift[0], 0, pept_act_image_smoothed.shape[0] - 1
                    ),
                    np.clip(
                        anchor[0][1] + shift[1], 0, pept_act_image_smoothed.shape[1] - 1
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
                msms_pos=anchor[0] if reference_image is None else None,
                snapped_msms_pos=(
                    snapped_anchor if "snapped_anchor" in locals() else anchor[0]
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
            )
        return pept_act_image_smoothed, None
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
                msms_pos=anchor[0] if reference_image is None else None,
                snapped_msms_pos=(
                    snapped_anchor if "snapped_anchor" in locals() else anchor[0]
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
                    if propA is not None
                    else None
                ),
            )

        return pept_act_image_smoothed, peak_properties


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


def align_images(reference_image, aligned_image, mask_threshold=25):
    """Align two images using phase cross-correlation and return the aligned image and the calculated shift.
    Parameters
    ----------
    reference_image : 2D array
        Reference image to align to.
    image_b : 2D array
        Image to be aligned.
    Returns
    -------
    aligned_image_b : 2D array
        Aligned version of image_b.
    shift : tuple
        Calculated shift applied to image_b.
    """
    mask1 = reference_image > np.percentile(reference_image, mask_threshold)
    mask2 = aligned_image > np.percentile(aligned_image, mask_threshold)
    shift, _, phasediff = phase_cross_correlation(
        mask1,
        mask2,
        reference_mask=mask1.astype(bool),
        moving_mask=mask2.astype(bool),
        upsample_factor=1,
        normalization=None,
    )

    aligned_image_b = ndi_shift(aligned_image, shift)

    return aligned_image_b, shift, phasediff


def compare_windowed_cosine(img1, img2, peak_coords, window_size=21):
    y, x = peak_coords
    r = window_size // 2

    # Pad to handle peaks near the boundary
    pad1 = np.pad(img1, r, mode="edge")
    pad2 = np.pad(img2, r, mode="edge")

    # Adjust coords for padding
    py, px = y + r, x + r

    # Extract and flatten patches
    patch1 = pad1[py - r : py + r + 1, px - r : px + r + 1].ravel()
    patch2 = pad2[py - r : py + r + 1, px - r : px + r + 1].ravel()

    # Cosine Similarity Formula
    norm = np.linalg.norm(patch1) * np.linalg.norm(patch2)
    return np.dot(patch1, patch2) / norm if norm != 0 else 0.0


def compare_gaussian_weighted(img1, img2, peak_coords, sigma=15):
    y, x = peak_coords
    yy, xx = np.indices(img1.shape)

    # Create Gaussian weight centered at coordinates
    dist_sq = (yy - y) ** 2 + (xx - x) ** 2
    weights = np.exp(-dist_sq / (2 * sigma**2))

    # Normalize to 0-1 for SSIM consistency
    i1_n = cv2.normalize(img1.astype(float), None, 0, 1, cv2.NORM_MINMAX)
    i2_n = cv2.normalize(img2.astype(float), None, 0, 1, cv2.NORM_MINMAX)

    # Structural Similarity weighted by the Gaussian mask
    return ssim(i1_n * weights, i2_n * weights, data_range=1.0)


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


def compare_orb_descriptors(des1, des2):
    """
    Compares two pre-computed descriptors using Hamming distance.
    Returns a similarity score between 0.0 and 1.0.
    """
    if des1 is None or des2 is None:
        return 0.0

    # Hamming distance: count bit differences
    # Lower distance = Higher similarity
    dist = cv2.norm(des1, des2, cv2.NORM_HAMMING)

    # ORB descriptors are 256 bits (32 bytes)
    return 1.0 - (dist / 256.0)


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
    roi_norm = (roi - roi.min()) / (roi.max() - roi.min())

    # Hu Moments
    moments = cv2.moments(roi_norm)
    hu = cv2.HuMoments(moments).flatten()
    # Log transform Hu moments (they span huge ranges)
    hu = -np.sign(hu) * np.log10(np.abs(hu))

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
    pp_quant_only_pivoted = pp_quant_only.pivot(
        index="mz_rank",
        columns="Run_name",
        values="intensity_sum",
    ).reset_index()
    pp_reference_pivoted = pp_reference.pivot(
        index="mz_rank",
        columns="Run_name",
        values="intensity_sum",
    ).reset_index()
    # Log-transform numeric columns and compute pairwise Pearson correlations (pairwise complete cases)
    pp_match_target_pivoted = pp_match_target.pivot(
        index="mz_rank",
        columns="Run_name",
        values="intensity_sum",
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
