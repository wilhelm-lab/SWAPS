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
from scipy.ndimage import shift as ndi_shift
import cv2
from concurrent.futures import ProcessPoolExecutor
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
    smooth_kwargs: dict | None = None,
    peak_kwargs: dict | None = None,
    align_kwargs: dict | None = None,
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
                smooth_kwargs,
                peak_kwargs,
                align_kwargs,
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
    smooth_kwargs=None,
    peak_kwargs=None,
    align_kwargs=None,
):
    pept_act, rt_center, im_center = get_pept_act_from_parquet(
        act_df.loc[act_df["mz_rank"] == pept_idx], pept_idx, dict_ref, run_name
    )

    match case:
        case "Reference":
            # Boundary Check
            if rt_center >= pept_act.shape[0] or im_center >= pept_act.shape[1]:
                logging.info(
                    "Pept_act shape: %s, rt_center: %s, im_center: %s",
                    pept_act.shape,
                    rt_center,
                    im_center,
                )
                logging.warning(
                    "Skipping reference for mz_rank %s due to center out of bounds",
                    pept_idx,
                )
                return None, None, None, None
            smooth_a, prop_a = quantify_from_coords(
                pept_act,
                anchor=(rt_center, im_center),
                patch_size=min(pept_act.shape),
            )
            if prop_a is not None:
                prop_a["Run_name"] = run_name
                prop_a["mz_rank"] = pept_idx
            return smooth_a, prop_a, rt_center, im_center
        case "Quant_Only":
            # Boundary Check
            if rt_center >= pept_act.shape[0] or im_center >= pept_act.shape[1]:
                logging.warning(
                    "Skipping quant only for mz_rank %s due to center out of bounds",
                    pept_idx,
                )
                return None
            prop_a = quantify_from_coords(
                pept_act,
                anchor=(rt_center, im_center),
                patch_size=min(pept_act.shape),
            )[1]
            if prop_a is not None:
                prop_a["Run_name"] = run_name
                prop_a["mz_rank"] = pept_idx
            return prop_a
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
                patch_size=min(pept_act.shape),
                smooth_kwargs=smooth_kwargs,
                peak_kwargs=peak_kwargs,
                align_kwargs=align_kwargs,
            )

            # Decoy quantification
            _, prop_d = quantify_from_coords(
                pept_act_decoy,
                anchor=(rt_ref, im_ref),
                reference_image=smooth_ref,
                patch_size=min(pept_act.shape),
                smooth_kwargs=smooth_kwargs,
                peak_kwargs=peak_kwargs,
                align_kwargs=align_kwargs,
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
    smooth_kwargs: dict | None = None,
    peak_kwargs: dict | None = None,
    align_kwargs: dict | None = None,
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
        # Logger.info(
        #     f"Raw files for reference, quant_only, match for mz_rank {pept_idx}: {reference_raw_file}, {quant_only_raw_file}, {match_raw_file}"
        # )
        # Get reference
        smooth_a, prop_a, rt_center, im_center = process_pept_run(
            act_dfs[reference_raw_file].loc[
                act_dfs[reference_raw_file]["mz_rank"] == pept_idx
            ],
            pept_idx,
            dict_ref,
            run_name=reference_raw_file,
            case="Reference",
        )
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

        # Quant only
        if len(quant_only_raw_file) > 0:
            for raw_file in quant_only_raw_file:
                prop_a = process_pept_run(
                    act_dfs[raw_file].loc[act_dfs[raw_file]["mz_rank"] == pept_idx],
                    pept_idx,
                    dict_ref,
                    run_name=raw_file,
                    case="Quant_Only",
                    smooth_kwargs=smooth_kwargs,
                    peak_kwargs=peak_kwargs,
                    align_kwargs=align_kwargs,
                )
                if prop_a is not None:
                    pp_quant_only_list.append(prop_a)
                else:
                    no_quant_log.append(
                        {
                            "mz_rank": pept_idx,
                            "run_name": raw_file,
                            "type": "quant_only",
                        }
                    )

        # Matches
        if len(match_raw_file) > 0 and smooth_a is not None and prop_a is not None:

            for raw_file in match_raw_file:
                batch_exclude = batch[batch != pept_idx]
                decoy_pept_idx = np.random.choice(batch_exclude)
                prop_t, prop_d, match_t, match_d = process_pept_run(
                    act_dfs[raw_file].loc[
                        act_dfs[raw_file]["mz_rank"].isin([pept_idx, decoy_pept_idx])
                    ],
                    pept_idx,
                    dict_ref,
                    run_name=raw_file,
                    case="Match",
                    decoy_pept_idx=decoy_pept_idx,
                    rt_ref=rt_center,
                    im_ref=im_center,
                    smooth_ref=smooth_a,
                    prop_ref=prop_a,
                    smooth_kwargs=smooth_kwargs,
                    peak_kwargs=peak_kwargs,
                    align_kwargs=align_kwargs,
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


def quantify_from_coords(
    pept_act_image,
    anchor,
    reference_image: np.ndarray | None = None,
    smooth_kwargs: dict | None = None,
    peak_kwargs: dict | None = None,
    align_kwargs: dict | None = None,
    patch_size: int | None = None,
):
    assert (
        anchor[0] < pept_act_image.shape[0] and anchor[1] < pept_act_image.shape[1]
    ), "Anchor coordinates are out of bounds of the image dimensions."
    anchor = np.array([(anchor[0].astype(int), anchor[1].astype(int))])
    smooth_kwargs = {} if smooth_kwargs is None else dict(smooth_kwargs)
    peak_kwargs = {} if peak_kwargs is None else dict(peak_kwargs)
    align_kwargs = {} if align_kwargs is None else dict(align_kwargs)
    if "int_threshold" not in peak_kwargs:
        peak_kwargs["int_threshold"] = 1
    if "threshold_rel" not in peak_kwargs:
        peak_kwargs["threshold_rel"] = 0.2
    if "min_distance" not in peak_kwargs:
        peak_kwargs["min_distance"] = 10

    pept_act_image_smoothed = smooth_and_denoise_image(pept_act_image, **smooth_kwargs)

    if reference_image is not None:
        pept_act_image_smoothed_aligned, shift, phasediff = align_images(
            aligned_image=pept_act_image_smoothed,
            reference_image=reference_image,
            **align_kwargs,
        )
        pept_act_image_aligned = ndi_shift(pept_act_image, shift)
    else:
        pept_act_image_smoothed_aligned = pept_act_image_smoothed
        pept_act_image_aligned = pept_act_image
    _, labels, _ = detect_2d_peak_with_watershed(
        pept_act_image_smoothed_aligned,
        **peak_kwargs,
        coordinates=anchor,
    )
    peak_properties = calculate_peak_property_from_labels_and_image(
        labels, pept_act_image_aligned, min_peak_sum_intensity=500
    )
    if peak_properties is None:
        return pept_act_image_smoothed_aligned, None
    else:
        peak_properties["orb_des"] = None
        peak_properties.at[0, "orb_des"] = get_sift_descriptor(
            pept_act_image_aligned, anchor[0], patch_size=patch_size
        )
        if reference_image is not None:
            peak_properties["shift_rt"] = shift[0]
            peak_properties["shift_im"] = shift[1]
        else:
            peak_properties["shift_rt"] = 0
            peak_properties["shift_im"] = 0
        return pept_act_image_smoothed_aligned, peak_properties


def compare_peak_properties(peak_properties_a, peak_properties_b):
    return {
        "orb_similarity": compare_sift_descriptors(
            peak_properties_a["orb_des"].values[0],
            peak_properties_b["orb_des"].values[0],
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
        gaussian_kwargs["sigma"] = 2
    if "size" not in uniform_kwargs:
        uniform_kwargs["size"] = (1, 10)
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


def align_images(reference_image, aligned_image, mask_threshold=25, upsample_factor=10):
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
        reference_image,
        aligned_image,
        reference_mask=mask1,
        moving_mask=mask2,
        upsample_factor=upsample_factor,
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
                fontsize=7,
                color="white",
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
    df, colors=None, labels=None, stack_order=None, fig_dir=None
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
            os.path.join(fig_dir, "match_type_counts.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
    plt.show()
