import os
import logging
from typing import Literal, List, Sequence
import sparse
from sparse import SparseArray, asnumpy, COO
import pandas as pd
import numpy as np
from scipy.ndimage import maximum_filter
from scipy.ndimage import generate_binary_structure, binary_erosion
from scipy.spatial.distance import euclidean

Logger = logging.getLogger(__name__)


def get_ref_rt_im_range(
    pept_mz_rank: int,
    maxquant_result_dict: pd.DataFrame,
    ms1scans: pd.DataFrame,
    mobility_values_df: pd.DataFrame,
    delta_im: float = 0.04,
    ref_rt_range: tuple | None = None,
    ref_im_range: tuple | None = None,
    # bbox: tuple | None = None,
):
    """
    Get the RT and IM range from the maxquant result dictionary
    """
    reference_entry = []
    if ref_rt_range is not None:
        rt_min, rt_max = ref_rt_range
        rt_center = (rt_min + rt_max) / 2
        reference_entry.append(rt_center)
    else:
        (rt_min, rt_max, rt_center) = maxquant_result_dict.loc[
            maxquant_result_dict["mz_rank"] == pept_mz_rank,
            ["RT_search_left", "RT_search_right", "RT_search_center"],
        ].values[0]
        reference_entry.append(rt_center)
        Logger.debug(
            "No reference RT range given, using dictionary entries: %s, (%s, %s).",
            rt_center,
            rt_min,
            rt_max,
        )
    if ref_im_range is not None:
        im_min, im_max = ref_im_range
        reference_entry.append((im_min + im_max) / 2)
    else:
        im_center = maxquant_result_dict.loc[
            maxquant_result_dict["mz_rank"] == pept_mz_rank,
            ["mobility_values_center_exp"],
        ].values[0][0]
        reference_entry.append(im_center)
        im_min, im_max = im_center - delta_im, im_center + delta_im
        Logger.debug(
            "No reference IM range given, using dictionary entries: %s, (%s, %s).",
            im_center,
            im_min,
            im_max,
        )

    rt_array = ms1scans["Time_minute"].to_numpy()
    rt_min_idx = max(np.searchsorted(rt_array, rt_min, side="left") - 1, 0)
    rt_max_idx = np.searchsorted(rt_array, rt_max, side="right")
    im_array = mobility_values_df["mobility_values"].to_numpy()
    im_min_idx = max(np.searchsorted(im_array, im_min, side="left") - 1, 0)
    im_max_idx = np.searchsorted(im_array, im_max, side="right")
    im_center_idx = np.abs(
        mobility_values_df["mobility_values"].values - im_center[0]
    ).argmin()
    rt_center_idx = np.abs(ms1scans["Time_minute"].values - rt_center).argmin()

    return (
        [rt_min_idx, rt_max_idx + 1],
        [im_min_idx, im_max_idx + 1],
        reference_entry,
        [rt_center_idx, im_center_idx],
    )


def slice_pept_act(
    pept_act_sparse: COO,
    plot_range: Literal["nonzero", "custom"] = "nonzero",
    rt_idx_range: list | None = None,
    im_idx_range: list | None = None,
):
    match plot_range:
        case "nonzero":
            nonzero_indices = np.nonzero(pept_act_sparse)
            rt_idx_range = [nonzero_indices[0].min(), nonzero_indices[0].max() + 1]
            im_idx_range = [nonzero_indices[1].min(), nonzero_indices[1].max() + 1]
        case "custom":
            assert rt_idx_range is not None
            assert im_idx_range is not None
            # Logger.debug("rt_idx_range, im_idx_range %s %s", rt_idx_range, im_idx_range)
    slice_pept_act_sparse = pept_act_sparse[
        rt_idx_range[0] : rt_idx_range[1], im_idx_range[0] : im_idx_range[1]
    ]
    return slice_pept_act_sparse, rt_idx_range, im_idx_range


def prepare_slice_pept_act_df(
    slice_pept_act_sparse: SparseArray,
    rt_idx_range: Sequence[int],
    im_idx_range: Sequence[int],
    mobility_values_df: pd.DataFrame,
    ms1scans: pd.DataFrame,
    convert_idx_to_values: bool = True,
):
    slice_pept_act_df = pd.DataFrame(asnumpy(slice_pept_act_sparse))
    if convert_idx_to_values:
        slice_pept_act_df.columns = mobility_values_df["mobility_values"].values[
            im_idx_range[0] : im_idx_range[1]
        ]
        slice_pept_act_df.index = pd.Index(
            ms1scans["Time_minute"].values[rt_idx_range[0] : rt_idx_range[1]]
        )
    slice_pept_act_df.replace(0, np.nan, inplace=True)
    return slice_pept_act_df


def detect_2d_peaks_apex(pept_act_df, filter_size: tuple = (5, 55)):
    """
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """

    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(2, 2)

    # apply the local maximum filter; all pixel of maximal value
    # in their neighborhood are set to 1
    local_max = maximum_filter(pept_act_df, size=filter_size) == pept_act_df

    # local_max is a mask that contains the peaks we are
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.

    # we create the mask of the background
    background = pept_act_df == 0

    # a little technicality: we must erode the background in order to
    # successfully subtract it form local_max, otherwise a line will
    # appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(
        background, structure=neighborhood, border_value=1
    )

    # we obtain the final mask, containing only peaks,
    # by removing the background from the local_max mask (xor operation)
    detected_peaks_apex_mat = local_max ^ eroded_background
    Logger.info(
        "Number of detected apex with size %s: %s",
        filter_size,
        detected_peaks_apex_mat.values.sum(),
    )
    detected_peaks_apex_indices = np.where(detected_peaks_apex_mat)
    detected_peaks_apex_rt_im = (
        pept_act_df.index[detected_peaks_apex_indices[0]],
        pept_act_df.columns[detected_peaks_apex_indices[1]],
    )
    return detected_peaks_apex_mat, detected_peaks_apex_rt_im


def score_peak_apex(peak_apex_list: List[tuple], reference_entry: list):
    """
    Takes a list of peak apexes and a reference entry and scores the peak apexes based on the reference entry
    """
    score_list = []
    for peak_apex in zip(*peak_apex_list):
        score_list.append(euclidean(peak_apex, reference_entry, w=[1, 10000]))
    return score_list


def get_highest_score_peak_apex(peak_apex_list, reference_entry):
    """
    Takes a list of peak apexes and a reference entry and returns the peak apex with the highest score
    """
    score_list = score_peak_apex(peak_apex_list, reference_entry)
    return (
        peak_apex_list[0][score_list.index(min(score_list))],
        peak_apex_list[1][score_list.index(min(score_list))],
    )


def nearest_na_index(series, given_index):
    # Find all NA indices
    na_indices = series.index[series.isna()]

    # Calculate distances to NA indices to the left and right of the given index
    distances = na_indices - given_index

    # Filter out negative distances (indices on the left)
    distances_left = np.where(distances <= 0, distances, -np.inf)
    distances_right = np.where(distances >= 0, distances, np.inf)

    # Find the nearest NA indices to the left and right
    nearest_na_index_left = na_indices[distances_left.argmax()]
    nearest_na_index_right = na_indices[distances_right.argmin()]

    return nearest_na_index_left, nearest_na_index_right


def get_peak_rt_im_range(peak_apex: tuple, pept_ref_slice):
    """Get the nearest NA value to the peak apex in the reference slice"""
    im_slice = pept_ref_slice.loc[peak_apex[0]]
    rt_slice = pept_ref_slice[peak_apex[1]]
    # Logger.debug("show rt slice: %s", rt_slice)
    rt_indices = nearest_na_index(rt_slice, peak_apex[0])
    im_indices = nearest_na_index(im_slice, peak_apex[1])
    return rt_indices, im_indices


def get_bbox_from_mq_exp(
    maxquant_result_exp_row: pd.Series,
    rt_start_col: str = "Calibrated retention time start",
    rt_end_col: str = "Calibrated retention time finish",
):
    """
    Get bounding box from MaxQuant result experiment table, in absolute values
    """
    min_rt = maxquant_result_exp_row[rt_start_col].values[0]
    max_rt = maxquant_result_exp_row[rt_end_col].values[0]
    center_rt = (min_rt + max_rt) / 2
    rt_width = max_rt - min_rt
    center_im = maxquant_result_exp_row["1/K0"].values[0]
    im_width = maxquant_result_exp_row["1/K0 length"].values[0]
    min_im, max_im = center_im - im_width / 2, center_im + im_width / 2
    return [min_rt, max_rt, rt_width, min_im, max_im, im_width]


