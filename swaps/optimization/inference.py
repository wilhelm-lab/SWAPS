import logging
import os
from multiprocessing import cpu_count
from typing import Callable, List, Union, Literal, Optional
import itertools
from operator import itemgetter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import spatial
from sparse import COO
from scipy.signal import find_peaks
from sklearn.decomposition import sparse_encode
from math import floor
from functools import wraps

# from optimization.dictionary import Dict
from optimization.custom_models import CustomLinearModel, mean_square_root_error
from utils.plot import plot_comparison, plot_isopattern_and_obs
from utils.constants import (
    _algo,
    _alpha_criteria,
    _alpha_opt_metric,
    _loss,
    _pp_method,
)
import sparse

Logger = logging.getLogger(__name__)


def sparse_encode_divide_and_conquer_with_residual_stats(
    frame_array, candidate_array, return_act_res=False, target_block_size=6000
):
    """
    Perform sparse encoding using divide-and-conquer on candidate blocks.
    Optionally compute residual statistics in measurement and candidate spaces.

    Parameters
    ----------
    frame_array : np.ndarray
        Measurement data (m_z, im)
    candidate_array : np.ndarray
        Candidate dictionary (m_z, pept)
    return_act_res : bool, optional
        Whether to compute activation residuals (per candidate and im)
    target_block_size : int, optional
        Target number of rows per candidate block

    Returns
    -------
    frame_act : np.ndarray
        Activation matrix (im, pept)
    frame_res : np.ndarray, optional
        Residual in candidate space (pept, im)
    """
    # --- Slice candidate array into blocks (rows = m/z) ---
    candidate_coo_blocks, col_start, col_end = slice_candidate_blocks_by_pept(
        candidate_array, target_block_size=target_block_size
    )

    im = frame_array.shape[0]  # im, m_z
    pept = candidate_array.shape[0]  # pept, m_z

    # Preallocate outputs
    frame_act = np.zeros((im, pept), dtype=np.float32)
    Logger.debug("Frame activation shape: %s", frame_act.shape)
    reconstruction = np.zeros_like(frame_array, dtype=np.float32)

    # Precompute offsets for placing peptide activations
    block_sizes = [cb.shape[1] for cb in candidate_coo_blocks]
    col_offsets = np.cumsum([0] + block_sizes[:-1])

    # --- Encoding loop ---
    for idx, candidate_block in enumerate(candidate_coo_blocks):
        start, end = col_start[idx], col_end[idx]  # m/z row slice for this block
        frame_block = frame_array[:, start:end]  # (m_z_block, im)
        Logger.debug(
            "Frame block shape: %s, candidate block shape: %s",
            frame_block.shape,
            candidate_block.shape,
        )
        # sparse_encode expects: frame_block (m_z_block, im), candidate_block (m_z_block, pept_block)
        im_pept_act = sparse_encode(
            frame_block,
            candidate_block,
            algorithm="threshold",
            alpha=0,
            positive=True,
        )  # returns (im, pept_block)
        Logger.debug("im_pept_act shape: %s", im_pept_act.shape)
        # Place activations in correct columns
        col_offset = col_offsets[idx]
        Logger.debug(
            "Processing block %s: m/z rows %s-%s, col offset %s",
            idx,
            start,
            end,
            col_offset,
        )
        frame_act[:, col_offset : col_offset + candidate_block.shape[1]] += im_pept_act

        if return_act_res:
            # Add reconstruction contribution for residual computation
            reconstruction[start:end, :] += im_pept_act @ candidate_block
    Logger.debug("frame_act non-zero count: %s", np.count_nonzero(frame_act))
    if return_act_res:
        # --- Measurement-space residual ---
        residual_im_mz = frame_array - reconstruction  # (im, m_z)

        # # --- Candidate-space residual stats ---

        # Mean residual per peptide
        frame_res = (
            candidate_array > 0
        ) @ residual_im_mz.T  # (pept, im) same dim as frame_act.T
        return frame_act, frame_res

    else:
        return frame_act


def _decide_row_cuts(n_rows, target_block_size=6000):
    """
    Decide row cut indices intelligently based on target block size.
    Tries to split evenly if leftover is small.
    """
    if n_rows <= target_block_size:
        return [n_rows]  # no slicing needed

    num_blocks = round(n_rows / target_block_size)
    num_blocks = max(1, num_blocks)  # ensure at least one block

    block_size = n_rows / num_blocks
    row_cut_indices = [round(block_size * (i + 1)) for i in range(num_blocks)]

    # Ensure final cut exactly matches n_rows
    row_cut_indices[-1] = n_rows

    return row_cut_indices


def slice_candidate_blocks_by_pept(matrix, target_block_size):
    """
    Slices candidate matrix into row blocks without splitting isotope envelopes.
    Returns blocks, start and end column indices for each block.
    """
    n_rows = matrix.shape[0]
    row_cut_indices = _decide_row_cuts(n_rows, target_block_size)

    blocks = []
    col_cut_indices_start = []
    col_cut_indices_end = []

    prev_row_cut = 0
    for row_cut_index in row_cut_indices:
        block_rows = matrix[prev_row_cut:row_cut_index, :]
        blocks_col_sum = block_rows.sum(axis=0)

        non_zero_indices = np.flatnonzero(blocks_col_sum)

        if non_zero_indices.size > 0:
            col_cut_index_start = non_zero_indices[0]
            col_cut_index_end = (
                non_zero_indices[-1] + 1
            )  # +1 because Python slicing is exclusive
            block = matrix[
                prev_row_cut:row_cut_index, col_cut_index_start:col_cut_index_end
            ]
        else:
            col_cut_index_start, col_cut_index_end = 0, 0
            block = matrix[prev_row_cut:row_cut_index, 0:0]

        blocks.append(block)
        col_cut_indices_start.append(col_cut_index_start)
        col_cut_indices_end.append(col_cut_index_end)

        prev_row_cut = row_cut_index

    assert sum(block.shape[0] for block in blocks) == n_rows, "Row slicing mismatch"
    return blocks, col_cut_indices_start, col_cut_indices_end


def _find_first_nonzero(arr, axis, invalid_val=-1):
    mask = arr != 0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)


def _find_last_nonzero(arr, axis, invalid_val=-1):
    mask = arr != 0
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)


def slice_candidate_blocks_by_mz(matrix):
    # TODO: finding valid slice position not done
    assert matrix.shape[1] >= 6000
    Logger.debug("Candidate matrix shape %s", matrix.shape)
    # slice candidate block by not splitting up isotope envlopes
    n_blocks = matrix.shape[1] // 3000
    Logger.debug("Slice candidate blocks into %s blocks.", n_blocks)
    block_size = matrix.shape[1] // n_blocks
    ref_col_cut_indices = [block_size * (i + 1) for i in range(n_blocks)]

    col_last_nonzero = _find_last_nonzero(arr=matrix, axis=0, invalid_val=-1)
    col_first_nonzero = _find_first_nonzero(arr=matrix, axis=0, invalid_val=-1)

    # comparison only starts with index 1, for actual indexing needs to +1
    valid_slice_pos = np.where(col_last_nonzero[0:-1] < col_first_nonzero[1:])[0] + 1
    # Logger.debug("valid slice pos %s", valid_slice_pos)
    col_cut_indices = []
    for i in range(n_blocks - 1):
        Logger.debug("column slice idx %s for block %s", ref_col_cut_indices[i], i + 1)
        idx = (np.abs(valid_slice_pos - ref_col_cut_indices[i])).argmin()
        col_cut_indices.append(valid_slice_pos[idx])
        valid_slice_pos = np.delete(valid_slice_pos, list(range(0, idx + 1)))
    if col_cut_indices[-1] != matrix.shape[1] + 1:
        col_cut_indices.append(matrix.shape[1] + 1)
    blocks = []
    prev_col_cut = 0
    col_cut_indices_start = []
    col_cut_indices_end = []
    Logger.debug("col cut indices: %s", col_cut_indices)
    for block_idx, col_cut_index in enumerate(col_cut_indices):
        Logger.debug(
            "Block %s, col indices start and end %s %s",
            block_idx,
            prev_col_cut,
            col_cut_index,
        )
        block_cols = matrix[:, prev_col_cut:col_cut_index]
        blocks_row_sum = block_cols.sum(axis=1)
        # get the first and last col with non-zero entries
        non_zero_indices = np.flatnonzero(blocks_row_sum)
        Logger.debug("Block row sum shape %s", blocks_row_sum.shape)

        # Find index of the first non-zero value
        if non_zero_indices.size > 0:
            row_cut_index_start = non_zero_indices[0]
            # Find index of the last non-zero value
            row_cut_index_end = non_zero_indices[-1]
            Logger.info(
                "row cut index start and end %s %s",
                row_cut_index_start,
                row_cut_index_end,
            )
            block = matrix[
                row_cut_index_start : row_cut_index_end + 1, prev_col_cut:col_cut_index
            ]
        else:
            # No non-zero rows in this column block: create an empty (0-row) block
            row_cut_index_start = 0
            row_cut_index_end = -1
            Logger.info(
                "No non-zero rows in block cols %s:%s; creating empty block",
                prev_col_cut,
                col_cut_index,
            )
            block = matrix[0:0, prev_col_cut:col_cut_index]

        blocks.append(block)
        col_cut_indices_start.append(prev_col_cut)
        col_cut_indices_end.append(col_cut_index)
        prev_col_cut = col_cut_index
        # row_cut_indices_start.append(row_cut_index_start)
        # row_cut_indices_end.append(row_cut_index_end + 1)
        Logger.info("block shape %s", block.shape)
    assert sum([block.shape[0] for block in blocks]) == matrix.shape[0]
    return blocks, col_cut_indices_start, col_cut_indices_end


def slice_frame_data_blocks(frame_data, col_cut_indices_start, col_cut_indices_end):
    frame_data_blocks = []
    for start, end in zip(col_cut_indices_start, col_cut_indices_end):
        frame_data_block = frame_data[:, start:end]
        frame_data_blocks.append(frame_data_block)
        # Logger.info(frame_data_block.shape)
    return frame_data_blocks


def process_one_frame(
    ms1scans: pd.DataFrame,
    ms1_frame_idx: int,
    maxquant_result_ref_with_im_index_sortmz: pd.DataFrame,
    mz_bin_digits: int = 3,
    debug: bool = False,
):
    """Process one frame data without IMS dimension with sparse encoding and peak selection."""
    Logger.debug("Start data preparation.")
    # prepare data
    frame_data = ms1scans.loc[ms1scans["MS1_frame_idx"] == ms1_frame_idx]
    Logger.debug("Frame data shape: %s", frame_data.shape[0])
    peaks_df = pd.DataFrame()
    pept_act_coo = {
        "coord_frame_indices": [],
        "coord_pept_indices": [],
        "data": [],
    }
    if frame_data.shape[0] > 0:
        scan_time = np.round(ms1scans.loc[ms1_frame_idx, "Time_minute"], decimals=4)  # type: ignore[arg-type]
        Logger.info("Scan time: %s", scan_time)
        candidate_precursor_by_rt = maxquant_result_ref_with_im_index_sortmz.loc[
            (maxquant_result_ref_with_im_index_sortmz["RT_search_left"] <= scan_time)
            & (maxquant_result_ref_with_im_index_sortmz["RT_search_right"] >= scan_time)
        ]
        Logger.info(
            "Number of candidates by RT in frame %s: %s",
            ms1_frame_idx,
            candidate_precursor_by_rt.shape[0],
        )
        if candidate_precursor_by_rt.shape[0] > 0:
            candidate_precursor_by_rt.sort_values(
                "mz_rank", ascending=True, inplace=True
            )
            all_frame_pept_idx = candidate_precursor_by_rt.mz_rank.values
            (
                frame_array,
                candidate_array,
            ) = _prepare_sparse_matrices(
                candidate_precursor_by_rt=candidate_precursor_by_rt,
                frame_data=frame_data,
                all_id=all_frame_pept_idx,
                mz_bin_digits=mz_bin_digits,
                use_ims=False,
            )

            Logger.debug("Start optimization with sparse encoding.")

            im_pept_act = sparse_encode_divide_and_conquer_with_residual_stats(
                frame_array, candidate_array
            )  # TODO: not yet hanlding returning residues

            Logger.debug("Start peak selection.")

            nonzero_indices = np.nonzero(im_pept_act)
            pept_act_coo["data"] = im_pept_act[nonzero_indices].tolist()  # type: ignore[assignment]
            pept_act_coo["coord_frame_indices"] = np.repeat(
                ms1_frame_idx, len(pept_act_coo["data"])
            ).tolist()
            pept_act_coo["coord_pept_indices"] = all_frame_pept_idx[
                nonzero_indices[1]
            ].tolist()
        else:
            Logger.info("No candidate precursor by RT from frame %s", ms1_frame_idx)
    else:
        Logger.info("No data for frame index %s", ms1_frame_idx)
    if debug:
        return (
            peaks_df,
            pept_act_coo,
            frame_array,
            candidate_array,
            im_pept_act,
            candidate_precursor_by_rt,
            all_frame_pept_idx,
        )
    else:
        return (
            peaks_df,
            pept_act_coo,
            # candidate_array,
            # frame_array,
            # im_pept_act,
        )  # TODO: remove candidate array


def process_one_frame_ims(
    data: pd.DataFrame,
    ms1scans: pd.DataFrame,
    ms1_frame_idx: int,
    maxquant_result_ref_with_im_index_sortmz: pd.DataFrame,
    mobility_values: pd.DataFrame,
    mz_bin_digits: int = 3,
    extract_im_peak: bool = False,
    debug: bool = False,
    return_res_coo_dict: bool = False,
    **im_peak_selection_kwargs,
):
    """Process one frame data with IMS dimension with sparse encoding and peak selection."""
    Logger.debug("Start data preparation.")
    # prepare data
    frame_data = data[
        {
            "frame_indices": [ms1scans.loc[ms1_frame_idx, "Id"]],
            "precursor_indices": [0],
        }
    ]
    Logger.debug("Finished data indexing, frame data shape: %s", frame_data.shape[0])
    peaks_df = pd.DataFrame()
    im_pept_act_coo_dict = {
        "coord_frame_indices": [],
        "coord_im_indices": [],
        "coord_pept_indices": [],
        "data": [],
    }
    im_pept_act_coo_dict = {}
    im_pept_res_coo_dict = {}

    if frame_data.shape[0] > 0:
        scan_time = np.round(ms1scans.loc[ms1_frame_idx, "Time_minute"], decimals=4)  # type: ignore[arg-type]
        candidate_precursor_by_rt = maxquant_result_ref_with_im_index_sortmz.loc[
            (maxquant_result_ref_with_im_index_sortmz["RT_search_left"] <= scan_time)
            & (maxquant_result_ref_with_im_index_sortmz["RT_search_right"] >= scan_time)
        ]
        Logger.info(
            "Finished filtering candidate precursors. Number of candidates by RT in frame %s: %s",
            ms1_frame_idx,
            candidate_precursor_by_rt.shape[0],
        )
        if candidate_precursor_by_rt.shape[0] > 0:
            candidate_precursor_by_rt.sort_values(
                "mz_rank", ascending=True, inplace=True
            )
            all_frame_pept_idx = candidate_precursor_by_rt.mz_rank.values
            (
                frame_array,
                candidate_array,
            ) = _prepare_sparse_matrices(
                candidate_precursor_by_rt=candidate_precursor_by_rt,
                frame_data=frame_data,
                mobility_values=mobility_values,
                all_id=all_frame_pept_idx,
                mz_bin_digits=mz_bin_digits,
                use_ims=True,
            )

            Logger.debug(
                "Finished preparing sparse matrix. Start optimization with sparse encoding."
            )

            deconv_results = sparse_encode_divide_and_conquer_with_residual_stats(
                frame_array, candidate_array, return_act_res=return_res_stats
            )
            Logger.debug("Finished sparse encoding.")
            if return_res_stats:
                im_pept_act, im_pept_res = deconv_results  # type: ignore[assignment]
                im_pept_res_coo_dict["data"] = im_pept_res[np.nonzero(im_pept_res)]  # type: ignore[assignment]
                im_pept_res_coo_dict["coord_frame_indices"] = np.repeat(
                    ms1_frame_idx, len(im_pept_res_coo_dict["data"])
                )
                im_pept_res_coo_dict["coord_im_indices"] = np.nonzero(im_pept_res)[0]  # type: ignore[assignment]
                im_pept_res_coo_dict["coord_pept_indices"] = all_frame_pept_idx[
                    np.nonzero(im_pept_res)[1]  # type: ignore[assignment]
                ]
                Logger.debug("Finished preparing residual COO dict.")
            else:
                im_pept_act = deconv_results  # type: ignore[assignment]
            assert isinstance(im_pept_act, np.ndarray)
            if extract_im_peak:
                peaks_df = _select_im_peak_from_frame_act(
                    im_pept_act=im_pept_act,
                    all_pept_mzrank=all_frame_pept_idx,
                    maxquant_result_dict_with_im_index=candidate_precursor_by_rt,
                    # delta_mobility_thres=delta_mobility_thres,
                    **im_peak_selection_kwargs,
                )
                peaks_df["frame_indices"] = ms1_frame_idx

            nonzero_indices = np.nonzero(im_pept_act)
            im_pept_act_coo_dict["data"] = im_pept_act[nonzero_indices].tolist()  # type: ignore[assignment]
            im_pept_act_coo_dict["coord_frame_indices"] = np.repeat(
                ms1_frame_idx, len(im_pept_act_coo_dict["data"])
            ).tolist()
            im_pept_act_coo_dict["coord_im_indices"] = nonzero_indices[0].tolist()
            im_pept_act_coo_dict["coord_pept_indices"] = all_frame_pept_idx[
                nonzero_indices[1]
            ].tolist()
            Logger.debug("Finished preparing activation COO dict.")
        else:
            Logger.info("No candidate precursor by RT from frame %s", ms1_frame_idx)
    else:
        Logger.info("No data for frame index %s", ms1_frame_idx)
    if debug:
        return (
            peaks_df,
            im_pept_act_coo_dict,
            frame_array,
            candidate_array,
            im_pept_act,
            candidate_precursor_by_rt,
            all_frame_pept_idx,
        )
    else:
        if return_res_stats:
            return (peaks_df, im_pept_act_coo_dict, im_pept_res_coo_dict)
        else:
            return (
                peaks_df,
                im_pept_act_coo_dict,
            )


def make_coo_from_dict(data_dict, shape: tuple, cutoff: List[int]):
    Logger.info("Shape of COO matrix: %s", shape)
    if len(cutoff) > 1:
        coo_list = []
        # n_pept_in_blocks = shape[2] // n_blocks_by_pept
        # cutoff = [(n_pept_in_blocks * (i + 1)) for i in range(n_blocks_by_pept - 1)]
        # cutoff.append(shape[2] + 1)
        Logger.debug("cutoff list %s", cutoff)
        prev_cutoff = 0
        for cutoff_i in cutoff:
            block_idx = np.where(
                (prev_cutoff <= np.array(data_dict["coord_pept_indices"]))
                & (np.array(data_dict["coord_pept_indices"]) < cutoff_i)
            )[0].astype(int)
            Logger.debug("block index %s", block_idx)
            coo_list.append(
                sparse.COO(
                    coords=[
                        list(itemgetter(*block_idx)(data_dict["coord_frame_indices"])),
                        list(itemgetter(*block_idx)(data_dict["coord_im_indices"])),
                        list(itemgetter(*block_idx)(data_dict["coord_pept_indices"])),
                    ],
                    data=list(itemgetter(*block_idx)(data_dict["data"])),
                    shape=shape,
                )
            )
            prev_cutoff = cutoff_i
        return coo_list
    else:
        return sparse.COO(
            coords=[
                data_dict["coord_frame_indices"],
                data_dict["coord_im_indices"],
                data_dict["coord_pept_indices"],
            ],
            data=data_dict["data"],
            shape=shape,
        )


def make_coo_from_dict_no_ims(data_dict, shape: tuple, cutoff: List[int]):
    Logger.info("Shape of COO matrix: %s", shape)
    if len(cutoff) > 1:
        coo_list = []
        # n_pept_in_blocks = shape[2] // n_blocks_by_pept
        # cutoff = [(n_pept_in_blocks * (i + 1)) for i in range(n_blocks_by_pept - 1)]
        # cutoff.append(shape[2] + 1)
        Logger.debug("cutoff list %s", cutoff)
        prev_cutoff = 0
        for cutoff_i in cutoff:
            block_idx = np.where(
                (prev_cutoff <= np.array(data_dict["coord_pept_indices"]))
                & (np.array(data_dict["coord_pept_indices"]) < cutoff_i)
            )[0].astype(int)
            Logger.info("block index %s", block_idx)
            if len(block_idx) > 0:
                coo_list.append(
                    sparse.COO(
                        coords=[
                            list(
                                itemgetter(*block_idx)(data_dict["coord_frame_indices"])
                            ),
                            # list(itemgetter(*block_idx)(data_dict["coord_im_indices"])),
                            list(
                                itemgetter(*block_idx)(data_dict["coord_pept_indices"])
                            ),
                        ],
                        data=list(itemgetter(*block_idx)(data_dict["data"])),
                        shape=shape,
                    )
                )
            prev_cutoff = cutoff_i
        return coo_list
    else:
        return sparse.COO(
            coords=[
                data_dict["coord_frame_indices"],
                # data_dict["coord_im_indices"],
                data_dict["coord_pept_indices"],
            ],
            data=data_dict["data"],
            shape=shape,
        )


def process_batch_frame(
    data: pd.DataFrame,
    ms1scans: pd.DataFrame,
    batch_scan_idx: list,
    maxquant_result_ref_with_im_index_sortmz: pd.DataFrame,
    mobility_values: Optional[pd.DataFrame],
    cutoff: List[int],
    delta_mobility_thres: int = 100,
    batch_num: int = 0,
    save_dir: str = "",
    return_im_pept_act: bool = False,
    extract_im_peak: bool = False,
    use_ims: bool = True,
    return_res_coo_dict: bool = False,
    **im_peak_selection_kwargs,
):
    batch_peaks_df = []
    if use_ims:
        batch_im_rt_pept_act_coo_dict = {
            "coord_frame_indices": [],
            "coord_im_indices": [],
            "coord_pept_indices": [],
            "data": [],
        }
    else:
        batch_rt_pept_act_coo_dict = {
            "coord_frame_indices": [],
            "coord_pept_indices": [],
            "data": [],
        }
    for scan_idx in batch_scan_idx:
        Logger.debug("Start processing frame index %s", scan_idx)
        if use_ims:
            assert mobility_values is not None
            peaks_df, frame_im_pept_act_coo = process_one_frame_ims(  # type: ignore[assignment]
                data=data,
                ms1scans=ms1scans,
                ms1_frame_idx=scan_idx,
                maxquant_result_ref_with_im_index_sortmz=maxquant_result_ref_with_im_index_sortmz,
                mobility_values=mobility_values,
                delta_mobility_thres=delta_mobility_thres,
                extract_im_peak=extract_im_peak,
                debug=False,
                **kwargs,
            )
            if extract_im_peak:
                batch_peaks_df.append(peaks_df)
            if return_im_pept_act:
                for key in batch_im_rt_pept_act_coo_dict.keys():
                    batch_im_rt_pept_act_coo_dict[key].extend(
                        frame_im_pept_act_coo[key]
                    )
        else:
            peaks_df, frame_im_pept_act_coo = process_one_frame(  # type: ignore[assignment]
                ms1scans=ms1scans,
                ms1_frame_idx=scan_idx,
                maxquant_result_ref_with_im_index_sortmz=maxquant_result_ref_with_im_index_sortmz,
                return_pept_act=return_im_pept_act,
                debug=False,
                **kwargs,
            )
            if return_im_pept_act:
                for key in batch_rt_pept_act_coo_dict.keys():
                    batch_rt_pept_act_coo_dict[key].extend(frame_im_pept_act_coo[key])

    if use_ims and extract_im_peak:
        batch_peaks_df = pd.concat(batch_peaks_df).reset_index(drop=True)
        batch_peaks_df.to_csv(
            os.path.join(save_dir, f"batch_peaks_df_{batch_num}.csv"), index=False
        )
    if use_ims:
        assert mobility_values is not None
        batch_im_rt_pept_act_coo = make_coo_from_dict(
            batch_im_rt_pept_act_coo_dict,
            shape=(
                len(ms1scans.index.values)
                + 1,  # this index is rank, starting from 1, add 1 for the last frame
                len(mobility_values),
                len(maxquant_result_ref_with_im_index_sortmz.mz_rank)
                + 1,  # this index is rank, starting from 1, add 1 for the last frame
            ),
            cutoff=cutoff,
        )
        if return_res_coo_dict:
            batch_im_rt_pept_res_coo = make_coo_from_dict(
                batch_im_rt_pept_res_coo_dict,
                shape=(
                    len(ms1scans.index.values)
                    + 1,  # this index is rank, starting from 1, add 1 for the last frame
                    len(mobility_values),
                    len(maxquant_result_ref_with_im_index_sortmz.mz_rank)
                    + 1,  # this index is rank, starting from 1, add 1 for the last frame
                ),
                cutoff=cutoff,
            )
    else:
        batch_im_rt_pept_act_coo = make_coo_from_dict_no_ims(
            batch_rt_pept_act_coo_dict,
            shape=(
                len(ms1scans.index.values)
                + 1,  # this index is rank, starting from 1, add 1 for the last frame
                len(maxquant_result_ref_with_im_index_sortmz.mz_rank)
                + 1,  # this index is rank, starting from 1, add 1 for the last frame
            ),
            cutoff=cutoff,
        )

    if isinstance(batch_im_rt_pept_act_coo, list):
        for pept_batch_idx, pept_batch_dict in enumerate(batch_im_rt_pept_act_coo):
            sparse.save_npz(
                os.path.join(
                    save_dir,
                    f"im_rt_pept_act_coo_batch{batch_num}_peptbatch{pept_batch_idx}.npz",
                ),
                pept_batch_dict,
            )
            Logger.info(
                "Size of COO matrix in batch %s, peptide batch %s: %s Mb",
                batch_num,
                pept_batch_idx,
                pept_batch_dict.nbytes / 1e6,
            )
    else:
        sparse.save_npz(
            os.path.join(
                save_dir, f"im_rt_pept_act_coo_batch{batch_num}_peptbatch0.npz"
            ),
            batch_im_rt_pept_act_coo,
        )
        Logger.info(
            "Size of COO matrix in batch %s: %s Mb",
            batch_num,
            batch_im_rt_pept_act_coo.nbytes / 1e6,
        )
            )


def _prepare_sparse_matrices(
    candidate_precursor_by_rt,
    frame_data,
    all_id,
    mz_bin_digits: int = 3,
    use_ims: bool = True,
    mobility_values: Optional[pd.DataFrame] = None,
):
    # --- Candidate arrays ---
    candidate_id = np.repeat(
        candidate_precursor_by_rt.mz_rank.values,
        candidate_precursor_by_rt.mz_length.values,
    )
    candidate_mz = np.round(
        np.concatenate(candidate_precursor_by_rt.IsoMZ.values),
        decimals=mz_bin_digits,
    )
    candidate_abundance = np.concatenate(candidate_precursor_by_rt.IsoAbundance.values)
    Logger.debug(
        "Candidate array shapes for id, mz and abundance: %s %s %s",
        candidate_id.shape,
        candidate_mz.shape,
        candidate_abundance.shape,
    )
    candidate_id_index = np.searchsorted(all_id, candidate_id)

    if use_ims:
        frame_mz = np.round(frame_data["mz_values"], decimals=mz_bin_digits)
    else:
        frame_mz = np.round(frame_data["mzarray"].values[0], decimals=mz_bin_digits)
    all_mz = np.union1d(frame_mz, candidate_mz)
    Logger.debug(
        "Number of mz values in candidate, frame and joint:%s, %s, %s",
        len(set(candidate_mz)),
        len(set(frame_mz)),
        len(all_mz),
    )
    candidate_mz_index = np.searchsorted(all_mz, candidate_mz)
    frame_mz_index = np.searchsorted(all_mz, frame_mz)

    # prepare arrays from sparse matrices
    min_mz_index = max(candidate_mz_index.min(), frame_mz_index.min())
    max_mz_index = min(candidate_mz_index.max(), frame_mz_index.max())
    Logger.debug("min and max mz index: %s %s", min_mz_index, max_mz_index)

    # make sure candidate mz index is not out of range of observed mz in frame
    mask = (candidate_mz_index >= min_mz_index) & (candidate_mz_index <= max_mz_index)
    candidate_mz_index_filtered = candidate_mz_index[mask]
    candidate_abundance_filtered = candidate_abundance[mask]
    candidate_id_index_filtered = candidate_id_index[mask]
    Logger.debug(
        "Shape of mask, candidate mz index, abundance and id index: %s, %s, %s, %s, sum of mask %s",
        mask.shape,
        candidate_mz_index_filtered.shape,
        candidate_abundance_filtered.shape,
        candidate_id_index_filtered.shape,
        sum(mask),
    )

    # Now build compact column set + mapping
    unique_idx, mapped_idx = np.unique(candidate_mz_index_filtered, return_inverse=True)

    # Allocate dense matrix directly
    candidate_array = np.zeros(
        (all_id.size, unique_idx.size), dtype=candidate_abundance.dtype
    )

    # Fill with filtered data
    candidate_array[candidate_id_index_filtered, mapped_idx] = (
        candidate_abundance_filtered
    )

    Logger.debug(
        "Number of mz values in filtered candidate index: %s",
        len(candidate_mz_index_filtered),
    )
    if use_ims:
        assert mobility_values is not None
        all_im = np.sort(mobility_values["mobility_values"])
        frame_im_index = np.searchsorted(all_im, frame_data["mobility_values"])

        frame_coo = COO(
            (frame_data["intensity_values"], (frame_im_index, frame_mz_index)),
        )  # TODO: frame array can be improved the same way
    else:
        intarray = frame_data["intarray"].values[0]
        frame_coo = COO(
            (
                intarray,
                (
                    np.zeros(len(intarray)).astype(int),
                    frame_mz_index,
                ),
            ),
            shape=(1, len(all_mz)),
        )

    # only candidate mz is considered
    frame_array = frame_coo.todense()[
        :, np.unique(candidate_mz_index_filtered).tolist()
    ]
    assert (
        frame_array.shape[1] == candidate_array.shape[1]
    ), "m/z dimension of frame array and candidate array mismatch %s, %s" % (
        frame_array.shape[1],
        candidate_array.shape[1],
    )
    return (
        frame_array,
        candidate_array,
    )  # , pd.DataFrame({"mz_index":frame_mz_index, "mz_value":frame_mz}), pd.DataFrame({"mz_index":candidate_mz_index, "mz_value":candidate_mz}) #TODO: remove extra returns


def _select_im_peak_from_frame_act(
    im_pept_act: np.ndarray,
    all_pept_mzrank: np.ndarray,
    maxquant_result_dict_with_im_index: pd.DataFrame,
    delta_mobility_thres: int = 100,
    **kwargs,
):
    peak_properties_list = _select_peaks_from_im_pept_act(
        im_pept_act, pept_mzrank=all_pept_mzrank, **kwargs
    )
    if peak_properties_list is None or all(v is None for v in peak_properties_list):
        peaks_df = pd.DataFrame()
        Logger.info("No peaks extracted.")
    else:
        peaks_df = pd.concat(peak_properties_list).reset_index(drop=True)
        Logger.info(
            "Number of peaks before delta mobility filter: %s", peaks_df.shape[0]
        )
        peaks_df = pd.merge(
            left=peaks_df,
            right=maxquant_result_dict_with_im_index[
                ["mobility_values_index", "mz_rank"]
            ],
            left_on="pept_mzrank",
            right_on="mz_rank",
            how="left",
        )
        peaks_df["delta_mobility"] = abs(
            peaks_df["peak"] - peaks_df["mobility_values_index"]
        )
        peaks_df = peaks_df.loc[peaks_df["delta_mobility"] <= delta_mobility_thres]
        if peaks_df.shape[0] > 0:
            Logger.info(
                "Number of peaks and peptide after delta mobility filter: %s %s",
                peaks_df.shape[0],
                peaks_df["pept_mzrank"].nunique(),
            )
            peaks_df = peaks_df.loc[
                peaks_df.groupby("pept_mzrank")["delta_mobility"].idxmin()
            ]
            peaks_df["peak_sum"] = peaks_df.apply(
                lambda x: _sum_im_act_per_pept(x, im_pept_act), axis=1
            )
        else:
            Logger.info("No peaks found after delta mobility filter.")
            peaks_df = pd.DataFrame()

    return peaks_df


def _sum_im_act_per_pept(peak_row, im_pept_act):
    pept_id_index = int(peak_row["pept_mzrank_index"])
    peak_start = int(peak_row["left_ips"])
    peak_end = int(peak_row["right_ips"])
    peak_sum = im_pept_act[peak_start : peak_end + 1, pept_id_index].sum()
    # Logger.debug("shape of peak_sum: %s", peak_sum.shape)
    return peak_sum


def _select_peaks_from_im_pept_act(
    im_pept_act: np.ndarray, pept_mzrank: np.ndarray, **kwargs
):
    # filtered only the columns with nonzero value > = 3
    pept_nonzero_count = np.count_nonzero(im_pept_act, axis=0)
    pept_valid_idx = np.where(pept_nonzero_count >= 3)[0]
    if pept_valid_idx.size > 0:
        Logger.info(
            "Number of peptides with nonzero mobility value >= 3: %s",
            len(pept_valid_idx),
        )
        peak_properties = [
            _extract_peaks_in_im(
                im_pept_act_array=im_pept_act[:, idx],
                index=idx,
                pept_mzrank=int(pept_mzrank[idx]),
                **kwargs,
            )
            for idx in pept_valid_idx
        ]
        if all(v is None for v in peak_properties):
            Logger.info("No peaks extracted.")
            peak_properties = None
    else:
        Logger.info("No peptides with nonzero mobility value >= 3.")
        peak_properties = None
    return peak_properties


def _extract_peaks_in_im(
    im_pept_act_array, index: int, pept_mzrank: int, height=0.1, width=4, rel_height=1
):
    # Logger.debug("Peak extration width %s", width)
    peaks, peak_properties = find_peaks(
        im_pept_act_array, height=height, width=width, rel_height=rel_height
    )
    if peaks.size > 0:
        peak_properties["pept_mzrank_index"] = np.repeat(index, len(peaks))
        peak_properties["pept_mzrank"] = np.repeat(pept_mzrank, len(peaks))
        peak_properties["peak"] = peaks
        peak_properties = pd.DataFrame(peak_properties)
        # Logger.debug(
        #     "Number of peaks extracted from pept_id %s: %s", pept_id, len(peaks)
        # )
    else:
        # Logger.debug("No peak extracted from pept_id %s in this frame.", pept_id)
        peak_properties = None
    return peak_properties


def parallel(func=None, args=(), merge_func=lambda x: x, parallelism=cpu_count()):
    def decorator(func: Callable):
        def inner(*args, **kwargs):
            results = Parallel(n_jobs=parallelism)(
                delayed(func)(*args, **kwargs) for i in range(parallelism)
            )
            return merge_func(results)

        return inner

    if func is None:
        # decorator was used like @parallel(...)
        return decorator
    else:
        # decorator was used like @parallel, without parens
        return decorator(func)


def generate_id_partitions(
    id_array,
    n_batch,
    how: Literal["round_robin", "block"] = "block",
    n_edge_counts: int = 50,
):
    id_partitions = [[] for _ in range(n_batch)]
    if how == "round_robin":
        Logger.info("Generate id partitions by round robin.")
        for i in range(n_batch):
            batch_idx = np.arange(i, len(id_array), n_batch)
            id_partitions[i] = id_array[batch_idx]
    elif how == "block":
        Logger.info("Generate id partitions by block.")
        block_size = (len(id_array) - 2 * n_edge_counts) // n_batch
        for i in range(n_batch):
            if i == 0:
                mark = block_size + n_edge_counts
                id_partitions[i] = id_array[:mark]
            elif i == n_batch - 1:
                id_partitions[i] = id_array[mark:]
            else:
                id_partitions[i] = id_array[mark : mark + block_size]
                mark = block_size * (i + 1) + n_edge_counts
    return id_partitions


def process_frames_parallel(
    n_jobs: int,
    batch_scan_indices: list,
    **kwargs,
):
    """
    Process frames in parallel by splitting frame indices into batches.
    Parameters
    ----------
    n_jobs : int
        Number of parallel jobs.
    batch_scan_indices : list
        List of list of frame indices for each batch.
    **kwargs : dict
        Additional arguments for `process_batch_frame` function.
    """
    list_batch_im_pept_act_coo_dict = Parallel(n_jobs=n_jobs)(
        delayed(process_batch_frame)(
            batch_scan_idx=batch,
            batch_num=batch[0],
            **kwargs,
        )
        for batch in batch_scan_indices
    )


def get_apex_from_im_rt_pept_act_coo(im_rt_pept_act_coo: sparse.COO):
    """
    Get RT and IM apex from im_rt_pept_act_coo sparse matrix.
    Parameters
    ----------
    im_rt_pept_act_coo : sparse.COO
        Sparse COO matrix with shape (n_frames, n_ims, n_peptides).
    Returns
    -------
    apex_df : pd.DataFrame
        DataFrame with columns 'peptide_idx', 'rt_apex', and 'im_apex'.
    """
    N = im_rt_pept_act_coo.shape[2]  # number of peptides
    coords = im_rt_pept_act_coo.coords
    data = im_rt_pept_act_coo.data

    # Step 1: sort by peptide index
    sorted_idx = np.argsort(coords[2])
    coords_sorted = coords[:, sorted_idx]
    data_sorted = data[sorted_idx]

    # Step 2: get unique peptide indices and boundaries
    unique_pept, start_idx, counts = np.unique(
        coords_sorted[2], return_index=True, return_counts=True
    )

    # Step 3: initialize arrays
    rt_apex = np.full(N, -1, dtype=int)
    im_apex = np.full(N, -1, dtype=int)

    # Step 4: loop only over peptides that have activations
    for idx, pept_idx in enumerate(unique_pept):
        start = start_idx[idx]
        end = start + counts[idx]

        rt_sum = np.bincount(
            coords_sorted[0, start:end], weights=data_sorted[start:end]
        )
        im_sum = np.bincount(
            coords_sorted[1, start:end], weights=data_sorted[start:end]
        )

        rt_apex[pept_idx] = np.argmax(rt_sum)
        im_apex[pept_idx] = np.argmax(im_sum)

    # Step 5: DataFrame
    apex_df = pd.DataFrame(
        {"peptide_idx": np.arange(N), "rt_apex": rt_apex, "im_apex": im_apex}
    )
    return apex_df


# def process_scans_parallel(
#     n_jobs: int,
#     ms1scans: pd.DataFrame,
#     maxquant_ref: pd.DataFrame,
#     abundance_missing_threshold: float = 0.4,
#     alpha_criteria: _alpha_criteria = "convergence",
#     alphas: Union[List, np.ndarray] = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
#     loss: _loss = "lasso",
#     opt_algo: _algo = "lasso_cd",
#     metric: _alpha_opt_metric = "cos_dist",
#     preprocessing_method: _pp_method = "raw",
#     corr_thres: float = 0.95,
#     max_iter: int = 1000,
#     return_precursor_scan_cos_dist: bool = False,
# ):
#     scan_result_list = Parallel(n_jobs=n_jobs)(
#         delayed(process_one_scan)(
#             scan_idx=scan_idx,
#             OneScan=OneScan,
#             Maxquant_result=maxquant_ref,
#             AbundanceMissingThres=abundance_missing_threshold,
#             alpha_criteria=alpha_criteria,
#             alphas=alphas,
#             metric=metric,
#             loss=loss,
#             opt_algo=opt_algo,
#             preprocessing_method=preprocessing_method,
#             corr_thres=corr_thres,
#             max_iter=max_iter,
#             return_interim_results=False,
#             return_precursor_scan_cos_dist=return_precursor_scan_cos_dist,
#         )
#         for scan_idx, OneScan in ms1scans.iterrows()
#     )
#     scan_result_dict = dict(pair for d in scan_result_list for pair in d.items())
#     return scan_result_dict
