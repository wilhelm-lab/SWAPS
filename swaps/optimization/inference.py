import logging
import os
from multiprocessing import cpu_count
from typing import List, Literal, Optional
import itertools
from operator import itemgetter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sparse import COO
from scipy.signal import find_peaks
from sklearn.decomposition import sparse_encode
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm

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
    # Logger.debug("Frame activation shape: %s", frame_act.shape)
    reconstruction = np.zeros_like(frame_array, dtype=np.float32)

    # Precompute offsets for placing peptide activations
    block_sizes = [cb.shape[0] for cb in candidate_coo_blocks]
    col_offsets = np.cumsum([0] + block_sizes[:-1])

    # --- Encoding loop ---
    for idx, candidate_block in enumerate(candidate_coo_blocks):
        start, end = col_start[idx], col_end[idx]  # m/z row slice for this block
        frame_block = frame_array[:, start:end]  # (m_z_block, im)

        # sparse_encode expects: frame_block (m_z_block, im), candidate_block (m_z_block, pept_block)
        im_pept_act = sparse_encode(
            frame_block,
            candidate_block,
            algorithm="threshold",
            alpha=0,
            positive=True,
        )  # returns (im, pept_block)

        # Place activations in correct columns
        col_offset = col_offsets[idx]

        try:
            frame_act[
                :, col_offset : col_offset + candidate_block.shape[0]
            ] += im_pept_act
        except:
            Logger.info(
                "Processing block %s: m/z rows %s-%s, col offset %s",
                idx,
                start,
                end,
                col_offset,
            )
            Logger.info("start and end of frame block %s %s", start, end)
            Logger.info(
                "Frame block shape: %s, candidate block shape: %s",
                frame_block.shape,
                candidate_block.shape,
            )
            Logger.info("im_pept_act shape: %s", im_pept_act.shape)

        if return_act_res:
            # Add reconstruction contribution for residual computation
            reconstruction += im_pept_act @ candidate_block
    # Logger.debug("frame_act non-zero count: %s", np.count_nonzero(frame_act))
    if return_act_res:
        # --- Measurement-space residual ---
        residual_im_mz = frame_array - reconstruction  # (im, m_z)

        # # --- Candidate-space residual stats ---

        # Mean residual per peptide
        frame_res = (
            candidate_array > 0
        ) @ residual_im_mz.T  # (pept, im) same dim as frame_act.T
        Logger.info(
            "frame_res shape: %s; frame_act shape: %s", frame_res.shape, frame_act.shape
        )
        return frame_act, frame_res.T  # transpose to (im, pept)

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


def process_one_frame(
    ms1scans: pd.DataFrame,
    ms1_frame_idx: int,
    maxquant_result_ref_with_im_index_sortmz: pd.DataFrame,
    ppm_tol: int = 20,
    bin_frame_mz: bool = True,
    bin_width: float = 0.01,
    debug: bool = False,
    return_res_coo_dict: bool = False,
    **process_frame_kwargs,
):
    """Process one frame data without IMS dimension with sparse encoding."""
    # prepare data
    frame_data = ms1scans.loc[ms1scans["MS1_frame_idx"] == ms1_frame_idx]
    pept_act_coo = {
        "coord_frame_indices": [],
        "coord_pept_indices": [],
        "data": [],
    }
    pept_res_coo = {
        "coord_frame_indices": [],
        "coord_pept_indices": [],
        "data": [],
    }
    if frame_data.shape[0] > 0:
        scan_time = np.round(ms1scans.loc[ms1_frame_idx, "Time_minute"], decimals=4)  # type: ignore[arg-type]
        candidate_precursor_by_rt = maxquant_result_ref_with_im_index_sortmz.loc[
            (maxquant_result_ref_with_im_index_sortmz["RT_search_left"] <= scan_time)
            & (maxquant_result_ref_with_im_index_sortmz["RT_search_right"] >= scan_time)
        ]
        if candidate_precursor_by_rt.shape[0] > 0:
            candidate_precursor_by_rt = candidate_precursor_by_rt.sort_values(
                "mz_rank", ascending=True
            )
            all_frame_pept_idx = candidate_precursor_by_rt.mz_rank.values
            (
                frame_array,
                candidate_array,
            ) = _prepare_sparse_matrices(
                candidate_precursor_by_rt=candidate_precursor_by_rt,
                frame_data=frame_data,
                all_id=all_frame_pept_idx,
                bin_frame_mz=bin_frame_mz,
                ppm_tol=ppm_tol,
                bin_width=bin_width,
                use_ims=False,
            )
            if frame_array is None:
                pept_act = None
                Logger.warning("No sparse matrices return for frame %s", ms1_frame_idx)
            else:
                deconv_results = sparse_encode_divide_and_conquer_with_residual_stats(
                    frame_array, candidate_array, return_act_res=return_res_coo_dict
                )
                if return_res_coo_dict:
                    pept_act, pept_res = deconv_results  # type: ignore[assignment]
                    nonzero_indices_res = np.nonzero(pept_res)
                    pept_res_coo["data"] = pept_res[nonzero_indices_res].tolist()  # type: ignore[assignment]
                    pept_res_coo["coord_frame_indices"] = np.repeat(
                        ms1_frame_idx, len(pept_res_coo["data"])
                    ).tolist()
                    pept_res_coo["coord_pept_indices"] = all_frame_pept_idx[
                        nonzero_indices_res[1]
                    ].tolist()
                else:
                    pept_act = deconv_results  # type: ignore[assignment]
                assert isinstance(pept_act, np.ndarray)

                nonzero_indices = np.nonzero(pept_act)
                pept_act_coo["data"] = pept_act[nonzero_indices].tolist()  # type: ignore[assignment]
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
            pept_act_coo,
            frame_array,
            candidate_array,
            pept_act,
            candidate_precursor_by_rt,
            all_frame_pept_idx,
        )
    else:
        if return_res_coo_dict:
            return (
                pept_act_coo,
                pept_res_coo,
            )
        else:
            return (pept_act_coo,)


def process_one_frame_ims(
    data: pd.DataFrame,
    ms1scans: pd.DataFrame,
    ms1_frame_idx: int,
    maxquant_result_ref_with_im_index_sortmz: pd.DataFrame,
    mobility_values: pd.DataFrame,
    ppm_tol: int = 20,
    bin_frame_mz: bool = True,
    bin_width: float = 0.01,
    extract_im_peak: bool = False,
    debug: bool = False,
    return_res_coo_dict: bool = False,
    parquet_file: Optional[str] = None,
    writer=None,
    merge_confounders: bool = False,
    # zarr_path: Optional[str] = None,
    # zarr_shape: Optional[tuple] = None,
    # zarr_chunks: tuple = (256, 128, 1),
    **im_peak_selection_kwargs,
):
    """Process one frame data with IMS dimension with sparse encoding and peak selection."""
    # prepare data
    frame_data = data[
        {
            "frame_indices": [ms1scans.loc[ms1_frame_idx, "Id"]],
            "precursor_indices": [0],
        }
    ]
    # Logger.debug("Finished data indexing, frame data shape: %s", frame_data.shape[0])
    peaks_df = pd.DataFrame()
    if writer is None:
        im_pept_act_coo_dict = {
            "coord_frame_indices": [],
            "coord_im_indices": [],
            "coord_pept_indices": [],
            "data": [],
        }
        im_pept_res_coo_dict = {
            "coord_frame_indices": [],
            "coord_im_indices": [],
            "coord_pept_indices": [],
            "data": [],
        }
    # else:
    #     z = zarr.open(
    #         zarr_path,
    #         mode="a",
    #         shape=zarr_shape,  # will be resized when appending data
    #         chunks=zarr_chunks,
    #         dtype=np.float32,
    #         compressor=zarr.Blosc(cname="zstd", clevel=5, shuffle=2),
    #     )
    if frame_data.shape[0] > 0:
        scan_time = np.round(ms1scans.loc[ms1_frame_idx, "Time_minute"], decimals=4)  # type: ignore[arg-type]
        candidate_precursor_by_rt = maxquant_result_ref_with_im_index_sortmz.loc[
            (maxquant_result_ref_with_im_index_sortmz["RT_search_left"] <= scan_time)
            & (maxquant_result_ref_with_im_index_sortmz["RT_search_right"] >= scan_time)
        ].copy()
        # Logger.debug(
        #     "Finished filtering candidate precursors. Number of candidates by RT in frame %s: %s",
        #     ms1_frame_idx,
        #     candidate_precursor_by_rt.shape[0],
        # )
        if candidate_precursor_by_rt.shape[0] > 0:
            candidate_precursor_by_rt = candidate_precursor_by_rt.sort_values(
                "mz_rank", ascending=True
            )
            if merge_confounders:
                candidate_precursor_by_rt = collapse_candidates_by_confounder_group(
                    candidate_precursor_by_rt
                )
                candidate_precursor_by_rt = candidate_precursor_by_rt.sort_values(
                    "mz_rank", ascending=True
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
                bin_frame_mz=bin_frame_mz,
                ppm_tol=ppm_tol,
                bin_width=bin_width,
                use_ims=True,
            )
            if frame_array is None:
                im_pept_act = None
                Logger.warning("No sparse matrices return for frame %s", ms1_frame_idx)
            else:
                deconv_results = sparse_encode_divide_and_conquer_with_residual_stats(
                    frame_array, candidate_array, return_act_res=return_res_coo_dict
                )
                if return_res_coo_dict:
                    im_pept_act, im_pept_res = deconv_results  # type: ignore[assignment]
                    nonzero_indices_res = np.nonzero(im_pept_res)
                    im_pept_res_coo_dict["data"] = im_pept_res[nonzero_indices_res].tolist()  # type: ignore[assignment]
                    im_pept_res_coo_dict["coord_frame_indices"] = np.repeat(
                        ms1_frame_idx, len(im_pept_res_coo_dict["data"])
                    ).tolist()
                    im_pept_res_coo_dict["coord_im_indices"] = nonzero_indices_res[0].tolist()  # type: ignore[assignment]
                    im_pept_res_coo_dict["coord_pept_indices"] = all_frame_pept_idx[
                        nonzero_indices_res[1]
                    ].tolist()
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

                if writer is not None:
                    writer.write_frame(ms1_frame_idx, all_frame_pept_idx, im_pept_act)
                else:
                    nonzero_indices = np.nonzero(im_pept_act)
                    im_pept_act_coo_dict["data"] = im_pept_act[nonzero_indices].tolist()  # type: ignore[assignment]
                    im_pept_act_coo_dict["coord_frame_indices"] = np.repeat(
                        ms1_frame_idx, len(im_pept_act_coo_dict["data"])
                    ).tolist()
                    im_pept_act_coo_dict["coord_im_indices"] = nonzero_indices[
                        0
                    ].tolist()
                    im_pept_act_coo_dict["coord_pept_indices"] = all_frame_pept_idx[
                        nonzero_indices[1]
                    ].tolist()
        else:
            Logger.info("No candidate precursor by RT from frame %s", ms1_frame_idx)
    else:
        Logger.info("No data for frame index %s", ms1_frame_idx)
    if writer is not None:
        return None
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
        if return_res_coo_dict:
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
        # Logger.debug("cutoff list %s", cutoff)
        prev_cutoff = 0
        for cutoff_i in cutoff:
            block_idx = np.where(
                (prev_cutoff <= np.array(data_dict["coord_pept_indices"]))
                & (np.array(data_dict["coord_pept_indices"]) < cutoff_i)
            )[0].astype(int)
            # Logger.debug("block index %s", block_idx)
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
        # Logger.debug("cutoff list %s", cutoff)
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
    extract_im_peak: bool = False,
    use_ims: bool = True,
    return_res_coo_dict: bool = False,
    max_mz_rank: int = 0,
    **process_frame_kwargs,
):
    batch_peaks_df = []
    if max_mz_rank == 0:
        max_mz_rank = maxquant_result_ref_with_im_index_sortmz.mz_rank.max()
    if use_ims:
        batch_im_rt_pept_act_coo_dict = {
            "coord_frame_indices": [],
            "coord_im_indices": [],
            "coord_pept_indices": [],
            "data": [],
        }
        batch_im_rt_pept_res_coo_dict = {
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
        # Logger.debug("Start processing frame index %s", scan_idx)
        if use_ims:
            assert mobility_values is not None
            one_frame_results = process_one_frame_ims(  # type: ignore[assignment]
                data=data,
                ms1scans=ms1scans,
                ms1_frame_idx=scan_idx,
                maxquant_result_ref_with_im_index_sortmz=maxquant_result_ref_with_im_index_sortmz,
                mobility_values=mobility_values,
                delta_mobility_thres=delta_mobility_thres,
                extract_im_peak=extract_im_peak,
                debug=False,
                return_res_coo_dict=return_res_coo_dict,
                **process_frame_kwargs,
            )
            if return_res_coo_dict:
                peaks_df, frame_im_pept_act_coo_dict, frame_im_pept_res_coo_dict = one_frame_results  # type: ignore[assignment]
            else:
                peaks_df, frame_im_pept_act_coo_dict = one_frame_results  # type: ignore[assignment]
            if extract_im_peak:
                batch_peaks_df.append(peaks_df)

            for key in batch_im_rt_pept_act_coo_dict.keys():
                batch_im_rt_pept_act_coo_dict[key].extend(
                    frame_im_pept_act_coo_dict[key]
                )
                if return_res_coo_dict:
                    batch_im_rt_pept_res_coo_dict[key].extend(
                        frame_im_pept_res_coo_dict[key]  # type: ignore[assignment]
                    )  # type: ignore[assignment]
        else:
            one_frame_results = process_one_frame(  # type: ignore[assignment]
                ms1scans=ms1scans,
                ms1_frame_idx=scan_idx,
                maxquant_result_ref_with_im_index_sortmz=maxquant_result_ref_with_im_index_sortmz,
                debug=False,
                return_res_coo_dict=return_res_coo_dict,
                **process_frame_kwargs,
            )
            if return_res_coo_dict:
                frame_rt_pept_act_coo, frame_rt_pept_res_coo = one_frame_results  # type: ignore[assignment]
            else:
                frame_rt_pept_act_coo = one_frame_results[0]  # type: ignore[assignment]

            for key in batch_rt_pept_act_coo_dict.keys():
                batch_rt_pept_act_coo_dict[key].extend(frame_rt_pept_act_coo[key])

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
                max_mz_rank
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
                    max_mz_rank
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
                max_mz_rank
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
    if return_res_coo_dict and use_ims:
        if isinstance(batch_im_rt_pept_res_coo, list):
            for pept_batch_idx, pept_batch_dict in enumerate(batch_im_rt_pept_res_coo):
                sparse.save_npz(
                    os.path.join(
                        save_dir,
                        f"im_rt_pept_res_coo_batch{batch_num}_peptbatch{pept_batch_idx}.npz",
                    ),
                    pept_batch_dict,
                )
                Logger.info(
                    "Size of residual COO matrix in batch %s, peptide batch %s: %s Mb",
                    batch_num,
                    pept_batch_idx,
                    pept_batch_dict.nbytes / 1e6,
                )
        else:
            sparse.save_npz(
                os.path.join(
                    save_dir, f"im_rt_pept_res_coo_batch{batch_num}_peptbatch0.npz"
                ),
                batch_im_rt_pept_res_coo,
            )
            Logger.info(
                "Size of residual COO matrix in batch %s: %s Mb",
                batch_num,
                batch_im_rt_pept_res_coo.nbytes / 1e6,
            )


def process_batch_frame_save_parquet(
    data: pd.DataFrame,
    ms1scans: pd.DataFrame,
    batch_scan_idx: list,
    maxquant_result_ref_with_im_index_sortmz: pd.DataFrame,
    parquet_file: str,
    mobility_values: Optional[pd.DataFrame],
    delta_mobility_thres: int = 100,
    batch_num: int = 0,
    save_dir: str = "",
    extract_im_peak: bool = False,
    use_ims: bool = True,
    return_res_coo_dict: bool = False,
    max_mz_rank: int = 0,
    **process_frame_kwargs,
):
    batch_peaks_df = []
    if max_mz_rank == 0:
        max_mz_rank = maxquant_result_ref_with_im_index_sortmz.mz_rank.max()
    if use_ims:
        batch_im_rt_pept_act_coo_dict = {
            "coord_frame_indices": [],
            "coord_im_indices": [],
            "coord_pept_indices": [],
            "data": [],
        }
        batch_im_rt_pept_res_coo_dict = {
            "coord_frame_indices": [],
            "coord_im_indices": [],
            "coord_pept_indices": [],
            "data": [],
        }
    else:  # TODO: fix non ims module
        batch_rt_pept_act_coo_dict = {
            "coord_frame_indices": [],
            "coord_pept_indices": [],
            "data": [],
        }
    path_act = f"{parquet_file}_activation.parquet"
    path_res = f"{parquet_file}_residue.parquet"
    schema_act = init_parquet_scheme(data_col_name="activation")
    schema_res = init_parquet_scheme(data_col_name="residue")
    with pq.ParquetWriter(path_act, schema_act) as writer_act, pq.ParquetWriter(
        path_res, schema_res
    ) as writer_res:
        for scan_idx in tqdm.tqdm(
            batch_scan_idx,
            desc=f"Processing batch starting {batch_scan_idx[0]}",
            total=len(batch_scan_idx),
        ):
            if use_ims:
                assert mobility_values is not None
                one_frame_results = process_one_frame_ims(  # type: ignore[assignment]
                    data=data,
                    ms1scans=ms1scans,
                    ms1_frame_idx=scan_idx,
                    maxquant_result_ref_with_im_index_sortmz=maxquant_result_ref_with_im_index_sortmz,
                    mobility_values=mobility_values,
                    delta_mobility_thres=delta_mobility_thres,
                    extract_im_peak=extract_im_peak,
                    debug=False,
                    return_res_coo_dict=return_res_coo_dict,
                    **process_frame_kwargs,
                )
                if return_res_coo_dict:
                    peaks_df, frame_im_pept_act_coo_dict, frame_im_pept_res_coo_dict = one_frame_results  # type: ignore[assignment]
                else:
                    peaks_df, frame_im_pept_act_coo_dict = one_frame_results  # type: ignore[assignment]
                if extract_im_peak:
                    batch_peaks_df.append(peaks_df)

                for key in batch_im_rt_pept_act_coo_dict.keys():
                    batch_im_rt_pept_act_coo_dict[key].extend(
                        frame_im_pept_act_coo_dict[key]
                    )
                    if return_res_coo_dict:
                        batch_im_rt_pept_res_coo_dict[key].extend(
                            frame_im_pept_res_coo_dict[key]  # type: ignore[assignment]
                        )  # type: ignore[assignment]
            else:
                one_frame_results = process_one_frame(  # type: ignore[assignment]
                    ms1scans=ms1scans,
                    ms1_frame_idx=scan_idx,
                    maxquant_result_ref_with_im_index_sortmz=maxquant_result_ref_with_im_index_sortmz,
                    debug=False,
                    return_res_coo_dict=return_res_coo_dict,
                    **process_frame_kwargs,
                )
                if return_res_coo_dict:
                    frame_rt_pept_act_coo, frame_rt_pept_res_coo = one_frame_results  # type: ignore[assignment]
                else:
                    frame_rt_pept_act_coo = one_frame_results[0]  # type: ignore[assignment]

                for key in batch_rt_pept_act_coo_dict.keys():
                    batch_rt_pept_act_coo_dict[key].extend(frame_rt_pept_act_coo[key])

        if use_ims and extract_im_peak:
            batch_peaks_df = pd.concat(batch_peaks_df).reset_index(drop=True)
            batch_peaks_df.to_csv(
                os.path.join(save_dir, f"batch_peaks_df_{batch_num}.csv"), index=False
            )
        if use_ims:
            assert mobility_values is not None
            table_act = pa.Table.from_pydict(
                {
                    "frame_idx": np.array(
                        batch_im_rt_pept_act_coo_dict["coord_frame_indices"],
                        dtype=np.uint16,
                    ),
                    "im_idx": np.array(
                        batch_im_rt_pept_act_coo_dict["coord_im_indices"],
                        dtype=np.uint16,
                    ),
                    "mz_rank": np.array(
                        batch_im_rt_pept_act_coo_dict["coord_pept_indices"],
                        dtype=np.uint32,
                    ),
                    "activation": np.array(
                        batch_im_rt_pept_act_coo_dict["data"], dtype=np.float32
                    ),
                }
            )

            writer_act.write_table(table_act)
            if return_res_coo_dict:
                table_res = pa.Table.from_pydict(
                    {
                        "frame_idx": np.array(
                            batch_im_rt_pept_res_coo_dict["coord_frame_indices"],
                            dtype=np.uint16,
                        ),
                        "im_idx": np.array(
                            batch_im_rt_pept_res_coo_dict["coord_im_indices"],
                            dtype=np.uint16,
                        ),
                        "mz_rank": np.array(
                            batch_im_rt_pept_res_coo_dict["coord_pept_indices"],
                            dtype=np.uint32,
                        ),
                        "residue": np.array(
                            batch_im_rt_pept_res_coo_dict["data"], dtype=np.float32
                        ),
                    }
                )
                writer_res.write_table(table_res)
            else:
                os.remove(path_res)  # remove residue parquet if not return res coo dict


def _match_candidate_mz_and_binned_frame_mz_by_ppm(
    candidate_mz,
    frame_mz,
    frame_int: Optional[np.ndarray] = None,
    ppm_tol: float = 10,
    bin_width: float = 0.01,  # <-- new: bin size in Daltons
):
    """
    Match candidate m/z values to frame m/z values within ppm tolerance.

    Args:
        candidate_mz: Array of candidate m/z values.
        frame_mz: Array of frame m/z values.
        frame_int: Optional array of frame intensity values for weighted binning.
        ppm_tol: PPM tolerance for matching.
        bin_width: Width of m/z bins in Daltons.

    Returns:
        uniform_mz_idx: Indices of matched frame m/z values.
        candidate_mz_idx: Indices of (binned) candidate m/z values.
        mapped_candidate_mz_idx: Mapped indices of candidate m/z to unique frame m/z.
        frame_mz_idx: Indices of (binned) frame m/z values.
        mapped_frame_mz_idx: Mapped indices of frame m/z to unique frame m/z.
        candidate_mask: Boolean mask of candidates m/z that are kept.
        frame_mask: Boolean mask of frame m/z that are kept.
    """
    # --- Bin frame m/z into stable bins ---
    mz_min, mz_max = frame_mz.min(), frame_mz.max()
    bin_edges = np.arange(mz_min, mz_max + bin_width, bin_width)
    frame_bin_idx = np.digitize(frame_mz, bin_edges) - 1

    # Sum intensities per bin
    # binned_frame_int = np.bincount(frame_bin_idx, weights=frame_int, minlength=len(bin_edges))
    if frame_int is None:
        binned_frame_mz = (bin_edges[:-1] + bin_edges[1:]) / 2  # bin centers
    else:
        # 2. weighted sum of m/z per bin
        weighted_sum = np.bincount(
            frame_bin_idx, weights=frame_mz * frame_int, minlength=len(bin_edges) - 1
        )

        # 3. total intensity per bin
        weight_total = np.bincount(
            frame_bin_idx, weights=frame_int, minlength=len(bin_edges) - 1
        )

        # 4. intensity-weighted bin m/z (avoid division by zero)
        with np.errstate(divide="ignore", invalid="ignore"):
            binned_frame_mz = np.where(
                weight_total > 0, weighted_sum / weight_total, 0.0
            )

    # --- Map candidate m/z into bins (with ppm tolerance) ---
    candidate_mz_idx = np.searchsorted(binned_frame_mz, candidate_mz)
    candidate_mz_idx = np.clip(candidate_mz_idx, 0, len(binned_frame_mz) - 1)

    # ppm check: keep only if candidate fits within tolerance to bin (weighted) center
    dist = np.abs(binned_frame_mz[candidate_mz_idx] - candidate_mz)
    tol = candidate_mz * ppm_tol / 1e6
    candidate_mz_mask = dist <= tol
    if candidate_mz_mask.sum() == 0:
        Logger.warning("No candidate mz left after ppm filtering!")
        return None
    candidate_mz_idx_filtered = candidate_mz_idx[candidate_mz_mask]
    unique_candidate_mz_idx, mapped_unique_candidate_mz_idx = np.unique(
        candidate_mz_idx_filtered, return_inverse=True
    )

    frame_mask = np.isin(frame_bin_idx, unique_candidate_mz_idx)
    # Map reduced bin indices into compact index space [0 .. len(unique_idx)-1]
    bin_mapping = {b: i for i, b in enumerate(unique_candidate_mz_idx)}
    frame_bin_idx_filtered = frame_bin_idx[frame_mask]
    mapped_bins = np.array([bin_mapping[b] for b in frame_bin_idx_filtered])

    return (
        unique_candidate_mz_idx,
        candidate_mz_idx,
        mapped_unique_candidate_mz_idx,
        frame_bin_idx,
        mapped_bins,
        candidate_mz_mask,
        frame_mask,
    )


def _match_merged_candidate_mz_and_frame_mz_by_ppm(
    candidate_mz,
    candidate_abundance,
    frame_mz,
    ppm_tol=10.0,
):
    """
    Match candidate m/z to frame m/z directly (no binning), within ppm tolerance.
    Returns index mappings and masks compatible with downstream array construction.
    """

    if len(candidate_mz) == 0 or len(frame_mz) == 0:
        return None
    candidate_mz_idx = np.arange(candidate_mz.size)
    (
        anchor_mz,
        unique_candidate_mz_idx,
        mapped_unique_candidate_mz_idx,
        candidate_mask,
    ) = anchor_bin_hybrid_with_assignments(
        candidate_mz, candidate_abundance, ppm_tol=ppm_tol
    )
    # Logger.debug(
    #     "Number of anchors selected: %s from %s", len(anchor_mz), len(candidate_mz)
    # )
    frame_mz_idx = np.arange(frame_mz.size)
    frame_mask, mapped_frame_mz_idx = match_frame_to_anchors(
        frame_mz, anchor_mz, ppm_tol=ppm_tol
    )
    # Logger.debug(
    #     "Number of frame m/z matched to anchors: %s from %s",
    #     frame_mask.sum(),
    #     len(frame_mz),
    # )

    return (
        unique_candidate_mz_idx,
        candidate_mz_idx,
        mapped_unique_candidate_mz_idx,
        frame_mz_idx,
        mapped_frame_mz_idx[mapped_frame_mz_idx != -1],
        candidate_mask,
        frame_mask,
    )


def anchor_bin_hybrid_with_assignments(
    candidate_mz, candidate_abundance, ppm_tol: float = 10
):
    """
    Hybrid anchor selection and assignment:
    1. Isolated peaks (no neighbor within ppm_tol) become anchors directly.
    2. Crowded peaks are greedily anchored by abundance.
    Returns:
        anchors: array of anchor m/z
        assignments: array of same length as candidate_mz
                     giving index of the assigned anchor
    """
    candidate_mz = np.asarray(candidate_mz, dtype=np.float64)
    candidate_abundance = np.asarray(candidate_abundance, dtype=np.float64)
    n = len(candidate_mz)

    # Sort by m/z
    sort_idx = np.argsort(candidate_mz)
    mz_sorted = candidate_mz[sort_idx]
    abundance_sorted = candidate_abundance[sort_idx]
    ppm_factor = ppm_tol * 1e-6

    # Compute ppm-based neighbor differences
    left_diff = np.full(n, np.inf)
    right_diff = np.full(n, np.inf)
    left_diff[1:] = (mz_sorted[1:] - mz_sorted[:-1]) / mz_sorted[1:] * 1e6
    right_diff[:-1] = (mz_sorted[1:] - mz_sorted[:-1]) / mz_sorted[:-1] * 1e6

    # Identify isolated peaks
    isolated_mask = (left_diff > ppm_tol) & (right_diff > ppm_tol)
    anchors_mask = isolated_mask.copy()
    assigned = isolated_mask.copy()

    # Prepare assignment array (anchor index per candidate)
    assignments = np.full(n, -1, dtype=int)
    anchor_ids = np.full(n, -1, dtype=int)

    # Initialize isolated anchors
    anchor_count = 0
    for i in np.where(isolated_mask)[0]:
        assignments[i] = anchor_count
        anchor_ids[i] = anchor_count
        anchor_count += 1

    # Process crowded peaks (greedy by abundance)
    crowded_idx = np.where(~isolated_mask)[0]
    for i in crowded_idx[np.argsort(-abundance_sorted[crowded_idx])]:
        if assigned[i]:
            continue

        anchor_mz = mz_sorted[i]
        tol = anchor_mz * ppm_factor
        lower = anchor_mz - tol
        upper = anchor_mz + tol

        # Find neighbors in tolerance
        left = np.searchsorted(mz_sorted, lower, side="left")
        right = np.searchsorted(mz_sorted, upper, side="right")

        # Assign all unassigned peaks in this region
        unassigned = ~assigned[left:right]
        if np.any(unassigned):
            assignments[left:right][unassigned] = anchor_count
            assigned[left:right][unassigned] = True

        anchors_mask[i] = True
        anchor_ids[i] = anchor_count
        anchor_count += 1

    anchors_sorted = mz_sorted[anchors_mask]

    # Map assignments back to original candidate order
    reverse_idx = np.empty_like(sort_idx)
    reverse_idx[sort_idx] = np.arange(n)
    assignments_original = assignments[reverse_idx]
    anchor_mz = anchors_sorted
    candidate_mz_idx = np.arange(candidate_mz.size)
    candidate_mask = assignments_original != -1
    unique_candidate_mz_idx, mapped_unique_candidate_mz_idx = np.unique(
        candidate_mz_idx[candidate_mask], return_inverse=True
    )
    return (
        anchor_mz,
        unique_candidate_mz_idx,
        mapped_unique_candidate_mz_idx,
        candidate_mask,
    )


def match_frame_to_anchors(frame_mz, anchors, ppm_tol: float = 10):
    """
    Match frame_mz values to anchor m/z values within ±ppm tolerance.

    Parameters
    ----------
    frame_mz : array-like
        Array of m/z values to match.
    anchors : array-like
        Sorted array of anchor m/z values.
    ppm_tol : float
        Tolerance in ppm.

    Returns
    -------
    match_mask : ndarray of bool
        True for frame_mz entries that matched at least one anchor.
    matched_anchor_idx : ndarray of int (optional)
        Index of the matching anchor for each frame_mz (–1 if no match).
    """
    frame_mz = np.asarray(frame_mz, dtype=np.float64)
    anchors = np.asarray(anchors, dtype=np.float64)
    anchors.sort()

    # Nearest anchor indices via binary search
    idx_right = np.searchsorted(anchors, frame_mz, side="left")

    # Candidate indices (left and right neighbor anchors)
    idx_left = np.clip(idx_right - 1, 0, len(anchors) - 1)
    idx_right = np.clip(idx_right, 0, len(anchors) - 1)

    # Distances in ppm to left and right anchors
    ppm_left = np.abs(frame_mz - anchors[idx_left]) / anchors[idx_left] * 1e6
    ppm_right = np.abs(frame_mz - anchors[idx_right]) / anchors[idx_right] * 1e6

    # Choose the closer anchor
    use_right = ppm_right < ppm_left
    nearest_idx = np.where(use_right, idx_right, idx_left)
    nearest_ppm = np.where(use_right, ppm_right, ppm_left)

    # Determine matches within tolerance
    match_mask = nearest_ppm <= ppm_tol
    matched_anchor_idx = np.full(frame_mz.shape, -1, dtype=int)
    matched_anchor_idx[match_mask] = nearest_idx[match_mask]

    # matched_anchor_idx = matched_anchor_idx[match_mask]

    return match_mask, matched_anchor_idx


def collapse_candidates_by_confounder_group(
    candidate_precursor_by_rt: pd.DataFrame,
) -> pd.DataFrame:
    """
    Collapse multi-member confounder groups present in this frame's candidate
    set into a single representative row keyed by confounder_group_id, using
    the group's precomputed merged isotope pattern (GroupIsoMZ/
    GroupIsoAbundance from dict_add_merged_confounder_pattern). This removes
    the collinear/competing rows a confounder group would otherwise produce
    in the per-frame SWA candidate matrix -- one activation trace is solved
    for the whole group instead.

    Solo candidates (confounder_group_id == -1, or the column absent) pass
    through unchanged, keyed by their own mz_rank. The returned "mz_rank"
    column is therefore a mix of real mz_ranks (solo) and group ids (merged
    groups) -- used as the row-key for _prepare_sparse_matrices. Caller must
    re-sort by "mz_rank" after calling this (group ids, being far larger than
    real mz_ranks, sort to the end -- harmless for _prepare_sparse_matrices,
    which only needs candidate_precursor_by_rt.mz_rank sorted ascending and
    self-consistent).
    """
    if "confounder_group_id" not in candidate_precursor_by_rt.columns:
        return candidate_precursor_by_rt
    grouped_mask = candidate_precursor_by_rt["confounder_group_id"] != -1
    if not grouped_mask.any():
        return candidate_precursor_by_rt

    solo = candidate_precursor_by_rt.loc[~grouped_mask]
    grouped = candidate_precursor_by_rt.loc[grouped_mask]
    representatives = grouped.drop_duplicates(
        subset="confounder_group_id", keep="first"
    ).copy()
    representatives["mz_rank"] = representatives["confounder_group_id"]
    representatives["IsoMZ"] = representatives["GroupIsoMZ"]
    representatives["IsoAbundance"] = representatives["GroupIsoAbundance"]
    representatives["mz_length"] = representatives["GroupMzLength"]
    representatives["RT_search_left"] = representatives["GroupRT_search_left"]
    representatives["RT_search_right"] = representatives["GroupRT_search_right"]
    return pd.concat([solo, representatives], ignore_index=True)


def _prepare_sparse_matrices(
    candidate_precursor_by_rt,
    frame_data,
    all_id,
    ppm_tol: float = 10,
    bin_frame_mz: bool = True,  # TODO: change the default to False later
    bin_width: float = 0.01,  # <-- new: bin size in Daltons
    use_ims: bool = True,
    mobility_values: Optional[pd.DataFrame] = None,
):
    """
    Prepare frame and candidate arrays for sparse encoding.

    Args:
        candidate_precursor_by_rt: DataFrame with candidate precursors for the frame.
        frame_data: DataFrame with m/z and intensity values for the frame.
        all_id: Array of all candidate IDs (mz_rank).
        ppm_tol: PPM tolerance for matching m/z values.
        bin_width: Width of m/z bins in Daltons.
        use_ims: Whether to include IMS dimension.
        mobility_values: DataFrame with mobility values if use_ims is True.

    Returns:
        frame_array: 2D array (im, m/z bins) or (1, m/z bins) if no IMS.
        candidate_array: 2D array (candidates, m/z bins).
    """
    # --- Candidate arrays ---
    candidate_id = np.repeat(
        candidate_precursor_by_rt.mz_rank.values,
        candidate_precursor_by_rt.mz_length.values,
    )
    candidate_mz = np.concatenate(candidate_precursor_by_rt.IsoMZ.values)
    candidate_abundance = np.concatenate(candidate_precursor_by_rt.IsoAbundance.values)
    candidate_id_idx = np.searchsorted(all_id, candidate_id)

    # --- Frame m/z values ---
    if use_ims:
        frame_mz = frame_data["mz_values"].values
        frame_int = frame_data["intensity_values"]
    else:
        frame_mz = np.asarray(frame_data["mzarray"].values[0])
        frame_int = np.asarray(frame_data["intarray"].values[0])

    # --- Match candidate m/z to frame m/z with ppm tolerance and binning ---
    if bin_frame_mz:
        match_results = _match_candidate_mz_and_binned_frame_mz_by_ppm(
            candidate_mz=candidate_mz,
            frame_mz=frame_mz,
            frame_int=None,
            ppm_tol=ppm_tol,
            bin_width=bin_width,
        )
    else:
        # TODO: this is not yet correct! When other candidate m/z
        # are put inside the anchor mz bin, the anchor mz bin width
        # remain the same meaning mz values towards bin edge are not really properly dealt with!
        match_results = _match_merged_candidate_mz_and_frame_mz_by_ppm(
            candidate_mz=candidate_mz,
            candidate_abundance=candidate_abundance,
            frame_mz=frame_mz,
            ppm_tol=ppm_tol,
        )
    if match_results is None:
        Logger.warning("No matched candidates and frame m/z!")
        return None, None
    else:
        (
            uniform_mz_idx,
            candidate_mz_idx,
            mapped_unique_candidate_mz_idx,
            frame_bin_idx,
            mapped_bins,
            candidate_mask,
            frame_mask,
        ) = match_results

    # --- Build candidate array ---
    candidate_mz_idx_filtered = candidate_mz_idx[candidate_mask]
    candidate_abundance_filtered = candidate_abundance[candidate_mask]
    candidate_id_idx_filtered = candidate_id_idx[candidate_mask]
    # Logger.debug(
    #     "After ppm+binning: %s candidates mz idx kept (out of %s)",
    #     len(candidate_mz_idx_filtered),
    #     len(candidate_mz),
    # )
    candidate_array = np.zeros(
        (all_id.size, uniform_mz_idx.size), dtype=candidate_abundance.dtype
    )
    np.add.at(
        candidate_array,
        (candidate_id_idx_filtered, mapped_unique_candidate_mz_idx),
        candidate_abundance_filtered,
    )

    # --- Build frame array ---
    if frame_mask.sum() == 0:
        Logger.warning("No frame data left after candidate_bin filtering!")
        return None, None
    frame_int_filtered = frame_int[frame_mask]
    if use_ims:
        assert mobility_values is not None
        all_im = np.sort(mobility_values["mobility_values"])
        frame_im_index = np.searchsorted(all_im, frame_data["mobility_values"])
        frame_im_idx_filtered = frame_im_index[frame_mask]

        frame_array = np.zeros(
            (all_im.size, len(uniform_mz_idx)), dtype=frame_int.dtype
        )
        np.add.at(frame_array, (frame_im_idx_filtered, mapped_bins), frame_int_filtered)

    else:
        frame_array = np.zeros((1, len(uniform_mz_idx)), dtype=frame_int.dtype)
        np.add.at(
            frame_array,
            (np.zeros(len(frame_int_filtered), dtype=int), mapped_bins),
            frame_int_filtered,
        )

    return frame_array, candidate_array


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
    peaks, peak_properties = find_peaks(
        im_pept_act_array, height=height, width=width, rel_height=rel_height
    )
    if peaks.size > 0:
        peak_properties["pept_mzrank_index"] = np.repeat(index, len(peaks))
        peak_properties["pept_mzrank"] = np.repeat(pept_mzrank, len(peaks))
        peak_properties["peak"] = peaks
        peak_properties = pd.DataFrame(peak_properties)
    else:
        # Logger.debug("No peak extracted from pept_id %s in this frame.", pept_id)
        peak_properties = None
    return peak_properties


def generate_id_partitions(
    id_array,
    n_batch,
    how: Literal["round_robin", "block"] = "block",
    n_edge_counts: int = 50,
    *,
    rt_chunk: int | None = None,
):
    """
    Generate partitions of scan IDs.

    If rt_chunk is provided:
      - round_robin is done over RT chunks (SAFE)
      - block is done over contiguous RT chunks (SAFE)

    Returns
    -------
    List[List[int]]
    """

    # --------------------------------------------------
    # RT-chunk aware mode
    # --------------------------------------------------
    if rt_chunk is not None:
        # 1) group scan IDs by RT chunk
        chunk_map = {}
        for rt in id_array:
            cid = rt // rt_chunk
            chunk_map.setdefault(cid, []).append(rt)

        # ensure deterministic order
        chunk_ids = sorted(chunk_map.keys())
        chunk_lists = [chunk_map[cid] for cid in chunk_ids]

        partitions = [[] for _ in range(n_batch)]

        if how == "round_robin":
            Logger.info("Generate RT-chunk partitions by round robin.")
            for i, chunk in enumerate(chunk_lists):
                partitions[i % n_batch].extend(chunk)

        elif how == "block":
            Logger.info("Generate RT-chunk partitions by block.")
            n_chunks = len(chunk_lists)
            block_size = (n_chunks - 2 * n_edge_counts) // n_batch

            mark = 0
            for i in range(n_batch):
                if i == 0:
                    end = block_size + n_edge_counts
                elif i == n_batch - 1:
                    end = n_chunks
                else:
                    end = mark + block_size

                for chunk in chunk_lists[mark:end]:
                    partitions[i].extend(chunk)

                mark = end

        return partitions

    # --------------------------------------------------
    # ORIGINAL behavior (unchanged)
    # --------------------------------------------------
    id_partitions = [[] for _ in range(n_batch)]

    if how == "round_robin":
        Logger.info("Generate id partitions by round robin.")
        for i in range(n_batch):
            batch_idx = np.arange(i, len(id_array), n_batch)
            id_partitions[i] = id_array[batch_idx].tolist()

    elif how == "block":
        Logger.info("Generate id partitions by block.")

        total = len(id_array)
        core_size = total - 2 * n_edge_counts
        block_size = core_size // n_batch

        start = 0

        for i in range(n_batch):
            if i == 0:
                end = block_size + n_edge_counts
            elif i == n_batch - 1:
                end = total
            else:
                end = start + block_size

            id_partitions[i] = id_array[start:end].tolist()
            start = end

    return id_partitions


def process_frames_parallel(
    n_jobs: int,
    batch_scan_indices: list,
    parquet_file_stem: Optional[str] = None,
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
    if parquet_file_stem is not None:
        Logger.info("Processing frames in parallel with Parquet output.")
        Parallel(n_jobs=n_jobs)(
            delayed(process_batch_frame_save_parquet)(
                batch_scan_idx=batch,
                parquet_file=f"{parquet_file_stem}_frame_batch_{i}",
                **kwargs,
            )
            for i, batch in enumerate(batch_scan_indices)
        )
    else:
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


def init_parquet_scheme(data_col_name: Literal["activation", "residue"] = "activation"):
    schema = pa.schema(
        [
            ("frame_idx", pa.uint16()),  # maximum 65535 frames
            ("im_idx", pa.uint16()),  # maximum 65535 mobility bins
            (
                "mz_rank",
                pa.uint32(),
            ),  # maximum 4 billion peptides, should be enough for now
            (data_col_name, pa.float32()),
        ]
    )
    return schema
