import sparse
import os
import numpy as np
import logging
import pandas as pd
import alphatims.bruker


Logger = logging.getLogger(__name__)


def load_dotd_data(dotd_file_path: str, swaps_result_dir: str = ""):
    """
    Load .d file data and save hdf5 if not already exists.
    :param: dotd_file_path: str, path to the .d file
    :param: swaps_result_dir: str, path to the directory to save the hdf5 file, optional, default .d directory
    :return: data: alphatims.bruker.TimsTOF, data object
    :return: hdf_file_name: str, path to the saved hdf5 file
    """
    data = alphatims.bruker.TimsTOF(dotd_file_path)
    if swaps_result_dir == "":
        Logger.info("No output directory provided, using the directory of the .d file")
        swaps_result_dir = os.path.dirname(dotd_file_path)
    os.makedirs(os.path.join(swaps_result_dir), exist_ok=True)
    hdf_path = os.path.join(swaps_result_dir, f"{data.sample_name}.hdf")
    if not os.path.isfile(hdf_path):
        hdf_file_name = data.save_as_hdf(
            directory=swaps_result_dir,
            file_name=f"{data.sample_name}.hdf",
            overwrite=False,
        )
        Logger.info("HDF file saved as %s", hdf_file_name)
    else:
        hdf_file_name = hdf_path
        Logger.info("HDF file %s already exists", hdf_file_name)
    return data, hdf_file_name


def export_im_and_ms1scans(
    data: alphatims.bruker.TimsTOF, swaps_result_dir: str = None
):
    """
    Export IM and MS1 scans to csv files.
    :param: data: alphatims.bruker.TimsTOF, data object
    :param: swaps_result_dir: str, path to the directory to save the csv files, optional, default None then no export
    :return: ms1scans: pd.DataFrame, MS1 scans
    :return: mobility_values_df: pd.DataFrame, mobility values
    """
    # ms1scans
    ms1scans = data.frames.loc[data.frames.MsMsType == 0]
    ms1scans["Time_minute"] = ms1scans["Time"] / 60
    ms1scans["MS1_frame_idx"] = (
        ms1scans["Time"].rank(axis=0, method="first", ascending=True).astype(int) - 1
    )  # 0-based index
    ms1scans.set_index("MS1_frame_idx", inplace=True, drop=False)
    Logger.info(
        "Double check MS1 frame index range: %s - %s",
        ms1scans["MS1_frame_idx"].min(),
        ms1scans["MS1_frame_idx"].max(),
    )

    # mobilty values
    mobility_values = np.sort(data.mobility_values)
    mobility_values_df = pd.DataFrame(
        mobility_values, columns=["mobility_values"]
    ).reset_index()
    mobility_values_df = mobility_values_df.rename(
        columns={"index": "mobility_values_index"}
    )
    Logger.info(
        "Double check mobility values index range: %s - %s",
        mobility_values_df["mobility_values_index"].min(),
        mobility_values_df["mobility_values_index"].max(),
    )

    # export if swaps_result_dir is not None
    if swaps_result_dir is not None:
        os.makedirs(os.path.join(swaps_result_dir), exist_ok=True)
        ms1scans.to_csv(os.path.join(swaps_result_dir, "ms1scans.csv"))
        mobility_values_df.to_csv(os.path.join(swaps_result_dir, "mobility_values.csv"))
    return ms1scans, mobility_values_df


def sum_pept_act_by_peptbatch(n_blocks_by_pept: int, act_dir):
    """Sum activation intensity for each peptide batch."""
    for pept_block_num in range(n_blocks_by_pept):
        act_3d = sparse.load_npz(
            os.path.join(act_dir, f"im_rt_pept_act_coo_peptbatch{pept_block_num}.npz")
        )
        if pept_block_num == 0:
            pept_act_sum_all = act_3d.sum(axis=(0, 1))
        else:
            pept_act_sum_all += act_3d.sum(axis=(0, 1))
        del act_3d
        Logger.info("sum activation intensity for pept batch %s", pept_block_num)
    sparse.save_npz(os.path.join(act_dir, "pept_act_sum_all.npz"), pept_act_sum_all)
    pept_act_sum_array = sparse.asnumpy(pept_act_sum_all)
    pept_act_sum_df = pd.DataFrame(
        pept_act_sum_array,
        columns=["pept_act_sum"],
        index=np.arange(pept_act_sum_array.shape[0]),
    )
    pept_act_sum_df.to_csv(os.path.join(act_dir, "pept_act_sum.csv"))


def combine_3d_act_and_sum_int(
    n_blocks_by_pept: int,
    n_batch: int,
    act_dir: str,
    remove_batch_file: bool = False,
    calc_pept_act_sum_filter_by_im: bool = False,
    maxquant_result_ref: pd.DataFrame = None,
):
    """
    Combine peptide blocks of 3D activation intensity data.
    :param n_blocks_by_pept: int, number of blocks by peptide
    :param n_batch: int, number of batch by ms1 scans
    :param act_dir: str, path to the directory of activation intensity data
    :param remove_batch_file: bool, whether to remove batch files, default False
    :param calc_pept_act_sum_filter_by_im: bool, whether to calculate summed activation intensity filtered by IM, default False
    :param maxquant_result_ref: pd.DataFrame, MaxQuant reference data, default None, only used when calc_pept_act_sum_filter_by_im is True
    """
    if calc_pept_act_sum_filter_by_im:
        assert maxquant_result_ref is not None
        maxquant_result_ref_sorted = maxquant_result_ref.copy()
        maxquant_result_ref_sorted.sort_values("mz_rank", inplace=True)
        prev_cutoff = 0
    # assert n_blocks_by_pept > 1
    # pept_act_sum_all_array = np.array([])

    for pept_block_num in range(n_blocks_by_pept):
        try:
            act_3d_all = sparse.load_npz(
                os.path.join(
                    act_dir, f"im_rt_pept_act_coo_peptbatch{pept_block_num}.npz"
                )
            )
            if pept_block_num == 0:
                pept_act_sum_all = act_3d_all.sum(axis=(0, 1))
            else:
                pept_act_sum_all += act_3d_all.sum(axis=(0, 1))
        # act_3d_all = None
        except FileNotFoundError:
            for batch_num in range(n_batch):
                # logging.info("Batch %s, output file path %s", batch_num, conf.output_file)
                act_3d = sparse.load_npz(
                    os.path.join(
                        act_dir,
                        f"im_rt_pept_act_coo_batch{batch_num}_peptbatch{pept_block_num}.npz",
                    )
                )
                pept_act_sum = act_3d.sum(axis=(0, 1))
                logging.info("NNZ size of batch %s act_3d %s", batch_num, act_3d.nnz)
                if batch_num == 0:
                    act_3d_all = act_3d
                    if pept_block_num == 0:
                        pept_act_sum_all = pept_act_sum
                    else:
                        pept_act_sum_all += pept_act_sum
                    del act_3d, pept_act_sum
                else:
                    act_3d_all += act_3d
                    pept_act_sum_all += pept_act_sum
                    logging.info("NNZ size of act_3d_all %s", act_3d_all.nnz)
                    del act_3d, pept_act_sum

            sparse.save_npz(
                os.path.join(
                    act_dir, f"im_rt_pept_act_coo_peptbatch{pept_block_num}.npz"
                ),
                act_3d_all,
            )
        if calc_pept_act_sum_filter_by_im:
            shape = act_3d_all.shape
            n_pept_in_blocks = shape[2] // n_blocks_by_pept
            cutoff = n_pept_in_blocks * (pept_block_num + 1)

            pept_act_sum_filter_by_im = sum_3d_act_filter_by_im_fast(
                act_3d_all,
                maxquant_result_ref.loc[
                    maxquant_result_ref["mz_rank"].isin(range(prev_cutoff, cutoff))
                ],
                return_df=False,
            )
            Logger.debug(
                "pept_act_sum_filter_by_im sum %s", pept_act_sum_filter_by_im.sum()
            )
            Logger.debug("pept_act_sum_filter_by_im %s", pept_act_sum_filter_by_im)
            if pept_block_num == 0:
                pept_act_sum_filter_by_im_array = pept_act_sum_filter_by_im
            else:
                pept_act_sum_filter_by_im_array += pept_act_sum_filter_by_im
            Logger.debug(
                "pept_act_sum_filter_by_im_array sum %s",
                pept_act_sum_filter_by_im_array.sum(),
            )
            prev_cutoff = cutoff

        if remove_batch_file:
            for batch_num in range(n_batch):
                os.remove(
                    os.path.join(
                        act_dir,
                        f"im_rt_pept_act_coo_batch{batch_num}_peptbatch{pept_block_num}.npz",
                    )
                )
    pept_act_sum_array = sparse.asnumpy(pept_act_sum_all)

    del pept_act_sum_all
    # pept_act_sum_all_array = np.append(pept_act_sum_all_array, pept_act_sum_array)
    pept_act_sum_df = pd.DataFrame(
        pept_act_sum_array[:],
        columns=["pept_act_sum"],
        index=np.arange(pept_act_sum_array.shape[0]),
    )
    pept_act_sum_df["mz_rank"] = pept_act_sum_df.index
    pept_act_sum_df.to_csv(os.path.join(act_dir, "pept_act_sum.csv"), index=False)
    if calc_pept_act_sum_filter_by_im:
        pept_act_sum_filter_by_im_df = pd.DataFrame(
            pept_act_sum_filter_by_im_array[:],
            columns=["pept_act_sum_filter_by_im"],
            index=np.arange(pept_act_sum_filter_by_im_array.shape[0]),
        )
        Logger.debug(
            "pept_act_sum_filter_by_im_df sum %s",
            pept_act_sum_filter_by_im_df["pept_act_sum_filter_by_im"].sum(),
        )
        pept_act_sum_filter_by_im_df["mz_rank"] = pept_act_sum_filter_by_im_df.index
        pept_act_sum_filter_by_im_df.to_csv(
            os.path.join(act_dir, "pept_act_sum_filter_by_im.csv"), index=False
        )

        return pept_act_sum_filter_by_im_df  # TODO: remove later


def sum_3d_act_filter_by_im_fast(
    im_rt_pept_act_coo_peptbatch,
    maxquant_result_ref: pd.DataFrame,
    chunk_size: int = 200,
    return_df: bool = True,
):
    """
    Sum activation intensity for each peptide batch and filter by accurate 1/K0 range.
    :param im_rt_pept_act_coo_peptbatch: sparse.coo_matrix, 3D activation intensity data
    :param maxquant_result_ref: pd.DataFrame, MaxQuant reference data
    :param chunk_size: int, chunk size for summing, default 200, when peptide number is larger this needs to be small
    :param return_df: bool, whether to return a DataFrame, default True, if false return numpy array
    :return: pept_act_sum_df: pd.DataFrame, summed activation intensity data filtered by IM dimension according to MaxQuant reference data
    """
    # TODO: what if the peptbatch is also in chunks?
    assert (
        "Ion mobility length" in maxquant_result_ref.columns
        and "mobility_values_index" in maxquant_result_ref.columns
    )
    maxquant_result_ref["mobility_values_index_start"] = np.minimum(
        np.maximum(
            0,
            maxquant_result_ref["mobility_values_index"]
            - maxquant_result_ref["Ion mobility length"] // 2,
        ),
        im_rt_pept_act_coo_peptbatch.shape[1],
    )
    maxquant_result_ref["mobility_values_index_end"] = np.minimum(
        np.maximum(
            0,
            maxquant_result_ref["mobility_values_index"]
            + maxquant_result_ref["Ion mobility length"] // 2,
        )
        + 1,
        im_rt_pept_act_coo_peptbatch.shape[1],
    )
    # Vectorized approach using list comprehension and numpy array
    maxquant_result_ref["mobility_values_coo"] = [
        np.arange(start, end)
        for start, end in zip(
            maxquant_result_ref["mobility_values_index_start"],
            maxquant_result_ref["mobility_values_index_end"],
        )
    ]
    maxquant_result_ref = maxquant_result_ref[
        [
            "mobility_values_index_start",
            "mobility_values_index_end",
            "mobility_values_coo",
            "mz_rank",
        ]
    ]

    # generate a sparse mask for filtering mobility values

    mobility_lengths = [len(coo) for coo in maxquant_result_ref["mobility_values_coo"]]
    repeated_mz_rank = np.repeat(
        maxquant_result_ref["mz_rank"].values, mobility_lengths
    )

    # Explode the DataFrame to align repeated mz_rank values with the corresponding mobility values
    maxquant_result_ref_exploded = maxquant_result_ref.explode(
        "mobility_values_coo"
    ).reset_index(drop=True)

    maxquant_result_ref_exploded["pept_coo"] = repeated_mz_rank
    im_coords = np.concatenate(
        maxquant_result_ref_exploded["mobility_values_coo"], axis=None
    )
    Logger.debug("im_coords min and max:  %s  %s", im_coords.min(), im_coords.max())
    pept_coords = np.concatenate(maxquant_result_ref_exploded["pept_coo"], axis=None)
    Logger.debug("pept_coords min and max: %s %s", pept_coords.min(), pept_coords.max())
    mask_coords = np.stack([im_coords, pept_coords], axis=0)
    mask_data = 1
    mask_sparse = sparse.COO(
        mask_coords, mask_data, shape=im_rt_pept_act_coo_peptbatch.shape[1:]
    )
    if im_rt_pept_act_coo_peptbatch.shape[0] > chunk_size:
        Logger.info("Summing in chunks..")
        chunk_number = im_rt_pept_act_coo_peptbatch.shape[0] // chunk_size

        mask_sparse_chunk = mask_sparse.broadcast_to(
            shape=(
                chunk_size,
                im_rt_pept_act_coo_peptbatch.shape[1],
                im_rt_pept_act_coo_peptbatch.shape[2],
            )
        )

        for i in range(chunk_number):
            Logger.info("Chunk %s", i)
            if i == 0:
                pept_act_sum_array = (
                    im_rt_pept_act_coo_peptbatch[
                        i * chunk_size : (i + 1) * chunk_size, :, :
                    ]
                    * mask_sparse_chunk
                ).sum(
                    axis=(0, 1)
                )  # Multiply needs to be in () before sum
                Logger.debug("pept_act_sum_array shape %s", pept_act_sum_array.shape)
            else:
                pept_act_sum_array += (
                    im_rt_pept_act_coo_peptbatch[
                        i * chunk_size : (i + 1) * chunk_size, :, :
                    ]
                    * mask_sparse_chunk
                ).sum(axis=(0, 1))
        if chunk_number * chunk_size < im_rt_pept_act_coo_peptbatch.shape[0]:
            Logger.info("Last chunk")
            mask_sparse_chunk = mask_sparse.broadcast_to(
                shape=(
                    im_rt_pept_act_coo_peptbatch.shape[0] - chunk_number * chunk_size,
                    im_rt_pept_act_coo_peptbatch.shape[1],
                    im_rt_pept_act_coo_peptbatch.shape[2],
                )
            )
            last_chunk_act = im_rt_pept_act_coo_peptbatch[
                chunk_number * chunk_size :, :, :
            ]
            Logger.debug("Last chunk act shape %s", last_chunk_act.shape)
            Logger.debug("mask_sparse_chunk shape %s", mask_sparse_chunk.shape)
            Logger.debug("pept_act_sum_array shape %s", pept_act_sum_array.shape)
            pept_act_sum_array += (last_chunk_act * mask_sparse_chunk).sum(
                axis=(0, 1)
            )  # TODO: bug: shape mismatch
            Logger.debug("pept_act_sum_array shape %s", pept_act_sum_array.shape)
    else:
        pept_act_sum_array = im_rt_pept_act_coo_peptbatch * mask_sparse.sum(axis=(0, 1))
    pept_act_sum_array = pept_act_sum_array.todense()
    if return_df:
        Logger.debug("Returning DataFrame")
        pept_act_sum_df = pd.DataFrame(
            pept_act_sum_array[:],
            columns=["pept_act_sum"],
            index=np.arange(pept_act_sum_array.shape[0]),
        )
        pept_act_sum_df["mz_rank"] = pept_act_sum_df.index
        # pept_act_sum_df.to_csv(os.path.join(act_dir, "pept_act_sum.csv"), index=False)
        return pept_act_sum_df
    else:
        Logger.debug("Returning numpy array")
        return pept_act_sum_array


def sum_3d_act_filter_by_im_improved(
    im_rt_pept_act_coo_peptbatch, maxquant_result_ref: pd.DataFrame
):
    """
    Sum activation intensity for each peptide batch and filter by IM.
    :param im_rt_pept_act_coo_peptbatch: sparse.coo_matrix, 3D activation intensity data
    :param maxquant_result_ref: pd.DataFrame, MaxQuant reference data
    :return: pept_act_sum_df: pd.DataFrame, summed activation intensity data filtered by IM dimension according to MaxQuant reference data
    """
    assert (
        "Ion mobility length" in maxquant_result_ref.columns
        and "mobility_values_index" in maxquant_result_ref.columns
    )

    # Calculate the start and end indices for the mobility values
    mobility_start = np.round(
        (
            maxquant_result_ref["mobility_values_index"]
            - maxquant_result_ref["Ion mobility length"] // 2
        ),
        decimals=0,
    ).values
    mobility_end = np.round(
        (
            maxquant_result_ref["mobility_values_index"]
            + maxquant_result_ref["Ion mobility length"] // 2
            + 1
        ),
        decimals=0,
    ).values
    # mz_rank = maxquant_result_ref["mz_rank"]
    left_minus = im_rt_pept_act_coo_peptbatch[:, :mobility_start, :].sum(axis=(0, 1))

    right_minus = im_rt_pept_act_coo_peptbatch[:, mobility_end:, :].sum(axis=(0, 1))
    total_value = im_rt_pept_act_coo_peptbatch[:, :, :].sum(axis=(0, 1))
    pept_act_sum_array = total_value - left_minus - right_minus

    # pept_act_sum_array = im_rt_pept_act_coo_peptbatch.sum(axis=(0, 1))
    pept_act_sum_df = pd.DataFrame(
        pept_act_sum_array[:],
        columns=["pept_act_sum"],
        index=np.arange(pept_act_sum_array.shape[0]),
    )
    pept_act_sum_df["mz_rank"] = pept_act_sum_df.index

    return pept_act_sum_df
