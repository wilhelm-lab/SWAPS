import logging
from typing import Optional, Literal
import sparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt

from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import regionprops_table
from skimage.filters import sobel
from scipy.ndimage import gaussian_filter1d, uniform_filter, gaussian_filter
from skimage.morphology import h_minima, remove_small_objects
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
    data: alphatims.bruker.TimsTOF, swaps_result_dir: Optional[str] = None
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
    ms1scans = pd.DataFrame(ms1scans)
    # export if swaps_result_dir is not None
    if swaps_result_dir is not None:
        os.makedirs(os.path.join(swaps_result_dir), exist_ok=True)
        ms1scans.to_csv(os.path.join(swaps_result_dir, "ms1scans.csv"))
        mobility_values_df.to_csv(os.path.join(swaps_result_dir, "mobility_values.csv"))
    return ms1scans, mobility_values_df


# def sum_pept_act_by_peptbatch(n_blocks_by_pept: int, act_dir):
#     """Sum activation intensity for each peptide batch."""
#     for pept_block_num in range(n_blocks_by_pept):
#         act_3d = sparse.load_npz(
#             os.path.join(act_dir, f"im_rt_pept_act_coo_peptbatch{pept_block_num}.npz")
#         )
#         if pept_block_num == 0:
#             pept_act_sum_all = act_3d.sum(axis=(0, 1))
#         else:
#             pept_act_sum_all += act_3d.sum(axis=(0, 1))
#         del act_3d
#         Logger.info("sum activation intensity for pept batch %s", pept_block_num)
#     sparse.save_npz(os.path.join(act_dir, "pept_act_sum_all.npz"), pept_act_sum_all)
#     pept_act_sum_array = sparse.asnumpy(pept_act_sum_all)
#     pept_act_sum_df = pd.DataFrame(
#         pept_act_sum_array,
#         columns=["pept_act_sum"],
#         index=np.arange(pept_act_sum_array.shape[0]),
#     )
#     pept_act_sum_df.to_csv(os.path.join(act_dir, "pept_act_sum.csv"))


def combine_3d_act_and_detect_peak(  # TODO: integrate res coo
    n_blocks_by_pept: int,
    n_batch: int,
    act_dir: str,
    maxquant_result_ref: pd.DataFrame,
    remove_batch_file: bool = False,
    calc_pept_act_sum_filter_by_im: bool = False,
    use_ims: bool = True,
    im_ref: str = "exp",
    n_cpu: int = 1,
    chunk_size: int = 1000,
    rt_group: int = 1,
):
    """
    Combine peptide blocks of 3D activation intensity data, \
        do peak detection and calculate peak properties

    :param n_blocks_by_pept: int, number of blocks by peptide
    :param n_batch: int, number of batch by ms1 scans
    :param act_dir: str, path to the directory of activation intensity data
    :param remove_batch_file: bool, whether to remove batch files, default False
    :param calc_pept_act_sum_filter_by_im: bool, whether to calculate summed activation intensity filtered by IM, default False
    :param maxquant_result_ref: pd.DataFrame, MaxQuant reference data, default None, only used when calc_pept_act_sum_filter_by_im is True
    :param use_ims: bool, whether the data contains ion mobility dimesion.
    :param im_ref: str, which ion mobility values to use for filtering, "exp" or "ref", default "exp"
    :param n_cpu: int, number of cpu to use for parallel processing, default 1
    :param chunk_size: int, chunk size for peak detection and property calculation, default 500, larger values require larger memory
    :param rt_group: int, number of RT groups to split for processing, default 1, larger values makes sense when there is many MS1 frames

    :return: peak_property_all_pept_batches: pd.DataFrame, peak properties for all peptide batches
    """
    if calc_pept_act_sum_filter_by_im:
        maxquant_result_ref_sorted = maxquant_result_ref.copy()
        maxquant_result_ref_sorted.sort_values("mz_rank", inplace=True)

    maxquant_result_ref["RT_group"] = pd.qcut(
        maxquant_result_ref["MS1_frame_idx_center_ref"], q=rt_group, labels=False
    )
    # assert n_blocks_by_pept > 1
    # pept_act_sum_all_array = np.array([])
    apex_df_arrays = None
    prev_cutoff = 0
    peak_property_all_pept_batches = pd.DataFrame()
    for pept_block_num in range(n_blocks_by_pept):
        try:
            act_3d_all = sparse.load_npz(
                os.path.join(
                    act_dir, f"im_rt_pept_act_coo_peptbatch{pept_block_num}.npz"
                )
            )
            Logger.info(
                "Loaded 3D activation intensity data for pept batch %s with shape %s",
                pept_block_num,
                act_3d_all.shape,
            )
            if use_ims:
                if pept_block_num == 0:
                    pept_act_sum_all = act_3d_all.sum(axis=(0, 1))
                else:
                    pept_act_sum_all += act_3d_all.sum(axis=(0, 1))
            else:
                if pept_block_num == 0:
                    pept_act_sum_all = act_3d_all.sum(axis=0)
                else:
                    pept_act_sum_all += act_3d_all.sum(axis=0)
        # act_3d_all = None
        except FileNotFoundError:
            for batch_num in range(n_batch):
                act_3d = sparse.load_npz(
                    os.path.join(
                        act_dir,
                        f"im_rt_pept_act_coo_batch{batch_num}_peptbatch{pept_block_num}.npz",
                    )
                )
                if use_ims:
                    pept_act_sum = act_3d.sum(axis=(0, 1))
                else:
                    pept_act_sum = act_3d.sum(axis=0)
                Logger.info("NNZ size of batch %s act_3d %s", batch_num, act_3d.nnz)
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
                    Logger.info("NNZ size of act_3d_all %s", act_3d_all.nnz)
                    del act_3d, pept_act_sum

            sparse.save_npz(
                os.path.join(
                    act_dir, f"im_rt_pept_act_coo_peptbatch{pept_block_num}.npz"
                ),
                act_3d_all,
            )

        # -----------Peak detection and property calculation-----------
        shape = act_3d_all.shape
        n_pept_in_blocks = shape[2] // n_blocks_by_pept
        cutoff = n_pept_in_blocks * (pept_block_num + 1)
        try:
            peak_property = pd.read_csv(
                os.path.join(act_dir, f"peptbatch{pept_block_num}_peak_properties.csv")
            )
            Logger.info(
                "Loaded existing peak properties for pept batch %s", pept_block_num
            )
        except FileNotFoundError:

            # Split mz ranks into chunks and compute start/end indices for each chunk
            chunk_indices = [
                (i, min(i + chunk_size, cutoff))
                for i in range(prev_cutoff, cutoff, chunk_size)
            ]
            Logger.info(
                "Processing pept batch %s from %s to %s with chunks: %s",
                pept_block_num,
                prev_cutoff,
                cutoff,
                chunk_indices,
            )

            collected_peak_results = []
            with ProcessPoolExecutor(max_workers=n_cpu) as executor:
                futures = [
                    executor.submit(
                        detect_2d_peak_and_calculate_peak_property,
                        act_3d_all,
                        maxquant_result_ref,
                        start_idx,
                        end_idx,
                    )
                    for start_idx, end_idx in chunk_indices
                ]

                for f in tqdm(futures, desc="Processing chunks"):
                    collected_peak_results.append(f.result())
            peak_property = pd.concat(collected_peak_results, ignore_index=True)
            peak_property.to_csv(
                os.path.join(act_dir, f"peptbatch{pept_block_num}_peak_properties.csv"),
                index=False,
            )
        peak_property_all_pept_batches = pd.concat(
            [peak_property_all_pept_batches, peak_property], ignore_index=True
        )

        if calc_pept_act_sum_filter_by_im:
            assert maxquant_result_ref is not None

            pept_act_sum_filter_by_im = _sum_3d_act_filter_by_im_fast(
                im_rt_pept_act_coo_peptbatch=act_3d_all,
                maxquant_result_ref=maxquant_result_ref.loc[
                    maxquant_result_ref["mz_rank"].isin(range(prev_cutoff, cutoff))
                ],
                return_df=False,
                im_ref=im_ref,
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

        if remove_batch_file:
            Logger.info("Removing batch files for pept batch %s", pept_block_num)
            for batch_num in range(n_batch):
                if os.path.exists(
                    os.path.join(
                        act_dir,
                        f"im_rt_pept_act_coo_batch{batch_num}_peptbatch{pept_block_num}.npz",
                    )
                ):
                    os.remove(
                        os.path.join(
                            act_dir,
                            f"im_rt_pept_act_coo_batch{batch_num}_peptbatch{pept_block_num}.npz",
                        )
                    )
                else:
                    Logger.warning(
                        "Batch file for batch %s and pept batch %s does not exist, skipping removal.",
                        batch_num,
                        pept_block_num,
                    )
                    continue

        prev_cutoff = cutoff
        Logger.info("Finished processing pept batch %s", pept_block_num)

    peak_property_all_pept_batches.to_csv(
        os.path.join(act_dir, "all_pept_batches_peak_properties.csv"), index=False
    )
    Logger.info(
        "Peak property calculation done for all peptide batches. Total peaks: %s",
        peak_property_all_pept_batches.shape[0],
    )
    peak_property_all_pept_batches.groupby("mz_rank").size().hist(bins=58)
    plt.xlabel("Number of peaks detected per image")
    plt.ylabel("Frequency")
    xlim_right = plt.xlim()[1] * 0.6
    ylim_top = plt.ylim()[1] * 0.8
    plt.text(
        xlim_right,
        ylim_top,
        f"n_peaks={len(peak_property_all_pept_batches)}\nn_images={peak_property_all_pept_batches['mz_rank'].nunique()}",
        fontsize=12,
    )
    plt.savefig(os.path.join(act_dir, "peak_count_per_image_histogram.png"), dpi=300)
    plt.close()

    # Peak activation sum without filtering
    pept_act_sum_array = sparse.asnumpy(pept_act_sum_all)
    Logger.info("pept_act_sum_all sum %s", pept_act_sum_array.shape)
    del pept_act_sum_all

    pept_act_sum_df = pd.DataFrame(
        pept_act_sum_array[:],
        columns=["pept_act_sum"],
        index=np.arange(pept_act_sum_array.shape[0]),
    )
    pept_act_sum_df["mz_rank"] = pept_act_sum_df.index
    pept_act_sum_df.to_csv(os.path.join(act_dir, "pept_act_sum.csv"), index=False)

    if use_ims and calc_pept_act_sum_filter_by_im:
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

    return peak_property_all_pept_batches


def _sum_3d_act_filter_by_im_fast(
    im_rt_pept_act_coo_peptbatch,
    maxquant_result_ref: pd.DataFrame,
    chunk_size: int = 200,
    return_df: bool = True,
    im_ref: str = "exp",
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
    # assert (
    #     "Ion mobility length" in maxquant_result_ref.columns
    #     and "mobility_values_index" in maxquant_result_ref.columns
    # )
    # maxquant_result_ref["mobility_values_index_start"] = np.minimum(
    #     np.maximum(
    #         0,
    #         maxquant_result_ref["mobility_values_index"]
    #         - maxquant_result_ref["Ion mobility length"] // 2,
    #     ),
    #     im_rt_pept_act_coo_peptbatch.shape[1],
    # )
    # maxquant_result_ref["mobility_values_index_end"] = np.minimum(
    #     np.maximum(
    #         0,
    #         maxquant_result_ref["mobility_values_index"]
    #         + maxquant_result_ref["Ion mobility length"] // 2,
    #     )
    #     + 1,
    #     im_rt_pept_act_coo_peptbatch.shape[1],
    # )
    # Vectorized approach using list comprehension and numpy array
    match im_ref:
        case "exp":
            left_col = "mobility_values_index_left_exp"
            right_col = "mobility_values_index_right_exp"
        case "ref":
            left_col = "mobility_values_index_left_ref"
            right_col = "mobility_values_index_right_ref"
    maxquant_result_ref[left_col] = maxquant_result_ref[left_col].astype(int)
    maxquant_result_ref[right_col] = maxquant_result_ref[right_col].astype(int)
    maxquant_result_ref["mobility_values_coo"] = [
        np.arange(start, end)
        for start, end in zip(
            maxquant_result_ref[left_col],
            maxquant_result_ref[right_col],
        )
    ]
    maxquant_result_ref = maxquant_result_ref[
        [
            left_col,
            right_col,
            "mobility_values_coo",
            "mz_rank",
        ]
    ]

    # generate a sparse mask for filtering mobility values

    mobility_lengths = [len(coo) for coo in maxquant_result_ref["mobility_values_coo"]]
    repeated_mz_rank = np.repeat(
        maxquant_result_ref["mz_rank"].to_numpy(), mobility_lengths
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


# def sum_3d_act_filter_by_im_improved(
#     im_rt_pept_act_coo_peptbatch, maxquant_result_ref: pd.DataFrame
# ):
#     """
#     Sum activation intensity for each peptide batch and filter by IM.
#     :param im_rt_pept_act_coo_peptbatch: sparse.coo_matrix, 3D activation intensity data
#     :param maxquant_result_ref: pd.DataFrame, MaxQuant reference data
#     :return: pept_act_sum_df: pd.DataFrame, summed activation intensity data filtered by IM dimension according to MaxQuant reference data
#     """
#     assert (
#         "Ion mobility length" in maxquant_result_ref.columns
#         and "mobility_values_index" in maxquant_result_ref.columns
#     )

#     # Calculate the start and end indices for the mobility values
#     mobility_start = np.round(
#         (
#             maxquant_result_ref["mobility_values_index"]
#             - maxquant_result_ref["Ion mobility length"] // 2
#         ),
#         decimals=0,
#     ).astype(int)
#     mobility_end = np.round(
#         (
#             maxquant_result_ref["mobility_values_index"]
#             + maxquant_result_ref["Ion mobility length"] // 2
#             + 1
#         ),
#         decimals=0,
#     ).astype(int)
#     # mz_rank = maxquant_result_ref["mz_rank"]
#     left_minus = im_rt_pept_act_coo_peptbatch[:, :mobility_start, :].sum(axis=(0, 1))

#     right_minus = im_rt_pept_act_coo_peptbatch[:, mobility_end:, :].sum(axis=(0, 1))
#     total_value = im_rt_pept_act_coo_peptbatch[:, :, :].sum(axis=(0, 1))
#     pept_act_sum_array = total_value - left_minus - right_minus

#     # pept_act_sum_array = im_rt_pept_act_coo_peptbatch.sum(axis=(0, 1))
#     pept_act_sum_df = pd.DataFrame(
#         pept_act_sum_array[:],
#         columns=["pept_act_sum"],
#         index=np.arange(pept_act_sum_array.shape[0]),
#     )
#     pept_act_sum_df["mz_rank"] = pept_act_sum_df.index

#     return pept_act_sum_df


def detect_2d_peak_with_watershed(
    image,
    int_threshold=0.5,
    min_distance=15,
    threshold_rel=0.2,
    coordinates: Optional[np.ndarray] = None,
    seed_radius: int = 4,
    use_competing_peaks: bool = True,  # new: enable/disable the feature
    min_distance_to_true_seed: int = 20,  # new: minimum distance from competing peaks to the true seed
):
    """
    Detect peaks in a 2D image using the watershed algorithm.

    Parameters:
    - pept_act_log: 2D numpy array
        The input image in which to detect peaks. Usually log10 transformed intensity.
    - log_threshold: float
        Threshold for log-transformed intensity to create the signal mask.
    - min_distance: int
        Minimum distance between detected peaks.
    - threshold_rel: float
        Minimum intensity of peaks, calculated as max(image) * threshold_rel.
    Returns:
    - labels: 2D numpy array
        Labeled regions corresponding to detected peaks.
    """
    # 2. Compute distance (to background) transform inside signal
    distance = image
    mask_signal = distance > int_threshold

    if not mask_signal.any():
        distance[~mask_signal] = 0
        return (
            np.empty((0, 2), dtype=int),
            np.zeros_like(distance, dtype=int),
            np.zeros_like(distance, dtype=float),
        )

    if coordinates is None:
        # --- original behavior unchanged ---
        coordinates = peak_local_max(
            distance,
            min_distance=min_distance,
            threshold_rel=threshold_rel,
            labels=mask_signal,
        )

    if coordinates.size == 0:
        return (
            coordinates,
            np.zeros_like(image, dtype=int),
            np.zeros_like(image, dtype=float),
        )

    mask = np.zeros(image.shape, dtype=bool)
    mask[tuple(coordinates.T)] = True

    # --- new: inject competing background seeds when true seed is provided ---
    if use_competing_peaks and coordinates.shape[0] == 1:
        gradient = sobel(image)

        bg_peaks = peak_local_max(
            gradient,
            min_distance=min_distance,
            threshold_rel=threshold_rel,
        )

        true_seed = coordinates[0]  # shape (2,)
        for bg_peak in bg_peaks:
            dist = np.linalg.norm(bg_peak - true_seed)
            if dist >= min_distance_to_true_seed:
                mask[tuple(bg_peak)] = True
    # -------------------------------------------------------------------------

    markers, _ = ndi.label(mask)

    if seed_radius > 0:
        dist, (ri, ci) = ndi.distance_transform_edt(~mask, return_indices=True)
        markers = np.where(dist <= seed_radius, markers[ri, ci], 0)

    labels = watershed(-image, markers, mask=mask_signal, compactness=0.001)

    # --- new: when competing peaks were used, only return true seed's region ---
    if use_competing_peaks and coordinates.shape[0] == 1:
        true_marker = markers[tuple(coordinates[0])]
        labels = np.where(labels == true_marker, true_marker, 0)
    # --------------------------------------------------------------------------

    return coordinates, labels, image


def calculate_peak_property_from_labels_and_image(
    labels,
    image_2d,
    grad_mag: Optional[np.ndarray] = None,
    image_res_2d: Optional[np.ndarray] = None,
    min_peak_area=10,
    min_peak_sum_intensity=1000,
    return_dy: bool = False,
):
    """
    Calculate properties of detected peaks from coordinates of the local maximum and labels.

    Parameters:
    - labels: 2D numpy array
        The labeled regions corresponding to detected peaks.
    - image_2d: 2D numpy array
        The original image from which to calculate peak properties.
    - image_2d_log: 2D numpy array
        The log-transformed image from which to calculate peak properties.
    - min_peak_area: int
        Minimum area of peaks to be considered valid.
    - min_peak_sum_intensity: float
        Minimum sum intensity of peaks to be considered valid.

    Returns:
    - df: pd.DataFrame
        A DataFrame containing peak properties, including row-based smoothness.
    """
    num_labels = labels.max()
    if num_labels == 0:
        # Logger.debug("No peaks detected after watershed.")
        return None

    # Region properties from skimage (fast, C-based)
    props = regionprops_table(
        labels,
        intensity_image=image_2d,
        properties=(
            "label",
            "centroid",
            "area",
            "area_filled",
            "bbox",
            "intensity_max",
            "intensity_mean",
            "intensity_std",
            "solidity",
        ),
    )
    df = pd.DataFrame(props)
    # Filter out small/weak peaks
    df["intensity_sum"] = df["intensity_mean"] * df["area"]
    df = df[
        (df["area"] >= min_peak_area) & (df["intensity_sum"] >= min_peak_sum_intensity)
    ]
    if df.empty:
        Logger.debug(
            "All detected peaks are filtered out by min area %s and min intensity sum %s",
            min_peak_area,
            min_peak_sum_intensity,
        )
        return None
    # Derived properties

    df["intensity_cv"] = df["intensity_std"] / (df["intensity_mean"] + 1e-8)
    df["im_length"] = df["bbox-2"] - df["bbox-0"]
    df["rt_length"] = df["bbox-3"] - df["bbox-1"]

    # Logger.info("Unique label values after filtering: %s", df["label"].values)
    if image_res_2d is not None:
        res_act_ratio = np.log1p(image_res_2d / (image_2d + 1e-6))
        props_res = regionprops_table(
            labels,
            intensity_image=res_act_ratio,
            properties=(
                "label",
                "intensity_max",
                "intensity_min",
                "intensity_mean",
                "intensity_std",
            ),
        )
        df_res = pd.DataFrame(props_res)
        df = df.merge(df_res, on="label", how="left", suffixes=("", "_res"))
        # df.drop(columns=["label_res"], inplace=True)

    # Compute row-based smoothness if gradient magnitude is provided
    if grad_mag is not None:
        row_smoothness_df = compute_row_smoothness_and_apex_index(
            labels=labels,
            dy=grad_mag,
            pept_act=image_2d,
            label_values=df["label"].values,
            apply_gaussian_smoothing=False,
        )

        df = df.merge(row_smoothness_df, on="label", how="left")
        if return_dy:
            return df, grad_mag
    else:
        return df


def compute_row_smoothness_and_apex_index(
    labels,
    dy,
    pept_act,
    label_values,
    apply_gaussian_smoothing: bool = True,
    sigma: float = 1.0,
):
    """
    Compute row-wise smoothness metrics for each labeled region.

    Parameters
    ----------
    labels : 2D numpy array
        Labeled peak regions.
    dy : 2D numpy array
        Gradient along rows (RT dimension).
    label_values : list or array-like
        Unique label values to compute metrics for.
    apply_gaussian_smoothing : bool, optional
        Whether to apply Gaussian smoothing to column profiles before computing second derivatives. Default is True.
    sigma : float, optional
        Standard deviation for Gaussian kernel if smoothing is applied. Default is 1.0.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - label
        - within_row_consistency
        - across_row_score_smoothness
        - row_smoothness (geometric mean)
    """
    records = []

    for lbl in label_values:
        mask = labels == lbl
        rows = np.where(np.any(mask, axis=1))[0]
        cols = np.where(np.any(mask, axis=0))[0]
        if len(rows) == 0 or len(cols) == 0:
            records.append(
                {
                    "label": lbl,
                    "within_row_consistency": np.nan,
                    "across_row_score_smoothness": np.nan,
                    "row_smoothness": np.nan,
                }
            )
            continue
        rt_apex = rows[np.argmax(pept_act[rows, :][:, cols].sum(axis=1))]
        im_apex = cols[np.argmax(pept_act[:, cols][rows, :].sum(axis=0))]
        # 1️⃣ within-row score: stability of dy along each row
        std_vals = [np.std(dy[r, mask[r, :]]) for r in rows]
        within_row_consistency = 1 / (1 + np.mean(std_vals))

        # 2️⃣ across-row score: smooth Gaussian-like shape per column
        col_scores = []
        for c in cols:
            col_mask = mask[:, c]
            col_vals = dy[:, c][col_mask]
            if len(col_vals) < 3:
                continue  # need at least 3 points to compute second derivative
            if apply_gaussian_smoothing:
                col_vals = gaussian_filter1d(col_vals, sigma=sigma)
            # col_vals_smooth = gaussian_filter1d(col_vals, sigma=1.0)
            second_deriv = np.diff(col_vals, n=2)
            col_score = 1 / (1 + np.mean(np.abs(second_deriv)))
            col_scores.append(col_score)

        if len(col_scores) == 0:
            across_row_score_smoothness = np.nan
        else:
            across_row_score_smoothness = np.mean(col_scores)

        # 3️⃣ geometric mean as combined score
        geom_mean = np.sqrt(within_row_consistency * across_row_score_smoothness)

        records.append(
            {
                "label": lbl,
                "within_row_consistency": within_row_consistency,
                "across_row_score_smoothness": across_row_score_smoothness,
                "row_smoothness": geom_mean,
                "rt_apex_index": rt_apex,
                "im_apex_index": im_apex,
            }
        )

    return pd.DataFrame(records)


def detect_2d_peak_and_calculate_peak_property(
    im_rt_pept_act_coo,
    ref_dict: pd.DataFrame,
    start_mz_idx: int,
    end_mz_idx: int,
    im_rt_pept_res_coo: Optional[sparse.COO] = None,
    filter: Literal["gaussian", "uniform", None] = "uniform",
    detect_kwargs=None,
    calc_kwargs=None,
):
    """
    Detect 2D peaks and calculate their properties for a range of mz ranks.

    :param im_rt_pept_act_coo: 3D sparse COO array of ion intensity data.
    :type im_rt_pept_act_coo: np.ndarray
    :param start_idx: int
        Starting m/z rank index to process.
    :param end_idx: int
        Ending m/z rank index to process.
    :param detect_kwargs: dict, optional
        Keyword arguments for detect_2d_peak_with_watershed.
    :param calc_kwargs: dict, optional
        Keyword arguments for calculate_peak_property_from_coords_and_labels.
    :return: DataFrame containing peak properties for all detected peaks.
    :rtype: pd.DataFrame
    """
    dict_ref_filtered = ref_dict[
        (ref_dict["mz_rank"] >= start_mz_idx) & (ref_dict["mz_rank"] < end_mz_idx)
    ]
    im_min = max(dict_ref_filtered["IM_search_idx_left"].min() - 25, 0)
    im_max = dict_ref_filtered["IM_search_idx_right"].max() + 25

    all_peak_properties_in_chunk = []

    detect_kwargs = detect_kwargs or {}
    calc_kwargs = calc_kwargs or {}

    for g in dict_ref_filtered["RT_group"].unique():
        group_df = dict_ref_filtered[dict_ref_filtered["RT_group"] == g]
        mz_idx = group_df["mz_rank"].unique()
        if len(mz_idx) == 0:
            Logger.info("No mz ranks found for RT group %s, skipping.", g)
            continue
        rt_min = group_df["MS1_frame_idx_left_ref"].min()
        rt_max = group_df["MS1_frame_idx_right_ref"].max() + 1
        Logger.info("Processing RT group %s with RT range %s - %s", g, rt_min, rt_max)
        im_rt_pept_act_coo_dense = np.atleast_3d(
            im_rt_pept_act_coo[rt_min:rt_max, im_min:im_max, mz_idx].todense()
        )  # Convert to dense for easier indexing
        if im_rt_pept_res_coo is not None:
            im_rt_pept_res_coo_dense = np.abs(
                np.atleast_3d(
                    im_rt_pept_res_coo[rt_min:rt_max, im_min:im_max, mz_idx].todense()
                )
            )  # sometimes residue is negative
        else:
            im_rt_pept_res_coo_dense = None
        for rel_idx, mz_rank in enumerate(mz_idx):
            pept_act = im_rt_pept_act_coo_dense[:, :, rel_idx]

            # pept_act = im_rt_pept_act_coo[rt_start:rt_end, im_start:im_end, pept_idx].todense()
            cleaned_mask = remove_small_objects(
                pept_act >= 10, min_size=9
            )  # TODO: hardcoded threshold
            match filter:
                case "gaussian":
                    pept_act_smoothed = gaussian_filter(pept_act, sigma=1.0)
                case "uniform":
                    blurred = uniform_filter(pept_act, size=9)
                    pept_act_smoothed = np.maximum(pept_act, blurred)
                case None:
                    pept_act_smoothed = pept_act
            pept_act_smoothed = pept_act_smoothed * cleaned_mask
            pept_act_smoothed_log = np.log10(1 + pept_act_smoothed)

            # Calculate gradient mask
            dy, dx = np.gradient(pept_act_smoothed)
            grad_mag = np.sqrt(dx**2 + dy**2)
            # grad_mag_smooth = gaussian_filter(grad_mag, sigma=2.0)

            # Detect peaks with flexible kwargs
            coordinates, labels, distance = detect_2d_peak_with_watershed(
                pept_act_smoothed_log,
                int_threshold=1,
                threshold_rel=0.2,
                min_distance=10,
            )
            # Calculate peak properties with flexible kwargs
            peak_properties = calculate_peak_property_from_labels_and_image(
                labels, pept_act, grad_mag, **calc_kwargs
            )

            if isinstance(peak_properties, pd.DataFrame) and not peak_properties.empty:
                peak_properties["mz_rank"] = mz_rank
                peak_properties["rt_apex_index"] += rt_min
                peak_properties["im_apex_index"] += im_min
                all_peak_properties_in_chunk.append(peak_properties)
            else:
                Logger.debug("No peaks detected for mz_rank %s", mz_rank)
                continue

    if len(all_peak_properties_in_chunk) > 1:
        all_peak_properties_in_chunk_df = pd.concat(
            all_peak_properties_in_chunk, ignore_index=True
        )
        Logger.info(
            "Returning %s images/mzrank with %s peaks in chunk %s - %s",
            len(all_peak_properties_in_chunk),
            all_peak_properties_in_chunk_df.shape[0],
            start_mz_idx,
            end_mz_idx,
        )
        return all_peak_properties_in_chunk_df
    else:
        Logger.warning("No peaks returned in chunk %s - %s", start_mz_idx, end_mz_idx)
        return None
