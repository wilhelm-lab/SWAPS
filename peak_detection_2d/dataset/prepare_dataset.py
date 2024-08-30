import os
import logging
from typing import Literal, List
import glob
import fire
import sparse
import pandas as pd
import numpy as np

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from torchvision import tv_tensors
from ..utils import (
    save_data_points_to_hdf5,
)
from postprocessing.ims_3d import (
    get_ref_rt_im_range,
    slice_pept_act,
    prepare_slice_pept_act_df,
    get_bbox_from_mq_exp,
)


Logger = logging.getLogger(__name__)


def prepare_2d_act_and_bbox(
    pept_mz_rank: int,
    act_3d: sparse.SparseArray,
    maxquant_result_dict: pd.DataFrame,
    maxquant_result_exp: pd.DataFrame,
    mobility_values_df: pd.DataFrame,
    ms1scans: pd.DataFrame,
    plot_range: Literal["nonzero", "custom"] = "custom",
    pept_batch_idx: int = 0,
    pept_batch_size: int = 50000,
    remove_oob: bool = True,  # whether to remove out of bound values
):
    (modseq, charge) = maxquant_result_dict.loc[
        maxquant_result_dict["mz_rank"] == pept_mz_rank,
        ["Modified sequence", "Charge"],
    ].values[0]
    maxquant_result_exp_row = maxquant_result_exp.loc[
        (maxquant_result_exp["Modified sequence"] == modseq)
        & (maxquant_result_exp["Charge"] == charge)
    ]
    if maxquant_result_exp_row.empty:
        return None

    Logger.info("Prepare data for peptide mz rank %s", pept_mz_rank)
    # TODO: make sure maxquant_result_exp each peptide only has one RT
    bbox = get_bbox_from_mq_exp(
        maxquant_result_exp_row
    )  # bbox: [min_rt, max_rt, rt_width, min_im, max_im, im_width]
    Logger.debug("Bbox: %s", bbox)
    rt_idx_range, im_idx_range, hint_value, hint_idx = get_ref_rt_im_range(
        pept_mz_rank=pept_mz_rank,
        maxquant_result_dict=maxquant_result_dict,
        mobility_values_df=mobility_values_df,
        ms1scans=ms1scans,
    )
    Logger.debug("RT and IM range: %s, %s", rt_idx_range, im_idx_range)

    batch_corrected_pept_mz_rank = pept_mz_rank - pept_batch_idx * pept_batch_size

    slice_pept_act_sparse, rt_idx_range, im_idx_range = slice_pept_act(
        pept_act_sparse=act_3d[:, :, batch_corrected_pept_mz_rank],
        plot_range=plot_range,
        rt_idx_range=rt_idx_range,
        im_idx_range=im_idx_range,
    )
    data_3d_heatmap = prepare_slice_pept_act_df(
        slice_pept_act_sparse,
        rt_idx_range,
        im_idx_range,
        mobility_values_df,
        ms1scans,
        convert_idx_to_values=False,
    )
    # TODO: what if less than zero?
    bbox_rt_min_idx = (
        np.abs(ms1scans["Time_minute"].values - bbox[0]).argmin() - rt_idx_range[0]
    )
    bbox_rt_max_idx = (
        np.abs(ms1scans["Time_minute"].values - bbox[1]).argmin() - rt_idx_range[0]
    )
    bbox_im_min_idx = (
        np.abs(mobility_values_df["mobility_values"].values - bbox[3]).argmin()
        - im_idx_range[0]
    )
    bbox_im_max_idx = (
        np.abs(mobility_values_df["mobility_values"].values - bbox[4]).argmin()
        - im_idx_range[0]
    )
    hint_idx[0] -= rt_idx_range[0]
    hint_idx[1] -= im_idx_range[0]
    if remove_oob:
        if any(
            [
                bbox_rt_min_idx < 0,
                bbox_rt_max_idx < 0,
                bbox_im_min_idx < 0,
                bbox_im_max_idx < 0,
                hint_idx[0] < 0,
                hint_idx[1] < 0,
            ]
        ):
            Logger.warning(
                "Out of bound values detected for pept_mz_rank %s",
                batch_corrected_pept_mz_rank,
            )
            return None
    return {
        "data": data_3d_heatmap,
        "bbox": [bbox_im_min_idx, bbox_rt_min_idx, bbox_im_max_idx, bbox_rt_max_idx],
        "hint_idx": hint_idx,
        "pept_mz_rank": pept_mz_rank,
    }


def prepare_2d_act_and_mask_updated(
    pept_mz_rank: int,
    peptbatch_act: sparse.SparseArray,
    maxquant_dict: pd.DataFrame,
    hint_matrix: sparse.SparseArray,
    add_label_mask: bool = True,
):
    """
    Prepare 2D activation and mask for a peptide mz rank
    :param pept_mz_rank: int, peptide mz rank
    :param peptbatch_act: SparseArray, 3D activation matrix
    :param maxquant_dict: pd.DataFrame, maxquant results
    :param hint_matrix: SparseArray, hint matrix
    :param add_label_mask: bool, whether to add label mask
    """
    pept_mz_rank = int(pept_mz_rank)
    maxquant_row = maxquant_dict[maxquant_dict["mz_rank"] == pept_mz_rank]
    target = 0 if maxquant_row["Decoy"].values[0] else 1
    # Ensure that there's only one row
    if maxquant_row.shape[0] > 1:
        raise ValueError(
            f"Expected one row for mz_rank {pept_mz_rank}, but found {maxquant_row.shape[0]} rows."
        )
    elif maxquant_row.shape[0] == 0:
        Logger.warning("No maxquant result found for peptide mz rank %s", pept_mz_rank)
        rt_left = 0
        rt_right = peptbatch_act.shape[0]
        im_left = 0
        im_right = peptbatch_act.shape[1]
        mask = np.zeros(peptbatch_act.shape[:2])
    else:
        rt_left = maxquant_row["MS1_frame_idx_left_ref"].values[0]
        rt_right = maxquant_row["MS1_frame_idx_right_ref"].values[0]
        im_left = maxquant_row["IM_search_idx_left"].values[0]
        im_right = maxquant_row["IM_search_idx_right"].values[0]
        mask = np.zeros(peptbatch_act.shape[:2])
        if (add_label_mask) and (not maxquant_row["Decoy"].values[0]):

            # mask = np.zeros(peptbatch_act.shape[:2])
            mask[
                int(maxquant_row["MS1_frame_idx_left_exp"].values[0]) : int(
                    maxquant_row["MS1_frame_idx_right_exp"].values[0]
                ),
                int(maxquant_row["mobility_values_index_left_exp"].values[0]) : int(
                    maxquant_row["mobility_values_index_right_exp"].values[0]
                ),
            ] = 1  # accurate mask
            # mask = mask[rt_left:rt_right, im_left:im_right]

    img = sparse.asnumpy(
        peptbatch_act[
            rt_left:rt_right,
            im_left:im_right,
            pept_mz_rank,
        ]
    )
    hint = sparse.asnumpy(
        hint_matrix[
            rt_left:rt_right,
            im_left:im_right,
            pept_mz_rank,
        ]
    )
    Logger.debug("Peptide mz_rank %s", pept_mz_rank)
    Logger.debug("Hint sum %s", hint.sum())
    Logger.debug("Hint non zero %s", np.count_nonzero(hint))
    mask = mask[rt_left:rt_right, im_left:im_right]
    Logger.debug("Mask sum %s", mask.sum())
    # Wrap sample and targets into torchvision tv_tensors:
    img = tv_tensors.Image(img)
    hint = tv_tensors.Image(hint)
    mask = tv_tensors.Image(mask)
    # Logger.debug("img sum %s", img.sum())
    return {
        "data": img,
        "hint_channel": hint,
        "mask": mask,
        "pept_mz_rank": pept_mz_rank,
        "target": target,
    }


def process_pept_mz_ranks(
    pept_mz_ranks: List[int],
    peptbatch_act: sparse.COO,
    maxquant_dict: pd.DataFrame,
    hint_matrix: sparse.COO,
    num_workers: int = 4,
):
    maxquant_dict[
        [
            "MS1_frame_idx_left_ref",
            "MS1_frame_idx_right_ref",
            "IM_search_idx_left",
            "IM_search_idx_right",
        ]
    ] = maxquant_dict[
        [
            "MS1_frame_idx_left_ref",
            "MS1_frame_idx_right_ref",
            "IM_search_idx_left",
            "IM_search_idx_right",
        ]
    ].astype(
        int
    )
    # with h5py.File(hdf5_file_path, "a") as hdf5_file:
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                prepare_2d_act_and_mask_updated,
                pept_mz_rank,
                peptbatch_act,
                maxquant_dict,
                hint_matrix,
            )
            for pept_mz_rank in tqdm(pept_mz_ranks)
        ]
        # Logger.debug("Data preparation completed, writing to HDF5 file.")

        return futures


def prepare_2d_act_and_mask(
    pept_mz_rank: int,
    act_3d: sparse.SparseArray,
    maxquant_result_merged: pd.DataFrame,
    mobility_values_df: pd.DataFrame,
    ms1scans: pd.DataFrame,
    plot_range: Literal["nonzero", "custom"] = "custom",
    pept_batch_idx: int = 0,
    pept_batch_size: int = 50000,
    delta_im: float = 0.07,
    hint_channel: np.array = None,
):
    maxquant_result_row = maxquant_result_merged.loc[
        maxquant_result_merged["mz_rank"] == pept_mz_rank
    ]
    if maxquant_result_row.empty:
        Logger.warning("No maxquant result found for peptide mz rank %s", pept_mz_rank)
        return None
    elif maxquant_result_row.shape[0] > 1:
        Logger.warning(
            "Multiple maxquant results found for peptide mz rank %s", pept_mz_rank
        )
        return None

    Logger.info("Prepare data for peptide mz rank %s", pept_mz_rank)
    bbox = get_bbox_from_mq_exp(
        maxquant_result_row
    )  # bbox: [min_rt, max_rt, rt_width, min_im, max_im, im_width]
    Logger.debug("Bbox: %s", bbox)
    rt_array = ms1scans["Time_minute"].values
    bbox_rt_min_idx = max(np.searchsorted(rt_array, bbox[0], side="right") - 1, 0)
    bbox_rt_max_idx = np.searchsorted(rt_array, bbox[1], side="left")
    im_array = mobility_values_df["mobility_values"].values
    bbox_im_min_idx = max(np.searchsorted(im_array, bbox[3], side="right") - 1, 0)
    bbox_im_max_idx = np.searchsorted(im_array, bbox[4], side="left")
    Logger.debug(
        "Bbox indices: %s, %s, %s, %s",
        bbox_rt_min_idx,
        bbox_rt_max_idx,
        bbox_im_min_idx,
        bbox_im_max_idx,
    )
    rt_idx_range, im_idx_range, hint_value, hint_idx = get_ref_rt_im_range(
        pept_mz_rank=pept_mz_rank,
        maxquant_result_dict=maxquant_result_merged,
        mobility_values_df=mobility_values_df,
        ms1scans=ms1scans,
        delta_im=delta_im,
    )
    Logger.debug("RT and IM range: %s, %s", rt_idx_range, im_idx_range)

    batch_corrected_pept_mz_rank = int(pept_mz_rank - pept_batch_idx * pept_batch_size)
    pept_act_sparse = act_3d[:, :, batch_corrected_pept_mz_rank]
    Logger.debug("Peptide activation shape %s", pept_act_sparse.shape)
    slice_pept_act_sparse, rt_idx_range, im_idx_range = slice_pept_act(
        pept_act_sparse=pept_act_sparse,
        plot_range=plot_range,
        rt_idx_range=rt_idx_range,
        im_idx_range=im_idx_range,
    )

    data_3d_heatmap = prepare_slice_pept_act_df(
        slice_pept_act_sparse,
        rt_idx_range,
        im_idx_range,
        mobility_values_df,
        ms1scans,
        convert_idx_to_values=False,
    )
    mask = np.zeros(pept_act_sparse.shape)
    mask[
        bbox_rt_min_idx : bbox_rt_max_idx + 1, bbox_im_min_idx : bbox_im_max_idx + 1
    ] = 1

    slice_mask = mask[
        rt_idx_range[0] : rt_idx_range[1], im_idx_range[0] : im_idx_range[1]
    ]
    Logger.debug("Masked area %s", slice_mask.sum())

    Logger.debug(
        "Masked intensity sum %s",
        np.nansum(np.multiply(data_3d_heatmap.values, slice_mask)),
    )
    if hint_channel is not None:
        assert hint_channel.shape == pept_act_sparse.shape
        Logger.info("Hint channel provided, using hint channel with shape.")
        hint_channel_slice = hint_channel.copy()
        hint_channel_slice[hint_idx[0], hint_idx[1]] = 1
        hint_channel_slice = hint_channel_slice[
            rt_idx_range[0] : rt_idx_range[1], im_idx_range[0] : im_idx_range[1]
        ]
    hint_idx[0] -= rt_idx_range[0]
    hint_idx[1] -= im_idx_range[0]
    return {
        "data": data_3d_heatmap,
        "hint_channel": hint_channel_slice,
        "mask": slice_mask,
        "hint_idx": hint_idx,
        "pept_mz_rank": pept_mz_rank,
    }


def _cartesian_product(group):
    rt_im_combinations = group[
        ["MS1_frame_idx_center_ref", "IM_search_idx_center"]
    ].values
    mz_ranks = group[["mz_rank"]].values

    # Create the Cartesian product using broadcasting
    cartesian_rt = np.array(
        np.meshgrid(rt_im_combinations[:, 0], mz_ranks[:, 0])
    ).T.reshape(-1, 2)
    cartesian_im = np.array(
        np.meshgrid(rt_im_combinations[:, 1], mz_ranks[:, 0])
    ).T.reshape(-1, 2)
    # Logger.info("cartesian_rt: %s, cartesian_im: %s", cartesian_im, cartesian_im)
    return pd.DataFrame(
        np.hstack([cartesian_rt, cartesian_im]),
        columns=[
            "MS1_frame_idx_center_ref",
            "mz_rank",
            "IM_search_idx_center",
            "mz_rank_2",
        ],
    )


def generate_hint_sparse_matrix(maxquant_dict_df: pd.DataFrame, shape: tuple):
    maxquant_dict_df[
        ["mz_rank", "MS1_frame_idx_center_ref", "IM_search_idx_center"]
    ] = maxquant_dict_df[
        ["mz_rank", "MS1_frame_idx_center_ref", "IM_search_idx_center"]
    ].astype(
        int
    )
    COO_dict = (
        maxquant_dict_df.groupby("mz_bin")
        .apply(_cartesian_product)
        .reset_index(drop=True)
    )
    COO_dict = COO_dict.astype(int)
    Logger.debug("COO dict dtypes: %s", COO_dict.dtypes)
    neg_hint = sparse.COO(
        coords=[
            COO_dict["MS1_frame_idx_center_ref"],
            COO_dict["IM_search_idx_center"],
            COO_dict["mz_rank_2"],
        ],
        data=-1,
        shape=shape,
    )
    pos_hint = sparse.COO(
        coords=[
            maxquant_dict_df["MS1_frame_idx_center_ref"],
            maxquant_dict_df["IM_search_idx_center"],
            maxquant_dict_df["mz_rank"],
        ],
        data=2,
        shape=shape,
    )
    hint_sparse_matrix = neg_hint + pos_hint
    return hint_sparse_matrix


def prepare_training_dataset(
    result_dir: str,
    maxquant_dict: pd.DataFrame,
    n_workers: int = 1,
    include_decoys: bool = False,
    chunk_size: int = 5000,
):

    # Create output directory
    ps_dir = os.path.join(result_dir, "peak_selection")
    os.makedirs(ps_dir, exist_ok=True)
    ps_data_dir = os.path.join(ps_dir, "training_data")
    os.makedirs(ps_data_dir, exist_ok=True)

    # Load relevant data
    if n_workers <= 0:
        n_workers = os.cpu_count()
    first_pept_act_batch = sparse.load_npz(
        os.path.join(
            result_dir,
            "results",
            "activation",
            "im_rt_pept_act_coo_peptbatch0.npz",
        )
    )
    pept_act_shape = first_pept_act_batch.shape
    if os.path.exists(os.path.join(ps_data_dir, "hint_matrix.npz")):
        hint_matrix = sparse.load_npz(os.path.join(ps_data_dir, "hint_matrix.npz"))
        Logger.info("Hint matrix existed and loaded.")
    else:
        Logger.info("Hint matrix not found, generating...")

        # maxquant_dict = pd.read_pickle(cfg_prepare_dataset.DICT_PICKLE_PATH)
        hint_matrix = generate_hint_sparse_matrix(
            maxquant_dict_df=maxquant_dict, shape=pept_act_shape
        )
        sparse.save_npz(os.path.join(ps_data_dir, "hint_matrix.npz"), hint_matrix)
    # get the number of pept batches
    peptbatch_name = os.path.join(
        result_dir, "results", "activation", "im_rt_pept_act_coo_peptbatch*.npz"
    )
    peptbatch_files = glob.glob(peptbatch_name)
    n_blocks_by_pept = len(peptbatch_files)

    # Assign pept batch index
    if "pept_batch_idx" not in maxquant_dict.columns:
        Logger.info("Column 'pept_batch_idx' not found, assigning pept batch index...")
        pept_batch_size = pept_act_shape[2] // n_blocks_by_pept
        maxquant_dict["pept_batch_idx"] = (
            maxquant_dict["mz_rank"] // pept_batch_size
        ).astype(int)
        max_pept_batch_idx = maxquant_dict["pept_batch_idx"].max()
        maxquant_dict.loc[
            maxquant_dict["pept_batch_idx"] == max_pept_batch_idx, "pept_batch_idx"
        ] = (max_pept_batch_idx - 1)
    pept_batch_indicies = maxquant_dict["pept_batch_idx"].unique()
    Logger.info("Pept batch indices: %s", pept_batch_indicies)
    if include_decoys:
        maxquant_dict_for_training = maxquant_dict.loc[
            maxquant_dict["source"].isin(["both"])
        ]  # Only use the data points that are from both sources for training and eval
    else:
        maxquant_dict_for_training = maxquant_dict.loc[
            (maxquant_dict["source"].isin(["both"])) & ~(maxquant_dict["Decoy"])
        ]
    Logger.info(
        "Number of training data points: %s", maxquant_dict_for_training.shape[0]
    )
    hdf5_file_paths = []
    for pept_batch in pept_batch_indicies:

        pept_mz_ranks = maxquant_dict_for_training.loc[
            maxquant_dict_for_training["pept_batch_idx"] == pept_batch, "mz_rank"
        ].values
        Logger.info(
            "Peptide batch %s, pept_mz_ranks length %s", pept_batch, len(pept_mz_ranks)
        )
        hdf5_file_path = os.path.join(
            ps_data_dir, f"train_datapoints_TD_peptbatch{pept_batch}.hdf5"
        )
        if pept_batch > 0:
            pept_act_batch = sparse.load_npz(
                os.path.join(
                    result_dir,
                    "results",
                    "activation",
                    f"im_rt_pept_act_coo_peptbatch{pept_batch}.npz",
                )
            )

        else:
            pept_act_batch = first_pept_act_batch
        num_chunks = len(pept_mz_ranks) // chunk_size + 1
        for i in tqdm(range(num_chunks)):
            start = i * chunk_size
            end = (i + 1) * chunk_size
            chunk = pept_mz_ranks[start:end]
            Logger.info("Processing chunk %s, from index %s to %s", i, start, end)
            list_of_data_points = process_pept_mz_ranks(
                pept_mz_ranks=chunk,
                peptbatch_act=pept_act_batch,
                maxquant_dict=maxquant_dict_for_training,
                hint_matrix=hint_matrix,
                num_workers=n_workers,  # Adjust based on your system's capabilities
            )
            save_data_points_to_hdf5(list_of_data_points, hdf5_file_path)
            del list_of_data_points
        hdf5_file_paths.append(hdf5_file_path)
    return hdf5_file_paths


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    fire.Fire(prepare_training_dataset)
