import json
import os
import pandas as pd
import numpy as np
import logging
import sparse
from peak_detection_2d.utils import save_data_points_to_hdf5
from postprocessing.ims_3d import sum_peptbatch_from_act
from peak_detection_2d.dataset.dataset import prepare_2d_act_and_mask

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
# experiment specific
result_parent_dir = "/cmnfs/proj/ORIGINS/data/brain/FreshFrozenBrain/SingleShot/DDA/"
result_base_dir = "frame0_1830_ssDDA_P064428_Fresh1_5ug_R1_BD5_1_4921_ScanByScan_RTtol0.9_threshold_missabthres0.5_convergence_NoIntercept_pred_mzBinDigits2_imPeakWidth4_deltaMobilityThres80"
pept_batch_size = 50000
peak_data_dir = "peak_detection_mask_data_with_hint_channels"
result_dir = os.path.join(result_parent_dir, result_base_dir)
with open(os.path.join(result_dir, "param.json"), mode="r", encoding="utf-8") as file:
    config = json.load(file)

maxquant_file_exp = config["MQ_exp_path"]
maxquant_file_ref = os.path.join(result_dir, "maxquant_result_ref.pkl")
maxquant_result_dict = pd.read_pickle(filepath_or_buffer=maxquant_file_ref)
maxquant_result_merged_cleaned = pd.read_pickle(
    os.path.join(result_dir, "maxquant_result_merged_cleaned.pkl")
)
ms1scans = pd.read_csv(os.path.join(result_dir, "ms1scans.csv"), index_col=0)

mobility_values_df = pd.read_csv(
    os.path.join(result_dir, "mobility_values.csv"), index_col=0
)

# Modify maxquant result dict to generate hint channels
ms1scans["rt_coordinate"] = ms1scans["MS1_frame_idx.1"] + 1
maxquant_result_dict = maxquant_result_dict.sort_values("predicted_RT")
maxquant_result_dict = pd.merge_asof(
    left=maxquant_result_dict,
    right=ms1scans[["rt_coordinate", "Time_minute"]],
    left_on="predicted_RT",
    right_on="Time_minute",
    direction="nearest",
)
maxquant_result_dict["mz_bin_2digit"] = np.round(maxquant_result_dict["m/z"], 2)

# Genearte peptide batches for summing up activation
pept_batch_idx = list(range(1, 6))  # TODO: neglect the first batch
logging.info("Peptide batch idx: %s", pept_batch_idx)
maxquant_result_merged_cleaned["pept_batch_idx"] = (
    maxquant_result_merged_cleaned["mz_rank"] // pept_batch_size
)

logging.info(
    "Start summing up activation for each peptide batch with seperation %s.",
    pept_batch_idx,
)
sum_peptbatch_from_act(
    pept_batch_idx_list=pept_batch_idx, result_dir=result_dir, n_act_batch=10
)

# Prepare data points for peak detection
os.makedirs(os.path.join(result_dir, peak_data_dir), exist_ok=True)

# if make hint channels:
array_zero = np.zeros((1831, 937))

for idx in pept_batch_idx:
    if os.path.exists(
        os.path.join(result_dir, peak_data_dir, f"dp_peptbatch_mask{idx}.h5")
    ):
        logging.info("Peak selection data exists for peptide batch %s.", idx)
    else:
        pept_act_batch = sparse.load_npz(
            os.path.join(result_dir, f"output_im_rt_pept_act_coo_peptbatch{idx}.npz")
        )
        batch_maxquant_result = maxquant_result_merged_cleaned.loc[
            maxquant_result_merged_cleaned["pept_batch_idx"] == idx
        ]
        logging.info(
            "Peptide batch %s has %s peptides.", idx, batch_maxquant_result.shape[0]
        )
        data_points = []
        for i in batch_maxquant_result["mz_rank"].values:
            mz_ref = maxquant_result_dict.loc[
                maxquant_result_dict["mz_rank"] == i, "mz_bin_2digit"
            ].values[0]
            array = array_zero.copy()
            array[
                maxquant_result_dict.loc[
                    (maxquant_result_dict["mz_bin_2digit"] == mz_ref),
                    "rt_coordinate",
                ],
                maxquant_result_dict.loc[
                    (maxquant_result_dict["mz_bin_2digit"] == mz_ref),
                    "mobility_values_index",
                ],
                # maxquant_result_dict["mobility_values_index"]["mobility_values_index"],
            ] = -1
            logging.info("Number of non-zero elements in hint channel: %s", array.sum())
            dp_dict = prepare_2d_act_and_mask(
                pept_mz_rank=i,
                act_3d=pept_act_batch,
                maxquant_result_merged=batch_maxquant_result,
                mobility_values_df=mobility_values_df,
                ms1scans=ms1scans,
                pept_batch_idx=idx,
                pept_batch_size=pept_batch_size,
                delta_im=0.07,
                hint_channel=array,
            )
            if dp_dict is not None:
                data_points.append(dp_dict)

        save_data_points_to_hdf5(
            data_points,
            os.path.join(
                result_dir,
                peak_data_dir,
                f"dp_peptbatch_mask{idx}.h5",
            ),
        )
