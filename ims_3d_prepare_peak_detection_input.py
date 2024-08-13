import json
import os
import pandas as pd

import logging
import sparse
from peak_detection_2d.utils import save_data_points_to_hdf5
from postprocessing.ims_3d import sum_peptbatch_from_act
from peak_detection_2d.dataset.dataset import prepare_2d_act_and_bbox

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
# experiment specific
result_parent_dir = "/cmnfs/proj/ORIGINS/data/brain/FreshFrozenBrain/SingleShot/DDA/"
result_base_dir = "frame0_1830_ssDDA_P064428_Fresh1_5ug_R1_BD5_1_4921_ScanByScan_RTtol0.9_threshold_missabthres0.5_convergence_NoIntercept_pred_mzBinDigits2_imPeakWidth4_deltaMobilityThres80"
pept_batch_size = 50000

result_dir = os.path.join(result_parent_dir, result_base_dir)
with open(os.path.join(result_dir, "param.json"), mode="r", encoding="utf-8") as file:
    config = json.load(file)

maxquant_file_exp = config["MQ_exp_path"]
maxquant_file_ref = os.path.join(result_dir, "maxquant_result_ref.pkl")

maxquant_result_dict = pd.read_pickle(filepath_or_buffer=maxquant_file_ref)
maxquant_result_exp = pd.read_csv(maxquant_file_exp, sep="\t", low_memory=False)
ms1scans = pd.read_csv(os.path.join(result_dir, "ms1scans.csv"), index_col=0)
mobility_values_df = pd.read_csv(
    os.path.join(result_dir, "mobility_values.csv"), index_col=0
)

# Genearte peptide batches for summing up activation
n_pept = maxquant_result_dict.shape[0]
pept_batch_idx = list(range(0, n_pept, pept_batch_size))

logging.info(
    "Start summing up activation for each peptide batch with seperation %s.",
    pept_batch_idx,
)
sum_peptbatch_from_act(
    pept_batch_idx_list=pept_batch_idx, result_dir=result_dir, n_act_batch=10
)

# Prepare data points for peak detection
os.makedirs(os.path.join(result_dir, "peak_detection_data"), exist_ok=True)
pept_batch_end_idx = pept_batch_idx[1:]
pept_batch_end_idx.append(n_pept + 1)
for idx, pept_end_idx in enumerate(pept_batch_end_idx):
    if os.path.exists(
        os.path.join(result_dir, "peak_detection_data", f"dp_peptbatch_updated{idx}.h5")
    ):
        logging.info("Peak selection data exists for peptide batch %s.", idx)
    else:
        pept_act_batch = sparse.load_npz(
            os.path.join(result_dir, f"output_im_rt_pept_act_coo_peptbatch{idx}.npz")
        )
        data_points = []
        if idx == 0:
            for i in range(1, pept_end_idx):
                dp_dict = prepare_2d_act_and_bbox(
                    pept_mz_rank=i,
                    act_3d=pept_act_batch,
                    maxquant_result_dict=maxquant_result_dict,
                    maxquant_result_exp=maxquant_result_exp,
                    mobility_values_df=mobility_values_df,
                    ms1scans=ms1scans,
                    pept_batch_idx=idx,
                    pept_batch_size=pept_batch_size,
                    remove_oob=True,
                )
                if dp_dict is not None:
                    data_points.append(dp_dict)
        else:
            for i in range(pept_batch_idx[idx], pept_end_idx):
                dp_dict = prepare_2d_act_and_bbox(
                    pept_mz_rank=i,
                    act_3d=pept_act_batch,
                    maxquant_result_dict=maxquant_result_dict,
                    maxquant_result_exp=maxquant_result_exp,
                    mobility_values_df=mobility_values_df,
                    ms1scans=ms1scans,
                    pept_batch_idx=idx,
                    pept_batch_size=pept_batch_size,
                    remove_oob=True,
                )
                if dp_dict is not None:
                    data_points.append(dp_dict)
        save_data_points_to_hdf5(
            data_points,
            os.path.join(
                result_dir,
                "peak_detection_data",
                f"dp_peptbatch_updated_no_oob{idx}.h5",
            ),
        )
