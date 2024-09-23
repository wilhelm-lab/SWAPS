import logging


import pandas as pd
import numpy as np
import os

from peak_detection_2d.dataset.prepare_dataset import (
    prepare_training_dataset,
    generate_hint_sparse_matrix,
)
import sparse

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
# ====================== Load data ======================
## Experiment data
swaps_config_path = "/cmnfs/proj/ORIGINS/SWAPS_exp/mixture/A_5139_20240912_134700_126900/config_20240912_134700_126900.yaml"
peak_selection_config_path = "/cmnfs/proj/ORIGINS/SWAPS_exp/mixture/A_5135_20240910_142404_677731/peak_selection/exp_20240910_181220_575220/updated_peak_selection_config.yaml"
ps_exp_dir = os.path.join("peak_selection", "eval_A_5135")
eval_test = True

from utils.config import get_cfg_defaults
from utils.singleton_swaps_optimization import swaps_optimization_cfg

cfg = get_cfg_defaults(swaps_optimization_cfg)
cfg.merge_from_file(swaps_config_path)
ps_exp_dir = os.path.join(cfg.RESULT_PATH, ps_exp_dir)
os.makedirs(ps_exp_dir, exist_ok=True)
cfg_peak_selection_transferred = cfg.PEAK_SELECTION
cfg_peak_selection_transferred.merge_from_file(peak_selection_config_path)
maxquant_result_ref = pd.read_pickle(cfg.DICT_PICKLE_PATH)

mobility_values_df = pd.read_csv(os.path.join(cfg.RESULT_PATH, "mobility_values.csv"))
ms1scans = pd.read_csv(os.path.join(cfg.RESULT_PATH, "ms1scans.csv"))
if os.path.isfile(
    os.path.join(cfg.RESULT_PATH, "peak_selection", "training_data", "hint_matrix.npz")
):
    logging.info("Hint matrix already exists")
else:
    os.makedirs(
        os.path.join(cfg.RESULT_PATH, "peak_selection", "training_data"), exist_ok=True
    )
    hint_matrix = generate_hint_sparse_matrix(
        maxquant_dict_df=maxquant_result_ref, shape=cfg.OPTIMIZATION.PEPTACT_SHAPE[0]
    )
    sparse.save_npz(
        os.path.join(
            cfg.RESULT_PATH, "peak_selection", "training_data", "hint_matrix.npz"
        ),
        hint_matrix,
    )
if eval_test:
    logging.info("Start eval test")
    ## Prepare eval dataset
    maxquant_result_ref_eval_df = maxquant_result_ref.loc[
        maxquant_result_ref["source"] != "ref"
    ]
    maxquant_result_ref_eval_df_sample = maxquant_result_ref_eval_df.sample(2000)
    training_file_paths = prepare_training_dataset(
        result_dir=cfg.RESULT_PATH,
        maxquant_dict=maxquant_result_ref_eval_df_sample,
        n_workers=cfg.N_CPU,
        include_decoys=cfg.PEAK_SELECTION.INCLUDE_DECOYS,
        source=cfg.PEAK_SELECTION.TRAINING_DATA_SOURCE,
    )
    cfg.PEAK_SELECTION.TRAINING_DATA = training_file_paths
    from peak_detection_2d.dataset.dataset import MultiHDF5_MaskDataset
    from peak_detection_2d.dataset.dataset import build_transformation
    import torch

    transformation, cfg_peak_selection_transferred.DATASET = build_transformation(
        cfg_peak_selection_transferred.DATASET
    )
    test_dataset = MultiHDF5_MaskDataset(
        training_file_paths, use_hint_channel=True, transforms=transformation
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg_peak_selection_transferred.DATASET.TEST_BATCH_SIZE,
        shuffle=True,
    )
    # Testset setup eval
    from peak_detection_2d.train import testset_eval
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    testset_eval(
        best_seg_model_path=cfg_peak_selection_transferred.MODEL.RESUME_PATH,
        best_cls_model_path=cfg_peak_selection_transferred.CLSMODEL.RESUME_PATH,
        cfg_cls_model=cfg_peak_selection_transferred.CLSMODEL,
        cfg_seg_model=cfg_peak_selection_transferred.MODEL,
        test_dataset=test_dataset,
        test_dataloader=test_dataloader,
        maxquant_result_ref=maxquant_result_ref,
        result_dir=ps_exp_dir,
        device=device,
        exp=False,
    )
else:
    logging.info("Skip eval test")

# ====================== Infer with transferred model ======================
# Infering
from peak_detection_2d.infer_on_pept_act import infer_on_pept_act
import torch

# ps_exp_dir = os.path.join(cfg.RESULT_PATH, "peak_selection", "eval_B_5136")
device = "cuda" if torch.cuda.is_available() else "cpu"
cfg.PEAK_SELECTION = cfg_peak_selection_transferred
logging.info("Infering with transferred model from %s", peak_selection_config_path)
infer_on_pept_act(
    cfg=cfg,
    best_seg_model_path=cfg_peak_selection_transferred.MODEL.RESUME_PATH,
    best_cls_model_path=cfg_peak_selection_transferred.CLSMODEL.RESUME_PATH,
    maxquant_dict=maxquant_result_ref,
    ps_exp_dir=ps_exp_dir,
    sigmoid_cls_score=True,
)
logging.info("Infering with transferred model done")


# ====================== Postprocessing ======================
# Proper deduplication and signal competition
pept_act_sum_ps = pd.read_csv(
    os.path.join(ps_exp_dir, "pept_act_sum_ps.csv")
)  # TODO: fill in
maxquant_result_ref = pd.read_pickle(cfg.DICT_PICKLE_PATH)
pept_act_sum_ps["target_decoy_score"].fillna(
    pept_act_sum_ps["target_decoy_score"].min(), inplace=True
)  # fillna with min score
from peak_detection_2d.utils import (
    compete_target_decoy_pair,
)
from postprocessing.fdr import generate_signal_compete_pairs
from postprocessing.compete_signal import compete_candidates_for_signal


pept_act_sum_ps_full, pept_act_sum_ps_full_tdc = compete_target_decoy_pair(
    pept_act_sum_ps,
    maxquant_result_ref,
)
maxquant_result_ref_tdc = pd.merge(
    left=pept_act_sum_ps_full_tdc, right=maxquant_result_ref, on=["mz_rank", "Decoy"]
)
signal_compete_tdc = generate_signal_compete_pairs(
    maxquant_dict=maxquant_result_ref_tdc, groupby_columns="mz_bin"
)
pept_act_sum_ps_tdc_all, result_after_compete, result_filtered = (
    compete_candidates_for_signal(
        result=signal_compete_tdc,
        pept_act_sum_ps=pept_act_sum_ps_full_tdc,
        log_sum_intensity_thres=2,
        delta_log_sum_intensity_thres=0.01,
    )
)
## Get number of isolated decoys
from postprocessing.fdr import get_isolated_decoys_from_pairs
from postprocessing.fdr import get_isolated_decoy_from_mzbins

signal_compete_all = generate_signal_compete_pairs(
    maxquant_dict=maxquant_result_ref, groupby_columns="mz_bin"
)
decoy_mz_ranks = set(maxquant_result_ref.loc[maxquant_result_ref["Decoy"], "mz_rank"])
isolated_decoys_set_pairs_all = get_isolated_decoys_from_pairs(
    result=signal_compete_all, decoy_mz_ranks=decoy_mz_ranks
)
isolated_decoys_mzbins_set = get_isolated_decoy_from_mzbins(
    maxquant_result_ref=maxquant_result_ref,
)
isolated_decoys_all = isolated_decoys_set_pairs_all.union(isolated_decoys_mzbins_set)
import pickle

# Create a dictionary to store the variables
variables = {
    "isolated_decoys_all": isolated_decoys_all,
    "isolated_decoys_mzbins_set": isolated_decoys_mzbins_set,
    "isolated_decoys_set_pairs_all": isolated_decoys_set_pairs_all,
}

# Save the variables to a file
with open(os.path.join(cfg.RESULT_PATH, "isolated_decoys.pkl"), "wb") as f:
    pickle.dump(variables, f)

with open(os.path.join(cfg.RESULT_PATH, "isolated_decoys.pkl"), "rb") as f:
    isolated_decoys = pickle.load(f)

## FDR eval and result analysis
pept_act_sum_ps_tdc_all_no_loser = pept_act_sum_ps_tdc_all.loc[
    pept_act_sum_ps_tdc_all["competition"] != "loser"
]
pept_act_sum_ps_tdc_all_no_loser_int_filter = pept_act_sum_ps_tdc_all_no_loser.loc[
    pept_act_sum_ps_tdc_all_no_loser["log_sum_intensity"] >= 2
]

# ====================== Result Analysis ======================
# Number of decoys and targets
td_count = pept_act_sum_ps_tdc_all_no_loser_int_filter["Decoy"].value_counts()
# Number of isolated decoys
n_filtered_isolated_decoys = (
    pept_act_sum_ps_tdc_all_no_loser_int_filter.loc[
        pept_act_sum_ps_tdc_all_no_loser_int_filter["Decoy"], "mz_rank"
    ]
    .isin(isolated_decoys["isolated_decoys_all"])
    .sum()
)
logging.info("Final FDR: %s%%", np.round(td_count[True] / td_count[False] * 100, 2))
logging.info(
    "Percentage of isolated decoys in all decoys: %s%%",
    np.round(len(isolated_decoys_all) / len(decoy_mz_ranks) * 100, 2),
)
logging.info(
    "Percentage of isolated decoys in filtered decoys: %s%%",
    np.round(n_filtered_isolated_decoys / td_count[True] * 100, 2),
)
# Plot full set
from peak_detection_2d.utils import (
    compete_target_decoy_pair,
    plot_target_decoy_distr,
    plot_roc_auc,
    calc_fdr_and_thres,
)

## Full set w/o TDC
plot_target_decoy_distr(
    pept_act_sum_ps_full,
    threshold=None,
    save_dir=os.path.join(ps_exp_dir, "results"),
    dataset_name="fullset",
    main_plot_type="scatter",
)
plot_roc_auc(
    pept_act_sum_ps_full,
    save_dir=os.path.join(ps_exp_dir, "results"),
    dataset_name="fullset",
)
pept_act_sum_ps_full_new = calc_fdr_and_thres(
    pept_act_sum_ps_full,
    score_col="target_decoy_score",
    filter_dict={"log_sum_intensity": [2, 100]},
    return_plot=True,
    save_dir=os.path.join(ps_exp_dir, "results"),
    dataset_name="fullset",
)
pept_act_sum_ps_full_new.to_csv(
    os.path.join(ps_exp_dir, "pept_act_sum_ps_full_fdr_thres.csv")
)

## Full set w TDC
plot_target_decoy_distr(
    pept_act_sum_ps_tdc_all_no_loser_int_filter,
    threshold=None,
    save_dir=os.path.join(ps_exp_dir, "results"),
    dataset_name="fullset_tdc",
    main_plot_type="scatter",
)
plot_roc_auc(
    pept_act_sum_ps_tdc_all_no_loser_int_filter,
    save_dir=os.path.join(ps_exp_dir, "results"),
    dataset_name="fullset_tdc",
)
pept_act_sum_ps_full_tdc_new = calc_fdr_and_thres(
    pept_act_sum_ps_tdc_all_no_loser_int_filter,
    score_col="target_decoy_score",
    filter_dict={"log_sum_intensity": [2, 100]},
    return_plot=True,
    save_dir=os.path.join(ps_exp_dir, "results"),
    dataset_name="fullset_tdc",
)
pept_act_sum_ps_full_tdc_new.to_csv(
    os.path.join(ps_exp_dir, "pept_act_sum_ps_full_tdc_fdr_thres.csv")
)
from result_analysis import result_analysis

logging.info("==================Result Analaysis==================")
if cfg.PEAK_SELECTION.ENABLE:
    eval_dir = os.path.join(ps_exp_dir, "results", "evaluation")
else:
    eval_dir = os.path.join(cfg.RESULT_PATH, "results", "evaluation")
act_dir = os.path.join(cfg.RESULT_PATH, "results", "activation")
pept_act_sum_df = pd.read_csv(os.path.join(act_dir, "pept_act_sum.csv"))
infer_int_col = "pept_act_sum"
# TODO: fix im filter config
if cfg.RESULT_ANALYSIS.POST_PROCESSING.FILTER_BY_IM:
    pept_act_sum_filter_by_im_df = pd.read_csv(
        os.path.join(act_dir, "pept_act_sum_filter_by_im.csv")
    )
    pept_act_sum_df = pd.merge(
        left=pept_act_sum_df,
        right=pept_act_sum_filter_by_im_df,
        on=["mz_rank"],
        how="left",
        suffixes=("", "_filter_by_im"),
    )
    infer_int_col = "pept_act_sum_filter_by_im"

if cfg.PEAK_SELECTION.ENABLE:
    pept_act_sum_ps = pd.read_csv(
        os.path.join(ps_exp_dir, "pept_act_sum_ps_full_tdc_fdr_thres.csv")
    )
    pept_act_sum_ps = pept_act_sum_ps.rename(
        {"sum_intensity": "sum_intensity_ps"}, axis=1
    )
    pept_act_sum_df = pd.merge(
        left=pept_act_sum_df,
        right=pept_act_sum_ps,
        on=["mz_rank"],
        how="left",
        suffixes=("", "_ps"),
    )
    infer_int_col = "sum_intensity_ps"

swaps_result = result_analysis.SWAPSResult(
    maxquant_dict=maxquant_result_ref,
    pept_act_sum_df=pept_act_sum_df,
    infer_intensity_col=infer_int_col,
    fdr_thres=0.3,
    log_sum_intensity_thres=cfg.RESULT_ANALYSIS.LOG_SUM_INTENSITY_THRESHOLD,
    save_dir=eval_dir,
    include_decoys=cfg.PREPARE_DICT.GENERATE_DECOY,
)
swaps_result.plot_intensity_corr()
swaps_result.plot_intensity_corr(contour=True)
swaps_result.plot_overlap_with_MQ(show_ref=False, level="precursor")
swaps_result.plot_overlap_with_MQ(show_ref=False, level="peptide")
swaps_result.plot_overlap_with_MQ(show_ref=False, level="protein")
