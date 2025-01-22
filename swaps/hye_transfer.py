import logging

import os
import pandas as pd
import fire
import sparse

from peak_detection_2d.dataset.prepare_dataset import (
    prepare_training_dataset,
    generate_hint_sparse_matrix,
)
from sbs_runner_ims import opt_scan_by_scan
from utils.config import get_cfg_defaults
from utils.singleton_swaps_optimization import swaps_optimization_cfg


def hye_model_transfer(
    swaps_result_folder,
    eval_test: bool = True,
    mix_a_folder: str = "120min_library_im_ref_A_5135_20240923_100204_771863",
    mix_b_folder: str = "120min_library_im_ref_B_5136_20240925_143619_467125",
):

    ## Experiment data
    timestamp = (
        swaps_result_folder.split("_")[-3]
        + "_"
        + swaps_result_folder.split("_")[-2]
        + "_"
        + swaps_result_folder.split("_")[-1]
    )
    logging.info("Timestamp extracted from swaps result folder: %s", timestamp)
    mixture = swaps_result_folder.split("ref_")[1][0]
    logging.info("swaps result folder %s with mixture %s", swaps_result_folder, mixture)

    swaps_config_path = os.path.join(
        "/cmnfs/proj/ORIGINS/SWAPS_exp/mixture",
        swaps_result_folder,
        f"config_{timestamp}.yaml",
    )
    if mixture == "A":
        ps_model_swaps_folder = mix_a_folder
    elif mixture == "B":
        ps_model_swaps_folder = mix_b_folder
    peak_selection_config_path = os.path.join(
        "/cmnfs/proj/ORIGINS/SWAPS_exp/mixture/",
        ps_model_swaps_folder,
        "peak_selection/cls_with_seg_output_dropout/updated_peak_selection_config.yaml",
    )
    ps_exp_dir = "eval_model_transfer"

    cfg = get_cfg_defaults(swaps_optimization_cfg)
    cfg.merge_from_file(swaps_config_path)
    # cfg_peak_selection_transferred = cfg.PEAK_SELECTION
    cfg.PEAK_SELECTION.merge_from_file(peak_selection_config_path)
    cfg.PEAK_SELECTION.EXP_DIR_NAME = ps_exp_dir
    ps_exp_dir = os.path.join(cfg.RESULT_PATH, "peak_selection", ps_exp_dir)
    os.makedirs(ps_exp_dir, exist_ok=True)
    try:
        ## Run swaps
        opt_scan_by_scan(
            os.path.join(
                ps_exp_dir,
                "config_eval_model_transfer.yaml",
            )
        )
    except FileNotFoundError:
        logging.info("Config file not found, generating...")
        ## Load data
        maxquant_result_ref = pd.read_pickle(cfg.DICT_PICKLE_PATH)

        ## Prepare hint matrix
        if os.path.isfile(
            os.path.join(
                cfg.RESULT_PATH, "peak_selection", "training_data", "hint_matrix.npz"
            )
        ):
            logging.info("Hint matrix already exists")
        else:
            os.makedirs(
                os.path.join(cfg.RESULT_PATH, "peak_selection", "training_data"),
                exist_ok=True,
            )
            hint_matrix = generate_hint_sparse_matrix(
                maxquant_dict_df=maxquant_result_ref,
                shape=cfg.OPTIMIZATION.PEPTACT_SHAPE[0],
            )
            sparse.save_npz(
                os.path.join(
                    cfg.RESULT_PATH,
                    "peak_selection",
                    "training_data",
                    "hint_matrix.npz",
                ),
                hint_matrix,
            )

        ## Prepare test data
        if eval_test:
            logging.info("Start eval test")
            ## Prepare eval dataset
            maxquant_result_ref_eval_df = maxquant_result_ref.loc[
                maxquant_result_ref["source"] != "ref"
            ]
            maxquant_result_ref_eval_df_sample = maxquant_result_ref_eval_df.sample(
                2000
            )
            logging.info("No training data found, preparing training data")
            training_file_paths = prepare_training_dataset(
                result_dir=cfg.RESULT_PATH,
                maxquant_dict=maxquant_result_ref_eval_df_sample,
                n_workers=cfg.N_CPU,
                include_decoys=cfg.PEAK_SELECTION.INCLUDE_DECOYS,
                source=cfg.PEAK_SELECTION.TRAINING_DATA_SOURCE,
                dataset_name="eval_transfer_sample_datapoints_TD",
            )
            cfg.PEAK_SELECTION.TRAINING_DATA = training_file_paths
            cfg.dump(
                stream=open(
                    os.path.join(
                        ps_exp_dir,
                        "config_eval_model_transfer.yaml",
                    ),
                    "w",
                    encoding="utf-8",
                )
            )
            ## Run swaps
            opt_scan_by_scan(
                os.path.join(
                    ps_exp_dir,
                    "config_eval_model_transfer.yaml",
                )
            )


if __name__ == "__main__":
    fire.Fire(hye_model_transfer)
