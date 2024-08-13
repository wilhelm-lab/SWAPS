"""Module providing a function calling the scan by scan optimization."""

import logging
import os
from datetime import datetime
import time
import fire
import pandas as pd
from utils.ims_utils import (
    load_dotd_data,
    export_im_and_ms1scans,
    combine_3d_act_and_sum_int,
)
from utils.config import get_cfg_defaults
from utils.singleton_swaps_optimization import swaps_optimization_cfg
from optimization.inference import process_ims_frames_parallel, generate_id_partitions
from peak_detection_2d.dataset.prepare_dataset import prepare_training_dataset
from peak_detection_2d.infer_on_pept_act import infer_on_pept_act
from peak_detection_2d.train import train
from result_analysis import result_analysis

# from result_analysis import result_analysis
from prepare_dict.prepare_dict import construct_dict, get_mzrank_batch_cutoff

# os.environ["NUMEXPR_MAX_THREADS"] = "8"


def opt_scan_by_scan(config_path: str):
    """Scan by scan optimization for joint identification and quantification."""
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    cfg = get_cfg_defaults(swaps_optimization_cfg)
    name_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if config_path is not None:
        cfg.merge_from_file(config_path)
        logging.info("merge with cfg file %s", config_path)
    if cfg.ADD_TIMESTAMP_TO_RESULT_PATH:
        cfg.RESULT_PATH = cfg.RESULT_PATH + "_" + name_timestamp
        cfg.ADD_TIMESTAMP_TO_RESULT_PATH = False  # in case of reuse of config file
    logging.info("==================Load data==================")
    os.makedirs(cfg.RESULT_PATH, exist_ok=True)
    if cfg.N_CPU < 0:
        cfg.N_CPU = int(os.getenv("SLURM_CPUS_PER_TASK", 1))
        logging.info("Number of CPUs: %s", cfg.N_CPU)
    if cfg.OPTIMIZATION.N_BATCH < 0:
        cfg.OPTIMIZATION.N_BATCH = cfg.N_CPU  # set batches as the same as N_CPU
    if cfg.DICT_PICKLE_PATH != "":
        maxquant_result_ref = pd.read_pickle(filepath_or_buffer=cfg.DICT_PICKLE_PATH)
    else:
        # Load data
        data, hdf_file_name = load_dotd_data(
            cfg.DATA_PATH, swaps_result_dir=cfg.EXPORT_DATA_HDF5_DIR
        )

        # Get the lowest level directory name with .d extension
        dir_with_extension = os.path.basename(os.path.normpath(cfg.DATA_PATH))
        if (
            cfg.FILTER_EXP_BY_RAW_FILE == ""
        ):  # if not specified, get the lowest level directory name with .d extension, by default None
            cfg.FILTER_EXP_BY_RAW_FILE = dir_with_extension.rstrip(".d")

        ms1scans, mobility_values_df = export_im_and_ms1scans(
            data=data, swaps_result_dir=cfg.RESULT_PATH
        )
        maxquant_result_exp = pd.read_csv(cfg.MQ_EXP_PATH, sep="\t", low_memory=False)
        maxquant_result_exp = maxquant_result_exp.loc[
            maxquant_result_exp["Raw file"] == cfg.FILTER_EXP_BY_RAW_FILE,
            :,
        ]
        maxquant_result_ref = pd.read_csv(cfg.MQ_REF_PATH, sep="\t", low_memory=False)
        # TODO filter ref df if needed

        maxquant_result_ref, dict_pickle_path, cfg_prepare_dict = construct_dict(
            cfg_prepare_dict=cfg.PREPARE_DICT,
            maxquant_exp_df=maxquant_result_exp,
            maxquant_ref_df=maxquant_result_ref,
            result_dir=os.path.join(cfg.RESULT_PATH),
            mobility_values_df=mobility_values_df,
            rt_values_df=ms1scans,
            random_seed=cfg.RANDOM_SEED,
            n_blocks_by_pept=cfg.OPTIMIZATION.N_BLOCKS_BY_PEPT,
        )
        peptact_shape = (
            (
                len(ms1scans.index.values)
                + 1,  # this index is rank, starting from 1, add 1 for the last frame
                len(mobility_values_df),
                len(maxquant_result_ref.mz_rank)
                + 1,  # this index is rank, starting from 1, add 1 for the last frame
            ),
        )
        cfg.PREPARE_DICT = cfg_prepare_dict
        cfg.DICT_PICKLE_PATH = dict_pickle_path
        cfg.OPTIMIZATION.PEPTACT_SHAPE = peptact_shape
        cfg.dump(
            stream=open(
                os.path.join(cfg.RESULT_PATH, f"config_{name_timestamp}.yaml"),
                "w",
                encoding="utf-8",
            )
        )
        logging.info(
            "Finished dictionary preparation and saved config to %s",
            os.path.join(cfg.RESULT_PATH, f"config_{name_timestamp}.yaml"),
        )
    act_dir = os.path.join(cfg.RESULT_PATH, "results", "activation")
    try:  # try and read results
        pept_act_sum_df = pd.read_csv(
            os.path.join(act_dir, "pept_act_sum.csv"), index_col=0
        )  # TODO: pept_act_sum is not the end
        if cfg.RESULT_ANALYSIS.POST_PROCESSING.FILTER_BY_IM:
            pept_act_sum_filter_by_im_df = pd.read_csv(
                os.path.join(act_dir, "pept_act_sum_filter_by_im.csv"), index_col=0
            )
        logging.info("Loaded pre-calculated optimization.")
    except FileNotFoundError:
        try:
            combine_3d_act_and_sum_int(
                n_blocks_by_pept=cfg.OPTIMIZATION.N_BLOCKS_BY_PEPT,
                n_batch=cfg.OPTIMIZATION.N_BATCH,
                act_dir=act_dir,
                remove_batch_file=False,
                calc_pept_act_sum_filter_by_im=cfg.RESULT_ANALYSIS.POST_PROCESSING.FILTER_BY_IM,
                maxquant_result_ref=maxquant_result_ref,
            )
            logging.info("Loaded pre-calculated activation")
        except FileNotFoundError:
            logging.info("Precalculated activation not found, start Scan By Scan.")

            logging.info("==================Scan By Scan==================")
            # act_dir = os.path.join(cfg.RESULT_PATH, "results", "activation")
            os.makedirs(act_dir, exist_ok=True)
            # Optimization
            start_time = time.time()
            logging.info("-----------------Scan by Scan Optimization-----------------")

            n_batch = cfg.OPTIMIZATION.N_BATCH
            logging.info("Number of batches: %s", n_batch)
            batch_scan_indices = generate_id_partitions(
                n_batch=n_batch,
                id_array=ms1scans.index.values,
                how="round_robin",
            )  # for small scale testing: ms1scans["Id"].iloc[0:500]
            logging.info("indices in first batch: %s", batch_scan_indices[0])
            # process scans
            cutoff = get_mzrank_batch_cutoff(maxquant_result_ref)
            process_ims_frames_parallel(
                data=data,
                n_jobs=cfg.N_CPU,
                ms1scans=ms1scans,
                batch_scan_indices=batch_scan_indices,
                maxquant_ref=maxquant_result_ref,
                mobility_values=mobility_values_df,
                cutoff=cutoff,
                delta_mobility_thres=cfg.OPTIMIZATION.DELTA_MOBILITY_INDEX_THRES,
                mz_bin_digits=cfg.PREPARE_DICT.MZ_BIN_DIGITS,
                process_in_blocks=True,
                width=cfg.OPTIMIZATION.IM_PEAK_EXTRACTION_WIDTH,
                save_dir=act_dir,
                return_im_pept_act=True,
                extract_im_peak=False,
            )

            minutes, seconds = divmod(time.time() - start_time, 60)
            logging.info(
                "Process scans - Script execution time: %dm %ds",
                int(minutes),
                int(seconds),
            )

            logging.info("=================Post Processing==================")
            # TODO: test when pept_batch_number > 1
            combine_3d_act_and_sum_int(
                n_blocks_by_pept=cfg.OPTIMIZATION.N_BLOCKS_BY_PEPT,
                n_batch=cfg.OPTIMIZATION.N_BATCH,
                act_dir=act_dir,
                remove_batch_file=False,
                calc_pept_act_sum_filter_by_im=cfg.RESULT_ANALYSIS.POST_PROCESSING.FILTER_BY_IM,
                maxquant_result_ref=maxquant_result_ref,
            )

    if cfg.PEAK_SELECTION.ENABLE:
        logging.info("==================Peak Selection==================")
        if len(cfg.PEAK_SELECTION.TRAINING_DATA) == 0:
            logging.info("No training data provided, start preparing training data.")
            training_file_paths = prepare_training_dataset(
                result_dir=cfg.RESULT_PATH,
                maxquant_dict=maxquant_result_ref,
                n_workers=cfg.N_CPU,
                include_decoys=cfg.PEAK_SELECTION.INCLUDE_DECOYS,
            )
            cfg.PEAK_SELECTION.TRAINING_DATA = training_file_paths
            cfg.dump(
                stream=open(
                    os.path.join(
                        cfg.RESULT_PATH,
                        f"config_{name_timestamp}.yaml",
                    ),
                    "w",
                    encoding="utf-8",
                )
            )
            logging.info(
                "Finished peak selection dataset preparation and saved config to %s",
                os.path.join(cfg.RESULT_PATH, f"config_{name_timestamp}.yaml"),
            )
        train_name_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        ps_exp_dir = os.path.join(
            cfg.RESULT_PATH, "peak_selection", "exp_" + train_name_timestamp
        )
        if not os.path.exists(ps_exp_dir):
            os.mkdir(ps_exp_dir)
        best_model_path = train(
            cfg_peak_selection=cfg.PEAK_SELECTION,
            ps_exp_dir=ps_exp_dir,
            random_state=cfg.RANDOM_SEED,
            maxquant_dict=maxquant_result_ref,
        )
        # Inference
        logging.info("Finished training peak selection model, start inference...")
        infer_on_pept_act(
            cfg=cfg,
            best_model_path=best_model_path,
            maxquant_dict=maxquant_result_ref,
            ps_exp_dir=ps_exp_dir,
        )
    if cfg.RESULT_ANALYSIS.ENABLE:  # TODO: haven't cleaned up the code
        logging.info("==================Result Analaysis==================")

        if cfg.RESULT_ANALYSIS.MQ_EXP_PATH == "":
            maxquant_result_exp = maxquant_result_ref
            logging.info(
                "Experiment data not given, using reference intensity as experiment"
                " data!"
            )
        elif cfg.RESULT_ANALYSIS.MQ_EXP_PATH[-4:] == ".txt":
            maxquant_result_exp = pd.read_csv(cfg.RESULT_ANALYSIS.MQ_EXP_PATH, sep="\t")
        elif cfg.RESULT_ANALYSIS.MQ_EXP_PATH[-4:] == ".pkl":
            maxquant_result_exp = pd.read_pickle(cfg.RESULT_ANALYSIS.MQ_EXP_PATH)
        elif cfg.RESULT_ANALYSIS.MQ_EXP_PATH[-4:] == ".csv":
            maxquant_result_exp = pd.read_csv(cfg.RESULT_ANALYSIS.MQ_EXP_PATH)

        if cfg.RESULT_ANALYSIS.FILTER_BY_RAW_FILE is not None:
            if cfg.RESULT_ANALYSIS.FILTER_BY_RAW_FILE == "":
                cfg.RESULT_ANALYSIS.FILTER_BY_RAW_FILE = dir_with_extension.rstrip(".d")
            logging.info(
                "Filtering experiment data by raw file: %s",
                cfg.RESULT_ANALYSIS.FILTER_BY_RAW_FILE,
            )
            maxquant_result_exp = maxquant_result_exp[
                maxquant_result_exp["Raw file"]
                == cfg.RESULT_ANALYSIS.FILTER_BY_RAW_FILE
            ]
        eval_dir = os.path.join(cfg.RESULT_PATH, "results", "evaluation")
        os.makedirs(eval_dir, exist_ok=True)
        # if "predicted_RT" not in maxquant_result_ref.columns:
        #     maxquant_result_ref["predicted_RT"] = maxquant_result_ref[
        #         "RT_search_center"
        #     ]

        pept_act_sum_df = pd.read_csv(os.path.join(act_dir, "pept_act_sum.csv"))
        pept_act_sum_df_list = [pept_act_sum_df]
        if cfg.RESULT_ANALYSIS.POST_PROCESSING.FILTER_BY_IM:
            pept_act_sum_filter_by_im_df = pd.read_csv(
                os.path.join(act_dir, "pept_act_sum_filter_by_im.csv")
            )
            pept_act_sum_df_list.append(pept_act_sum_filter_by_im_df)

        sbs_result = result_analysis.SBSResult(
            maxquant_ref_df=maxquant_result_ref,
            # maxquant_merge_df=maxquant_result_ref,
            maxquant_exp_df=maxquant_result_exp,
            filter_by_rt_ovelap=cfg.RESULT_ANALYSIS.FILTER_BY_RT_OVERLAP,
            pept_act_sum_df_list=pept_act_sum_df_list,
            ims=True,
            save_dir=eval_dir,
        )
        for col in sbs_result.sum_cols:
            if col != "mz_rank":
                sbs_result.plot_intensity_corr(
                    ref_col="Intensity",
                    inf_col=col,
                    contour=False,
                    # save_dir=None,
                    # interactive = True, hover_data = ['Modified sequence', 'Charge', 'mz_rank']
                )

        # Overlap with MQ
        sbs_result.plot_overlap_with_MQ(show_ref=True)


if __name__ == "__main__":
    fire.Fire(opt_scan_by_scan)
