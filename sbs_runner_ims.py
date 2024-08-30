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
from peak_detection_2d.utils import (
    compete_target_decoy_pair,
    plot_target_decoy_distr,
    plot_roc_auc,
    calc_fdr_and_thres,
)
from result_analysis import result_analysis
from prepare_dict.prepare_dict import construct_dict, get_mzrank_batch_cutoff


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
            len(cfg.FILTER_EXP_BY_RAW_FILE) == 0
        ):  # if not specified, get the lowest level directory name with .d extension, by default None
            cfg.FILTER_EXP_BY_RAW_FILE.append(dir_with_extension.rstrip(".d"))

        ms1scans, mobility_values_df = export_im_and_ms1scans(
            data=data, swaps_result_dir=cfg.RESULT_PATH
        )
        # maxquant_result_exp = pd.read_csv(cfg.MQ_EXP_PATH, sep="\t", low_memory=False)
        # maxquant_result_exp = maxquant_result_exp.loc[
        #     maxquant_result_exp["Raw file"] == cfg.FILTER_EXP_BY_RAW_FILE,
        #     :,
        # ]
        maxquant_result_ref = pd.read_csv(cfg.MQ_REF_PATH, sep="\t", low_memory=False)
        # TODO filter ref df if needed

        maxquant_result_ref, dict_pickle_path, cfg_prepare_dict = construct_dict(
            cfg_prepare_dict=cfg.PREPARE_DICT,
            filter_exp_by_raw_file=cfg.FILTER_EXP_BY_RAW_FILE,
            maxquant_exp_path=cfg.MQ_EXP_PATH,
            # maxquant_exp_df=maxquant_result_exp,
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

        # Inference eval
        if cfg.PREPARE_DICT.GENERATE_DECOY:
            logging.info(
                "==================Peak Selection and FDR eval on full dataset=================="
            )
            pept_act_sum_ps = pd.read_csv(
                os.path.join(ps_exp_dir, "pept_act_sum_ps.csv")
            )
            pept_act_sum_ps_full, pept_act_sum_ps_full_tdc = compete_target_decoy_pair(
                pept_act_sum_ps,
                maxquant_result_ref,
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
                filter_dict={"log_sum_intensity": [1, 100]},
                return_plot=True,
                save_dir=os.path.join(ps_exp_dir, "results"),
                dataset_name="fullset",
            )
            pept_act_sum_ps_full_new.to_csv(
                os.path.join(ps_exp_dir, "pept_act_sum_ps_full_fdr_thres.csv")
            )

            ## Full set w TDC
            plot_target_decoy_distr(
                pept_act_sum_ps_full_tdc,
                threshold=None,
                save_dir=os.path.join(ps_exp_dir, "results"),
                dataset_name="fullset_tdc",
                main_plot_type="scatter",
            )
            plot_roc_auc(
                pept_act_sum_ps_full_tdc,
                save_dir=os.path.join(ps_exp_dir, "results"),
                dataset_name="fullset_tdc",
            )
            pept_act_sum_ps_full_tdc_new = calc_fdr_and_thres(
                pept_act_sum_ps_full_tdc,
                score_col="target_decoy_score",
                filter_dict={"log_sum_intensity": [1, 100]},
                return_plot=True,
                save_dir=os.path.join(ps_exp_dir, "results"),
                dataset_name="fullset_tdc",
            )
            pept_act_sum_ps_full_tdc_new.to_csv(
                os.path.join(ps_exp_dir, "pept_act_sum_ps_full_tdc_fdr_thres.csv")
            )

    if cfg.RESULT_ANALYSIS.ENABLE:  # TODO: haven't cleaned up the code
        logging.info("==================Result Analaysis==================")
        if cfg.PEAK_SELECTION.ENABLE:
            eval_dir = os.path.join(ps_exp_dir, "results", "evaluation")
        else:
            eval_dir = os.path.join(cfg.RESULT_PATH, "results", "evaluation")
        os.makedirs(eval_dir, exist_ok=True)

        pept_act_sum_df = pd.read_csv(os.path.join(act_dir, "pept_act_sum.csv"))
        infer_int_col = "pept_act_sum"
        # TODO: fix im filter config
        if cfg.RESULT_ANALYSIS.POST_PROCESSING.FILTER_BY_IM:
            pept_act_sum_filter_by_im_df = pd.read_csv(
                os.path.join(act_dir, "pept_act_sum_filter_by_im.csv")
            )
            pept_act_sum_filter_by_im_df = pept_act_sum_filter_by_im_df.rename(
                {"sum_intensity": "sum_intensity_filter_by_im"}, axis=1
            )
            pept_act_sum_df = pd.merge(
                left=pept_act_sum_df,
                right=pept_act_sum_filter_by_im_df,
                on=["mz_rank"],
                how="left",
                suffixes=("", "_filter_by_im"),
            )
            infer_int_col = "sum_intensity_filter_by_im"

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
            fdr_thres=cfg.RESULT_ANALYSIS.FDR_THRESHOLD,
            log_sum_intensity_thres=1,
            save_dir=eval_dir,
        )
        swaps_result.plot_intensity_corr()
        swaps_result.plot_overlap_with_MQ(show_ref=False, level="precursor")
        swaps_result.plot_overlap_with_MQ(show_ref=False, level="peptide")
        swaps_result.plot_overlap_with_MQ(show_ref=False, level="protein")


if __name__ == "__main__":
    fire.Fire(opt_scan_by_scan)
