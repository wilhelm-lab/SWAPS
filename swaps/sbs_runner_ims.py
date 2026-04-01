"""Module providing a function calling the scan by scan optimization."""

import logging
import os
from datetime import datetime
import time
import argparse
import yaml
import pandas as pd
import torch

# Disable cuDNN globally BEFORE anything else
torch.backends.cudnn.enabled = False

from utils.ims_utils import (
    load_dotd_data,
    export_im_and_ms1scans,
)
from utils.config import get_cfg_defaults
from utils.singleton_swaps_optimization import swaps_optimization_cfg
from optimization.inference import process_frames_parallel, generate_id_partitions
from prepare_dict.prepare_dict import (
    construct_dict_from_search_pivoted,
    dict_add_index_to_raw_file,
    dict_add_im_index,
    dict_add_rt_index,
)
from prepare_dict.search_engine_output_parser import sage_parser, fragpipe_psm_parser
from postprocessing.rescore import (
    brew_with_mokapot,
    normalize_shift_by_runs,
    combine_matches_target_decoy,
)
from postprocessing.match_features import (
    match_features_batches_parallel,
    calc_quant_corr,
    plot_match_type_from_combined,
)
from postprocessing.helper import build_pivot

# Clear existing logging handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


def _save_effective_cfg(cfg, processing_kwargs: dict, result_path: str):
    """Write the fully-resolved config (YACS + processing_kwargs) to result_path."""
    import yaml as _yaml

    cfg.defrost()
    out = _yaml.safe_load(cfg.dump())  # convert YACS to plain dict
    cfg.freeze()
    out["MATCH_FEATURES_KWARGS"] = processing_kwargs
    save_path = os.path.join(result_path, "effective_config.yaml")
    with open(save_path, "w") as f:
        _yaml.dump(out, f, default_flow_style=False, sort_keys=False)
    logging.info("Saved effective config to %s", save_path)


def opt_scan_by_scan(config_path: str):
    """Scan by scan optimization for joint identification and quantification."""

    # --------------Load config-----------------#
    cfg = get_cfg_defaults(swaps_optimization_cfg)  # type: ignore
    name_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if config_path is not None:
        cfg.merge_from_file(config_path)
        logging.info("merge with cfg file %s", config_path)
    processing_kwargs = yaml.safe_load(cfg.MATCH_FEATURES_KWARGS.dump())

    if cfg.ADD_TIMESTAMP_TO_RESULT_PATH:
        cfg.RESULT_PATH = cfg.RESULT_PATH + "_" + name_timestamp
        cfg.ADD_TIMESTAMP_TO_RESULT_PATH = False  # in case of reuse of config file
    os.makedirs(cfg.RESULT_PATH, exist_ok=True)
    if cfg.N_CPU < 0:
        cfg.N_CPU = int(
            os.getenv("SLURM_CPUS_PER_TASK", 1)
        )  # TODO: only work in slurm jobs
        logging.info("Number of CPUs: %s", cfg.N_CPU)
    if cfg.OPTIMIZATION.N_BATCH < 0:
        cfg.OPTIMIZATION.N_BATCH = cfg.N_CPU  # set batches as the same as N_CPU

    _save_effective_cfg(cfg, processing_kwargs, cfg.RESULT_PATH)

    # -------------Prepare general dictionary---------#
    try:
        dict_ref = pd.read_pickle(os.path.join(cfg.RESULT_PATH, "dict_ref.pkl"))
        logging.info("Loaded prepared dictionary with %s entries", len(dict_ref))
    except FileNotFoundError:
        logging.info("==================Prepare Dictionary==================")

        match cfg.PREPARE_DICT.SEARCH_ENGINE.lower():
            case "maxquant":
                evidence = pd.read_csv(
                    cfg.SEARCH_OUTPUT_PATH, sep="\t", low_memory=False
                )
            case "sage":
                evidence = pd.read_csv(
                    cfg.SEARCH_OUTPUT_PATH, sep="\t", low_memory=False
                )
                evidence = sage_parser(evidence)
            case "fragpipe":
                evidence = pd.DataFrame()
                for exp_dir in os.listdir(cfg.SEARCH_OUTPUT_PATH):
                    if (
                        os.path.isdir(os.path.join(cfg.SEARCH_OUTPUT_PATH, exp_dir))
                        and exp_dir != "MSBooster"
                    ):
                        tmp = pd.read_csv(
                            os.path.join(cfg.SEARCH_OUTPUT_PATH, exp_dir, "psm.tsv"),
                            sep="\t",
                        )
                        evidence = pd.concat([evidence, tmp], ignore_index=True)
                        logging.info(
                            "Loaded evidence with %s rows from %s", len(tmp), exp_dir
                        )

                evidence = fragpipe_psm_parser(evidence)
        dict_ref = construct_dict_from_search_pivoted(
            cfg_prepare_dict=cfg.PREPARE_DICT,
            evidence=evidence,
            n_blocks_by_pept=1,
        )
        dict_ref.to_pickle(os.path.join(cfg.RESULT_PATH, "dict_ref.pkl"))

    # -------------Scan-Wise Activation for each .d dataset------------#
    # added_im_and_rt_index = False
    if cfg.SWA:
        logging.info(
            "Processing Scan-Wise Activation for each .d dataset in %s excluding %s",
            cfg.DATA_PATH,
            cfg.EXCLUDE_DATASET_NAME,
        )
        for dot_d_dir in os.listdir(cfg.DATA_PATH):
            logging.info("process dataset: %s", dot_d_dir)
            if dot_d_dir.endswith(".d") and dot_d_dir not in cfg.EXCLUDE_DATASET_NAME:
                logging.info(
                    "============Scan-Wise Activation for dataset: %s============",
                    dot_d_dir,
                )
                logging.info(
                    "--------------------------Load Data--------------------------"
                )
                # Get the lowest level directory name with .d extension
                dir_wo_extension = dot_d_dir.split(".")[0]
                act_dir = os.path.join(cfg.RESULT_PATH, dir_wo_extension, "activation")
                os.makedirs(act_dir, exist_ok=True)

                # Load data
                if cfg.USE_IMS:
                    data, hdf_file_name = load_dotd_data(
                        os.path.join(cfg.DATA_PATH, dot_d_dir),
                        swaps_result_dir=cfg.EXPORT_DATA_HDF5_DIR,
                    )
                    ms1scans, mobility_values_df = export_im_and_ms1scans(
                        data=data,  # type: ignore
                        swaps_result_dir=os.path.join(
                            cfg.RESULT_PATH, dir_wo_extension
                        ),
                    )
                    dict_ref = dict_add_index_to_raw_file(
                        dict_ref,
                        mobility_values_df,
                        ms1scans,
                        dir_wo_extension,
                    )
                    # if not added_im_and_rt_index:
                    dict_ref = dict_add_im_index(
                        dict_ref,
                        mobility_values_df,
                        "IM_search_left",
                        "IM_search_center",
                        "IM_search_right",
                        idx_suffix=f"_ref_{dir_wo_extension}",
                    )
                    dict_ref = dict_add_rt_index(
                        dict_ref,
                        ms1scans,
                        idx_suffix=f"_ref_{dir_wo_extension}",
                    )
                    # added_im_and_rt_index = True

                    # Save the updated dict_ref with added activation info to the result directory for downstream processing
                    dict_ref.to_pickle(
                        os.path.join(cfg.RESULT_PATH, "dict_ref_with_activation.pkl")
                    )
                else:
                    raise NotImplementedError(
                        "Currently only support IMS data. Please set USE_IMS to True."
                    )
                    # mzml_data = load_mzml(cfg.DATA_PATH, unify_format=True)
                    # ms1scans = mzml_data
                    # mobility_values_df = None

                # Optimization
                start_time = time.time()
                logging.info(
                    "-----------------Scan by Scan Optimization-----------------"
                )

                n_batch = cfg.OPTIMIZATION.N_BATCH
                max_mz_rank = dict_ref.mz_rank.max()
                logging.info("Number of batches: %s", n_batch)
                batch_scan_indices = generate_id_partitions(
                    n_batch=n_batch,
                    id_array=ms1scans.index.values,
                    how="round_robin",
                )  # for small scale testing: ms1scans["Id"].iloc[0:500]
                logging.debug("indices in first batch: %s", batch_scan_indices[0])
                # process scans
                process_frames_parallel(
                    data=data,
                    n_jobs=cfg.N_CPU,
                    ms1scans=ms1scans,
                    parquet_file_stem=os.path.join(act_dir, "swa"),
                    batch_scan_indices=batch_scan_indices,
                    maxquant_result_ref_with_im_index_sortmz=dict_ref,
                    mobility_values=mobility_values_df,
                    # cutoff=cutoff,
                    delta_mobility_thres=cfg.OPTIMIZATION.DELTA_MOBILITY_INDEX_THRES,
                    ppm_tol=cfg.PREPARE_DICT.PPM_TOL,
                    bin_width=cfg.PREPARE_DICT.BIN_WIDTH,
                    process_in_blocks=True,
                    width=cfg.OPTIMIZATION.IM_PEAK_EXTRACTION_WIDTH,
                    # save_dir=os.path.join(act_dir, "target"),
                    extract_im_peak=False,
                    use_ims=cfg.USE_IMS,
                    return_res_coo_dict=False,
                    max_mz_rank=max_mz_rank,
                )
    else:
        logging.info(
            "Scan-Wise Activation is skipped as SWA is set to False in config. Please make sure *.parquet files exists in RESULT_PATH/<raw_file>/activation/ for downstream processing."
        )
        assert os.path.exists(
            os.path.join(cfg.RESULT_PATH, "dict_ref_with_activation.pkl")
        ), "dict_ref_with_activation.pkl not found, please run with SWA enabled or make sure the file exists for downstream processing."
        dict_ref = pd.read_pickle(
            os.path.join(cfg.RESULT_PATH, "dict_ref_with_activation.pkl")
        )

    # Once finished, write the updated dict_ref with added activation info to the result directory for downstream processing

    # -----------------Quantification and Feature-Feature Match--------------#
    logging.info(
        "==================Quantification and Feature-Feature Match=================="
    )
    (
        matches_target,
        matches_decoy,
        pp_reference,
        pp_quant_only,
        pp_match_target,
        pp_match_decoy,
        df_no_quant,
        df_no_match,
    ) = match_features_batches_parallel(
        dict_ref=dict_ref,
        raw_file_list=[
            d
            for d in os.listdir(cfg.RESULT_PATH)
            if os.path.isdir(os.path.join(cfg.RESULT_PATH, d))
            and not d.startswith("quantification")
        ],
        result_dir=cfg.RESULT_PATH,
        peptide_indicies=dict_ref["mz_rank"].values,  # type: ignore
        batch_size=100,
        max_workers=cfg.N_CPU,
        processing_kwargs=processing_kwargs,
    )
    quant_dir = os.path.join(cfg.RESULT_PATH, "quantification")
    os.makedirs(quant_dir, exist_ok=True)
    dfs_to_save = {
        "matches_target.csv": matches_target,
        "matches_decoy.csv": matches_decoy,
        "pp_reference.csv": pp_reference,
        "pp_quant_only.csv": pp_quant_only,
        "pp_match_target.csv": pp_match_target,
        "pp_match_decoy.csv": pp_match_decoy,
        "no_quant_log.csv": df_no_quant,
        "no_match_log.csv": df_no_match,
    }

    for filename, df in dfs_to_save.items():
        df.to_csv(os.path.join(quant_dir, filename), index=False)

    logging.info("=================FDR control==================")
    matches_target_normalized, matches_decoy_normalized = normalize_shift_by_runs(
        matches_target, matches_decoy
    )
    tdc_df = combine_matches_target_decoy(
        matches_target_normalized, matches_decoy_normalized, dict_ref
    )
    result, model, psms = brew_with_mokapot(
        tdc_df,
        feature_cols=[
            "im_shift_abs_scaled",
            "rt_shift_abs_scaled",
            "rt_shift",
            "im_shift",
            "im_shift_scaled",
            "rt_shift_scaled",
            "sift_similarities",
            "hu_similarities",
            "zernike_similarities",
            "sift_distance",
            "hu_distance",
            "zernike_distance",
            "template_matching_score",
        ],
        normalize_features=False,
        psmid_col="Sequence_with_runs",
        peptide_col="Sequence",
        protein_col="Proteins",
        decoy_col="Decoy",
        model=None,
        train_fdr=0.05,
        test_fdr=0.01,
        filename_col="matched_run",
        work_dir=quant_dir,
    )
    # Filter for the columns passed the makopot filter
    psms_filtered = psms.loc[(psms["mokapot q-value"] < 0.01) & (psms["label"] == True)]
    psms_filtered["mz_rank"] = psms_filtered["scannr"].str.split("_").str[0].astype(int)
    pp_match_target_filtered = pp_match_target.merge(
        psms_filtered[["filename", "mz_rank"]],
        left_on=["mz_rank", "Run_name"],
        right_on=["mz_rank", "filename"],
        how="inner",
    )
    pp_match_target_filtered.to_csv(
        os.path.join(quant_dir, "pp_match_target_filtered.csv"), index=False
    )
    dfs_to_concat = {
        "MBR": pp_match_target_filtered.drop(columns=["filename"]),
        "MS/MS Ref": pp_reference,
        "MS/MS Quant": pp_quant_only,
    }
    pp_all = pd.DataFrame()
    for df_type, df in dfs_to_concat.items():
        df["Match Type"] = df_type
        pp_all = pd.concat([pp_all, df], ignore_index=True)
    pp_all.to_csv(
        os.path.join(quant_dir, "pp_reference_quant_only_match_target_filtered.csv"),
        index=False,
    )
    pivot = build_pivot(pp_all, dict_ref)
    pivot.to_csv(os.path.join(quant_dir, "swaps_combined_ions.csv"))

    logging.info("=================Result Analysis==================")
    calc_quant_corr(
        pp_quant_only,
        pp_reference,
        pp_match_target_filtered,
        quant_dir,
    )
    plot_match_type_from_combined(df=pivot, fig_dir=quant_dir)
    try:
        # compare with IonQuant results
        combined_ionquant = pd.read_csv(
            os.path.join(cfg.SEARCH_OUTPUT_PATH, "combined_ion.tsv"),
            sep="\t",
        )
        plot_match_type_from_combined(
            df=combined_ionquant, fig_dir=quant_dir, fig_name_suffix="_ionquant"
        )
    except FileNotFoundError:
        logging.info(
            "IonQuant combined_ion.csv not found in search output path. Skipping comparison with IonQuant results. Skipping"
        )


def main():
    """
    Entry point for the CLI tool. Wraps the opt_scan_by_scan function.

    Args:
        config_path (str): Path to the configuration YAML file.
    """
    parser = argparse.ArgumentParser(description="Run ScanByScan processing.")
    parser.add_argument("config_path", help="Path to the configuration YAML file")
    args = parser.parse_args()

    # Call the actual function with the parsed argument
    opt_scan_by_scan(args.config_path)


if __name__ == "__main__":
    # fire.Fire(main)
    main()
