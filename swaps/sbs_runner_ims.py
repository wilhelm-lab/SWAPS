"""Module providing a function calling the scan by scan optimization."""

import logging
import os
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed as tpe_as_completed
from datetime import datetime
import time
import argparse
import yaml
import pandas as pd
import numpy as np
import directlfq.lfq_manager as lfq_manager

from utils.tools import get_dot_d_paths, report_snap_log_collection
from utils.ims_utils import (
    load_dotd_data,
    export_im_and_ms1scans,
)
from utils.config import get_cfg_defaults, merge_cfg_from_file
from utils.singleton_swaps_optimization import swaps_optimization_cfg
from optimization.inference import process_frames_parallel, generate_id_partitions
from prepare_dict.prepare_dict import (
    construct_dict_from_search_pivoted,
    dict_add_index_to_raw_file,
    dict_add_im_index,
    dict_add_rt_index,
    filter_maxquant_by_ok,
)
from prepare_dict.search_engine_output_parser import sage_parser, fragpipe_psm_parser
from postprocessing.rescore import (
    brew_with_mokapot,
    normalize_shift_by_runs,
    combine_matches_target_decoy,
)
from postprocessing.match_features import (
    match_features_batches_parallel,
)
from utils.plot import (
    calc_quant_corr,
    plot_match_type_from_combined,
    plot_quantification_by_run,
)
from postprocessing.helper import build_pivot, build_mz_sorted_activation
from postprocessing.direct_lfq import (
    reformat_swaps_combined_for_directlfq,
)

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
        merge_cfg_from_file(cfg, config_path)
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
                evidence = sage_parser(
                    evidence,
                    rt_window=cfg.PREPARE_DICT.SAGE.RT_WINDOW,
                    im_window=cfg.PREPARE_DICT.SAGE.IM_WINDOW,
                )
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
        if len(cfg.PREPARE_DICT.OK.DIR) > 0:
            evidence = filter_maxquant_by_ok(
                evidence_100fdr=evidence,  # type: ignore
                ok_dir=cfg.PREPARE_DICT.OK.DIR,
                ok_output_type=cfg.PREPARE_DICT.OK.OUTPUT,
                rescore_fdr=cfg.PREPARE_DICT.OK.FDR,
            )
            logging.info(
                "Evidence filtered by Oktoberfest rescoring: %d rows retained",
                len(evidence),
            )
        if cfg.EXCLUDE_DATASET_NAME:
            excluded_raw = {name.split(".")[0] for name in cfg.EXCLUDE_DATASET_NAME}
            before = len(evidence)  # type: ignore
            evidence = evidence.loc[~evidence["Raw file"].isin(excluded_raw)]  # type: ignore
            logging.info(
                "Excluded %d evidence rows from %s",
                before - len(evidence),
                excluded_raw,
            )
        dict_ref = construct_dict_from_search_pivoted(
            cfg_prepare_dict=cfg.PREPARE_DICT,
            evidence=evidence,  # type: ignore
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
        dot_d_paths = get_dot_d_paths(cfg.DATA_PATH, cfg.EXCLUDE_DATASET_NAME)
        for dot_d_path in dot_d_paths:
            dot_d_dir = os.path.basename(dot_d_path)
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
                    dot_d_path,
                    swaps_result_dir=cfg.EXPORT_DATA_HDF5_DIR,
                )
                ms1scans, mobility_values_df = export_im_and_ms1scans(
                    data=data,  # type: ignore
                    swaps_result_dir=os.path.join(cfg.RESULT_PATH, dir_wo_extension),
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
            logging.info("-----------------Scan by Scan Optimization-----------------")

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
    raw_file_list = [
        d
        for d in os.listdir(cfg.RESULT_PATH)
        if os.path.isdir(os.path.join(cfg.RESULT_PATH, d))
        and not d.startswith("quantification")
    ]

    # One-time preprocessing: merge frame-partitioned parquets into a single
    # mz_rank-sorted parquet per directory so workers can skip row groups.
    # build_mz_sorted_activation is idempotent — skips if the file exists.
    logging.info("Building mz-sorted activation parquets (skips if already done)...")
    act_dirs = [
        os.path.join(cfg.RESULT_PATH, raw_file, "activation")
        for raw_file in raw_file_list
    ]

    with ThreadPoolExecutor(
        max_workers=min(
            5, min(len(act_dirs), cfg.N_CPU)
        )  # TODO: dirty fix to avoid OOM
    ) as tpe:
        futures = {tpe.submit(build_mz_sorted_activation, d): d for d in act_dirs}
        for fut in tpe_as_completed(futures):
            d = futures[fut]
            exc = fut.exception()
            if exc:
                logging.error("build_mz_sorted_activation failed for %s: %s", d, exc)
            else:
                logging.info("Done: %s", d)

    (
        matches_target,
        matches_decoy,
        pp_reference,
        pp_match_target,
        pp_match_decoy,
        df_no_quant,
        df_no_match,
        snap_log_collection,
    ) = match_features_batches_parallel(
        dict_ref=dict_ref,
        raw_file_list=raw_file_list,
        result_dir=cfg.RESULT_PATH,
        peptide_indicies=dict_ref["mz_rank"].values,  # type: ignore
        batch_size_max=1500,
        max_workers=cfg.N_CPU,
        processing_kwargs=processing_kwargs,
    )
    quant_dir = os.path.join(cfg.RESULT_PATH, cfg.MATCH_FEATURES_KWARGS.dir_name)
    os.makedirs(quant_dir, exist_ok=True)
    dfs_to_save = {
        "no_quant_log.parquet": df_no_quant,
        "no_match_log.parquet": df_no_match,
        "matches_target.parquet": matches_target,
        "matches_decoy.parquet": matches_decoy,
        "pp_reference.parquet": pp_reference,
        "pp_match_target.parquet": pp_match_target,
        "pp_match_decoy.parquet": pp_match_decoy,
    }

    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = [
            ex.submit(df.to_parquet, os.path.join(quant_dir, fn), index=False)
            for fn, df in dfs_to_save.items()
        ]
        for f in futs:
            f.result()
    with open(os.path.join(quant_dir, "snap_log_collection.pkl"), "wb") as f:
        pickle.dump(snap_log_collection, f, protocol=pickle.HIGHEST_PROTOCOL)
    report_snap_log_collection(snap_log_collection, quant_dir)
    _save_effective_cfg(cfg, processing_kwargs, quant_dir)
    logging.info("=================FDR control==================")

    if cfg.FDR.INT_THRES > 0:
        pp_target_passing = pp_match_target.loc[
            pp_match_target["intensity_sum"] >= cfg.FDR.INT_THRES,
            ["feature_instance_id", "Run_name"],
        ].drop_duplicates()
        pp_decoy_passing = pp_match_decoy.loc[
            pp_match_decoy["intensity_sum"] >= cfg.FDR.INT_THRES,
            ["feature_instance_id", "Run_name"],
        ].drop_duplicates()
        valid_target = pd.MultiIndex.from_frame(pp_target_passing)
        valid_decoy = pd.MultiIndex.from_frame(pp_decoy_passing)
        matches_target = matches_target[
            pd.MultiIndex.from_arrays(
                [matches_target["feature_instance_id"], matches_target["matched_run"]]
            ).isin(valid_target)
        ]
        matches_decoy = matches_decoy[
            pd.MultiIndex.from_arrays(
                [matches_decoy["feature_instance_id"], matches_decoy["matched_run"]]
            ).isin(valid_decoy)
        ]
        logging.info(
            "After intensity filtering (>= %s): %d target, %d decoy matches",
            cfg.FDR.INT_THRES,
            len(matches_target),
            len(matches_decoy),
        )

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
            "zernike_similarities",
            "sift_distance",
            "zernike_distance",
            "template_matching_score",
        ],
        normalize_feature_cols=[
            "sift_similarities",
            "zernike_similarities",
            "sift_distance",
            "zernike_distance",
            "template_matching_score",
        ],
        scannr_col="feature_instance_id_with_label",  # No deduplication btw target and decoy for now
        psmid_col="Sequence_with_runs",
        specid_col="Sequence_with_runs",
        peptide_col="Sequence",
        protein_col="Proteins",
        decoy_col="Decoy",
        model=None,
        train_fdr=cfg.FDR.TRAIN,
        test_fdr=cfg.FDR.TEST,
        filename_col="matched_run",
        work_dir=quant_dir,
    )
    # Filter for the columns passed the makopot filter
    psms_filtered = psms.loc[(psms["mokapot q-value"] < 0.01) & (psms["label"] == True)]
    psms_filtered["feature_instance_id"] = psms_filtered["scannr"].str.replace(
        "_True", "", regex=True
    )
    psms_filtered["mz_rank"] = (
        psms_filtered["feature_instance_id"].str.split("_").str[0].astype(int)
    )
    pp_match_target_filtered = pp_match_target.merge(
        psms_filtered[["filename", "mz_rank", "feature_instance_id"]],
        left_on=["mz_rank", "Run_name", "feature_instance_id"],
        right_on=["mz_rank", "filename", "feature_instance_id"],
        how="inner",
    )
    # TODO: filter for intensity threshold
    pp_match_target_filtered.to_parquet(
        os.path.join(quant_dir, "pp_match_target_filtered.parquet"), index=False
    )
    dfs_to_concat = {
        "MBR": pp_match_target_filtered.drop(columns=["filename"]),
        "MS/MS Ref": pp_reference,
        # "MS/MS Quant": pp_quant_only,
    }
    pp_all = pd.DataFrame()
    for df_type, df in dfs_to_concat.items():
        df["Match Type"] = df_type
        pp_all = pd.concat([pp_all, df], ignore_index=True)
    pivot = build_pivot(pp_all, dict_ref)
    with ThreadPoolExecutor(max_workers=2) as ex:
        futs = [
            ex.submit(
                pp_all.to_parquet,
                os.path.join(
                    quant_dir, "pp_reference_quant_only_match_target_filtered.parquet"
                ),
                index=False,
            ),
            ex.submit(
                pivot.to_parquet, os.path.join(quant_dir, "swaps_combined_ions.parquet")
            ),
        ]
        for f in futs:
            f.result()

    logging.info("=================DirectLFQ Analysis==================")
    _ = reformat_swaps_combined_for_directlfq(
        pivot,
        dict_ref,
        output_dir=quant_dir,
        ion_id_col="mz_rank",
        protein_id_col="Proteins",
    )
    lfq_manager.run_lfq(input_file=os.path.join(quant_dir, "swaps.aq_reformat.tsv"))

    logging.info("=================Result Analysis==================")
    calc_quant_corr(
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
    plot_quantification_by_run(
        pivot,
        dataset_name="SWAPS_ions_raw",
        int_col_keyword="Intensity",
        label_char_range=(0, 10),
        fig_dir=quant_dir,
    )
    lfq_swaps_ion = pd.read_csv(
        os.path.join(quant_dir, "swaps.aq_reformat.tsv.ion_intensities.tsv"), sep="\t"
    )
    plot_quantification_by_run(
        lfq_swaps_ion,
        dataset_name="SWAPS_ions_directLFQ",
        label_char_range=(0, 14),
        id_cols=["protein", "ion"],
        fig_dir=quant_dir,
    )
    lfq_swaps_protein = pd.read_csv(
        os.path.join(quant_dir, "swaps.aq_reformat.tsv.protein_intensities.tsv"),
        sep="\t",
    )
    plot_quantification_by_run(
        lfq_swaps_protein,
        dataset_name="SWAPS_proteins_directLFQ",
        label_char_range=(0, 14),
        id_cols=["protein", "Protein"],
        fig_dir=quant_dir,
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
