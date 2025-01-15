"""Module providing a function calling the scan by scan optimization."""

import logging
import os

import time
from multiprocessing import cpu_count
from typing import Literal
import fire
import numpy as np
import pandas as pd

import postprocessing.no_ims_2d as no_ims_2d
from postprocessing.peak_selection import match_peaks_to_exp
from utils.tools import load_mzml
from utils.config_json import Config
from optimization.inference import process_scans_parallel
from result_analysis import result_analysis

os.environ["NUMEXPR_MAX_THREADS"] = "32"


def _define_rt_search_range(
    maxquant_result_dict: pd.DataFrame,
    rt_tol: float,
    rt_ref: Literal["exp", "pred", "mix"],
):
    """Define the search range for the precursor RT."""
    if rt_ref == "exp":
        maxquant_result_dict["RT_search_left"] = (
            maxquant_result_dict["Calibrated retention time start"] - rt_tol
        )
        maxquant_result_dict["RT_search_right"] = (
            maxquant_result_dict["Calibrated retention time finish"] + rt_tol
        )
        rt_ref_act_peak = "Calibrated retention time"
    elif rt_ref == "pred":
        maxquant_result_dict["RT_search_left"] = (
            maxquant_result_dict["predicted_RT"] - rt_tol
        )
        maxquant_result_dict["RT_search_right"] = (
            maxquant_result_dict["predicted_RT"] + rt_tol
        )
        rt_ref_act_peak = "predicted_RT"
    elif rt_ref == "mix":
        maxquant_result_dict["RT_search_left"] = (
            maxquant_result_dict["Retention time new"] - rt_tol
        )
        maxquant_result_dict["RT_search_right"] = (
            maxquant_result_dict["Retention time new"] + rt_tol
        )
        rt_ref_act_peak = "Retention time new"
    maxquant_result_dict["RT_search_center"] = maxquant_result_dict[rt_ref_act_peak]
    return maxquant_result_dict


def _merge_activation_results(
    processed_scan_dict: dict, ref_id: pd.Series, n_ms1scans: int
):
    """Merge the activation results."""
    activation = pd.DataFrame(index=ref_id, columns=range(n_ms1scans))
    precursor_scan_cos_dist = pd.DataFrame(index=ref_id, columns=range(n_ms1scans))
    precursor_collinear_sets = pd.DataFrame(index=ref_id, columns=range(n_ms1scans))
    scan_record_list = []
    for scan_idx, result_dict_scan in processed_scan_dict.items():
        if result_dict_scan["activation"] is not None:
            activation.loc[result_dict_scan["activation"]["precursor"], scan_idx] = (
                result_dict_scan["activation"]["activation"]
            )
        if result_dict_scan["precursor_cos_dist"] is not None:
            precursor_scan_cos_dist.loc[
                result_dict_scan["precursor_cos_dist"]["precursor"], scan_idx
            ] = result_dict_scan["precursor_cos_dist"]["cos_dist"]
        if result_dict_scan["precursor_collinear_sets"] is not None:
            precursor_collinear_sets.loc[
                result_dict_scan["precursor_collinear_sets"]["precursor"], scan_idx
            ] = result_dict_scan["precursor_collinear_sets"]["collinear_candidates"]
        scan_record_list.append(result_dict_scan["scans_record"])
    scan_record = pd.DataFrame(
        scan_record_list,
        columns=[
            "Scan",
            "Time",
            "CandidatePrecursorByRT",
            "FilteredPrecursor",
            "NumberHighlyCorrDictCandidate",
            "BestAlpha",
            "Cosine Dist",
            "IntensityExplained",
        ],
    )
    return activation, precursor_scan_cos_dist, scan_record, precursor_collinear_sets


def opt_scan_by_scan(config_path: str):
    """Scan by scan optimization for joint identification and quantification."""
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    conf = Config(config_path)
    conf.make_result_dirs()

    # start analysis
    start_time_init = time.time()
    logging.info("==================Load data==================")

    # Load data
    maxquant_result_ref = pd.read_pickle(filepath_or_buffer=conf.mq_ref_path)
    maxquant_result_exp = pd.read_csv(filepath_or_buffer=conf.mq_exp_path, sep="\t")
    ms1scans = load_mzml(msconvert_file=conf.mzml_path)
    n_ms1scans = ms1scans.shape[0]
    minutes, seconds = divmod(time.time() - start_time_init, 60)
    logging.info("Script execution time: %dm %ds", int(minutes), int(seconds))

    # deifne RT search range
    maxquant_result_ref = _define_rt_search_range(
        maxquant_result_ref, conf.rt_tol, conf.rt_ref
    )
    maxquant_result_ref.to_pickle(
        os.path.join(conf.result_dir, "maxquant_result_ref.pkl")
    )
    ref_id = maxquant_result_ref["id"]
    try:  # try and read results
        scan_record = pd.read_pickle(conf.output_file + "_scan_record.pkl")
        activation = np.load(conf.output_file + "_activationByScanFromLasso.npy")
        logging.info("Load pre-calculated optimization.")
    except FileNotFoundError:
        logging.info("Precalculated optimization not found, start Scan By Scan.")
        logging.info("==================Scan By Scan==================")
        # Optimization
        start_time = time.time()
        logging.info("-----------------Scan by Scan Optimization-----------------")

        # process scans
        processed_scan_dict = process_scans_parallel(
            n_jobs=cpu_count(),
            ms1scans=ms1scans,  # for small scale testing: MS1Scans.iloc[1000:1050, :]
            maxquant_ref=maxquant_result_ref,
            loss="lasso",
            opt_algo=conf.opt_algo,
            alphas=conf.alphas,
            alpha_criteria=conf.alpha_criteria,
            abundance_missing_threshold=conf.iso_ab_mis_thres,
            return_precursor_scan_cos_dist=conf.peak_sel_cos_dist,
        )

        minutes, seconds = divmod(time.time() - start_time, 60)
        logging.info(
            "Process scans - Script execution time: %dm %ds", int(minutes), int(seconds)
        )

        # merge results
        (
            activation,
            precursor_scan_cos_dist,
            scan_record,
            precursor_collinear_sets,
        ) = _merge_activation_results(processed_scan_dict, ref_id, n_ms1scans)

        minutes, seconds = divmod(time.time() - start_time, 60)
        logging.info(
            "Merge results - Script execution time: %dm %ds", int(minutes), int(seconds)
        )

        # save results
        activation = activation.fillna(0)
        np.save(conf.output_file + "_activationByScanFromLasso.npy", activation.values)
        np.save(
            conf.output_file + "_collinearPrecursors", precursor_collinear_sets.values
        )
        if conf.peak_sel_cos_dist:
            precursor_scan_cos_dist = precursor_scan_cos_dist.fillna(0)
            np.save(
                conf.output_file + "_precursor_scan_CosDist.npy",
                precursor_scan_cos_dist.values,
            )
        scan_record.to_pickle(conf.output_file + "_scan_record.pkl")

        minutes, seconds = divmod(time.time() - start_time, 60)
        logging.info(
            "Save results - Script execution time: %dm %ds", int(minutes), int(seconds)
        )

    logging.info("=================Post Processing==================")

    # calc activation sum w/o smoothing, w/ Gaussian and local minima smoothing
    ms1cans_no_array = pd.read_csv(
        os.path.join(conf.dirname, conf.ms1scans_no_array_name)
    )
    try:
        sum_raw = pd.read_csv(os.path.join(conf.result_dir, "sum_raw.csv"))
    except FileNotFoundError:
        _, sum_raw = no_ims_2d.smooth_act_mat(
            activation=activation,
            ms1scans_no_array=ms1cans_no_array,
            method="Raw",
        )
        sum_raw.to_csv(os.path.join(conf.result_dir, "sum_raw.csv"), index=False)

    try:
        refit_activation_minima = np.load(conf.output_file + "_activationMinima.npy")
        sum_minima = pd.read_csv(os.path.join(conf.result_dir, "sum_minima.csv"))
    except FileNotFoundError:
        refit_activation_minima, sum_minima = no_ims_2d.smooth_act_mat(
            activation=activation,
            ms1scans_no_array=ms1cans_no_array,
            method="LocalMinima",
        )
        np.save(conf.output_file + "_activationMinima.npy", refit_activation_minima)
        sum_minima.to_csv(os.path.join(conf.result_dir, "sum_minima.csv"), index=False)

    try:
        refit_activation_gaussian = np.load(
            conf.output_file + "_activationGaussian.npy"
        )
        sum_gaussian = pd.read_csv(os.path.join(conf.result_dir, "sum_gaussian.csv"))
    except FileNotFoundError:
        (
            refit_activation_gaussian,
            sum_gaussian,
        ) = no_ims_2d.smooth_act_mat(
            activation=activation,
            ms1scans_no_array=ms1cans_no_array,
            method="GaussianKernel",
        )
        np.save(conf.output_file + "_activationGaussian.npy", refit_activation_gaussian)
        sum_gaussian.to_csv(
            os.path.join(conf.result_dir, "sum_gaussian.csv"), index=False
        )

    # Elution peak preservation
    try:
        sum_peak = pd.read_csv(os.path.join(conf.result_dir, "sum_peak.csv"))
        peak_results = pd.read_csv(os.path.join(conf.result_dir, "peak_results.csv"))
    except FileNotFoundError:
        sum_peak, peak_results = no_ims_2d.select_peak_from_activation(
            maxquant_result_ref=maxquant_result_ref,
            ms1scans_no_array=ms1cans_no_array,
            activation=refit_activation_minima,
            return_peak_result=True,  # default find peaks setting, minimal peak_width = 2
        )
        sum_peak.to_csv(os.path.join(conf.result_dir, "sum_peak.csv"), index=False)
        peak_results.to_csv(
            os.path.join(conf.result_dir, "peak_results.csv"), index=False
        )

    logging.debug(
        "dimension of sum_raw, sum_gaussiam, sum_minima, sum_peak: %s, %s, %s, %s",
        sum_raw.shape,
        sum_gaussian.shape,
        sum_minima.shape,
        sum_peak.shape,
    )

    logging.info("==================Result Analaysis==================")
    sbs_result = result_analysis.SBSResult(
        maxquant_ref_df=maxquant_result_ref,
        maxquant_exp_df=maxquant_result_exp,
        pept_act_sum_df_list=sum_raw,
        sum_gaussian=sum_gaussian,
        sum_minima=sum_minima,
        sum_peak=sum_peak,
    )

    sbs_result.compare_with_maxquant_exp_int(
        filter_by_rt_overlap=None, handle_mul_exp_pcm="drop", save_dir=conf.report_dir
    )
    merged_df = sbs_result.ref_exp_df_inner

    peak_results_matched = match_peaks_to_exp(
        ref_exp_inner_df=merged_df, peak_results=peak_results
    )
    peak_results_matched.to_csv(
        os.path.join(conf.result_dir, "peak_results_matched.csv")
    )

    # Correlation
    for sum_col in sbs_result.sum_cols:
        sbs_result.plot_intensity_corr(
            inf_col=sum_col, interactive=False, save_dir=conf.report_dir
        )

    # Overlap with MQ
    sbs_result.plot_overlap_with_MQ(save_dir=conf.report_dir)

    # evaluate target and decoy
    sbs_result.eval_target_decoy(save_dir=conf.report_dir)

    # selected alpha
    if conf.opt_algo == "lasso_cd":
        result_analysis.plot_alphas_across_scan(
            scan_record=scan_record, x="Time", save_dir=conf.report_dir
        )

    # Report
    scan_record = result_analysis.generate_result_report(
        scan_record=scan_record,
        intensity_cols=[sbs_result.ref_df[col] for col in sbs_result.sum_cols]
        + [sbs_result.ref_exp_df_inner["Intensity"]],
        save_dir=conf.report_dir,
    )
    scan_record.to_csv(conf.output_file + "_scan_record.csv")


if __name__ == "__main__":
    fire.Fire(opt_scan_by_scan)
