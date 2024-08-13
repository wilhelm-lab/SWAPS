import os
import logging
import pickle
import time
import argparse
import pandas as pd
import numpy as np
import postprocessing.no_ims_2d as no_ims_2d
from optimization.dictionary import LoadMZML
from multiprocessing import cpu_count
from optimization.inference import process_scans_parallel
from result_analysis import result_analysis

os.environ["NUMEXPR_MAX_THREADS"] = "32"


def main():
    parser = argparse.ArgumentParser(description="Running Scan By Scan pipeline")

    # Add arguments
    parser.add_argument(
        "-ml",
        "--mzML_path",
        type=str,
        required=True,
        help="path to the data input in mzML format, containing only MS1 level data",
    )
    parser.add_argument(
        "-MQ",
        "--MQ_path",
        type=str,
        required=True,
        help=(
            "path to MaxQuant results (evidence.txt) of the same RAW file,             "
            "                used for constructing reference dictionary"
        ),
    )
    parser.add_argument(
        "-RT_tol",
        "--RT_tol",
        type=float,
        required=True,
        help=(
            "Tolerance when selecting candidate precursor by retention time, unit is"
            " minute"
        ),
    )
    parser.add_argument(
        "-cond",
        "--condition",
        type=str,
        required=False,
        default="peakRange",
        help="The method for doing mz alignment.",
    )
    parser.add_argument(
        "-mz_tol",
        "--mz_tol",
        type=float,
        required=False,
        default=0,
        help=(
            "Tolerance when matching theoratical isotope mz value to observed mz value"
        ),
    )
    # required if cond == '2stepNN'
    parser.add_argument(
        "-ac",
        "--alpha_criteria",
        type=str,
        required=False,
        default="convergence",
        help="Criteria for choosing the best alpha, either min or convergence",
    )
    parser.add_argument(
        "-alpha",
        "--alpha",
        type=float,
        required=False,
        default=0,
        help="Penalty strength for Lasso regression.",
    )  # TODO: change to alpha list
    parser.add_argument(
        "-opt",
        "--opt_algo",
        type=str,
        required=False,
        default="lasso_cd",
        help=(
            "Algorithm to use for sparse encoding, default lasso with coordinate"
            " descent."
        ),
    )
    parser.add_argument(
        "-IE_ab",
        "--IE_ab_thres",
        type=float,
        required=False,
        default=0.001,
        help=(
            "Abundance threshold for generating Isotope Envelops when constructing"
            " dictionary"
        ),
    )
    parser.add_argument(
        "-IE_mab",
        "--IE_AbundanceMissingThres",
        required=True,
        type=float,
        default=0.4,
        help=(
            "Threshold for maximum allowed missing isotope abundance for a precursor   "
            "                          to be included into candidate list"
        ),
    )
    parser.add_argument(
        "-RT_ref",
        "--RT_ref",
        type=str,
        required=False,
        default="exp",
        help=(
            "Reference retention time, exp - experimental retention time (from MQ),    "
            "                         pred - prediction from DeepLC, mix - exp RT from"
            " MQ if available, else from DeepLC"
        ),
    )
    parser.add_argument(
        "-MQexp",
        "--MQ_exp_path",
        type=str,
        required=False,
        help=(
            "path to MaxQuant experiment                         results (evidence.txt)"
            " of the same RAW file, used for comparing inferred intensity"
        ),
    )
    # Parse the command-line arguments
    args = parser.parse_args()
    msconvert_file = args.mzML_path
    maxquant_file = args.MQ_path
    RT_tol = float(args.RT_tol)
    mz_tol = float(args.mz_tol)
    mzalign_method = str(args.condition)
    alpha = float(args.alpha)
    ab_thres = float(args.IE_ab_thres)
    AbundanceMissingThres = float(args.IE_AbundanceMissingThres)
    alpha_criteria = args.alpha_criteria
    RT_ref = args.RT_ref
    maxquant_file_exp = args.MQ_exp_path
    opt_algo = args.opt_algo

    # define paths
    dirname = os.path.dirname(msconvert_file)
    basename = os.path.basename(msconvert_file)
    MS1Scans_NoArray_name = basename[:-5] + "_MS1Scans_NoArray.csv"
    filename = basename[:-5] + "_ScanByScan"
    filename += (
        "_RTtol"
        + str(RT_tol)
        + "_MZtol"
        + str(mz_tol)
        + "_cond"
        + mzalign_method
        + "_alpha"
        + str(alpha)
        + "_"
        + opt_algo
        + "_abthres"
        + str(ab_thres)
        + "_missabthres"
        + str(AbundanceMissingThres)
        + "_"
        + alpha_criteria
        + "_NoIntercept"
        + "_"
        + RT_ref
    )

    if not os.path.exists(os.path.join(dirname, filename)):
        os.makedirs(os.path.join(dirname, filename))
    filename_full = os.path.join(dirname, filename)
    output_file = os.path.join(filename_full, filename + "_output")  # filename

    logging.basicConfig(
        filename=os.path.join(filename_full, filename + ".log"),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    logging.info(
        "Args: msconvert filepath - %s, maxquant result path - %s.",
        msconvert_file,
        maxquant_file,
    )
    logging.info(
        "Args: RT tolerance - %s, mz tolerance(not relevant) - %s.", RT_tol, mz_tol
    )
    logging.info(
        "Args: mz align method - %s, alpha - %s, isotope abundance threshold - %s.",
        mzalign_method,
        alpha,
        ab_thres,
    )
    logging.info(
        "Args: maximum missing isotope abundance - %s, alpha criteria - %s.",
        AbundanceMissingThres,
        alpha_criteria,
    )
    logging.info("Output directory - %s", filename_full)

    # start analysis
    start_time_init = time.time()
    logging.info("==================Load data==================")

    # Load reference data
    # Maxquant_result = pd.read_csv(filepath_or_buffer=maxquant_file, sep='\t')
    Maxquant_result_dict = pd.read_pickle(filepath_or_buffer=maxquant_file)
    Maxquant_result_exp = pd.read_csv(filepath_or_buffer=maxquant_file_exp, sep="\t")
    # Load MS1 scans from pkl or mzml file
    try:
        with open(msconvert_file[:-5] + ".pkl", "rb") as f:
            MS1Scans = pickle.load(f)
        logging.info("Load pre-stored pickle results")
        if not os.path.isfile(os.path.join(dirname, MS1Scans_NoArray_name)):
            MS1Scans_NoArray = MS1Scans.iloc[:, 1:5].copy()
            MS1Scans_NoArray.to_csv(
                os.path.join(dirname, MS1Scans_NoArray_name), index=0
            )
    except:
        logging.info("Pickle result not found, load mzml file.")
        MS1Scans = LoadMZML(msconvert_file)
        MS1Scans.to_pickle(msconvert_file[:-5] + ".pkl")
    minutes, seconds = divmod(time.time() - start_time_init, 60)
    logging.info("Script execution time: {}m {}s".format(int(minutes), int(seconds)))

    # Set left and right edge for scan search range
    if RT_ref == "exp":
        Maxquant_result_dict["RT_search_left"] = (
            Maxquant_result_dict["Calibrated retention time start"] - RT_tol
        )
        Maxquant_result_dict["RT_search_right"] = (
            Maxquant_result_dict["Calibrated retention time finish"] + RT_tol
        )
        RT_ref_act_peak = "Calibrated retention time"
    elif RT_ref == "pred":
        Maxquant_result_dict["RT_search_left"] = (
            Maxquant_result_dict["predicted_RT"] - RT_tol
        )
        Maxquant_result_dict["RT_search_right"] = (
            Maxquant_result_dict["predicted_RT"] + RT_tol
        )
        RT_ref_act_peak = "predicted_RT"
    elif RT_ref == "mix":
        Maxquant_result_dict["RT_search_left"] = (
            Maxquant_result_dict["Retention time new"] - RT_tol
        )
        Maxquant_result_dict["RT_search_right"] = (
            Maxquant_result_dict["Retention time new"] + RT_tol
        )
        RT_ref_act_peak = "Retention time new"
    try:  # try and read results
        scan_record = pd.read_pickle(output_file + "_scan_record.pkl")
        activation = np.load(output_file + "_activationByScanFromLasso.npy")

        logging.info("Load pre-calculated optimization.")
    except FileNotFoundError:
        logging.info("Precalculated optimization not found, start Scan By Scan.")
        logging.info("==================Scan By Scan==================")
        start_time = time.time()
        logging.info("-----------------Calculate Isotope Pattern-----------------")
        # Calculate isotope pattern,
        # on slurm notes isospecpy return 'illegal instruction, core dumped error',
        # use precomputed isospec result

        # Maxquant_result['IsoMZ'], Maxquant_result['IsoAbundance'] = \
        #     zip(*Maxquant_result.apply(lambda row: CalcModpeptIsopattern(modpept=row['Modified sequence'],
        #                                                                 charge=row['Charge'],
        #                                                                 ab_thres = ab_thres),
        #                                                                 axis=1))
        minutes, seconds = divmod(time.time() - start_time, 60)
        logging.info(
            "Script execution time: {}m {}s".format(int(minutes), int(seconds))
        )

        # Optimization
        start_time = time.time()
        logging.info("-----------------Scan by Scan Optimization-----------------")

        # process scans
        result_dict = process_scans_parallel(
            n_jobs=cpu_count(),
            ms1scans=MS1Scans,  # for small scale testing: MS1Scans.iloc[1000:1050, :]
            maxquant_ref=Maxquant_result_dict,
            loss="lasso",
            opt_algo=opt_algo,
            # alphas=[0],  # TODO: use alpha = 0 for threshold, change if needed!!
            alpha_criteria=alpha_criteria,
            abundance_missing_threshold=AbundanceMissingThres,
        )
        minutes, seconds = divmod(time.time() - start_time, 60)
        logging.info(
            "Script execution time: {}m {}s".format(int(minutes), int(seconds))
        )
        # merge results
        n_ms1scans = MS1Scans.shape[0]
        activation = pd.DataFrame(
            index=Maxquant_result_dict["id"], columns=range(n_ms1scans)
        )
        scan_record_list = []
        for scan_idx, result_dict_scan in result_dict.items():
            if result_dict_scan["activation"] is not None:
                activation.loc[
                    result_dict_scan["activation"]["precursor"], scan_idx
                ] = result_dict_scan["activation"]["activation"]
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

        minutes, seconds = divmod(time.time() - start_time, 60)
        logging.info(
            "Script execution time: {}m {}s".format(int(minutes), int(seconds))
        )
        activation = activation.fillna(0)
        np.save(output_file + "_activationByScanFromLasso.npy", activation.values)
        activation.to_pickle(output_file + "_activationByScanFromLasso.pkl")
        scan_record.to_pickle(output_file + "_scan_record.pkl")
        minutes, seconds = divmod(time.time() - start_time, 60)
        logging.info(
            "Script execution time: {}m {}s".format(int(minutes), int(seconds))
        )

    logging.info("==================Result Analaysis==================")
    # Make result directory
    MS1Scans_NoArray = pd.read_csv(os.path.join(dirname, MS1Scans_NoArray_name))
    result_dir = filename_full
    report_dir = os.path.join(result_dir, "report")
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
        os.makedirs(os.path.join(report_dir, "activation"))

    # Smoothing

    Maxquant_result_dict["SumActivation"] = activation.sum(
        axis=1
    )  # sample without smoothing
    _, sum_raw = no_ims_2d.SmoothActivationMatrix(
        activation=activation, MS1Scans_noArray=MS1Scans_NoArray, method="Raw"
    )
    sum_raw.to_csv(os.path.join(result_dir, "sum_raw.csv"), index=False)
    try:
        refit_activation_minima = np.load(output_file + "_activationMinima.npy")
        sum_minima = pd.read_csv(os.path.join(result_dir, "sum_minima.csv"))
    except FileNotFoundError:
        refit_activation_minima, sum_minima = no_ims_2d.SmoothActivationMatrix(
            activation=activation,
            MS1Scans_noArray=MS1Scans_NoArray,
            method="LocalMinima",
        )
        np.save(output_file + "_activationMinima.npy", refit_activation_minima)
        sum_minima.to_csv(os.path.join(result_dir, "sum_minima.csv"), index=False)

    try:
        refit_activation_gaussian = np.load(output_file + "_activationGaussian.npy")
        sum_gaussian = pd.read_csv(os.path.join(result_dir, "sum_gaussian.csv"))
    except FileNotFoundError:
        (
            refit_activation_gaussian,
            sum_gaussian,
        ) = no_ims_2d.SmoothActivationMatrix(
            activation=activation,
            MS1Scans_noArray=MS1Scans_NoArray,
            method="GaussianKernel",
        )
        np.save(output_file + "_activationGaussian.npy", refit_activation_gaussian)
        sum_gaussian.to_csv(os.path.join(result_dir, "sum_gaussian.csv"), index=False)
    logging.debug(
        "dimension of sum_raw, sum_gaussiam, sum_minima: %s, %s, %s",
        sum_raw.shape,
        sum_gaussian.shape,
        sum_minima.shape,
    )
    pp_sumactivation = pd.concat(
        [
            sum_raw.reset_index(
                drop=True
            ),  # concat 3 objects might reset index for the first two
            sum_gaussian.reset_index(drop=True),
            sum_minima.reset_index(drop=True),
        ],
        axis=1,
        ignore_index=False,
    )
    logging.debug("dimension of pp_sumactivation %s", pp_sumactivation.shape)
    logging.debug("dimension of Maxquant_result_dict %s", Maxquant_result_dict.shape)
    pp_sumactivation = pp_sumactivation.set_index(Maxquant_result_dict.index)
    Maxquant_result_dict = pd.concat([Maxquant_result_dict, pp_sumactivation], axis=1)
    SumActCol = [
        "SumActivationRaw",
        "AUCActivationRaw",
        "TRPZActivationRaw",
        "SumActivationGaussianKernel",
        "AUCActivationGaussianKernel",
        "TRPZActivationGaussianKernel",
        "SumActivationLocalMinima",
        "AUCActivationLocalMinima",
        "TRPZActivationLocalMinima",
        "AUCActivationPeak",
    ]

    # Elution peak properties
    activation_df = pd.DataFrame(activation, index=Maxquant_result_dict["id"])
    refit_activation_minima_df = pd.DataFrame(
        refit_activation_minima, index=Maxquant_result_dict["id"]
    )
    refit_activation_gaussian_df = pd.DataFrame(
        refit_activation_gaussian, index=Maxquant_result_dict["id"]
    )

    Maxquant_result_dict_act_peak = no_ims_2d.calculate_sum_activation(
        Maxquant_result=Maxquant_result_dict,
        activation_df=refit_activation_minima_df,
        MS1ScansNoArray=MS1Scans_NoArray,
        RT_ref=RT_ref_act_peak,
    )
    Maxquant_result_dict_act_peak.to_pickle(
        os.path.join(result_dir, "maxquant_dict_sbs_result.pkl")
    )

    # Correlation
    if RT_ref in ["pred", "mix"]:
        Maxquant_result = result_analysis.compare_act_sum_with_MQ(
            MQ_dict=Maxquant_result_dict_act_peak,
            MQ_exp=Maxquant_result_exp,
            RT_tol=RT_tol,
            MQ_dict_sum_act_col=SumActCol,
        )

    RegIntensity, AbsResidue, valid_idx = result_analysis.PlotCorr(
        Maxquant_result["Intensity"],
        Maxquant_result["AUCActivationRaw"],
        interactive=False,
        save_dir=report_dir,
    )
    Maxquant_result_filtered = pd.DataFrame(
        {"RegressionIntensity": RegIntensity, "AbsResidue": AbsResidue}
    )
    Maxquant_result_filtered = pd.concat(
        [Maxquant_result.loc[valid_idx[0], :], Maxquant_result_filtered],
        join="inner",
        axis=1,
    )
    Maxquant_result_filtered["IntensityLog"] = np.log10(
        Maxquant_result_filtered["Intensity"]
    )
    Maxquant_result_filtered.to_pickle(
        os.path.join(result_dir, "maxquant_filtered_sbs_result.pkl")
    )
    for sum_col in SumActCol:
        _, _, _ = result_analysis.PlotCorr(
            Maxquant_result["Intensity"], Maxquant_result[sum_col], save_dir=report_dir
        )

    # Report
    scan_record = result_analysis.GenerateResultReport(
        scan_record=scan_record,
        intensity_cols=[Maxquant_result_dict[col] for col in SumActCol]
        + [Maxquant_result["Intensity"]],
        save_dir=report_dir,
    )
    scan_record.to_csv(output_file + "_scan_record.csv")

    # Plot activation for selected samples
    Accurate50_idx = Maxquant_result_filtered.nsmallest(50, "AbsResidue")["id"].values
    Inaccurate50_idx = Maxquant_result_filtered.nlargest(50, "AbsResidue")["id"].values

    for idx in Accurate50_idx:
        print(idx)
        _ = result_analysis.PlotActivation(
            MaxquantEntry=Maxquant_result_filtered.loc[
                Maxquant_result_filtered["id"] == idx, :
            ],
            PrecursorTimeProfiles=[
                activation_df.loc[idx, :],
                refit_activation_minima_df.loc[idx, :],
                refit_activation_gaussian_df.loc[idx, :],
            ],
            PrecursorTimeProfileLabels=[
                "Raw",
                "LocalMinimaSmoothing",
                "GaussianSmoothing",
            ],
            MS1ScansNoArray=MS1Scans_NoArray,
            RT_tol=RT_tol,
            save_dir=os.path.join(report_dir, "activation", "accurate"),
        )
    for idx in Inaccurate50_idx:
        _ = result_analysis.PlotActivation(
            MaxquantEntry=Maxquant_result_filtered.loc[
                Maxquant_result_filtered["id"] == idx, :
            ],
            PrecursorTimeProfiles=[
                activation_df.loc[idx, :],
                refit_activation_minima_df.loc[idx, :],
                refit_activation_gaussian_df.loc[idx, :],
            ],
            PrecursorTimeProfileLabels=[
                "Raw",
                "LocalMinimaSmoothing",
                "GaussianSmoothing",
            ],
            MS1ScansNoArray=MS1Scans_NoArray,
            RT_tol=RT_tol,
            save_dir=os.path.join(report_dir, "activation", "inaccurate"),
        )


if __name__ == "__main__":
    main()
