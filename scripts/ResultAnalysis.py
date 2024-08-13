import os
import numpy as np
import pandas as pd
import argparse
from postprocessing import no_ims_2d


def main():
    parser = argparse.ArgumentParser(
        description="Running Scan By Scan optimization pipeline"
    )

    # Add arguments
    parser.add_argument(
        "-rdir",
        "--result_dir",
        required=True,
        help="path to the result from scan by scan optimization",
    )
    parser.add_argument(
        "-ddir",
        "--data_dir",
        required=True,
        help="path to data containing combined folder and MS1Scans data etc.",
    )
    parser.add_argument(
        "-MQ",
        "--MQ_path",
        required=True,
        help=(
            "path to MaxQuant results (evidence.txt) of the same RAW file,"
            " used for constructing reference dictionary"
        ),
    )

    # Process paths
    args = parser.parse_args()
    result_dir = args.result_dir
    data_dir = args.data_dir
    basename = os.path.basename(result_dir)
    output_file = os.path.join(result_dir, basename + "_output")  # filename
    maxquant_file = os.path.join(data_dir, "combined/txt/evidence.txt")
    MS1Scans_NoArray = os.path.join(data_dir, "MS1Scans_NoArray.csv")

    # Load data
    Maxquant_result = pd.read_csv(filepath_or_buffer=maxquant_file, sep="\t")
    activation = np.load(output_file + "_activationByScanFromLasso.npy")
    emptyScans = pd.read_csv(
        filepath_or_buffer=output_file + "_EmptyScans.csv", index_col=0
    )
    NonEmptyScans = pd.read_csv(
        filepath_or_buffer=output_file + "_NonEmptyScans.csv", index_col=0
    )
    MS1Scans_NoArray = pd.read_csv(MS1Scans_NoArray)

    # Make result directory
    report_dir = os.path.join(result_dir, "report")
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
        os.makedirs(os.path.join(report_dir, "activation"))

    # Smoothing
    Maxquant_result["SumActivation"] = activation.sum(
        axis=1
    )  # sample without smoothing
    try:
        refit_activation_minima = np.load(output_file + "_activationMinima.npy")
        sum_minima = pd.read_csv(os.path.join(result_dir, "sum_minima.csv"))
    except FileNotFoundError:
        refit_activation_minima, sum_minima = no_ims_2d.smooth_act_mat(
            activation=activation,
            ms1scans_no_array=MS1Scans_NoArray,
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
        ) = no_ims_2d.smooth_act_mat(
            activation=activation,
            ms1scans_no_array=MS1Scans_NoArray,
            method="GaussianKernel",
        )
        np.save(output_file + "_activationGaussian.npy", refit_activation_gaussian)
        sum_gaussian.to_csv(os.path.join(result_dir, "sum_gaussian.csv"), index=False)

    # Correlation
    Maxquant_result_filtered = no_ims_2d.TransformAndFilter(
        Maxquant_result=Maxquant_result
    )
    (
        Maxquant_result_filtered["RegressionIntensity"],
        Maxquant_result_filtered["AbsResidue"],
        _,
    ) = no_ims_2d.PlotCorr(
        Maxquant_result_filtered["Intensity"],
        Maxquant_result_filtered["SumActivation"],
        save_dir=report_dir,
    )
    for sum_col in [
        sum_minima.iloc[:, 0],
        sum_minima.iloc[:, 1],
        sum_gaussian.iloc[:, 0],
        sum_gaussian.iloc[:, 1],
    ]:
        _, _, _ = no_ims_2d.PlotCorr(
            Maxquant_result["Intensity"], sum_col, save_dir=report_dir
        )

    # Report
    NonEmptyScans = no_ims_2d.GenerateResultReport(
        Maxquant_result=Maxquant_result,
        MS1Scan_noArray=MS1Scans_NoArray,
        emptyScans=emptyScans,
        NonEmptyScans=NonEmptyScans,
        intensity_cols=[
            Maxquant_result["Intensity"],
            Maxquant_result["SumActivation"],
            sum_gaussian.iloc[:, 0],
            sum_gaussian.iloc[:, 1],
            sum_minima.iloc[:, 0],
            sum_minima.iloc[:, 1],
        ],
        save_dir=report_dir,
    )

    # Plot activation for selected samples
    Accurate50_idx = Maxquant_result_filtered["AbsResidue"].nsmallest(50).index
    Inaccurate50_idx = Maxquant_result_filtered["AbsResidue"].nlargest(50).index
    for idx in Accurate50_idx:
        _ = no_ims_2d.PlotActivation(
            MaxquantEntry=Maxquant_result.iloc[idx, :],
            PrecursorTimeProfiles=[
                activation[idx, :],
                refit_activation_minima[idx, :],
                refit_activation_gaussian[idx, :],
            ],
            PrecursorTimeProfileLabels=[
                "Raw",
                "LocalMinimaSmoothing",
                "GaussianSmoothing",
            ],
            MS1ScansNoArrary=MS1Scans_NoArray,
            save_dir=os.path.join(report_dir, "activation", "accurate"),
        )
    for idx in Inaccurate50_idx:
        _ = no_ims_2d.PlotActivation(
            MaxquantEntry=Maxquant_result.iloc[idx, :],
            PrecursorTimeProfiles=[
                activation[idx, :],
                refit_activation_minima[idx, :],
                refit_activation_gaussian[idx, :],
            ],
            PrecursorTimeProfileLabels=[
                "Raw",
                "LocalMinimaSmoothing",
                "GaussianSmoothing",
            ],
            MS1ScansNoArrary=MS1Scans_NoArray,
            save_dir=os.path.join(report_dir, "activation", "inaccurate"),
        )


if __name__ == "__main__":
    main()
