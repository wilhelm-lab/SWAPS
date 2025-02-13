import logging
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import IsoSpecPy as iso
from prepare_dict.prepare_dict import filter_maxquant_by_ok

Logger = logging.getLogger(__name__)


def get_precursor_intensity_by_category(
    mq_001_dir,
    precursors_df: pd.DataFrame,
    filter_by_raw_file: str = None,
    mq_100_dir: str = None,
    ok_output_dir: str = None,
    plot_type: str | None = "kde",
    title: str = "Precursor intensity by identification status",
    xlim: tuple = (3, 9),
    **kwargs,
):
    """
    Plot precursor intensity by category defined by identification status

    :mq_001_dir: str: path to MaxQuant 1% output directory
    :precursors_df: pd.DataFrame: DataFrame with precursor information from timstof .d file
    :filter_by_raw_file: str: filter by raw file name
    :mq_100_dir: str: path to MaxQuant 100% output directory
    :ok_output_dir: str: path to OK output directory
    :plot_type: str: type of plot to use, either "kde" or "hist"
    :title: str: title of the plot
    :xlim: tuple: x-axis limits
    :**kwargs: dict: additional arguments to pass to the plot
    :returns: pd.DataFrame: DataFrame with precursor information and identification status
    """
    acc_pasef_msms_scans = pd.read_csv(
        os.path.join(mq_001_dir, "accumulatedMsmsScans.txt"), sep="\t"
    )
    evidence_001 = pd.read_csv(os.path.join(mq_001_dir, "evidence.txt"), sep="\t")
    if filter_by_raw_file is not None:
        acc_pasef_msms_scans = acc_pasef_msms_scans[
            acc_pasef_msms_scans["Raw file"] == filter_by_raw_file
        ]
        evidence_001 = evidence_001[evidence_001["Raw file"] == filter_by_raw_file]
    acc_pasef_msms_scans["Identified"].fillna("-", inplace=True)
    precursors_mq = (
        acc_pasef_msms_scans["PASEF precursor IDs"]
        .str.split(";")
        .explode()
        .astype(int)
        .tolist()
    )
    if ok_output_dir is not None:
        assert mq_100_dir is not None, "Need to provide mq_100_dir to plot ok_output"
        evidence_100 = pd.read_csv(os.path.join(mq_100_dir, "evidence.txt"), sep="\t")
        if filter_by_raw_file is not None:
            evidence_100 = evidence_100[evidence_100["Raw file"] == filter_by_raw_file]
        evidence_rescore_001fdr = filter_maxquant_by_ok(evidence_100, ok_output_dir)
        evidence_001 = pd.merge(
            evidence_rescore_001fdr,
            evidence_001[
                ["Raw file", "Modified sequence", "Charge", "MS/MS scan number"]
            ],
            on=["Raw file", "Modified sequence", "Charge"],
            how="outer",
            indicator=True,
            suffixes=("_ok", "_mq"),
        )
        evidence_001["_merge"].value_counts()
        scan_number_ok = (
            evidence_001[evidence_001["_merge"] != "right_only"]["MS/MS scan number_ok"]
            .explode()
            .astype(int)
            .tolist()
        )
        scan_number_mq = (
            evidence_001[evidence_001["_merge"] != "left_only"]["MS/MS scan number_mq"]
            .explode()
            .astype(int)
            .tolist()
        )
        scan_number_both = (
            evidence_001[evidence_001["_merge"] == "both"]["MS/MS scan number_ok"]
            .explode()
            .astype(int)
            .tolist()
        )
        precursors_id_by_ok = (
            acc_pasef_msms_scans.loc[
                acc_pasef_msms_scans["Scan number"].isin(scan_number_ok),
                "PASEF precursor IDs",
            ]
            .str.split(";")
            .explode()
            .astype(int)
            .tolist()
        )
        precursors_id_by_both = (
            acc_pasef_msms_scans.loc[
                acc_pasef_msms_scans["Scan number"].isin(scan_number_both),
                "PASEF precursor IDs",
            ]
            .str.split(";")
            .explode()
            .astype(int)
            .tolist()
        )
    else:
        scan_number_mq = evidence_001["MS/MS scan number"].tolist()
    precursors_id_by_mq = (
        acc_pasef_msms_scans.loc[
            acc_pasef_msms_scans["Scan number"].isin(scan_number_mq),
            "PASEF precursor IDs",
        ]
        .str.split(";")
        .explode()
        .astype(int)
        .tolist()
    )
    precursors_df["log_intensity"] = np.log10(precursors_df["Intensity"])
    precursors_df["status"] = "Not identified"
    precursors_df.loc[~precursors_df["Id"].isin(precursors_mq), "status"] = "Not in MQ"
    precursors_df.loc[precursors_df["Id"].isin(precursors_id_by_ok), "status"] = (
        "Identified by OK"
    )
    precursors_df.loc[precursors_df["Id"].isin(precursors_id_by_mq), "status"] = (
        "Identified by MQ"
    )
    precursors_df.loc[precursors_df["Id"].isin(precursors_id_by_both), "status"] = (
        "Identified by both"
    )
    precursors_df["Identified"] = ~precursors_df["status"].str.contains("Not")
    hue_mapping = {
        "Not in MQ": "red",
        "Not identified": "orange",
        "Identified by MQ": "purple",
        "Identified by OK": "green",
        "Identified by both": "blue",
    }
    match plot_type:
        case "kde":
            sns.kdeplot(
                precursors_df,
                x="log_intensity",
                hue="status",
                common_norm=True,
                palette=hue_mapping,
                **kwargs,
            )
        case "hist":
            sns.histplot(precursors_df, x="log_intensity", hue="status", **kwargs)
        case None:
            Logger.info("No plot type provided, only return the DataFrame")
        case _:
            raise ValueError(f"Invalid plot type {plot_type}")
    if plot_type is not None:
        plt.xlim(xlim[0], xlim[1])
        plt.title(title)
        plt.xlabel("Precursor Intensity (Log10)")
    return precursors_df


def extract_precursor_features(
    precursor_id,
    data,
    precursor_df,
):
    dataframe = data[:, :, precursor_id, :, :]
    (frame_index_parent, precursor_intensity) = precursor_df.loc[
        precursor_df["Id"] == precursor_id, ["Parent", "Intensity"]
    ].values[0]
    Logger.info(
        "Checking precursor %s, intensity %s, frag spectra dataframe shape %s",
        precursor_id,
        precursor_intensity,
        dataframe.shape,
    )
    frame_index_parent = int(frame_index_parent)
    quad_low_mz = dataframe.loc[
        dataframe["precursor_indices"] == precursor_id, "quad_low_mz_values"
    ].min()
    quad_high_mz = dataframe.loc[
        dataframe["precursor_indices"] == precursor_id, "quad_high_mz_values"
    ].max()
    mobility_low = dataframe.loc[
        dataframe["precursor_indices"] == precursor_id, "scan_indices"
    ].min()
    mobility_high = dataframe.loc[
        dataframe["precursor_indices"] == precursor_id, "scan_indices"
    ].max()
    frame_index = dataframe.loc[
        dataframe["precursor_indices"] == precursor_id, "frame_indices"
    ].min()
    rt_values_min = dataframe["rt_values"].min()
    rt_values_max = dataframe["rt_values"].max()
    Logger.info(
        "Filter precursor feature with frame (frag) %s, frame (precursor parent) %s, quad mz %s, %s, mobility scans %s, %s, retention time (not used for filtering) %s, %s",
        frame_index,
        frame_index_parent,
        quad_low_mz,
        quad_high_mz,
        mobility_low,
        mobility_high,
        rt_values_min,
        rt_values_max,
    )
    filtered_dataframe = data[
        {
            "frame_indices": frame_index_parent,
            "scan_indices": slice(mobility_low, mobility_high + 1),
            "mz_values": slice(quad_low_mz - 0.1, quad_high_mz + 0.1),
            "precursor_indices": 0,
        }
    ]

    Logger.info("Filtered dataframe %s", filtered_dataframe.shape)
    return filtered_dataframe, quad_high_mz, quad_low_mz


def estimate_theoretical_isotope_pattern(mz_values, charge):
    """
    Estimates a theoretical isotope distribution based on the averagine model.

    Parameters:
        mz_values (np.array): Experimental m/z values.
        charge (int): Charge state of the peptide.

    Returns:
        tuple: (theoretical_mz_values, theoretical_intensity_values)
    """
    # Estimate the neutral mass of the peptide
    neutral_mass = mz_values[0] * charge  # Approximation

    # Compute averagine-based elemental composition
    num_averagine_units = neutral_mass / 111.125  # Average mass of an averagine unit
    avg_composition = {
        "C": round(num_averagine_units * 4.9384),
        "H": round(num_averagine_units * 7.7583),
        "N": round(num_averagine_units * 1.3577),
        "O": round(num_averagine_units * 1.4773),
        "S": round(num_averagine_units * 0.0417),
    }
    Logger.info(
        "avg composition %s with %s units", avg_composition, num_averagine_units
    )

    # Generate isotopic pattern using Pyteomics
    isotope_pattern = iso.IsoTotalProb(formula=avg_composition, prob_to_cover=0.9999)
    # isotope_pattern = mass.isotopologues(avg_composition)
    Logger.info("Isotope pattern %s", isotope_pattern)
    # Extract m/z values and intensities
    theoretical_mz_values = []
    theoretical_intensity_values = []
    for mz, prob in isotope_pattern:
        theoretical_mz_values += [mz / charge]
        theoretical_intensity_values += [prob]
    # theoretical_mz_values = np.array([mz / charge for mz, prob in isotope_pattern])
    # theoretical_intensity_values = np.array(list(isotope_pattern.values()))
    theoretical_intensity_values /= np.sum(theoretical_intensity_values)  # Normalize

    return theoretical_mz_values, theoretical_intensity_values


def calculate_averagine_fit(df, charge):
    """
    Computes the averagine fit metrics for an experimental isotopic pattern.

    Parameters:
        df (pd.DataFrame): DataFrame with columns "mz_values" and "intensity_values".
        charge (int): Charge state of the peptide.

    Returns:
        dict: RMS error, m/z bias, and explained intensity fraction.
    """
    # Extract experimental values
    mz_exp = df["mz_values"].values
    intensity_exp = df["intensity_values"].values.astype(float)
    intensity_exp /= np.sum(intensity_exp)  # Normalize intensities

    # Estimate theoretical pattern
    theoretical_mz_values, theoretical_intensity_values = (
        estimate_theoretical_isotope_pattern(mz_exp, charge)
    )

    # Interpolate theoretical intensities at experimental m/z values
    interp_func = interp1d(
        theoretical_mz_values,
        theoretical_intensity_values,
        kind="linear",
        fill_value=0,
        bounds_error=False,
    )
    interpolated_theoretical_intensities = interp_func(mz_exp)

    # Compute RMS error
    rms_error = np.sqrt(
        np.mean((intensity_exp - interpolated_theoretical_intensities) ** 2)
    )

    # Compute m/z bias (weighted mean shift)
    mz_bias = np.sum((mz_exp - theoretical_mz_values[0]) * intensity_exp)

    # Compute explained intensity fraction
    total_theoretical_intensity = np.sum(theoretical_intensity_values)
    total_matched_intensity = np.sum(interpolated_theoretical_intensities)
    explained_intensity_fraction = (
        total_matched_intensity / total_theoretical_intensity
        if total_theoretical_intensity > 0
        else 0
    )

    return {
        "RMS Error": rms_error,
        "m/z Bias": mz_bias,
        "Explained Intensity Fraction": explained_intensity_fraction,
    }
