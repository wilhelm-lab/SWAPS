import logging
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from prepare_dict.prepare_dict import filter_maxquant_by_ok

Logger = logging.getLogger(__name__)


def plot_precursor_intensity_by_category(
    mq_001_dir,
    precursors_df: pd.DataFrame,
    filter_by_raw_file: str = None,
    mq_100_dir: str = None,
    ok_output_dir: str = None,
    plot_type: str = "kde",
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
    plt.xlim(xlim[0], xlim[1])
    plt.title(title)
    plt.xlabel("Precursor Intensity (Log10)")
    return precursors_df
