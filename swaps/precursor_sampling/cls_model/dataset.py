import os
import pickle
import logging
from tqdm import tqdm
import numpy as np
import pandas as pd
import logging
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import IsoSpecPy as iso
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from prepare_dict.prepare_dict import filter_maxquant_by_ok
from utils.plot import plot_comparison

Logger = logging.getLogger(__name__)


def extract_precursor_features(
    precursor_id,
    data,
    precursor_df,
):
    dataframe = data[:, :, precursor_id, :, :]
    (frame_index_parent, precursor_intensity, charge) = precursor_df.loc[
        precursor_df["Id"] == precursor_id, ["Parent", "Intensity", "Charge"]
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
    return filtered_dataframe, quad_high_mz, quad_low_mz, charge


def preprocess_mz_intensity(
    df, min_mz, max_mz, n_digits=3, mz_values=None, intensity_values=None
):

    # Round m/z values and aggregate intensities
    df["rounded_mz"] = df["mz_values"].round(n_digits)
    binned_data = df.groupby("rounded_mz", as_index=True)["intensity_values"].sum()

    # Create a continuous index for all possible bins
    # min_mz, max_mz = (
    #     df["quad_low_mz_values"].min(),
    #     df["quad_high_mz_values"].max(),
    # )  # TODO: not to use min and max, but quad low and high
    Logger.debug("Min m/z: %s, Max m/z: %s", min_mz, max_mz)
    all_bins = np.round(
        np.arange(min_mz, max_mz + 10**-n_digits, 10**-n_digits), n_digits
    )

    # Efficiently fill missing bins with zero intensity
    binned_data = binned_data.reindex(all_bins, fill_value=0)
    Logger.debug("Length of binned data: %s", len(binned_data))
    return (
        binned_data.values,
        binned_data.index.values,
    )  # Returns intensity array and bins


def plot_binned_intensities(
    intensities, mz_bins=None, title="Binned Intensities", num_xticks=10
):
    plt.figure(figsize=(10, 5))
    if mz_bins is None:
        mz_bins = np.arange(len(intensities))
    sns.barplot(x=mz_bins, y=intensities, color="black")

    plt.xlabel("m/z (binned)")
    plt.ylabel("Intensity")
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Show only a subset of x-ticks
    xtick_indices = np.linspace(0, len(mz_bins) - 1, num_xticks, dtype=int)
    plt.xticks(xtick_indices, np.round(mz_bins[xtick_indices], 3), rotation=90)

    plt.show()


def plot_precursor_features(data_dict, precursor_id):
    plot_binned_intensities(
        data_dict[precursor_id]["features"],
        title=f"Precursor ID: {precursor_id}, Label: {data_dict[precursor_id]['label']}",
    )


def prepare_precursor_dataset(
    precursor_df,
    data,
    n_digits=3,
    save_path=None,
):
    data_dict = {}
    for idx, row in tqdm(
        precursor_df.iterrows(), total=len(precursor_df), desc="Processing precursors"
    ):
        precursor_id = int(row["Id"])
        ms1data, quad_high_mz, quad_low_mz = extract_precursor_features(
            precursor_id, data=data, precursor_df=precursor_df
        )
        if ms1data.shape[0] == 0:
            Logger.warning("No MS1 data found for precursor ID: %s", precursor_id)
            continue
        intensities, _ = preprocess_mz_intensity(
            ms1data, min_mz=quad_low_mz, max_mz=quad_high_mz, n_digits=n_digits
        )
        if sum(intensities) == 0:
            Logger.warning(
                "All intensities are zero for precursor ID: %s", precursor_id
            )
            continue
        else:
            Logger.debug(
                "Precursor ID: %s, length of intensities %s",
                precursor_id,
                len(intensities),
            )
        data_dict[precursor_id] = {
            "features": intensities,
            "length": len(intensities),
            "label": row["Identified"],
        }
    if save_path:
        with open(save_path, "wb") as f:
            pickle.dump(data_dict, f)
    else:
        return data_dict


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
class PrecursorDataset(Dataset):
    def __init__(self, data_dict, normalize=True, max_len=None):
        # Filter out entries with zero total intensity
        self.data = [(k, v) for k, v in data_dict.items() if sum(v["features"]) > 0]

        if max_len is None:
            self.max_len = max(len(v["features"]) for _, v in self.data)
        else:
            self.max_len = max_len
        self.normalize = normalize

        Logger.debug("Max sequence length: %s", self.max_len)
        Logger.info(
            "Filtered dataset size after removing zero-intensity entries: %s",
            len(self.data),
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        precursor_id, sample = self.data[idx]
        intensity_seq = np.array(sample["features"], dtype=np.float32)

        if self.normalize and intensity_seq.max() > 0:
            intensity_seq = intensity_seq / intensity_seq.max()  # Normalize

        label = int(sample["label"])  # Convert boolean to int

        # Pad sequence to max length
        padded_seq = np.zeros(self.max_len, dtype=np.float32)
        padded_seq[: len(intensity_seq)] = intensity_seq

        # add channel dimension
        padded_seq = np.expand_dims(padded_seq, axis=0)

        return torch.tensor(padded_seq), torch.tensor(label, dtype=torch.float32)
