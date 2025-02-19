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


def estimate_theoretical_isotope_pattern(mz_value, charge):
    """
    Estimates a theoretical isotope distribution based on the averagine model.

    Parameters:
        mz_values (np.array): Experimental m/z values.
        charge (int): Charge state of the peptide.

    Returns:
        tuple: (theoretical_mz_values, theoretical_intensity_values)
    """
    # Estimate the neutral mass of the peptide
    neutral_mass = mz_value * charge  # Approximate precursor mass

    # Compute averagine-based elemental composition
    num_averagine_units = neutral_mass / 111.125  # Average mass of one averagine unit
    avg_composition = {
        "C": round(num_averagine_units * 4.9384),
        "N": round(num_averagine_units * 1.3577),
        "O": round(num_averagine_units * 1.4773),
        "S": round(num_averagine_units * 0.0417),
    }

    # Adjust hydrogen count to correct rounding errors
    corrected_mass = (
        avg_composition["C"] * 12.0000
        + avg_composition["N"] * 14.0031
        + avg_composition["O"] * 15.9949
        + avg_composition["S"] * 31.9721
    )
    avg_composition["H"] = round((neutral_mass - corrected_mass) / 1.0078)

    Logger.info(
        "Final avg composition %s with %s averagine units",
        avg_composition,
        num_averagine_units,
    )

    # Generate isotopic pattern using IsoSpecPy
    isotope_pattern = iso.IsoTotalProb(formula=avg_composition, prob_to_cover=0.9999)

    # Extract m/z values and intensities
    theoretical_mz_values = []
    theoretical_intensity_values = []
    for mz, prob in isotope_pattern:
        theoretical_mz_values.append(mz / charge)  # Convert to m/z space
        theoretical_intensity_values.append(prob)

    # Normalize intensities
    theoretical_intensity_values = np.array(theoretical_intensity_values)
    theoretical_intensity_values /= np.sum(theoretical_intensity_values)

    return np.array(theoretical_mz_values), theoretical_intensity_values


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
        ms1data, quad_high_mz, quad_low_mz, charge = extract_precursor_features(
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


def calculate_averagine_fit(
    ms1data,
    charge,
    align_by: str = "interpolate",
    quad_low_mz: float = None,
    quad_high_mz: float = None,
    n_digits: int = 3,
    plot: bool = False,
):
    """
    Computes the averagine fit metrics for an experimental isotopic pattern.

    Parameters:
        df (pd.DataFrame): DataFrame with columns "mz_values" and "intensity_values".
        charge (int): Charge state of the peptide.

    Returns:
        dict: RMS error, m/z bias, and explained intensity fraction.
    """
    # Extract experimental values
    mz_exp = ms1data["mz_values"].values
    mz_max_int = ms1data.loc[ms1data["intensity_values"].argmax(), "mz_values"]
    intensity_exp = ms1data["intensity_values"].values.astype(float)
    intensity_exp /= np.sum(intensity_exp)  # Normalize intensities

    # Estimate theoretical pattern
    theoretical_mz_values, theoretical_intensity_values = (
        estimate_theoretical_isotope_pattern(mz_max_int, charge)
    )
    wd = wasserstein_distance(
        mz_exp,
        theoretical_mz_values,
        u_weights=intensity_exp,
        v_weights=theoretical_intensity_values,
    )
    if plot:
        plot_comparison(
            x_true=mz_exp,
            y_true=intensity_exp,
            x_pred=theoretical_mz_values,
            y_pred=theoretical_intensity_values,
            # title="Comparison of experimental and theoretical isotopic patterns",
        )

        plt.suptitle(f"Wasserstein Dist.: {wd:.3f}")
        # plt.savefig("comparison.png", dpi=300)
        # plt.close()
    match align_by:
        case "interpolate":
            # Interpolate theoretical intensities at experimental m/z values
            interp_func = interp1d(
                theoretical_mz_values,
                theoretical_intensity_values,
                kind="linear",
                fill_value=0,
                bounds_error=False,
            )
            aligned_theoretical_intensities = interp_func(mz_exp)
            Logger.debug(
                "Length of exp intensities %s, aligned theoretical intensities: %s",
                len(intensity_exp),
                len(aligned_theoretical_intensities),
            )
        case "bin":
            exp_df = pd.DataFrame(
                {"mz_values": mz_exp, "intensity_values": intensity_exp}
            )
            fit_df = pd.DataFrame(
                {
                    "mz_values": theoretical_mz_values,
                    "intensity_values": theoretical_intensity_values,
                }
            )
            intensity_exp, mz_exp = preprocess_mz_intensity(
                exp_df, min_mz=quad_low_mz, max_mz=quad_high_mz, n_digits=n_digits
            )
            aligned_theoretical_intensities, theoretical_mz_values = (
                preprocess_mz_intensity(
                    fit_df, min_mz=quad_low_mz, max_mz=quad_high_mz, n_digits=n_digits
                )
            )
            Logger.debug(
                "Length of exp intensities %s, aligned theoretical intensities: %s",
                len(intensity_exp),
                len(aligned_theoretical_intensities),
            )

    # Compute RMS error
    rms_error = np.sqrt(np.mean((intensity_exp - aligned_theoretical_intensities) ** 2))

    # Compute m/z bias (weighted mean shift)
    mz_bias = np.sum((mz_exp - theoretical_mz_values[0]) * intensity_exp)

    theo_non_zero = np.where(
        (aligned_theoretical_intensities > 0) | (intensity_exp > 0)
    )[0]
    Logger.debug("Theoretical non-zero values: %s", theo_non_zero)
    corr = spearmanr(
        intensity_exp[theo_non_zero], aligned_theoretical_intensities[theo_non_zero]
    )
    Logger.debug("Spearman correlation: %s", corr[0])

    # Compute explained intensity fraction
    # total_theoretical_intensity = np.sum(theoretical_intensity_values)
    # total_matched_intensity = np.sum(aligned_theoretical_intensities)
    # explained_intensity_fraction = (
    #     total_matched_intensity / total_theoretical_intensity
    #     if total_theoretical_intensity > 0
    #     else 0
    # )

    return {
        "RMS Error": rms_error,
        "m/z Bias": mz_bias,
        "corr": corr[0],
        "wd": wd,
        # "Explained Intensity Fraction": explained_intensity_fraction,
    }


def calculate_precursor_averagine_fit(precursor_df, data, n_digits=3, save_path=None):
    result = {
        "Precursor ID": [],
        "Bin RMS Error": [],
        "Bin m/z Bias": [],
        "Bin Spearman Corr": [],
        "Interpolate RMS Error": [],
        "Interpolate m/z Bias": [],
        "Interpolate Spearman Corr": [],
        "Wasserstein Distance": [],
    }
    for idx, row in tqdm(
        precursor_df.iterrows(), total=len(precursor_df), desc="Processing precursors"
    ):
        precursor_id = int(row["Id"])
        # Logger.debug("Processing precursor ID: %s", idx)
        ms1data, quad_high_mz, quad_low_mz, charge = extract_precursor_features(
            precursor_id=precursor_id,
            data=data,
            precursor_df=precursor_df,
        )
        if ms1data.shape[0] == 0:
            Logger.warning("No MS1 data found for precursor ID: %s", idx)
            continue
        bin_fit = calculate_averagine_fit(
            ms1data=ms1data,
            charge=charge,
            align_by="bin",
            quad_low_mz=quad_low_mz,
            quad_high_mz=quad_high_mz,
            n_digits=n_digits,
        )
        interpolate_fit = calculate_averagine_fit(
            ms1data=ms1data,
            charge=charge,
            align_by="interpolate",
        )
        result["Bin RMS Error"].append(bin_fit["RMS Error"])
        result["Bin m/z Bias"].append(bin_fit["m/z Bias"])
        result["Bin Spearman Corr"].append(bin_fit["corr"])
        result["Interpolate RMS Error"].append(interpolate_fit["RMS Error"])
        result["Interpolate m/z Bias"].append(interpolate_fit["m/z Bias"])
        result["Interpolate Spearman Corr"].append(interpolate_fit["corr"])
        result["Wasserstein Distance"].append(interpolate_fit["wd"])
        result["Precursor ID"].append(precursor_id)
    if save_path:
        pd.DataFrame(result).to_csv(save_path, index=False)
    else:
        return pd.DataFrame(result)


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


def eval_precursor_fit(
    precursor_df,
    fit_df,
    save_dir,
    hue="Identified",
    hue_mapping=None,
    common_norm=True,
    **kwargs,
):
    precursor_with_fit = pd.merge(
        left=precursor_df,
        right=fit_df,
        left_on="Id",
        right_on="Precursor ID",
    )
    # Logger.info(
    #     "Merged precursor and fit dataframes columns: %s", precursor_with_fit.columns
    # )
    precursor_with_fit["log_intensity"] = np.log10(precursor_with_fit["Intensity"] + 1)

    # save the dataframe
    precursor_with_fit.to_csv(
        os.path.join(save_dir, "precursor_fit_merged.csv"), index=False
    )
    # plot distribution results
    plt.rcParams.update({"font.size": 14})
    fig, axes = plt.subplots(3, 3, figsize=(20, 10))
    axes = axes.flatten()  # Flatten the 2D array of axes into 1D

    for idx, col in enumerate(
        [
            "Bin RMS Error",
            "Bin m/z Bias",
            "Bin Spearman Corr",
            "Interpolate RMS Error",
            "Interpolate m/z Bias",
            "Interpolate Spearman Corr",
            "Wasserstein Distance",
        ]
    ):
        sns.kdeplot(
            data=precursor_with_fit,
            x=col,
            hue=hue,
            ax=axes[idx],
            common_norm=common_norm,
            palette=hue_mapping,
            **kwargs,
        )
        # plt.title("Random 15min 3R")

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            save_dir,
            f"precursor_fit_metrics_hue_{hue}_common_norm_{str(common_norm)}.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    precursor_with_fit["NotIdentified"] = ~precursor_with_fit["Identified"]
    precursor_with_fit["Wasserstein Similarity"] = (
        precursor_with_fit["Wasserstein Distance"].max()
        - precursor_with_fit["Wasserstein Distance"]
    )
    # calc_fdr_and_thres(
    #     pred_df=precursor_with_fit,
    #     score_col="Wasserstein Similarity",
    #     return_plot=True,
    #     save_dir=save_dir,
    #     decoy_col="NotIdentified",
    # )
    return precursor_with_fit


class PrecursorDataset(Dataset):
    def __init__(
        self,
        data_dict,
        normalize=True,
        max_len=None,
        sampling_strategy=None,
        filter_class: int = None,
        scale_factor=1.0,
        add_channel_dim=True,
        return_ids=False,
    ):
        # Filter out entries with zero total intensity
        self.data = [(k, v) for k, v in data_dict.items() if sum(v["features"]) > 0]

        if max_len is None:
            self.max_len = max(len(v["features"]) for _, v in self.data)
        else:
            self.max_len = max_len

        self.normalize = normalize
        self.scale_factor = scale_factor
        self.return_ids = return_ids
        # Extract features and labels
        X = [v["features"] for _, v in self.data]
        y = [int(v["label"]) for _, v in self.data]
        ids = [k for k, _ in self.data]
        # Convert to numpy array
        X = np.array(
            [self._pad_and_normalize(seq, self.scale_factor) for seq in X],
            dtype=np.float32,
        )
        y = np.array(y, dtype=np.int64)
        ids = np.array(ids, dtype=np.int64)
        if filter_class is not None:
            filtered_indices = np.where(y == filter_class)[0]
            X_resampled = X[filtered_indices]
            y_resampled = y[filtered_indices]
            ids = ids[filtered_indices]
            Logger.info(
                "Dataset class counts after filtering for class %s: %s",
                filter_class,
                np.bincount(y_resampled),
            )
        # if pos_only:
        #     pos_indices = np.where(y == 1)[0]
        #     X_resampled = X[pos_indices]
        #     y_resampled = y[pos_indices]
        #     Logger.info(
        #         "Dataset class counts after filtering: %s", np.bincount(y_resampled)
        #     )
        else:
            # Apply sampling strategy if specified
            if sampling_strategy == "oversample":
                Logger.info(
                    "Dataset class counts before oversampling: %s", np.bincount(y)
                )
                sampler = RandomOverSampler(sampling_strategy="auto", random_state=42)
                # Combine X and ids for resampling
                X_combined = np.column_stack((X, ids))
                X_combined_resampled, y_resampled = sampler.fit_resample(X_combined, y)
                # Split back X and ids
                X_resampled = X_combined_resampled[:, :-1]
                ids = X_combined_resampled[:, -1].astype(np.int64)
                Logger.info(
                    "Dataset class counts after oversampling: %s",
                    np.bincount(y_resampled),
                )
            elif sampling_strategy == "undersample":
                Logger.info(
                    "Dataset class counts before undersampling: %s", np.bincount(y)
                )
                sampler = RandomUnderSampler(sampling_strategy="auto", random_state=42)
                # Combine X and ids for resampling
                X_combined = np.column_stack((X, ids))
                X_combined_resampled, y_resampled = sampler.fit_resample(X_combined, y)
                # Split back X and ids
                X_resampled = X_combined_resampled[:, :-1]
                ids = X_combined_resampled[:, -1].astype(np.int64)
                Logger.info(
                    "Dataset class counts after undersampling: %s",
                    np.bincount(y_resampled),
                )
            else:
                X_resampled, y_resampled = X, y  # No resampling
        if add_channel_dim:
            self.X = torch.tensor(X_resampled).unsqueeze(1)
        else:
            self.X = torch.tensor(X_resampled)
        self.y = torch.tensor(y_resampled, dtype=torch.float32)
        self.ids = torch.tensor(ids, dtype=torch.int64)
        Logger.info(f"Dataset size after preprocessing: {len(self.y)}")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if self.return_ids:
            return self.X[idx], self.y[idx], self.ids[idx]
        else:
            return self.X[idx], self.y[idx]

    def _pad_and_normalize(self, intensity_seq, scale_factor=1.0):
        """Helper function to normalize and pad sequences"""
        if self.normalize and np.max(intensity_seq) > 0:
            intensity_seq = (
                intensity_seq / np.max(intensity_seq) * scale_factor
            )  # Normalize

        padded_seq = np.zeros(self.max_len, dtype=np.float32)
        padded_seq[: len(intensity_seq)] = intensity_seq

        return padded_seq
