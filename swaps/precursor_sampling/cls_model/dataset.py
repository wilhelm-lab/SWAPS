import numpy as np
import pandas as pd
import logging
import seaborn as sns
import matplotlib.pyplot as plt
from ..explore_precursors import extract_precursor_features
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import pickle

Logger = logging.getLogger(__name__)


def preprocess_mz_intensity(df, min_mz, max_mz, n_digits=3):
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


# Custom Dataset
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
