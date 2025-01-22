from curses import meta
import logging
from typing import Literal, List

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import numpy as np
import pandas as pd
from numba import jit
import tensorflow as tf
from keras.layers import (
    Conv1D,
    Dense,
    Dropout,
    GlobalMaxPooling1D,
    GlobalAveragePooling1D,
    Masking,
)
from keras import metrics, regularizers
from keras.losses import BinaryCrossentropy
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import (
    minmax_scale,
    scale,
)
import math
from utils.tools import match_time_to_scan
from utils.plot import plot_pie, save_plot
from keras.models import Model
from keras.layers import (
    Input,
    Masking,
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    concatenate,
)
from keras import regularizers


Logger = logging.getLogger(__name__)

METRICS = [
    metrics.BinaryCrossentropy(name="cross entropy"),  # same as model's loss
    metrics.MeanSquaredError(name="Brier score"),
    metrics.TruePositives(name="tp"),
    metrics.FalsePositives(name="fp"),
    metrics.TrueNegatives(name="tn"),
    metrics.FalseNegatives(name="fn"),
    metrics.BinaryAccuracy(name="accuracy"),
    metrics.Precision(name="precision"),
    metrics.Recall(name="recall"),
    metrics.AUC(name="auc"),
    metrics.AUC(name="prc", curve="PR"),  # precision-recall curve
]

########################################################################################
# The following functions are used for annotating peak results


def match_peaks_to_exp(
    ref_exp_inner_df: pd.DataFrame,
    peak_results: pd.DataFrame,
    n_peaks: int = 1,
):
    """Match peaks to experimental peaks and return True or False in "matched" column"""
    # TODO: currently only supports one exp pcm RT per precursor id
    peak_results = pd.merge(
        left=peak_results,
        right=ref_exp_inner_df[
            [
                "id",
                "Calibrated retention time start",
                "Calibrated retention time",
                "Calibrated retention time finish",
            ]
        ],
        on="id",
        how="left",
    )
    peak_results["RT_apex_diff"] = abs(
        peak_results["apex_time"] - peak_results["Calibrated retention time"]
    )

    Logger.info("Use RT range as reference for peak selection.")
    peak_results["RT_start_diff"] = abs(
        peak_results["start_time"] - peak_results["Calibrated retention time start"]
    )
    peak_results["RT_end_diff"] = abs(
        peak_results["end_time"] - peak_results["Calibrated retention time finish"]
    )
    peak_results["RT_diff_sum"] = (
        peak_results["RT_start_diff"] + peak_results["RT_end_diff"]
    )

    # drop rows where "RT_diff_sum" is NaN
    peak_results.dropna(subset=["RT_diff_sum"], inplace=True)

    peak_results.loc[
        peak_results.groupby("id")["RT_diff_sum"]
        .nsmallest(n_peaks)
        .index.get_level_values(1),
        "matched",
    ] = True
    peak_results["matched"].fillna(False, inplace=True)

    return peak_results


def report_peak_property_distr(
    peak_results: pd.DataFrame, property_col: str, save_dir: str | None = None
):
    matched = peak_results[peak_results["matched"]][property_col]
    unmatched = peak_results[~peak_results["matched"]][property_col]

    plot_pie(
        sizes=matched.value_counts().values,
        labels=matched.value_counts().index.values,
        title=f"{property_col} Distribution for Matched Peaks",
        save_dir=save_dir,
        fig_spec_name=f"Matched_{property_col}",
        accumulative_threshold=0.95,
    )

    plot_pie(
        sizes=unmatched.value_counts().values,
        labels=unmatched.value_counts().index.values,
        title=f"{property_col} Distribution for Unmatched Peaks",
        save_dir=save_dir,
        fig_spec_name=f"Unmatched_{property_col}",
        accumulative_threshold=0.95,
    )

    top_values = peak_results[property_col].value_counts().nlargest(10)

    sns.countplot(
        data=peak_results[peak_results[property_col].isin(top_values.index)],
        x=property_col,
        hue="matched",
    )

    plt.title(f"{property_col} Distribution")
    plt.xlabel(property_col)
    plt.ylabel("Count")

    save_plot(save_dir, fig_type_name="Count", fig_spec_name=f"{property_col}")


def report_peak_width_distr(peak_results: pd.DataFrame, save_dir: str | None = None):
    """Generate a report on peak  for the peak results dataframe"""
    # Calculate width distribution for matched and unmatched peaks
    matched_widths = peak_results[peak_results["matched"]]["peak_width"]
    unmatched_widths = peak_results[~peak_results["matched"]]["peak_width"]

    plot_pie(
        sizes=matched_widths.value_counts().values,
        labels=matched_widths.value_counts().index.values,
        title="Peak Width Distribution for Matched Peaks",
        save_dir=save_dir,
        fig_spec_name="MatchedPeakWidth",
        accumulative_threshold=0.95,
    )

    # Generate pie chart for unmatched peaks
    plot_pie(
        sizes=unmatched_widths.value_counts().values,
        labels=unmatched_widths.value_counts().index.values,
        title="Peak Width Distribution for Unmatched Peaks",
        save_dir=save_dir,
        fig_spec_name="UnmatchedPeakWidth",
        accumulative_threshold=0.95,
    )

    # Get the top 10 width value counts
    top_widths = peak_results["peak_width"].value_counts().nlargest(10)

    # Plot the count plot for the top 10 widths
    sns.countplot(
        data=peak_results[peak_results["peak_width"].isin(top_widths.index)],
        x="peak_width",
        hue="matched",
    )

    # Set the title and labels
    plt.title("Peak Width Distribution")
    plt.xlabel("Peak Width")
    plt.ylabel("Count")

    save_plot(save_dir, fig_type_name="Count", fig_spec_name="PeakWidth")


########################################################################################
# The following functions are used for preparing the input for peak classification
def transform_activation(
    activation_df: pd.DataFrame,
    log_intensity: bool = False,
    standardize: Literal["std", "minmax"] | None = "std",
):
    """Transform the activation dataframe to a standardized (and/or log) dataframe"""
    activation_values = activation_df.values
    if log_intensity:
        activation_values = np.log(activation_values + 1)
    match standardize:
        case "std":
            activation_values_std = scale(activation_values, axis=1)
        case "minmax":
            activation_values_std = minmax_scale(
                activation_values, axis=1, feature_range=(0, 1)
            )
        case None:
            activation_values_std = activation_values
    activation_df_std = pd.DataFrame(
        data=np.array(activation_values_std),
        index=activation_df.index,
        columns=activation_df.columns,
    )
    return activation_df_std


def prepare_seq_and_label(
    activation_df: pd.DataFrame,
    log_intensity: bool = False,
    standardize: Literal["std", "minmax"] | None = "std",
    ms1scans_no_array: pd.DataFrame | None = None,
    data_unit: Literal["peak", "activation"] = "peak",
    peak_results: pd.DataFrame | None = None,
    margin: int = 3,
    method: Literal["mask", "clip"] = "mask",
    ref_exp_inner_df: pd.DataFrame | None = None,
    pad: bool = True,
    pad_value: float = -1.0,
    pad_pos: Literal["post", "two_side"] = "two_side",
    max_len: int | None = None,
) -> (np.ndarray, np.ndarray):
    """
    Prepare the input as arrays for peak classification.

    :param activation_df: DataFrame containing activation data.
    :param log_intensity: Flag indicating whether to apply logarithmic transformation to intensity values.
    :param standardize: Method for standardizing the data. Options are "std" for standardization, \
        "minmax" for min-max scaling, or None for no standardization.
    :param ms1scans_no_array: DataFrame containing MS1 scan data for matching time to scan. \
        Only required if scan information is not available.
    :param data_unit: Unit of the data to be prepared. \
        Options are "peak" for one peak per data point or \
             "activation" for one activation curve per data point.
    :param peak_results: DataFrame containing peak results. Only required for data_unit = "peak". \
        Must contain "id", "start_scan", "end_scan", \
            "RT_search_left_scan", and "RT_search_right_scan" columns.
    :param margin: Margin value used for peak sequence. Only required for data_unit = "peak".
    :param method: Method for peak selection. Only required for data_unit = "peak". \
        Options are "mask" for masking the peaks or "clip" for clipping the peaks.
    :param ref_exp_inner_df: DataFrame containing reference experiment inner data. \
        Only required for data_unit = "activation".
    
    :return: Arrays containing the prepared input sequences and labels.
    """
    activation_df_std = transform_activation(activation_df, log_intensity, standardize)
    if "matched" not in peak_results.columns:
        Logger.info("No matched results found. Set all to False.")
        peak_results["matched"] = False
    match data_unit:
        case "peak":
            if "RT_search_left_scan" not in peak_results.columns:
                Logger.info("Match time to scan for the peak results dataframe.")
                peak_results = match_time_to_scan(
                    peak_results,
                    time_cols=["RT_search_left", "RT_search_right"],
                    ms1scans_no_array=ms1scans_no_array,
                )
            peaks = peak_results.apply(
                _prepare_peak_roi_row,
                axis=1,
                activation_df_std=activation_df_std,
                margin=margin,
                method=method,
            )
            seq = peaks.values
            label = peak_results["matched"].values
            if pad:
                seq = pad_seq(seq, value=pad_value, pad_pos=pad_pos, maxlen=max_len)
        case "activation":
            label, seq = _prepare_activation_seq_and_label(
                ref_exp_inner_df=ref_exp_inner_df,
                activation_df_std=activation_df_std,
                ms1cans_no_array=ms1scans_no_array,
            )
            if pad:
                seq = pad_seq(seq, value=pad_value, pad_pos=pad_pos, maxlen=max_len)
                label = pad_seq(label, value=pad_value, pad_pos=pad_pos, maxlen=max_len)
    Logger.info("Sequence shape: %s", seq.shape)
    Logger.info("Label shape: %s", label.shape)
    assert seq.shape[0] == label.shape[0], "Sequence and label shape mismatch!"
    return seq, label.astype(np.int32)


def _prepare_peak_roi_row(
    peak_results_row: pd.Series,
    activation_df_std: pd.DataFrame,
    margin: int = 3,
    method: Literal["mask", "clip"] = "mask",
):
    """Get the region of interest from the activation dataframe"""
    start_index = peak_results_row["start_scan"] - margin
    end_index = peak_results_row["end_scan"] + margin
    match method:
        case "mask":  # get roi by masking out the rest
            original_row = activation_df_std.loc[peak_results_row["id"]].values
            new_row = np.zeros_like(original_row)
            new_row[start_index : end_index + 1] = original_row[
                start_index : end_index + 1
            ]
            return new_row[
                peak_results_row["RT_search_left_scan"] : peak_results_row[
                    "RT_search_right_scan"
                ]
                + 1
            ]
        case "clip":  # get roi by clipping the rest
            return activation_df_std.loc[
                peak_results_row["id"],
                start_index : end_index + 1,
            ].values


def _prepare_activation_seq_and_label(
    activation_df_std: pd.DataFrame,
    ref_exp_inner_df: pd.DataFrame,
    ms1cans_no_array: pd.DataFrame | None = None,
):
    """Prepare the label for data_unit = "activation" """

    ref_exp_inner_df.index = ref_exp_inner_df.index.astype(int)
    if "Calibrated retention time start_scan" not in ref_exp_inner_df.columns:
        Logger.info("Match time to scan for the reference experiment dataframe.")
        ref_exp_inner_df = match_time_to_scan(
            ref_exp_inner_df,
            time_cols=[
                "Calibrated retention time start",
                "Calibrated retention time finish",
                "RT_search_left",
                "RT_search_right",
            ],
            ms1scans_no_array=ms1cans_no_array,
        )
    label_and_seq = ref_exp_inner_df.apply(
        _prepare_activation_row,
        axis=1,
        result_type="expand",
        activation_df_std=activation_df_std,
    )
    label = label_and_seq[0].values
    seq = label_and_seq[1].values
    return label, seq


def _prepare_activation_row(
    ref_exp_inner_df_row: pd.Series, activation_df_std: pd.DataFrame
):
    precursor_id = ref_exp_inner_df_row.id
    example_seq = activation_df_std.loc[precursor_id].values
    label = np.zeros_like(example_seq)
    label[
        ref_exp_inner_df_row[
            "Calibrated retention time start_scan"
        ] : ref_exp_inner_df_row["Calibrated retention time finish_scan"]
    ] = 1
    seq = activation_df_std.loc[precursor_id].values
    seq = seq[
        ref_exp_inner_df_row["RT_search_left_scan"] : ref_exp_inner_df_row[
            "RT_search_right_scan"
        ]
        + 1
    ]
    label = label[
        ref_exp_inner_df_row["RT_search_left_scan"] : ref_exp_inner_df_row[
            "RT_search_right_scan"
        ]
        + 1
    ]
    assert seq.shape[0] == label.shape[0], "Sequence and label shape mismatch!"
    return label, seq


def pad_seq(
    seq: np.ndarray,
    maxlen: int | None = None,
    value: float = -1.0,
    pad_pos: Literal["post", "two_side"] = "two_side",
):
    """Pad the sequence to the same length."""
    if maxlen is None:
        maxlen = max(map(len, seq))
        Logger.debug("Max length: %s", maxlen)
    match pad_pos:
        case "post":
            padded_seq = np.array(
                [_pad_seq_post(s, maxlen, value) for s in seq]
            ).astype(np.float32)
        case "two_side":
            padded_seq = np.array(
                [_pad_seq_two_side(s, maxlen, value) for s in seq]
            ).astype(np.float32)
    Logger.info("Padded sequence shape: %s", padded_seq.shape)
    return padded_seq


@jit(nopython=True)
def _pad_seq_post(peak, maxlen, value):
    result = np.full((maxlen,), value)
    result[: len(peak)] = peak
    return result


@jit(nopython=True)
def _pad_seq_two_side(peak, maxlen, value):
    if len(peak) > maxlen:
        diff = len(peak) - maxlen
        start = diff // 2
        end = start + maxlen
        return peak[start:end]
    else:
        result = np.full((maxlen,), value)
        pad_length = maxlen - len(peak)
        before = pad_length // 2
        result[before : before + len(peak)] = peak
        return result


def split_data(
    seq: np.ndarray,
    # label: np.ndarray,
    # metadata: np.ndarray | None = None,
    precursor_id: np.ndarray | None = None,
    test_ratio: float = 0.2,
    val_ratio: float = 0.2,
    random_seed: int | None = None,
):
    """Split the data into training and testing sets.
    :param seq: Input sequences.
    :param label: Labels.
    :param precursor_id: Precursor id for splitting data by precursor id. \
        If not available, split data randomly.
    :param train_ratio: Ratio of the training set.
    :param random_seed: Random seed for reproducibility.

    :return: Training and testing sets.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    train_ratio = 1 - test_ratio - val_ratio
    if precursor_id is not None:
        Logger.info("Split data by precursor id.")
        unique_precursor_id = np.unique(precursor_id)
        np.random.shuffle(unique_precursor_id)
        train_precursor_id = unique_precursor_id[
            : int(len(unique_precursor_id) * train_ratio)
        ]
        test_precursor_id = unique_precursor_id[
            -int(len(unique_precursor_id) * test_ratio) :
        ]
        train_indices = np.where(np.isin(precursor_id, train_precursor_id))[0]
        test_indices = np.where(np.isin(precursor_id, test_precursor_id))[0]
        if val_ratio > 0:
            val_precursor_id = unique_precursor_id[
                int(len(unique_precursor_id) * train_ratio) : int(
                    len(unique_precursor_id) * (train_ratio + val_ratio)
                )
            ]
            val_indices = np.where(np.isin(precursor_id, val_precursor_id))[0]
    else:
        Logger.info("Precursor id not available, splitting data randomly.")
        indices = np.arange(len(seq))
        np.random.shuffle(indices)
        train_indices = indices[: int(len(indices) * train_ratio)]
        test_indices = indices[-int(len(indices) * test_ratio) :]
        if val_ratio > 0:
            val_indices = indices[
                int(len(indices) * train_ratio) : int(
                    len(indices) * (train_ratio + val_ratio)
                )
            ]

    if val_ratio > 0:
        return (
            train_indices,
            val_indices,
            test_indices,
        )
    return train_indices, test_indices


def evaluate_class_distribution(label: np.ndarray):
    """Evaluate the class distribution of the label.

    return initial_bias and class_weight
    """
    class_distribution = pd.Series(label).value_counts(normalize=False)
    Logger.info("Class distribution: %s", class_distribution)
    pos = class_distribution[1]
    neg = class_distribution[0]
    inital_bias = np.log([pos / neg])
    total = neg + pos
    weight_for_0 = (1 / neg) * (total / 2.0)
    weight_for_1 = (1 / pos) * (total / 2.0)
    class_weight = {0: weight_for_0, 1: 1 / weight_for_1}
    initial_distr = np.array([1 - (pos / total), pos / total])
    return inital_bias, class_weight, initial_distr


# TODO: the following resampling leads to val performance always classifying 0
# def make_tf_dataset(
#     seq: np.ndarray,
#     label: np.ndarray,
#     balance_by_resample: bool = False,
#     batch_size: int = 32,
# ):
#     """Make a tf.data.Dataset from the input sequences and labels."""

#     def make_ds(features, labels):
#         ds = tf.data.Dataset.from_tensor_slices((features, labels))
#         # ds = ds.shuffle(10000).repeat() # Don't shuffle since it was already done in split_data
#         return ds

#     if balance_by_resample:
#         pos_features = seq[label == 1]
#         neg_features = seq[label == 0]
#         pos_labels = label[label == 1]
#         neg_labels = label[label == 0]

#         pos_ds = make_ds(pos_features, pos_labels)
#         neg_ds = make_ds(neg_features, neg_labels)
#         resampled_ds = tf.data.Dataset.sample_from_datasets(
#             [pos_ds, neg_ds], weights=[0.5, 0.5]
#         )
#         Logger.info("Resampled dataset: %s", len(list(resampled_ds)))
#         resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
#         return resampled_ds
#     else:
#         ds = make_ds(seq, label)
#         Logger.info("Dataset: %s", len(list(ds)))
#         ds = ds.batch(batch_size).prefetch(2)
#         return ds


def rank_among_collinear_candidates(item, list_without_item):
    """Rank the item among collinear candidates."""
    list_without_item.append(item)
    list_length = len(list_without_item)
    item_rank = list_without_item.index(item) + 1
    if list_length == 1:
        return 1  # Maximum score for the only item in the list
    else:
        # Scale factor to ensure scores are between 0 and 1
        scale_factor = 1.0 / (1 - math.exp(-1))
        return scale_factor * (
            1 - math.exp(-(list_length - item_rank + 1) / list_length)
        )


########################################################################################
# The following functions are used for model training and tuning


def make_cnn_model(config: dict, output_bias=None):
    """Make the CNN model."""
    if output_bias is not None:
        Logger.info("Output bias using : %s", output_bias)
        output_bias = tf.keras.initializers.Constant(output_bias)

    model = Sequential()
    model.add(Masking(mask_value=-1))
    model.add(
        Conv1D(
            filters=config["conv1_n_filters"],
            kernel_size=config["conv1_kernel_size"],
            activation="relu",
        )
    )
    model.add(
        Conv1D(
            filters=config["conv2_n_filters"],
            kernel_size=config["conv2_kernel_size"],
            activation="relu",
        )
    )
    model.add(
        Conv1D(
            filters=config["conv3_n_filters"],
            kernel_size=config["conv3_kernel_size"],
            activation="relu",
        )
    )
    model.add(GlobalAveragePooling1D())
    model.add(
        Dense(
            config["dense1_n_units"],
            kernel_regularizer=regularizers.l2(config["dense1_reg_rate"]),
            activation="relu",
            name="dense_1",
        )
    )
    model.add(
        Dense(
            config["dense2_n_units"],
            kernel_regularizer=regularizers.l2(config["dense2_reg_rate"]),
            activation="relu",
            name="dense_2",
        )
    )
    model.add(
        Dense(
            config["dense3_n_units"],
            kernel_regularizer=regularizers.l2(config["dense3_reg_rate"]),
            activation="relu",
            name="dense_3",
        )
    )
    model.add(Dropout(config["dropout_rate"], name="dropout_1"))
    model.add(
        Dense(
            1,
            activation="sigmoid",
            name="dense_prediction",
            bias_initializer=output_bias,
        )
    )

    model.compile(
        optimizer=Adam(learning_rate=config["learning_rate"]),
        loss=BinaryCrossentropy(),
        metrics=METRICS,
    )
    return model


def make_conv1d_encoder(config: dict):
    peak_seq_input = Input(
        shape=(config["peak_seq_maxlen"], config["peak_seq_features"]),
        name="peak_seq",
    )
    peak_seq_features = Masking(mask_value=-1)(peak_seq_input)
    peak_seq_features = Conv1D(
        filters=config["conv1_n_filters"],
        kernel_size=config["conv1_kernel_size"],
        activation="relu",
    )(peak_seq_features)
    peak_seq_features = Conv1D(
        filters=config["conv2_n_filters"],
        kernel_size=config["conv2_kernel_size"],
        activation="relu",
    )(peak_seq_features)
    peak_seq_features = Conv1D(
        filters=config["conv3_n_filters"],
        kernel_size=config["conv3_kernel_size"],
        activation="relu",
    )(peak_seq_features)
    conv1d_output = GlobalAveragePooling1D()(peak_seq_features)
    # conv1d_block = Model(inputs=inputs, outputs=conv1d_output)
    return conv1d_output


def make_dense_classifier(config: dict, output_bias=None):
    metadata_input = Input(shape=config["metadata_input_shape"], name="metadata")
    peak_seq_input = Input(
        shape=(config["peak_seq_maxlen"], config["peak_seq_features"]),
        name="peak_seq",
    )
    peak_seq_features = Masking(mask_value=-1)(peak_seq_input)
    peak_seq_features = Conv1D(
        filters=config["conv1_n_filters"],
        kernel_size=config["conv1_kernel_size"],
        activation="relu",
    )(peak_seq_features)
    peak_seq_features = Conv1D(
        filters=config["conv2_n_filters"],
        kernel_size=config["conv2_kernel_size"],
        activation="relu",
    )(peak_seq_features)
    peak_seq_features = Conv1D(
        filters=config["conv3_n_filters"],
        kernel_size=config["conv3_kernel_size"],
        activation="relu",
    )(peak_seq_features)
    conv1d_output = GlobalAveragePooling1D()(peak_seq_features)

    x = concatenate([conv1d_output, metadata_input])
    x = Dense(
        config["dense1_n_units"],
        kernel_regularizer=regularizers.l2(config["dense1_reg_rate"]),
        activation="relu",
        name="dense_1",
    )(x)
    x = Dense(
        config["dense2_n_units"],
        kernel_regularizer=regularizers.l2(config["dense2_reg_rate"]),
        activation="relu",
        name="dense_2",
    )(x)
    x = Dense(
        config["dense3_n_units"],
        kernel_regularizer=regularizers.l2(config["dense3_reg_rate"]),
        activation="relu",
        name="dense_3",
    )(x)
    x = Dropout(config["dropout_rate"], name="dropout_1")(x)
    output = Dense(
        1, activation="sigmoid", name="dense_prediction", bias_initializer=output_bias
    )(x)

    model = Model(inputs=[peak_seq_input, metadata_input], outputs=output)
    # model.compile(
    #     optimizer=Adam(learning_rate=config["learning_rate"]),
    #     loss=BinaryCrossentropy(),
    #     metrics=METRICS,
    # )
    return model


def make_multi_input_model(config: dict, output_bias=None):
    """Make the multi input model."""
    if output_bias is not None:
        Logger.info("Output bias using : %s", output_bias)
        output_bias = tf.keras.initializers.Constant(output_bias)

    multi_input_model = make_dense_classifier(config, output_bias=output_bias)

    multi_input_model.compile(
        optimizer=Adam(learning_rate=config["learning_rate"]),
        loss=BinaryCrossentropy(),
        metrics=METRICS,
    )
    return multi_input_model


def model_tuner(hp, maxlen: int = 361, initial_bias: List[float] | None = -1.04778782):
    model = Sequential()
    model.add(Masking(mask_value=-1))  # Masking layer with the padding marker
    hp_conv1_n_filters = hp.Int("conv1_n_filters", min_value=8, max_value=128, step=8)
    hp_conv1_kernel_size = hp.Int("conv1_kernel_size", min_value=3, max_value=9, step=2)
    model.add(
        Conv1D(
            filters=hp_conv1_n_filters,
            kernel_size=hp_conv1_kernel_size,
            activation="relu",
            input_shape=(None, maxlen),
        )
    )
    hp_conv2_n_filters = hp.Int("conv2_n_filters", min_value=8, max_value=128, step=8)
    hp_conv2_kernel_size = hp.Int("conv2_kernel_size", min_value=3, max_value=9, step=2)
    model.add(
        Conv1D(
            filters=hp_conv2_n_filters,
            kernel_size=hp_conv2_kernel_size,
            activation="relu",
            input_shape=(None, maxlen),
        )
    )
    model.add(GlobalMaxPooling1D())
    hp_dense1_n = hp.Int("dense1_n", min_value=8, max_value=128, step=8)
    hp_dense1_reg_rate = hp.Choice("dense1_reg_rate", [0.0, 0.1, 0.01, 0.001, 0.0001])
    model.add(
        Dense(
            hp_dense1_n,
            kernel_regularizer=regularizers.l2(hp_dense1_reg_rate),
            activation="relu",
            name="dense_1",
        )
    )
    hp_dropout_rate = hp.Choice("dropout_rate", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    model.add(Dropout(hp_dropout_rate, name="dropout_1"))
    if isinstance(initial_bias, float):
        output_bias = tf.keras.initializers.Constant(initial_bias)
        Logger.debug("Output bias: %s", output_bias)
    else:
        output_bias = None
    model.add(
        Dense(
            1,
            activation="sigmoid",
            bias_initializer=output_bias,
            name="dense_prediction",
        )
    )

    hp_learning_rate = hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])
    model.compile(
        optimizer=Adam(learning_rate=hp_learning_rate),
        loss=BinaryCrossentropy(),
        metrics=METRICS,
    )
    return model


########################################################################################
# The following functions are used for evaluating the classification performance


def produce_pred_df(
    model: Sequential,
    data: tf.data.Dataset,
    precursor_id: np.ndarray,
    data_idx: np.ndarray,
    label: np.ndarray | None = None,
):
    """Produce a dataframe containing the predictions and true labels."""
    if label is None:
        label = np.zeros_like(data_idx)
    y_pred = model.predict(data)
    df = pd.DataFrame(
        {
            "y_pred_prob": y_pred.flatten(),
            "y_true": label.flatten(),
            "precursor_id": precursor_id[data_idx],
        }
    )
    return df


def plot_pred_distr(
    df: pd.DataFrame, save_dir: str | None = None, fig_spec_name: str = ""
):
    sns.kdeplot(
        data=df,
        x="y_pred_prob",
        hue="y_true",
        fill=True,
        common_norm=True,
        alpha=0.5,
        linewidth=0,
    )
    plt.title("Prediction Distribution")
    plt.xlabel("Prediction Score")
    save_plot(save_dir=save_dir, fig_type_name="PredDistr", fig_spec_name=fig_spec_name)


def evaluate_id_based_cls(df_test: pd.DataFrame, top_n: int = 1):
    """Evaluate the classification accuracy based on highest scored peak per precursor_id."""
    # In df_test, only consider precursor with true peak labblled
    grouped = df_test.groupby('precursor_id')['y_true'].sum()
    no_groundtruth_ids = grouped[grouped == 0].index
    print(f"Number of precursors without ground truth: {len(no_groundtruth_ids)}")
    df_test_truepeaks = df_test[~df_test['precursor_id'].isin(no_groundtruth_ids)]
    df_test_filtered = get_top_n_scored_peaks_by_precursor(df_test_truepeaks, top_n)

    # only keep the true peak of each precursor_id
    # df_test_filtered = df_test_filtered[df_test_filtered["y_true"] == 1]

    # Calculate accuracy
    if top_n > 1:
        df_test_filtered = df_test_filtered.groupby("precursor_id").agg(
            {"y_true": "sum"}
        )
        df_test_filtered["y_pred"] = 1

    accuracy = accuracy_score(df_test_filtered["y_true"], df_test_filtered["y_pred"])
    print(f"Accuracy for top {top_n} peaks: {accuracy:.4f}")
    return df_test_filtered


def get_top_n_scored_peaks_by_precursor(
    df_test, top_n, id_col: str = "precursor_id", score_col: str = "y_pred_prob"
):
    df_test_filtered = df_test.copy()
    if top_n == 1:
        # For each precursor_id, assign 1 to the highest probability and 0 to the rest
        df_test_filtered["y_pred"] = (
            df_test_filtered.groupby(id_col)[score_col]
            .transform(lambda x: x == x.max())
            .astype(int)
        )
    else:
        # For each precursor_id, assign 1 to the top_n highest probabilities and 0 to the rest
        df_test_filtered["rank"] = df_test_filtered.groupby(id_col)[score_col].rank(
            method="first", ascending=False
        )
        df_test_filtered["y_pred"] = (df_test_filtered["rank"] <= top_n).astype(int)
        df_test_filtered.drop(columns=["rank"], inplace=True)
    df_test_filtered = df_test_filtered[df_test_filtered["y_pred"] == 1]

    return df_test_filtered


def plot_activation_and_score(
    peak_results,
    df_test,
    activation,
    precursor_id: int,
    cos_dist: pd.DataFrame | None = None,
    save_dir: str | None = None,
    log_int: bool = True,
    show_pred_peak: bool = True,
):
    """Plot the activation and score for a precursor_id."""
    peak_results_filtered = peak_results[peak_results["id"] == precursor_id]
    df_test_filtered = df_test[df_test["precursor_id"] == precursor_id]
    peak_results_score = pd.merge(
        left=peak_results_filtered,
        right=df_test_filtered,
        right_index=True,
        left_index=True,
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(activation.loc[precursor_id])
    if cos_dist is not None:
        ax1 = ax.twinx()
        ax1.plot(cos_dist.loc[precursor_id], color="red")
    ax.set_title(f"Activation for precursor_id {precursor_id}")

    y_min, y_max = ax.get_ylim()
    ax.set_xlim(
        peak_results_score["RT_search_left_scan"].min(),
        peak_results_score["RT_search_right_scan"].max(),
    )
    for row in peak_results_score.iterrows():
        ax.fill_between(
            [
                row[1]["start_scan"],
                row[1]["end_scan"],
            ],
            [y_min, y_min],
            [y_max, y_max],
            color="grey",
            alpha=row[1]["y_pred_prob"],
            label="Peak Score",
        )
    true_peak = peak_results_score[peak_results_score["matched"] == 1]
    ax.add_patch(
        Rectangle(
            (true_peak["start_scan"].min(), y_min),
            true_peak["end_scan"].max() - true_peak["start_scan"].min(),
            y_max - y_min,
            color="green",
            alpha=1,
            linewidth=2.5,
            fill=False,
            label="True Peak",
        )
    )
    pred_peak = peak_results_score[
        peak_results_score["y_pred_prob"].max() == peak_results_score["y_pred_prob"]
    ]
    if show_pred_peak:
        ax.add_patch(
            Rectangle(
                (pred_peak["start_scan"].min(), y_min),
                pred_peak["end_scan"].max() - pred_peak["start_scan"].min(),
                y_max - y_min,
                color="yellow",
                alpha=1,
                linewidth=2.5,
                fill=False,
                label="True Peak",
            )
        )
    ax.vlines(
        np.mean([true_peak["RT_search_left_scan"], true_peak["RT_search_right_scan"]]),
        y_min,
        y_max,
        color="black",
        linestyle="--",
        label="Predicted RT",
    )
    if log_int:
        ax.set_yscale("log")
    ax.set_ylabel("Activation")
    ax.set_xlabel("Scan Index")
    # ax.legend()
    plt.tight_layout()
    save_plot(
        save_dir=save_dir,
        fig_type_name="ActivationPeakCls",
        fig_spec_name=f"Activation_{precursor_id}",
    )

def filter_and_aggregate_intensity(df, threshold=0.05, by_matched:bool=False):
    if by_matched:
        df["peak_intensity_auc"] = df.apply(
            lambda row: row["peak_intensity_auc"] if row["matched"] else 0, axis=1
        )
    else:
        df["peak_intensity_auc"] = df.apply(
            lambda row: row["peak_intensity_auc"] if row["pred_score"] > threshold else 0, axis=1
        )
    result = df.groupby("id")["peak_intensity_auc"].sum().reset_index()
    return result