"""

Prepare dictionary
Usage:
    generate_reference.py --config_path=<config_path>
    generate_reference.py -h | --help

Options:
    -h --help               show this screen help
    --version              show version
"""

from typing import List, Literal, Optional, Dict
import re
import os
import logging
from math import ceil
import pandas as pd
import matplotlib.pyplot as plt
import IsoSpecPy as iso
import torch
from utils.constants import decoy_mutation_rule
from utils.plot import save_plot
from utils.tools import cleanup_maxquant
from utils.metrics import RT_metrics
import numpy as np
import statsmodels.api as sm
from alphabase.psm_reader import psm_reader_provider
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from peptdeep.pretrained_models import ModelManager
from .search_engine_output_parser import sage_parser

Logger = logging.getLogger(__name__)


def merge_ref_and_exp(
    maxquant_ref_df: pd.DataFrame,
    maxquant_exp_df: pd.DataFrame,
    save_dir: str,
    ref_type: Literal["MQ", "pred"],
    use_ims: bool = True,
):
    """Merge dictionaries from multiple files using Modified sequence and Charge"""
    # evaluate the elution counts of the experiment file
    exp_elution_counts = maxquant_exp_df.groupby(["Modified sequence", "Charge"]).size()
    plt.bar(
        height=exp_elution_counts.value_counts().to_numpy(),
        x=exp_elution_counts.value_counts().index,
    )
    sizes = exp_elution_counts.value_counts().values
    total = sum(sizes)
    percentages = [(size / total) * 100 for size in sizes]

    for i, (size, percentage) in enumerate(zip(sizes, percentages)):
        plt.text(
            i + 1, size + 1, f"{percentage:.1f}%, {size}", ha="center"
        )  # starting from 1
    plt.ylabel("Number of Modified Peptides, Charge Combination")
    plt.xlabel("Number of Elution Counts")
    save_plot(
        save_dir=save_dir, fig_type_name="BarChart", fig_spec_name="exp_elution_counts"
    )

    # clean up both dataframe
    maxquant_ref_df = cleanup_maxquant(
        maxquant_ref_df, remove_decoys=True, how_duplicates="keep_highest_int"
    )
    Logger.debug(
        "NaN values in ref df Retention time start_exp/exp/finish_exp: %s, %s, %s",
        maxquant_ref_df["Retention time start"].isna().sum(),
        maxquant_ref_df["Retention time"].isna().sum(),
        maxquant_ref_df["Retention time finish"].isna().sum(),
    )
    maxquant_exp_df = cleanup_maxquant(
        maxquant_exp_df, remove_decoys=True, how_duplicates="keep_highest_int"
    )
    Logger.debug(
        "NaN values in exp df Retention time start_exp/exp/finish_exp: %s, %s, %s",
        maxquant_exp_df["Retention time start"].isna().sum(),
        maxquant_exp_df["Retention time"].isna().sum(),
        maxquant_exp_df["Retention time finish"].isna().sum(),
    )
    join_on_columns = ["Modified sequence", "Sequence", "Charge"]
    ref_selected = maxquant_ref_df[join_on_columns]
    Logger.debug("maxquant ref df columns: %s", maxquant_ref_df.columns)
    exp_selected = maxquant_exp_df[join_on_columns]

    # Perform a merge with an indicator
    merged_df = ref_selected.merge(
        exp_selected, on=join_on_columns, how="outer", indicator=True
    )

    # Filter rows that only exist in maxquant_ref_df
    ref_unique_rows = merged_df[merged_df["_merge"] == "left_only"]
    exp_unique_rows = merged_df[merged_df["_merge"] == "right_only"]
    both_unique_rows = merged_df[merged_df["_merge"] == "both"]
    Logger.debug(
        "ref_unique_rows, exp_unique_rows, both_unique_rows: %s, %s, %s",
        ref_unique_rows.shape,
        exp_unique_rows.shape,
        both_unique_rows.shape,
    )
    # Drop the indicator column
    ref_unique_rows = ref_unique_rows.drop(columns=["_merge"])
    exp_unique_rows = exp_unique_rows.drop(columns=["_merge"])
    both_unique_rows = both_unique_rows.drop(columns=["_merge"])

    # Optionally, you can merge the unique rows back to the original ref dataframe
    ref_unique_df = maxquant_ref_df.merge(
        ref_unique_rows, on=join_on_columns, how="inner"
    )
    ref_unique_df = ref_unique_df.rename(
        columns={
            "Retention time start": "Retention time start_ref",
            "Retention time": "Retention time_ref",
            "Retention time finish": "Retention time finish_ref",
        }
    )
    exp_unique_df = maxquant_exp_df.merge(
        exp_unique_rows, on=join_on_columns, how="inner"
    )
    exp_unique_df = exp_unique_df.rename(
        columns={
            "Retention time start": "Retention time start_exp",
            "Retention time": "Retention time_exp",
            "Retention time finish": "Retention time finish_exp",
        }
    )
    if ref_type == "MQ":
        ref_spec_columns = [
            "Retention time start",
            "Retention time",
            "Retention time finish",
        ]
        if use_ims:
            ref_spec_columns += [
                "mobility_values_index_left_ref",
                "mobility_values_index_right_ref",
                "mobility_values_index_center_ref",
                "mobility_values_center_ref",
            ]
        Logger.info("ref spec columns: %s", ref_spec_columns)
    elif ref_type in ["pred"]:
        ref_spec_columns = []
    Logger.debug("Maxquant_ref_df columns: %s", maxquant_ref_df.columns)
    Logger.debug("Maxquant_exp_df columns: %s", maxquant_exp_df.columns)
    both_unique_df = maxquant_exp_df.merge(
        maxquant_ref_df[join_on_columns + ref_spec_columns],
        on=join_on_columns,
        how="inner",
        suffixes=("_exp", "_ref"),
    )
    Logger.debug(
        "NaN values in both_unique_df Retention time start_exp/exp/finish_exp: %s, %s, %s",
        both_unique_df["Retention time start_exp"].isna().sum(),
        both_unique_df["Retention time_exp"].isna().sum(),
        both_unique_df["Retention time finish_exp"].isna().sum(),
    )
    exp_unique_df["source"] = "exp"
    ref_unique_df["source"] = "ref"
    both_unique_df["source"] = "both"
    Logger.debug(
        "Shape of ref_unique_df, exp_unique_df and both_unique_df: %s, %s, %s",
        ref_unique_df.shape,
        exp_unique_df.shape,
        both_unique_df.shape,
    )
    df_to_merge = []
    for df in [ref_unique_df, exp_unique_df, both_unique_df]:
        if df.shape[0] > 0:
            df_to_merge.append(df)
            Logger.info(
                "Concatenating %s entries from source %s",
                df.shape[0],
                df["source"].values[0],
            )
    maxquant_merge_df = pd.concat(
        df_to_merge,
        ignore_index=True,
        axis=0,
        join="outer",
    )
    Logger.debug(
        "NaN values in maxquant_merge_df (concat) Retention time start_exp/exp/finish_exp: %s, %s, %s",
        maxquant_merge_df["Retention time start_exp"].isna().sum(),
        maxquant_merge_df["Retention time_exp"].isna().sum(),
        maxquant_merge_df["Retention time finish_exp"].isna().sum(),
    )
    # evaluate
    plt.bar(
        height=maxquant_merge_df["source"].value_counts().to_numpy(),
        x=maxquant_merge_df["source"].value_counts().index,
    )
    plt.ylabel("Number of Modified Peptides, Charge Combination")
    plt.xlabel("Merge Status, left: reference, right: experiment")
    sizes = maxquant_merge_df["source"].value_counts().values
    total = sum(sizes)
    percentages = [(size / total) * 100 for size in sizes]
    for i, (size, percentage) in enumerate(zip(sizes, percentages)):
        plt.text(i, size + 1, f"{percentage:.1f}%, {size}", ha="center")
    save_plot(
        save_dir=save_dir, fig_type_name="BarChart", fig_spec_name="candidate_overlap"
    )

    return maxquant_merge_df


def _mutate_seq(seq):

    return (
        seq[0]
        + decoy_mutation_rule[seq[1]]
        + seq[2:-2]
        + decoy_mutation_rule[seq[-2]]
        + seq[-1]
    )


def _mutate_seq_enhance(seq):
    return (
        seq[:3]
        # + decoy_mutation_rule[seq[2]]
        + seq[3:-3]
        + decoy_mutation_rule[seq[-3]]
        + seq[-2:]
    )


def generate_decoy_seq(
    modseq, seq, method: Literal["mutate", "mutate_enhance"] = "mutate"
):
    # Track positions and extract content inside parentheses
    special_parts = []
    # cleaned_seq = []
    special_parts_start = []
    special_parts_end = []
    # Regex to find parentheses and underscores
    # Create a regex pattern to match special characters and sequences
    special_chars = [
        "_",
        "(Oxidation (M))",
        "M(ox)",
        "(Acetyl (Protein N-term))",
        "(ac)",
    ]  # TODO: update according to PTM list
    pattern = "|".join(re.escape(char) for char in special_chars)
    regex = re.compile(pattern)

    for match in regex.finditer(modseq):
        start, end = match.span()
        special_parts_start.append(start)
        special_parts_end.append(end)
        # Logger.debug("Match: %s, %s", start, end)

        special_parts.append(modseq[start:end])
    # Logger.debug("Special parts: %s", special_parts)
    match method:
        case "mutate":
            # Apply mutation
            mutated_seq = _mutate_seq(seq)
        case "mutate_enhance":
            # Apply enhance mutation
            mutated_seq = _mutate_seq_enhance(seq)
        case _:
            raise ValueError(f"Method {method} not supported")

    # Interleave mutated sequence and special parts
    mutated_modseq = mutated_seq
    for idx, part in enumerate(special_parts):
        mutated_modseq = (
            mutated_modseq[: special_parts_start[idx]]
            + part
            + mutated_modseq[special_parts_start[idx] :]
        )
        # Logger.debug("Mutated sequence: %s", mutated_modseq)

    return mutated_seq, mutated_modseq


def concat_decoy_and_target(maxquant_df: pd.DataFrame, mutation_method: str = "mutate"):
    """Concatenate target and decoy sequences"""
    maxquant_df["TD pair id"] = (
        maxquant_df["m/z"].rank(axis=0, method="first", ascending=True).astype(int)
    )
    maxquant_decoy_df = generate_decoy_df(maxquant_df, mutation_method="mutate")
    similar_td_pair = _eval_target_decoy_pair_mz(
        pd.concat([maxquant_df, maxquant_decoy_df])
    )
    if similar_td_pair["IsoMZ_identical"].sum() > 0:
        Logger.info("Some TD pairs have identical, mutate with enhanced method")
        maxquant_decoy_df_enhanced = generate_decoy_df(
            maxquant_decoy_df.loc[
                maxquant_decoy_df["TD pair id"].isin(similar_td_pair["TD pair id"])
            ],
            mutation_method="mutate_enhance",
        )
        similar_td_pair_enhanced = _eval_target_decoy_pair_mz(
            pd.concat(
                [
                    maxquant_df.loc[
                        maxquant_df["TD pair id"].isin(similar_td_pair["TD pair id"])
                    ],
                    maxquant_decoy_df_enhanced,
                ]
            )
        )
        maxquant_decoy_df = pd.concat(
            [
                maxquant_decoy_df.loc[
                    ~maxquant_decoy_df["TD pair id"].isin(similar_td_pair["TD pair id"])
                ],
                maxquant_decoy_df_enhanced,
            ]
        )
    n_before = maxquant_decoy_df.shape[0]
    Logger.info("Number of decoys generated: %s", n_before)
    maxquant_decoy_df = maxquant_decoy_df[
        ~maxquant_decoy_df["Modified sequence"].isin(maxquant_df["Modified sequence"])
    ]
    Logger.info(
        "Number of decoys after removing decoys identical to any targets: %s, %s are removed",
        maxquant_decoy_df.shape[0],
        n_before - maxquant_decoy_df.shape[0],
    )
    return pd.concat([maxquant_df, maxquant_decoy_df])


def _eval_target_decoy_pair_mz(maxquant_dict: pd.DataFrame):
    maxquant_dict = maxquant_dict.copy(deep=True)
    # Assuming maxquant_result_ref is your DataFrame
    if "IsoMZ" not in maxquant_dict.columns:
        maxquant_dict = dict_add_iso_pattern(maxquant_dict)

    # Group by 'TD pair id'
    grouped = maxquant_dict.groupby("TD pair id")

    # Ensure each group has exactly 2 rows
    # This will return groups with exactly 2 rows
    filtered_groups = grouped.filter(lambda x: len(x) == 2)

    # Calculate the difference between the two 'mz_bin' values in each group
    # We can use the diff function to get the difference, then take the absolute value
    # filtered_groups["mz_bin_difference"] = (
    #     filtered_groups.groupby("TD pair id")["mz_bin"].diff().abs()
    # )
    # Check if 'IsoAbundance' values are identical within each group
    filtered_groups["IsoMZ_identical"] = filtered_groups.groupby("TD pair id")[
        "IsoMZ"
    ].transform(lambda x: np.array_equal(x.iloc[0], x.iloc[1]))

    # Drop the NaN values that arise from the diff (because it shifts the series)
    result = filtered_groups.dropna(subset=["IsoMZ_identical"])

    # If you want the result as a new DataFrame with 'TD pair id' and the difference
    result = result[["TD pair id", "IsoMZ_identical"]].drop_duplicates()
    result = result.loc[result["IsoMZ_identical"] == 1]
    Logger.info("Number of TD pairs with identical iso mz: %s", result.shape[0])
    return result


def generate_decoy_df(maxquant_df, mutation_method):
    maxquant_decoy_df = maxquant_df.copy()
    maxquant_decoy_df.drop("m/z", inplace=True, axis=1)
    maxquant_df["Decoy"] = False
    maxquant_decoy_df["Decoy"] = True
    maxquant_decoy_df[["Sequence", "Modified sequence"]] = maxquant_df.apply(
        lambda x: generate_decoy_seq(
            x["Modified sequence"], x["Sequence"], mutation_method
        ),
        axis=1,
        result_type="expand",
    )
    maxquant_decoy_df["_merge"] = "decoy"
    maxquant_decoy_df["Raw file"] = "decoy"
    maxquant_decoy_df = dict_add_mz_mono(maxquant_decoy_df)
    return maxquant_decoy_df


def _check_td_pair_mass(maxquant_result_ref: pd.DataFrame):
    # Assuming maxquant_result_ref is your DataFrame

    # Group by 'TD pair id'
    grouped = maxquant_result_ref.groupby("TD pair id")

    # Ensure each group has exactly 2 rows
    # This will return groups with exactly 2 rows
    filtered_groups = grouped.filter(lambda x: len(x) == 2)

    # Calculate the difference between the two 'mz_bin' values in each group
    # We can use the diff function to get the difference, then take the absolute value
    filtered_groups["mz_bin_difference"] = (
        filtered_groups.groupby("TD pair id")["mz_bin"].diff().abs()
    )

    # Drop the NaN values that arise from the diff (because it shifts the series)
    result = filtered_groups.dropna(subset=["mz_bin_difference"])

    # If you want the result as a new DataFrame with 'TD pair id' and the difference
    result = result[["TD pair id", "mz_bin_difference"]].drop_duplicates()
    Logger.info(
        "TD pair id with mz_bin difference > 0: %s",
        result.loc[result["mz_bin_difference"] > 0].shape[0],
    )
    return result


def dict_add_index_to_raw_file(
    maxquant_df: pd.DataFrame,
    mobility_values_df: pd.DataFrame,
    rt_values_df: pd.DataFrame,
    raw_file: str,
):
    maxquant_df = maxquant_df.sort_values(f"{raw_file}_RT")
    maxquant_df_with_rt_index = pd.merge_asof(
        left=maxquant_df,
        right=rt_values_df[["Time_minute", "MS1_frame_idx"]],
        left_on=f"{raw_file}_RT",
        right_on="Time_minute",
        direction="nearest",
    )
    maxquant_df_with_rt_index = maxquant_df.merge(
        maxquant_df_with_rt_index[["mz_rank", "MS1_frame_idx"]],
        on="mz_rank",
        how="left",
    )
    for col in ["Time_minute", "MS1_frame_idx"]:
        if col in maxquant_df_with_rt_index.columns:
            maxquant_df_with_rt_index.rename(
                {col: f"{raw_file}_{col}_exp"}, axis=1, inplace=True
            )
    maxquant_df_with_rt_index = maxquant_df_with_rt_index.sort_values(f"{raw_file}_1K0")
    maxquant_df_with_rt_index = pd.merge_asof(
        left=maxquant_df_with_rt_index,
        right=mobility_values_df[["mobility_values_index", "mobility_values"]],
        left_on=f"{raw_file}_1K0",
        right_on="mobility_values",
        direction="nearest",
    )
    for col in ["mobility_values_index", "mobility_values"]:
        if col in maxquant_df_with_rt_index.columns:
            maxquant_df_with_rt_index.rename(
                {col: f"{raw_file}_{col}_exp"}, axis=1, inplace=True
            )
    return maxquant_df_with_rt_index


def dict_add_im_index(
    maxquant_df: pd.DataFrame,
    mobility_values_df: pd.DataFrame,
    mq_im_left_col: str | None = None,
    mq_im_center_col: str = "1/K0",
    mq_im_right_col: str | None = None,
    im_idx_length: int | None = None,
    idx_suffix: str = "",
):
    """Get IM index as indicated in mobility values from .d file for each row in maxquant_dict_df/
    has built-in control of int type for index"""
    maxquant_df = maxquant_df.sort_values(mq_im_center_col)
    maxquant_df_with_im_index = pd.merge_asof(
        left=maxquant_df,
        right=mobility_values_df[["mobility_values_index", "mobility_values"]],
        left_on=mq_im_center_col,
        right_on="mobility_values",
        suffixes=("", "_center" + idx_suffix),
        direction="nearest",
    )
    for col in ["mobility_values_index", "mobility_values"]:
        if col in maxquant_df_with_im_index.columns:
            maxquant_df_with_im_index.rename(
                {col: col + "_center" + idx_suffix}, axis=1, inplace=True
            )
    Logger.debug(
        "dict_add_im_index columns merging center: %s",
        maxquant_df_with_im_index.columns,
    )
    max_im_index = mobility_values_df["mobility_values_index"].max()
    if (im_idx_length is None) and (
        (mq_im_left_col is None) or (mq_im_right_col is None)
    ):
        im_idx_length = maxquant_df_with_im_index["Ion mobility length"] // 2 + 1  # type: ignore
        Logger.info(
            "mq_im_left_col or mq_im_right_col not given, IM index length required but not given, using peptide specific im length values"
        )
    if mq_im_left_col is None:
        maxquant_df_with_im_index["mobility_values_index_left" + idx_suffix] = (
            np.maximum(
                0,
                (
                    maxquant_df_with_im_index[
                        "mobility_values_index_center" + idx_suffix
                    ]
                    - im_idx_length
                ),  # type: ignore
            )
        )
        # mq_im_left_col = "mobility_values_index_left"
    else:
        maxquant_df_with_im_index = maxquant_df_with_im_index.sort_values(
            mq_im_left_col
        )
        maxquant_df_with_im_index = pd.merge_asof(
            left=maxquant_df_with_im_index,
            right=mobility_values_df[["mobility_values_index", "mobility_values"]],
            left_on=mq_im_left_col,
            right_on="mobility_values",
            suffixes=("", "_left" + idx_suffix),
            direction="nearest",
        )
        for col in ["mobility_values_index", "mobility_values"]:
            if col in maxquant_df_with_im_index.columns:
                maxquant_df_with_im_index.rename(
                    {col: col + "_left" + idx_suffix}, axis=1, inplace=True
                )
    if mq_im_right_col is None:
        maxquant_df_with_im_index["mobility_values_index_right" + idx_suffix] = (
            np.minimum(
                max_im_index,
                (
                    maxquant_df_with_im_index[
                        "mobility_values_index_center" + idx_suffix
                    ]
                    + im_idx_length
                ),  # type: ignore
            )
        )
        # mq_im_right_col = "mobility_values_index_right"
    else:
        maxquant_df_with_im_index = maxquant_df_with_im_index.sort_values(
            mq_im_right_col
        )
        maxquant_df_with_im_index = pd.merge_asof(
            left=maxquant_df_with_im_index,
            right=mobility_values_df[["mobility_values_index", "mobility_values"]],
            left_on=mq_im_right_col,
            right_on="mobility_values",
            suffixes=("", "_right" + idx_suffix),
            direction="nearest",
        )
        for col in ["mobility_values_index", "mobility_values"]:
            if col in maxquant_df_with_im_index.columns:
                maxquant_df_with_im_index.rename(
                    {col: col + "_right" + idx_suffix}, axis=1, inplace=True
                )

    Logger.debug("dict_add_im_index columns: %s", maxquant_df_with_im_index.columns)
    return maxquant_df_with_im_index


def dict_add_rt_index(
    maxquant_df: pd.DataFrame,
    rt_values_df: pd.DataFrame,
    mq_rt_left_col: Optional[str] = "RT_search_left",
    mq_rt_center_col: Optional[str] = "RT_search_center",
    mq_rt_right_col: Optional[str] = "RT_search_right",
    rt_values_col: str = "Time_minute",
    idx_suffix: str = "",
):
    """Get RT index of RT_search_left, RT_search_center and RT_search_right/
    as indicated in RT values from .d file for each row in maxquant_dict_df/
    has built-in control of int type for index"""
    Logger.debug("dict_add_rt_index columns: %s", maxquant_df.columns)
    if mq_rt_center_col is not None:
        Logger.info("mq_rt_center_col columns: %s", mq_rt_center_col)
        maxquant_df = maxquant_df.sort_values(mq_rt_center_col)
        maxquant_df = pd.merge_asof(
            left=maxquant_df,
            right=rt_values_df[[rt_values_col, "MS1_frame_idx"]],
            left_on=mq_rt_center_col,
            right_on=rt_values_col,
            direction="nearest",
            suffixes=("", "_center" + idx_suffix),
        )
        for col in [rt_values_col, "MS1_frame_idx"]:
            if col in maxquant_df.columns:
                maxquant_df.rename(
                    {col: col + "_center" + idx_suffix}, axis=1, inplace=True
                )
    if mq_rt_left_col is not None:
        maxquant_df = maxquant_df.sort_values(mq_rt_left_col)
        maxquant_df = pd.merge_asof(
            left=maxquant_df,
            right=rt_values_df[[rt_values_col, "MS1_frame_idx"]],
            left_on=mq_rt_left_col,
            right_on=rt_values_col,
            direction="nearest",
            suffixes=("", "_left" + idx_suffix),
        )
        for col in [rt_values_col, "MS1_frame_idx"]:
            if col in maxquant_df.columns:
                maxquant_df.rename(
                    {col: col + "_left" + idx_suffix}, axis=1, inplace=True
                )
    if mq_rt_right_col is not None:
        maxquant_df = maxquant_df.sort_values(mq_rt_right_col)
        maxquant_df = pd.merge_asof(
            left=maxquant_df,
            right=rt_values_df[[rt_values_col, "MS1_frame_idx"]],
            left_on=mq_rt_right_col,
            right_on=rt_values_col,
            direction="nearest",
            suffixes=("", "_right" + idx_suffix),
        )
        for col in [rt_values_col, "MS1_frame_idx"]:
            if col in maxquant_df.columns:
                maxquant_df.rename(
                    {col: col + "_right" + idx_suffix}, axis=1, inplace=True
                )

    Logger.debug("dict_add_rt_index columns: %s", maxquant_df.columns)
    return maxquant_df


def dict_add_mz_rank(maxquant_dict_df: pd.DataFrame):
    """Add m/z rank to maxquant_dict_df"""
    maxquant_dict_df["mz_rank"] = (
        maxquant_dict_df["m/z"].rank(axis=0, method="first", ascending=True).astype(int)
    )
    return maxquant_dict_df


def dict_add_mz_bin(maxquant_dict_df: pd.DataFrame, mz_bin_digits: int = 2):
    """Add m/z bin to maxquant_dict_df"""

    maxquant_dict_df["mz_bin"] = maxquant_dict_df["m/z"].apply(
        lambda x: round(x, mz_bin_digits)
    )
    return maxquant_dict_df


def dict_add_iso_pattern(maxquant_dict_df: pd.DataFrame, ab_thres: float = 0.01):
    """
    Get isotopic pattern as indicated in iso_pattern_df for each row in maxquant_dict_df

    :param maxquant_dict_df: pd.DataFrame
    :param ab_thres: float, default 0.01, the threshold for isotopic pattern absolute abundance
    :return: pd.DataFrame
    """

    maxquant_dict_df["IsoMZ"], maxquant_dict_df["IsoAbundance"] = zip(
        *maxquant_dict_df.apply(
            lambda row: calculate_modpept_isopattern(
                modpept=row["Modified sequence"],
                charge=row["Charge"],
                ab_thres=ab_thres,
            ),
            axis=1,
        )
    )
    return maxquant_dict_df


def dict_add_mz_len(maxquant_dict_df: pd.DataFrame):
    assert "IsoMZ" in maxquant_dict_df.columns
    maxquant_dict_df["mz_length"] = maxquant_dict_df["IsoMZ"].apply(lambda x: len(x))
    return maxquant_dict_df


def dict_add_mz_mono(maxquant_dict_df: pd.DataFrame):
    maxquant_dict_df["m/z"] = maxquant_dict_df.apply(
        lambda row: calculate_modpept_mz(row["Modified sequence"], row["Charge"]),
        axis=1,
    )
    return maxquant_dict_df


def dict_add_mass_mono(maxquant_dict_df: pd.DataFrame):
    maxquant_dict_df["mass"] = maxquant_dict_df.apply(
        lambda row: calculate_modpept_mz(row["Modified sequence"], 1),
        axis=1,
    )
    return maxquant_dict_df


def calculate_modpept_isopattern(
    modpept: str, charge: int, ab_thres: float = 0.005, mod_CAM: bool = True
):
    """takes a peptide sequence with modification and charge,
    calculate and return the two LISTs of isotope pattern with all isotopes m/z value
    with abundance larger than ab_thres, both sorted by isotope mass

    :modpept: str
    :charge: charge state of the percursor, int
    :mzrange: limitation of detection of mz range
    :mm: bin size of mz value, int
    :ab_thres: the threshold for filtering isotopes, float

    return: two list
    """

    # account for extra atoms from modification and water
    # count extra atoms
    n_H = 2 + charge  # 2 from water and others from charge (proton)
    n_Mox = (
        modpept.count("M(ox)") + modpept.count("Oxidation (M)") + modpept.count("[147]")
    )
    modpept = modpept.replace("(ox)", "")
    modpept = modpept.replace("(Oxidation (M))", "")
    n_acetylN = (
        modpept.count("(ac)")
        + modpept.count("(Acetyl (Protein N-term))")
        + modpept.count("[43]")
    )
    modpept = modpept.replace("(Acetyl (Protein N-term))", "")
    modpept = modpept.replace("(ac)", "")

    if mod_CAM:
        n_C = modpept.count("C")
    else:
        n_C = 0
    # addition of extra atoms
    atom_composition = iso.ParseFASTA(modpept)
    atom_composition["H"] += 3 * n_C + n_H + 2 * n_acetylN
    atom_composition["C"] += 2 * n_C + 2 * n_acetylN
    atom_composition["N"] += 1 * n_C
    atom_composition["O"] += 1 * n_C + 1 + n_acetylN + 1 * n_Mox

    # Isotope calculation
    formula = "".join([f"{key}{value}" for key, value in atom_composition.items()])
    iso_distr = iso.IsoThreshold(formula=formula, threshold=ab_thres, absolute=True)
    iso_distr.sort_by_mass()
    mz_sort_by_mass = iso_distr.np_masses() / charge
    probs_sort_by_mass = iso_distr.np_probs()

    return mz_sort_by_mass, probs_sort_by_mass


def calculate_modpept_mz(modpept: str, charge: int, mod_CAM: bool = True):
    """takes a peptide sequence with modification and charge,
    calculate and return the m/z of mono-isotopic peak

    :modpept: str
    :charge: charge state of the percursor, int
    :mod_CAM: bool, whether to consider CAM modification
    return: two list
    """

    # account for extra atoms from modification and water
    # count extra atoms
    n_H = 2 + charge  # 2 from water and others from charge (proton)
    n_Mox = modpept.count("M(ox)") + modpept.count("Oxidation (M)")
    modpept = modpept.replace("(ox)", "")
    modpept = modpept.replace("(Oxidation (M))", "")
    n_acetylN = modpept.count("(ac)") + modpept.count("(Acetyl (Protein N-term))")
    modpept = modpept.replace("(Acetyl (Protein N-term))", "")
    modpept = modpept.replace("(ac)", "")

    if mod_CAM:
        n_C = modpept.count("C")
    else:
        n_C = 0
    # addition of extra atoms
    atom_composition = iso.ParseFASTA(modpept)
    atom_composition["H"] += 3 * n_C + n_H + 2 * n_acetylN
    atom_composition["C"] += 2 * n_C + 2 * n_acetylN
    atom_composition["N"] += 1 * n_C
    atom_composition["O"] += 1 * n_C + 1 + n_acetylN + 1 * n_Mox

    # Isotope calculation
    formula = "".join([f"{key}{value}" for key, value in atom_composition.items()])
    iso_distr = iso.IsoThreshold(formula=formula, threshold=0.1, absolute=True)
    iso_distr.sort_by_prob()
    mz_mono = iso_distr.np_masses()[-1] / charge

    return mz_mono


def _define_rt_search_range(
    maxquant_result_dict: pd.DataFrame,
    rt_tol: float,
    rt_ref: Literal["exp", "pred", "mix", "align_lowess", "ref"],
    rt_range: tuple[float, float],
):
    """Define the search range for the precursor RT."""
    Logger.debug("define_rt_search_range: %s", maxquant_result_dict.columns)
    match rt_ref:
        case "exp":
            # maxquant_result_dict["Calibrated retention time start"] = (
            #     maxquant_result_dict["Calibrated retention time start"]
            #     - maxquant_result_dict["Retention time calibration"]
            # )
            # maxquant_result_dict["Calibrated retention time finish"] = (
            #     maxquant_result_dict["Calibrated retention time finish"]
            #     - maxquant_result_dict["Retention time calibration"]
            # )
            # maxquant_result_dict["Calibrated retention time start"].fillna(
            #     maxquant_result_dict["Retention time"]
            #     - 0.5 * maxquant_result_dict["Retention length"],
            #     inplace=True,
            # )
            # maxquant_result_dict["Calibrated retention time finish"].fillna(
            #     maxquant_result_dict["Retention time"]
            #     + 0.5 * maxquant_result_dict["Retention length"],
            #     inplace=True,
            # )
            maxquant_result_dict["RT_search_left"] = (
                maxquant_result_dict["Retention time start_exp"] - rt_tol
            )
            maxquant_result_dict["RT_search_right"] = (
                maxquant_result_dict["Retention time finish_exp"] + rt_tol
            )
            rt_ref_act_peak = "Retention time_exp"
        case "pred" | "align_lowess":
            maxquant_result_dict["RT_search_left"] = (
                maxquant_result_dict["predicted_RT"] - rt_tol
            )
            maxquant_result_dict["RT_search_right"] = (
                maxquant_result_dict["predicted_RT"] + rt_tol
            )
            rt_ref_act_peak = "predicted_RT"
        case "ref":
            maxquant_result_dict["RT_search_left"] = (
                maxquant_result_dict["Retention time start_ref"] - rt_tol
            )
            maxquant_result_dict["RT_search_right"] = (
                maxquant_result_dict["Retention time finish_ref"] + rt_tol
            )
            rt_ref_act_peak = "Retention time_ref"
        case "mix":
            maxquant_result_dict["RT_search_left"] = (
                maxquant_result_dict["Retention time start_ss"] - rt_tol
            )
            maxquant_result_dict["RT_search_right"] = (
                maxquant_result_dict["Retention time finish_ss"] + rt_tol
            )
            rt_ref_act_peak = "Retention time_ss"
        case _:
            raise ValueError(f"RT reference {rt_ref} not supported")
    maxquant_result_dict["RT_search_center"] = maxquant_result_dict[rt_ref_act_peak]
    if maxquant_result_dict["RT_search_center"].isna().sum() > 0:
        Logger.warning(
            maxquant_result_dict.loc[maxquant_result_dict["RT_search_center"].isna()]
        )
    Logger.debug(
        "NaN values, before clipping, in RT_search_left, right and center: %s, %s, %s",
        maxquant_result_dict["RT_search_left"].isna().sum(),
        maxquant_result_dict["RT_search_right"].isna().sum(),
        maxquant_result_dict["RT_search_center"].isna().sum(),
    )

    maxquant_result_dict[
        ["RT_search_left", "RT_search_center", "RT_search_right"]
    ].clip(rt_range[0], rt_range[1], inplace=True)
    Logger.debug(
        "NaN values, after clipping, in RT_search_left, right and center: %s, %s, %s",
        maxquant_result_dict["RT_search_left"].isna().sum(),
        maxquant_result_dict["RT_search_right"].isna().sum(),
        maxquant_result_dict["RT_search_center"].isna().sum(),
    )
    return maxquant_result_dict


def _get_im_idx_span_from_values(mobility_values_df: pd.DataFrame, im_length: float):
    """Get the span of the ion mobility index given im value length."""
    im_idx_span = ceil(im_length / mobility_values_df["mobility_values"].diff().mean())
    return im_idx_span


def _define_im_idx_search_range(
    maxquant_df: pd.DataFrame,
    im_length: int,
    im_ref: Literal["exp", "pred", "mix", "ref", "align_lr"],
    im_idx_range: tuple[float, float],
    delta_im_95: float | None = None,
    mobility_values_df: pd.DataFrame | None = None,
):
    """
    Ion mobility search range. It is not used in the activation optimization step, /
    where the full IM range is used. However, it is used in the post-processing step /
    to crop out only the activation in the relevant IM range, /
    hence it is REPRESENTED AS INDICES!
    """

    im_length = int(im_length)
    half_im_length = ceil(im_length / 2)
    match im_ref:
        case "exp":  # TODO: not checked
            maxquant_df["IM_search_idx_left"] = maxquant_df[
                "mobility_values_index_left_exp"
            ]
            maxquant_df["IM_search_idx_right"] = maxquant_df[
                "mobility_values_index_right_exp"
            ]
            maxquant_df["IM_search_idx_center"] = maxquant_df[
                "mobility_values_index_center_exp"
            ]
        case "pred" | "align_lr":  # Use predicted IM, expand with delta_im_95s
            assert "mobility_pred" in maxquant_df.columns
            assert mobility_values_df is not None
            assert delta_im_95 is not None and delta_im_95 > 0
            Logger.info(
                "Calculating prediction right and left with delta_im_95: %s",
                delta_im_95,
            )
            maxquant_df["mobility_pred_left"] = (
                maxquant_df["mobility_pred"] - delta_im_95
            )
            maxquant_df["mobility_pred_right"] = (
                maxquant_df["mobility_pred"] + delta_im_95
            )
            maxquant_df = dict_add_im_index(
                maxquant_df=maxquant_df,
                mobility_values_df=mobility_values_df,
                mq_im_center_col="mobility_pred",
                mq_im_left_col="mobility_pred_left",
                mq_im_right_col="mobility_pred_right",
                im_idx_length=None,
                idx_suffix="_pred",
            )
            # maxquant_df = maxquant_df.rename(
            #     mapper={
            #         "mobility_values_index_left_pred": "IM_search_idx_left",
            #         "mobility_values_index_right_pred": "IM_search_idx_right",
            #         "mobility_values_index_center_pred": "IM_search_idx_center",
            #     },
            #     axis=1,
            # )
            maxquant_df["IM_search_idx_left"] = (
                maxquant_df["mobility_values_index_left_pred"] - half_im_length
            )
            maxquant_df["IM_search_idx_right"] = (
                maxquant_df["mobility_values_index_right_pred"] + half_im_length
            )
            maxquant_df["IM_search_idx_center"] = maxquant_df[
                "mobility_values_index_center_pred"
            ]
        case (
            "mix"
        ):  # Use either exp IM or if not available, use ref IM, im_length is not used
            # TODO: whether to use im_length or not?
            maxquant_df["IM_search_idx_left"] = maxquant_df[
                "mobility_values_index_left_exp"
            ]
            maxquant_df["IM_search_idx_right"] = maxquant_df[
                "mobility_values_index_right_exp"
            ]
            maxquant_df["IM_search_idx_center"] = maxquant_df[
                "mobility_values_index_center_exp"
            ]
            maxquant_df["IM_search_idx_center"].fillna(
                maxquant_df["mobility_values_index_center_ref"], inplace=True
            )
            maxquant_df["IM_search_idx_left"].fillna(
                maxquant_df["mobility_values_index_left_ref"], inplace=True
            )
            maxquant_df["IM_search_idx_right"].fillna(
                maxquant_df["mobility_values_index_right_ref"], inplace=True
            )
        case (
            "ref"
        ):  # Use primarily ref IM, if not available, use exp IM, expand with delta_im_95
            Logger.info("Using primarily ref IM for IM search range.")
            _merge_df = maxquant_df.loc[
                (maxquant_df["source"] == "both") & (maxquant_df["Decoy"] == False)
            ]
            delta_im_idx_95_left = RT_metrics(
                _merge_df["mobility_values_index_left_ref"],
                _merge_df["mobility_values_index_left_exp"],
            ).CalcDeltaRTwidth()
            delta_im_idx_95_right = RT_metrics(
                _merge_df["mobility_values_index_right_ref"],
                _merge_df["mobility_values_index_right_exp"],
            ).CalcDeltaRTwidth()
            delta_1k0_95 = RT_metrics(
                _merge_df["mobility_values_index_center_ref"],
                _merge_df["mobility_values_index_center_exp"],
            ).CalcDeltaRTwidth()
            Logger.info(
                "Delta IM index 95 left: %s, right: %s, delta_1K0_95: %s",
                delta_im_idx_95_left,
                delta_im_idx_95_right,
                delta_1k0_95,
            )
            maxquant_df["IM_search_idx_center"] = maxquant_df[
                "mobility_values_index_center_ref"
            ]

            # If IM_search_idx_center is NaN, fill with exp IM with noise
            residual = (
                maxquant_df["mobility_values_index_center_ref"]
                - maxquant_df["mobility_values_index_center_exp"]
            ).dropna()
            noise = np.random.choice(
                residual, maxquant_df["IM_search_idx_center"].isna().sum()
            )
            # maxquant_df["IM_search_idx_center"].fillna(
            #     maxquant_df["mobility_values_index_center_exp"], inplace=True
            # )
            Logger.info(
                "IM Noise size: %s, noise mean: %s, noise std: %s",
                noise.size,
                noise.mean(),
                noise.std(),
            )
            maxquant_df.loc[
                maxquant_df["IM_search_idx_center"].isna(), "IM_search_idx_center"
            ] = (
                noise
                + maxquant_df.loc[
                    maxquant_df["IM_search_idx_center"].isna(),
                    "mobility_values_index_center_exp",
                ]
            )
            maxquant_df.loc[
                :,
                [
                    "IM_search_idx_center",
                ],
            ] = maxquant_df[["IM_search_idx_center"]].clip(
                im_idx_range[0] + 1, im_idx_range[1] - 1
            )  # After adding noise, clip to the range

            maxquant_df["IM_search_idx_center"] = maxquant_df[
                "IM_search_idx_center"
            ].astype(int)
            maxquant_df["IM_search_idx_left"] = (
                maxquant_df["IM_search_idx_center"]
                - half_im_length
                - int(delta_im_idx_95_left)
            )
            maxquant_df["IM_search_idx_right"] = (
                maxquant_df["IM_search_idx_center"]
                + half_im_length
                + int(delta_im_idx_95_right)
            )
        case _:
            raise ValueError(f"IM reference {im_ref} not supported")
    maxquant_df.loc[
        :,
        [
            "IM_search_idx_left",
            "IM_search_idx_center",
            "IM_search_idx_right",
        ],
    ] = maxquant_df[
        ["IM_search_idx_left", "IM_search_idx_center", "IM_search_idx_right"]
    ].clip(
        im_idx_range[0], im_idx_range[1]
    )
    return maxquant_df


def prepare_alpha_train_test_df(
    maxquant_path: str, train_frac: float, filter_dict: dict, random_state: int = 42
):
    # Load
    mq_reader = psm_reader_provider.get_reader("maxquant")
    mq_reader.load(
        maxquant_path,
    )
    psm_df = mq_reader.psm_df

    # Filter
    for key, value in filter_dict.items():
        psm_df = psm_df[psm_df[key].isin(value)]
        Logger.info(
            "psm_df size after filtering %s in %s: %s", key, value, psm_df.shape
        )

    # Normalize RT
    rt_max = psm_df["rt"].max()
    Logger.info(f"Max RT used for normalization: {rt_max}")
    psm_df["rt_norm"] = psm_df["rt"] / rt_max

    # Split
    train_df, test_df = train_test_split(
        psm_df, test_size=1 - train_frac, random_state=random_state
    )

    return train_df, test_df, rt_max


def _rescale_and_get_delta95(predict_df, lc_grad, obs_col, pred_col):
    if pred_col == "rt_pred":
        predict_df["rt_pred_norm"] = predict_df["rt_pred"]
        predict_df[pred_col] = predict_df["rt_pred_norm"] * lc_grad

    ori_rt_eval = RT_metrics(predict_df[obs_col], predict_df[pred_col])

    return ori_rt_eval.CalcDeltaRTwidth()


def update_alpha_pept_deep_model(
    pept_property: Literal["rt", "mobility"],
    train_df,
    test_df,
    save_dir: str,
    lc_grad: float,
    epoch=10,
    device: str = "cuda:0",
):
    # init model
    models = ModelManager(device=device)
    models.load_installed_models()
    match pept_property:
        case "rt":
            model = models.rt_model
        case "mobility":
            model = models.ccs_model
        case _:
            raise ValueError(f"pept_property {pept_property} not supported")

    # train
    model.train(train_df, epoch=epoch, verbose=True, verbose_each_epoch=False)

    # eval
    predict_train_df = model.predict(train_df)
    predict_test_df = model.predict(test_df)
    if pept_property == "mobility":
        predict_train_df = model.ccs_to_mobility_pred(train_df)  # type: ignore
        predict_test_df = model.ccs_to_mobility_pred(test_df)  # type: ignore

    delta_95_train = _rescale_and_get_delta95(
        predict_train_df,
        lc_grad=lc_grad,
        obs_col=pept_property,
        pred_col=f"{pept_property}_pred",
    )
    delta_95_test = _rescale_and_get_delta95(
        predict_test_df,
        lc_grad=lc_grad,
        obs_col=pept_property,
        pred_col=f"{pept_property}_pred",
    )
    Logger.info(f"Training Delta95: {delta_95_train}, Testing Delta95: {delta_95_test}")

    # save
    save_model_name = os.path.join(save_dir, f"{pept_property}_model")
    model.save(save_model_name)
    return delta_95_test, save_model_name


def dict_add_alpha_pept_pred(
    model_path,
    pept_property: Literal["rt", "mobility"],
    dict_for_pred_path,
    maxquant_dict: pd.DataFrame,
    lc_grad: float | None = None,
    device: str = "cuda:0",
):
    Logger.info(f"Device: {device}")
    # load dict

    Logger.info(f"dict size: {maxquant_dict.shape}")
    mq_reader = psm_reader_provider.get_reader("maxquant")
    mq_reader.load(dict_for_pred_path)
    dict_for_pred = mq_reader.psm_df
    Logger.info(f"dict_for_pred size: {dict_for_pred.shape}")

    # predict
    models = ModelManager(device=device)
    models.load_installed_models()
    if pept_property == "mobility":
        model = models.ccs_model
        model.load(model_path)
        predict_df = model.predict(dict_for_pred)
        predict_df = model.ccs_to_mobility_pred(predict_df)
    elif pept_property == "rt":
        assert lc_grad is not None and lc_grad > 0
        model = models.rt_model
        model.load(model_path)
        predict_df = model.predict(dict_for_pred)
        predict_df["rt_pred_norm"] = predict_df["rt_pred"]
        predict_df["rt_pred"] = predict_df["rt_pred_norm"] * lc_grad
    Logger.info("Columns in predict_df: %s", predict_df.columns)
    # merge
    maxquant_dict_new = pd.merge(
        left=maxquant_dict,
        right=predict_df[
            ["sequence", "charge", "scan_num", "raw_name", f"{pept_property}_pred"]
        ],
        left_on=["Sequence", "Charge", "MS/MS scan number", "Raw file"],
        right_on=["sequence", "charge", "scan_num", "raw_name"],
        how="left",
    )
    Logger.info("Columns in predict_df: %s", maxquant_dict_new.columns)
    # Logger.info(
    #     "Number of entries with empty prediction: %s",
    #     maxquant_dict_new[pept_property + "_pred"].isna().sum(),
    # )
    maxquant_dict_new = maxquant_dict_new.dropna(subset=[pept_property + "_pred"])
    maxquant_dict_new = maxquant_dict_new.drop(
        labels=["scan_num", "raw_name", "sequence", "charge"], axis=1
    )
    Logger.info(f"dict size after dropping empty prediction: {maxquant_dict_new.shape}")
    if pept_property == "rt":  # rename rt_pred to predicted_RT for compatibility
        assert lc_grad is not None and lc_grad > 0
        maxquant_dict_new["predicted_RT"] = np.minimum(
            maxquant_dict_new["rt_pred"], lc_grad
        )

    return maxquant_dict_new


def dict_add_im_align_lr(
    maxquant_dict: pd.DataFrame,
    train_frac: float,
    random_state: int,
):
    """
    Add experimental IM prediction from linear regression based
    on reference IM values.

    When reference IM values are not available,
    the prediction is inferred using experimental IM values.
    Which essentially adds noise to the true value.

    Outputs maxquant_dict with additional column 'mobility_pred'
    """

    # select the precursors appearing in both ref and exp for training and testing
    max_exp_im = maxquant_dict["mobility_values_center_exp"].max()
    min_exp_im = maxquant_dict["mobility_values_center_exp"].min()
    both = maxquant_dict.loc[
        (maxquant_dict["source"] == "both") & (~maxquant_dict["Decoy"])
    ].copy()
    Logger.info("both dataframe size: %s", both.shape)
    LR_base = RT_metrics(
        both["mobility_values_center_ref"], both["mobility_values_center_exp"]
    )
    base_delta95 = LR_base.CalcDeltaRTwidth()
    both_train, both_test = train_test_split(
        both, test_size=1 - train_frac, random_state=random_state
    )
    reg = LinearRegression().fit(
        both_train["mobility_values_center_ref"].values.reshape(-1, 1),
        both_train["mobility_values_center_exp"].values.reshape(-1, 1),
    )
    both_train["mobility_values_center_pred"] = reg.predict(
        both_train["mobility_values_center_ref"].values.reshape(-1, 1)
    )
    LR_train = RT_metrics(
        RT_obs=both_train["mobility_values_center_exp"],
        RT_pred=both_train["mobility_values_center_pred"],
    )
    train_delta95 = LR_train.CalcDeltaRTwidth()
    both_test["mobility_values_center_pred"] = reg.predict(
        both_test["mobility_values_center_ref"].values.reshape(-1, 1)
    )
    LR_test = RT_metrics(
        RT_obs=both_test["mobility_values_center_exp"],
        RT_pred=both_test["mobility_values_center_pred"],
    )
    test_delta95 = LR_test.CalcDeltaRTwidth()
    Logger.info(
        "Delta95 base: %s, for training set: %s, for testing set: %s",
        base_delta95,
        train_delta95,
        test_delta95,
    )
    maxquant_dict.loc[maxquant_dict["source"] != "exp", "mobility_pred"] = reg.predict(
        maxquant_dict.loc[
            maxquant_dict["source"] != "exp", "mobility_values_center_ref"
        ]
        .to_numpy()
        .reshape(-1, 1)
    )
    maxquant_dict.loc[maxquant_dict["source"] == "exp", "mobility_pred"] = reg.predict(
        maxquant_dict.loc[
            maxquant_dict["source"] == "exp", "mobility_values_center_exp"
        ]
        .to_numpy()
        .reshape(-1, 1)
    )
    maxquant_dict["mobility_pred"] = maxquant_dict["mobility_pred"].clip(
        lower=min_exp_im, upper=max_exp_im
    )
    return maxquant_dict, test_delta95


def dict_add_rt_align_lowess(
    maxquant_dict: pd.DataFrame,
    train_frac: float,
    random_state: int,
):
    """
    Add experimental RT prediction from LOWESS based
    on reference RT values.

    When reference RT values are not available,
    the prediction is inferred using experimental RT values.
    Which essentially adds noise to the true value.

    Outputs maxquant_dict with additional column 'mobility_pred'
    """

    # select the precursors appearing in both ref and exp for training and testing
    Logger.info("aligning RT using LOWESS")
    min_exp_rt = maxquant_dict["Retention time_exp"].min()
    max_exp_rt = maxquant_dict["Retention time_exp"].max()
    Logger.info("min_exp_rt: %s, max_exp_rt: %s", min_exp_rt, max_exp_rt)
    try:
        both = maxquant_dict.loc[
            (maxquant_dict["source"] == "both") & (~maxquant_dict["Decoy"])
        ].copy()
    except KeyError:
        both = maxquant_dict.loc[(maxquant_dict["source"] == "both")].copy()
    Logger.info("both dataframe size: %s", both.shape)
    both_train, both_test = train_test_split(
        both, test_size=1 - train_frac, random_state=random_state
    )
    lowess = sm.nonparametric.lowess
    both_test["predicted_RT"] = lowess(
        endog=both_train["Retention time_exp"],
        exog=both_train["Retention time_ref"],
        frac=0.05,
        return_sorted=False,
        xvals=both_test["Retention time_ref"],
    )
    lowess_eval = RT_metrics(
        both_test["Retention time_exp"],
        both_test["predicted_RT"],
    )

    test_delta95 = lowess_eval.CalcDeltaRTwidth()
    Logger.info(
        "Delta RT 95 for test set: %s",
        test_delta95,
    )
    maxquant_dict.loc[maxquant_dict["source"] != "exp", "predicted_RT"] = lowess(
        endog=both_train["Retention time_exp"],
        exog=both_train["Retention time_ref"],
        frac=0.05,
        return_sorted=False,
        xvals=maxquant_dict.loc[maxquant_dict["source"] != "exp", "Retention time_ref"],
    )
    # maxquant_dict.loc[maxquant_dict["source"] == "exp", "predicted_RT"] = (
    #     maxquant_dict.loc[
    #         maxquant_dict["source"] == "exp", "Calibrated retention time_exp"
    #     ]
    # )  # Shouldn't apply LOWESS to experimental since the lowess assumption doesn't hold

    # For the columns where the ref RT is NaN, use the exp RT with additional noise by boostrapping residuals
    lowess_fit_residuals = (
        maxquant_dict.loc[maxquant_dict["source"] == "both", "predicted_RT"]
        - maxquant_dict.loc[maxquant_dict["source"] == "both", "Retention time_exp"]
    )
    noise = np.random.choice(
        lowess_fit_residuals,
        size=len(maxquant_dict.loc[maxquant_dict["source"] == "exp"]),
    )
    Logger.info(
        "RT Noise size: %s, mean: %s, std: %s", len(noise), noise.mean(), noise.std()
    )
    maxquant_dict.loc[maxquant_dict["source"] == "exp", "predicted_RT"] = (
        maxquant_dict.loc[maxquant_dict["source"] == "exp", "Retention time_exp"]
        + noise
    )

    maxquant_dict["predicted_RT"] = maxquant_dict["predicted_RT"].clip(
        lower=min_exp_rt, upper=max_exp_rt
    )
    return maxquant_dict, test_delta95


def filter_maxquant_by_ok(
    evidence_100fdr: pd.DataFrame,
    ok_dir: str,
    ok_output_type: str = "psms",
    rescore_fdr: float = 0.01,
):
    """
    Filter maxquant by Oktoberfest rescoring results.
    Please ensure the evidence file is at 100% FDR.
    :param evidence_100fdr: pd.DataFrame, the evidence file at 100% FDR
    :param ok_dir: str, the directory of Oktoberfest results
    :param ok_output_type: str, one of "psms" or "peptides"
    :param rescore_fdr: float, the FDR threshold for rescoring
    :return: pd.DataFrame, the evidence file at 1% FDR after rescoring
    """
    Logger.info(
        "Filtering maxquant by Oktoberfest rescoring results. Please ensure the evidence file is at 100% FDR."
    )
    assert ok_output_type in ["psms", "peptides"]  # one of the oktoberfest output types
    ok_df = pd.read_csv(
        os.path.join(
            ok_dir, "results", "mokapot", f"rescore.mokapot.{ok_output_type}.txt"
        ),
        sep="\t",
    )
    ok_df_001fdr = ok_df[ok_df["mokapot q-value"] <= rescore_fdr]
    ok_df_001fdr.rename(
        {"Peptide": "Peptide_mq", "Proteins": "Proteins_mq"}, axis=1, inplace=True
    )
    evidence_rescore_001fdr = pd.merge(
        evidence_100fdr,
        ok_df_001fdr,
        left_on=["Raw file", "MS/MS scan number"],
        right_on=["filename", "ScanNr"],
        suffixes=("_mq", "_ok"),
    )
    evidence_rescore_001fdr["Proteins"] = evidence_rescore_001fdr["Proteins"].fillna(
        evidence_rescore_001fdr["Proteins_mq"]
    )  # fill in missing proteins
    return evidence_rescore_001fdr


def construct_dict_from_search_pivoted(
    cfg_prepare_dict,
    evidence: pd.DataFrame,
    # rt_values_df: pd.DataFrame,
    # mobility_values_df: pd.DataFrame,
    n_blocks_by_pept: int = 1,
):

    evidence_cleaned = cleanup_maxquant(
        evidence,
        id_cols=["Sequence", "Modifications", "Charge", "Raw file", "Proteins"],
        how_duplicates="keep_highest_int",
    )
    # MATCH/MBR rows may carry a null Modified sequence even though the same
    # peptide was directly identified (MSMS) in another run.  Propagate the
    # Modified sequence from the MSMS row to all rows in the same
    # (Sequence, Modifications, Charge) group so the aggregation in
    # get_rt_im_range always finds a non-null value.
    evidence_cleaned["Modified sequence"] = (
        evidence_cleaned
        .groupby(["Sequence", "Modifications", "Charge"], dropna=False)["Modified sequence"]
        .transform(lambda x: x.fillna(x.dropna().iloc[0]) if x.notna().any() else x)
    )
    # After propagation, rows that are still null are MATCH-only groups with no
    # MSMS donor present.  Recover unmodified peptides from the bare Sequence;
    # anything with actual modifications that still has no Modified sequence is
    # ambiguous — drop it with a warning.
    still_null = evidence_cleaned["Modified sequence"].isna()
    if still_null.any():
        unmod_mask = still_null & (evidence_cleaned["Modifications"] == "Unmodified")
        evidence_cleaned.loc[unmod_mask, "Modified sequence"] = (
            "_" + evidence_cleaned.loc[unmod_mask, "Sequence"] + "_"
        )
        still_null = evidence_cleaned["Modified sequence"].isna()
        if still_null.any():
            Logger.warning(
                "Dropping %d rows with irrecoverable null 'Modified sequence' "
                "(modified MATCH-only peptides with no MSMS donor)",
                still_null.sum(),
            )
            evidence_cleaned = evidence_cleaned[~still_null]
    evidence_pivoted = pivot_psm_by_mz_rank(evidence_cleaned)
    evidence_group_summary = get_rt_im_range(
        evidence_cleaned,
        id_cols=["Sequence", "Modifications", "Charge", "Proteins"],
        summarize_without_match=cfg_prepare_dict.SUMMARIZE_WITHOUT_MATCH,
    )

    # add extra columns
    evidence_group_summary = dict_add_iso_pattern(
        evidence_group_summary, ab_thres=cfg_prepare_dict.ISO_MIN_AB_THRES
    )
    evidence_group_summary = dict_add_mz_rank(maxquant_dict_df=evidence_group_summary)

    evidence_group_summary = dict_add_mz_bin(
        maxquant_dict_df=evidence_group_summary,
        mz_bin_digits=cfg_prepare_dict.MZ_BIN_DIGITS,
    )
    evidence_group_summary = dict_add_mz_len(maxquant_dict_df=evidence_group_summary)

    pept_batch_size = ceil(evidence_group_summary.shape[0] / n_blocks_by_pept) + 1
    evidence_group_summary["pept_batch_idx"] = (
        evidence_group_summary["mz_rank"] // pept_batch_size
    ).astype(int)
    # evidence_group_summary = dict_add_rt_index(
    #     evidence_group_summary,
    #     rt_values_df,
    #     "RT_search_left",
    #     "RT_search_center",
    #     "RT_search_right",
    #     idx_suffix="_ref",
    # )
    # evidence_group_summary = dict_add_im_index(
    #     evidence_group_summary,
    #     mobility_values_df,
    #     "IM_search_left",
    #     "IM_search_center",
    #     "IM_search_right",
    #     idx_suffix="_ref",
    # )
    dict_ref = pd.merge(
        evidence_pivoted,
        evidence_group_summary,
        on=["Sequence", "Modifications", "Charge"],
        how="left",
    )
    Logger.info("Total entries in dictionary: %s", dict_ref.shape)
    return dict_ref


def construct_dict(
    cfg_prepare_dict,
    filter_exp_by_raw_file: List[str],
    maxquant_ref_df: pd.DataFrame,
    maxquant_exp_path: str,
    result_dir: str,
    rt_values_df: pd.DataFrame,
    ref_type: Literal["MQ", "pred"] = "MQ",
    mobility_values_df: Optional[pd.DataFrame] = None,
    use_ims: bool = True,
    random_seed: int = 42,
    n_blocks_by_pept: int = 1,
    keep_matched_precursors: bool = False,
    # device: str = "gpu",
):
    gpu_count = torch.cuda.device_count()
    match gpu_count:
        case 0:
            device = "cpu"
            Logger.info("No GPU available, using CPU")
        case 1:
            device = "cuda"
            Logger.info("Using 1 GPU, device is %s", device)
        case _:
            device = "gpu"
            Logger.info("Using multiple GPUs, device is %s", device)
    maxquant_exp_df = pd.read_csv(maxquant_exp_path, sep="\t", low_memory=False)
    if (
        "sage_discriminant_score" in maxquant_exp_df.columns
    ):  # infer search engine, remap column names
        Logger.debug("Sage discriminant score found, parsing sage results")
        maxquant_exp_df = sage_parser(maxquant_exp_df)
        Logger.debug("Renamed columns: %s", maxquant_exp_df.columns)
    Logger.info("maxquant_exp_df size: %s", maxquant_exp_df.shape)
    maxquant_exp_df = maxquant_exp_df.loc[
        maxquant_exp_df["Raw file"].isin(filter_exp_by_raw_file)
    ]
    Logger.info(
        "maxquant_exp_df size after filter by raw file %s: %s",
        filter_exp_by_raw_file,
        maxquant_exp_df.shape,
    )
    if len(cfg_prepare_dict.EXP_OK_DIR) > 0:
        maxquant_exp_df = filter_maxquant_by_ok(
            evidence_100fdr=maxquant_exp_df,
            ok_dir=cfg_prepare_dict.EXP_OK_DIR,
            ok_output_type=cfg_prepare_dict.EXP_OK_OUTPUT,
            rescore_fdr=cfg_prepare_dict.EXP_OK_FDR,
        )
        Logger.info(
            "maxquant_exp_df size after filtering by Oktoberfest: %s",
            maxquant_exp_df.shape,
        )
    if len(cfg_prepare_dict.REF_OK_DIR) > 0:
        maxquant_ref_df = filter_maxquant_by_ok(
            evidence_100fdr=maxquant_ref_df,
            ok_dir=cfg_prepare_dict.REF_OK_DIR,
            ok_output_type=cfg_prepare_dict.REF_OK_OUTPUT,
            rescore_fdr=cfg_prepare_dict.REF_OK_FDR,
        )
        Logger.info(
            "maxquant_ref_df size after filtering by Oktoberfest: %s",
            maxquant_ref_df.shape,
        )
    if not keep_matched_precursors:
        if use_ims:
            maxquant_exp_df = maxquant_exp_df.loc[
                maxquant_exp_df["Type"].isin(["TIMS-MULTI-MSMS"])
            ]
            maxquant_ref_df = maxquant_ref_df.loc[
                maxquant_ref_df["Type"].isin(["TIMS-MULTI-MSMS"])
            ]
        else:
            maxquant_exp_df = maxquant_exp_df.loc[
                maxquant_exp_df["Type"].isin(["MULTI-MSMS", "MSMS"])
            ]
            maxquant_ref_df = maxquant_ref_df.loc[
                maxquant_ref_df["Type"].isin(["MULTI-MSMS", "MSMS"])
            ]
        Logger.info(
            "maxquant_exp_df size after removing matched precursors: %s",
            maxquant_exp_df.shape,
        )

        Logger.info(
            "maxquant_ref_df size after removing matched precursors: %s",
            maxquant_ref_df.shape,
        )
    rt_range = (
        rt_values_df["Time_minute"].min(),
        rt_values_df["Time_minute"].max(),
    )
    Logger.info("RT range: %s", rt_range)
    construct_dict_dir = os.path.join(result_dir, "construct_dict")
    rt_transfer_dir = os.path.join(construct_dict_dir, "RT_transfer_learn")
    os.makedirs(construct_dict_dir, exist_ok=True)
    os.makedirs(rt_transfer_dir, exist_ok=True)
    if use_ims:
        assert mobility_values_df is not None
        im_range = (
            mobility_values_df["mobility_values"].min(),
            mobility_values_df["mobility_values"].max(),
        )
        im_idx_range = (
            mobility_values_df["mobility_values_index"].min(),
            mobility_values_df["mobility_values_index"].max(),
        )
        Logger.info("IM range: %s", im_range)

        im_transfer_dir = os.path.join(construct_dict_dir, "IM_transfer_learn")
        os.makedirs(im_transfer_dir, exist_ok=True)
    maxquant_exp_filtered_path = os.path.join(
        construct_dict_dir, "maxquant_exp_filtered.txt"
    )
    maxquant_exp_df.to_csv(maxquant_exp_filtered_path, sep="\t")

    _LOADED_ALPHA_DATASET = False
    # RT
    if cfg_prepare_dict.RT_REF == "pred":
        if cfg_prepare_dict.UPDATED_RT_MODEL_PATH == "":
            if not _LOADED_ALPHA_DATASET:
                train_df, test_df, rt_max = prepare_alpha_train_test_df(
                    maxquant_exp_filtered_path,
                    train_frac=cfg_prepare_dict.TRAIN_FRAC,
                    filter_dict={"raw_name": filter_exp_by_raw_file},
                    random_state=random_seed,
                )
                cfg_prepare_dict.RT_MAX = rt_max.item()
                _LOADED_ALPHA_DATASET = True
            Logger.info("Retraining RT model with AlphaPeptDeep")
            delta_rt_95, model_path = update_alpha_pept_deep_model(
                pept_property="rt",
                train_df=train_df,
                test_df=test_df,
                save_dir=rt_transfer_dir,
                epoch=cfg_prepare_dict.RT_TRAIN_EPOCHS,
                lc_grad=rt_max,
                device=device,
            )
            cfg_prepare_dict.UPDATED_RT_MODEL_PATH = model_path
            if cfg_prepare_dict.RT_TOL < 0:
                cfg_prepare_dict.RT_TOL = delta_rt_95.item()
        else:
            Logger.info("Using existing RT model")
            delta_rt_95 = cfg_prepare_dict.RT_TOL
    # IM
    if use_ims and cfg_prepare_dict.IM_REF == "pred":
        if cfg_prepare_dict.UPDATED_IM_MODEL_PATH == "":
            Logger.info("Retraining IM model with AlphaPeptDeep")
            if not _LOADED_ALPHA_DATASET:
                train_df, test_df, rt_max = prepare_alpha_train_test_df(
                    maxquant_exp_filtered_path,
                    train_frac=cfg_prepare_dict.TRAIN_FRAC,
                    filter_dict={"raw_name": filter_exp_by_raw_file},
                    random_state=random_seed,
                )
                _LOADED_ALPHA_DATASET = True
                cfg_prepare_dict.RT_MAX = rt_max.item()
            delta_im_95, model_path = update_alpha_pept_deep_model(
                pept_property="mobility",
                train_df=train_df,
                test_df=test_df,
                save_dir=im_transfer_dir,
                epoch=cfg_prepare_dict.IM_TRAIN_EPOCHS,
                lc_grad=rt_max,
                device=device,
            )
            cfg_prepare_dict.UPDATED_IM_MODEL_PATH = model_path
            if cfg_prepare_dict.DELTA_IM_95 < 0:
                cfg_prepare_dict.DELTA_IM_95 = delta_im_95.item()
        else:
            Logger.info("Using existing IM model")
            delta_im_95 = cfg_prepare_dict.DELTA_IM_95

    maxquant_exp_df["Retention time start"] = (
        maxquant_exp_df["Calibrated retention time start"]
        - maxquant_exp_df["Retention time calibration"]
    )
    maxquant_exp_df["Retention time finish"] = (
        maxquant_exp_df["Calibrated retention time finish"]
        - maxquant_exp_df["Retention time calibration"]
    )
    maxquant_exp_df["Retention time start"].fillna(
        maxquant_exp_df["Calibrated retention time start"], inplace=True
    )  # if rt calibration is missing that rt should be the same before and after calib
    maxquant_exp_df["Retention time finish"].fillna(
        maxquant_exp_df["Calibrated retention time finish"], inplace=True
    )  # if rt calibration is missing that rt should be the same before and after calib
    # get idx of exp RT and IM values
    maxquant_exp_df = dict_add_rt_index(
        maxquant_df=maxquant_exp_df,
        rt_values_df=rt_values_df,
        mq_rt_left_col="Retention time start",
        mq_rt_center_col="Retention time",
        mq_rt_right_col="Retention time finish",
        idx_suffix="_exp",
    )  # exp values are based on the calibration
    Logger.debug(
        "maxquant_exp_df shape after adding rt index: %s", maxquant_exp_df.shape
    )
    if use_ims:
        assert mobility_values_df is not None
        maxquant_exp_df = dict_add_im_index(
            maxquant_df=maxquant_exp_df,
            mobility_values_df=mobility_values_df,
            mq_im_center_col="1/K0",
            idx_suffix="_exp",
        )
    if ref_type == "MQ" and use_ims:
        assert mobility_values_df is not None
        # dict_add_rt_index doesn't work for ref values
        maxquant_ref_df = dict_add_im_index(
            maxquant_df=maxquant_ref_df,
            mobility_values_df=mobility_values_df,
            mq_im_center_col="1/K0",
            idx_suffix="_ref",
            im_idx_length=None,
        )

    maxquant_ref_df["Retention time start"] = (
        maxquant_ref_df["Calibrated retention time start"]
        - maxquant_ref_df["Retention time calibration"]
    )
    maxquant_ref_df["Retention time finish"] = (
        maxquant_ref_df["Calibrated retention time finish"]
        - maxquant_ref_df["Retention time calibration"]
    )
    maxquant_ref_df["Retention time start"].fillna(
        maxquant_ref_df["Calibrated retention time start"], inplace=True
    )  # if rt calibration is missing that rt should be the same before and after calib
    maxquant_ref_df["Retention time finish"].fillna(
        maxquant_ref_df["Calibrated retention time finish"], inplace=True
    )  # if rt calibration is missing that rt should be the same before and after calib
    # merge reference and experiment
    maxquant_dict = merge_ref_and_exp(
        maxquant_ref_df=maxquant_ref_df,
        maxquant_exp_df=maxquant_exp_df,
        save_dir=construct_dict_dir,
        ref_type=ref_type,
        use_ims=use_ims,
    )

    # generate decoy first and then predict RT and IM
    if cfg_prepare_dict.GENERATE_DECOY:
        Logger.info("Generating decoy")
        maxquant_dict = concat_decoy_and_target(maxquant_dict)
    else:
        maxquant_dict["Decoy"] = False
    # Do prediction first and then concat target and decoys
    # IM/RT prediction for full dictionary, define RT and IM search range
    dict_path = os.path.join(construct_dict_dir, "maxquant_dict_for_pred.txt")
    maxquant_dict_to_pred = maxquant_dict.copy()
    # Logger.info("Maxquant_dict_to_pred columns: %s", maxquant_dict_to_pred.columns)
    # Logger.info("Maxquant_dict columns : %s", maxquant_dict.columns)
    # maxquant_dict_to_pred["Retention time"] = maxquant_dict["Retention time"].fillna(
    #     maxquant_dict["Retention time"].mean()
    # )
    maxquant_dict_to_pred.to_csv(
        dict_path,
        sep="\t",
        index=False,
    )
    # add rt pred
    match cfg_prepare_dict.RT_REF:
        case "pred":
            maxquant_dict = dict_add_alpha_pept_pred(
                model_path=cfg_prepare_dict.UPDATED_RT_MODEL_PATH,
                pept_property="rt",
                dict_for_pred_path=dict_path,
                maxquant_dict=maxquant_dict,
                lc_grad=cfg_prepare_dict.RT_MAX,
                device=device,
            )
        case "align_lr":
            raise NotImplementedError("RT align_lr not implemented yet")
        case "align_lowess":
            maxquant_dict, delta_rt_95 = dict_add_rt_align_lowess(
                maxquant_dict,
                train_frac=cfg_prepare_dict.TRAIN_FRAC,
                random_state=random_seed,
            )
            if cfg_prepare_dict.RT_TOL < 0:
                cfg_prepare_dict.RT_TOL = delta_rt_95.item()
        case "exp":
            Logger.info("Using exp RT for RT reference")
            pass
        case "ref":
            Logger.info("Using ref RT for RT reference")
            pass
        case "mix":
            Logger.info("Using mix RT for RT reference/prediction")
            pass
        case _:
            raise ValueError(f"RT reference {cfg_prepare_dict.RT_REF} not supported")

    # add im pred
    if use_ims:
        match cfg_prepare_dict.IM_REF:
            case "pred":
                maxquant_dict = dict_add_alpha_pept_pred(
                    model_path=cfg_prepare_dict.UPDATED_IM_MODEL_PATH,
                    pept_property="mobility",
                    dict_for_pred_path=dict_path,
                    maxquant_dict=maxquant_dict,
                    lc_grad=cfg_prepare_dict.RT_MAX,
                    device=device,
                )
            case "align_lr":
                maxquant_dict, delta_im_95 = dict_add_im_align_lr(
                    maxquant_dict,
                    train_frac=cfg_prepare_dict.TRAIN_FRAC,
                    random_state=random_seed,
                )
                cfg_prepare_dict.DELTA_IM_95 = delta_im_95.item()
            case "ref":
                Logger.info("Using ref IM for IM prediction")
                pass
            case "mix":
                Logger.info("Using mix IM for IM reference/prediction")
                pass
            case "exp":
                Logger.info("Using exp IM for IM reference")
                pass
            case _:
                raise ValueError(
                    f"IM reference {cfg_prepare_dict.IM_REF} not supported"
                )
        maxquant_dict = maxquant_dict.rename(
            mapper={"mobility_values_index": "mobility_pred_idx"}, axis=1
        )
        # specify im tolerence for search range (expected ion mobility length)
        if cfg_prepare_dict.IM_LENGTH < 0:
            Logger.info(
                "IM tolerance not specified, using 99.9 percentile of experiment IM length"
            )
            im_length = (
                int(maxquant_exp_df["Ion mobility length"].quantile(0.999) + 2) // 2
            )  # TODO: currently using only exp IM length
            cfg_prepare_dict.IM_LENGTH = im_length

        maxquant_dict = _define_im_idx_search_range(
            maxquant_df=maxquant_dict,
            im_length=cfg_prepare_dict.IM_LENGTH,
            im_ref=cfg_prepare_dict.IM_REF,
            im_idx_range=im_idx_range,
            delta_im_95=cfg_prepare_dict.DELTA_IM_95,
            mobility_values_df=mobility_values_df,
        )
    maxquant_dict = _define_rt_search_range(
        maxquant_result_dict=maxquant_dict,
        rt_tol=float(cfg_prepare_dict.RT_TOL),
        rt_ref=cfg_prepare_dict.RT_REF,
        rt_range=rt_range,
    )

    # get idx of all predicted RT and IM values
    maxquant_dict = dict_add_rt_index(
        maxquant_dict,
        rt_values_df,
        "RT_search_left",
        "RT_search_center",
        "RT_search_right",
        idx_suffix="_ref",
    )
    # add extra columns
    maxquant_dict = dict_add_iso_pattern(
        maxquant_dict, ab_thres=cfg_prepare_dict.ISO_MIN_AB_THRES
    )
    maxquant_dict = dict_add_mz_rank(maxquant_dict_df=maxquant_dict)
    if not cfg_prepare_dict.GENERATE_DECOY:
        maxquant_dict["TD pair id"] = maxquant_dict["mz_rank"]
    maxquant_dict = dict_add_mz_bin(
        maxquant_dict_df=maxquant_dict,
        mz_bin_digits=cfg_prepare_dict.MZ_BIN_DIGITS,
    )
    maxquant_dict = dict_add_mz_len(maxquant_dict_df=maxquant_dict)

    pept_batch_size = ceil(maxquant_dict.shape[0] / n_blocks_by_pept) + 1
    maxquant_dict["pept_batch_idx"] = (
        maxquant_dict["mz_rank"] // pept_batch_size
    ).astype(int)
    # split between target and decoys
    maxquant_dict_target = maxquant_dict.loc[~maxquant_dict["Decoy"]].copy()
    maxquant_dict_decoy = maxquant_dict.loc[maxquant_dict["Decoy"]].copy()
    # get batch cutoff based on target peptides
    # save results
    dict_pickle_path = os.path.join(result_dir, "maxquant_result_dict.pkl")
    dict_target_pickle_path = os.path.join(result_dir, "maxquant_result_target_ref.pkl")
    dict_decoy_pickle_path = os.path.join(result_dir, "maxquant_result_decoy_ref.pkl")
    maxquant_dict_target.to_pickle(dict_target_pickle_path)
    maxquant_dict_decoy.to_pickle(dict_decoy_pickle_path)
    maxquant_dict.to_pickle(dict_pickle_path)
    Logger.info(
        "Finish dictionary construction. Filtered prediction dataframe dimension: %s.",
        maxquant_dict.shape,
    )
    Logger.debug("Columns in maxquant_dict: %s", maxquant_dict.columns)
    return (
        maxquant_dict,
        maxquant_dict_target,
        maxquant_dict_decoy,
        dict_target_pickle_path,
        dict_decoy_pickle_path,
        dict_pickle_path,
        cfg_prepare_dict,
    )


def get_mzrank_batch_cutoff(maxquant_dict_df: pd.DataFrame):
    """Get the cutoff for each batch of peptides based on mz rank"""
    max_min_mz_rank = maxquant_dict_df.groupby("pept_batch_idx")["mz_rank"].agg(
        ["min", "max"]
    )
    # cutoff = [0]  # start from 0
    cutoff = []
    cutoff.extend(
        max_min_mz_rank["min"].values[1:].tolist()
    )  # take all except the first and last
    cutoff.append(
        max_min_mz_rank["max"].values[-1] + 1
    )  # end at the last, +1 to include the last
    return cutoff


def align_rt_from_multiple_source(
    original_dict: pd.DataFrame, target_col: str
) -> pd.DataFrame:
    # Filter first
    filtered = original_dict[original_dict["Type"] == "TIMS-MULTI-MSMS"]

    # Group and keep only the row with the highest Intensity in each group
    original_dict_agg_int_max = filtered.loc[
        filtered.groupby(["Modified sequence", "Charge", "Raw file"])[
            "Intensity"
        ].idxmax()
    ]
    # pivot rt table
    rt_pivot = original_dict_agg_int_max.pivot_table(
        index=["Modified sequence", "Charge"],
        columns=["Raw file"],
        values="Retention time",
        aggfunc="mean",
    )

    # New DataFrame for aligned results
    aligned_df = pd.DataFrame(index=rt_pivot.index)

    # Loop over all other columns
    for col in rt_pivot.columns:
        if col == target_col:
            aligned_df[col] = rt_pivot[col]  # keep original reference as-is
            continue

        # Get non-missing pairs between reference and current column
        valid = rt_pivot[[target_col, col]].dropna()
        if len(valid) < 3:
            aligned_df[col] = np.nan  # too few points to fit LOWESS
            continue

        xvals = rt_pivot[col]  # your x values (may have NaN)
        mask_valid_x = xvals.notna()  # remember where xvals are valid

        # Prepare output array filled with NaN
        aligned_col = np.full(len(xvals), np.nan)

        # Fit LOWESS using valid data only
        fitted = sm.nonparametric.lowess(
            endog=valid[target_col],
            exog=valid[col],
            frac=0.05,
            return_sorted=False,
            xvals=xvals[mask_valid_x],
        )

        # Insert fitted values only where xvals were valid
        aligned_col[mask_valid_x] = fitted

        # Store in result DataFrame
        aligned_df[col] = aligned_col
    ref_rt = aligned_df.drop(columns=[target_col]).mean(axis=1)
    ref_rt.fillna(aligned_df[target_col], inplace=True)
    return ref_rt.reset_index().rename(columns={0: "Retention time_ref"})


def get_rt_im_range(
    evidence: pd.DataFrame,
    match_col: str = "Type",
    id_cols: List[str] = ["Sequence", "Modifications", "Charge"],
    rt_col: str = "Retention time",
    # rt_start_col: str = "Calibrated retention time start",
    # rt_finish_col: str = "Calibrated retention time finish",
    rt_length_col: str = "Retention length",
    mobility_col: str = "1/K0",
    mobility_length_col: str = "1/K0 length",
    summarize_without_match: bool = False,
):
    # --- Step 7: Get rt and ion mobility experimental statistic
    if summarize_without_match:
        evidence = evidence[
            ~evidence[match_col].str.contains("MATCH", na=False)
        ]  # only keep matched for summary
    evidence_group_summary = (
        evidence.groupby(id_cols)
        .agg(
            modified_sequence=(
                "Modified sequence",
                lambda x: x.dropna().iloc[0] if x.notna().any() else np.nan,
            ),
            mz=("m/z", "mean"),
            count=(rt_col, "count"),
            rt_apex_mean=(rt_col, "mean"),
            rt_apex_std=(rt_col, "std"),
            rt_apex_min=(rt_col, "min"),
            rt_apex_max=(rt_col, "max"),
            mobility_exp_mean=(mobility_col, "mean"),
            mobility_exp_std=(mobility_col, "std"),
            mobility_exp_min=(mobility_col, "min"),
            mobility_exp_max=(mobility_col, "max"),
            # rt_start_mean=(rt_start_col, "mean"),
            # rt_start_std=(rt_start_col, "std"),
            # rt_start_min=(rt_start_col, "min"),
            # rt_finish_mean=(rt_finish_col, "mean"),
            # rt_finish_std=(rt_finish_col, "std"),
            # rt_finish_max=(rt_finish_col, "max"),
            rt_length_mean=(rt_length_col, "mean"),
            rt_length_std=(rt_length_col, "std"),
            rt_length_min=(rt_length_col, "min"),
            rt_length_max=(rt_length_col, "max"),
            mobility_length_mean=(mobility_length_col, "mean"),
            mobility_length_std=(mobility_length_col, "std"),
            mobility_length_min=(mobility_length_col, "min"),
            mobility_length_max=(mobility_length_col, "max"),
        )
        .reset_index()
    )
    evidence_group_summary = evidence_group_summary.rename(
        columns={"modified_sequence": "Modified sequence", "mz": "m/z"}
    )
    evidence_group_summary["rt_apex_std"] = evidence_group_summary[
        "rt_apex_std"
    ].fillna(evidence_group_summary["rt_apex_std"].mean())
    evidence_group_summary["rt_length_std"] = evidence_group_summary[
        "rt_length_std"
    ].fillna(evidence_group_summary["rt_length_std"].mean())
    evidence_group_summary["mobility_exp_std"] = evidence_group_summary[
        "mobility_exp_std"
    ].fillna(evidence_group_summary["mobility_exp_std"].mean())
    evidence_group_summary["mobility_length_std"] = evidence_group_summary[
        "mobility_length_std"
    ].fillna(evidence_group_summary["mobility_length_std"].mean())
    rt_length_max_left_clip = evidence_group_summary.loc[
        evidence_group_summary["rt_length_max"] > 0, "rt_length_max"
    ].quantile(0.01)
    rt_length_max_right_clip = evidence_group_summary.loc[
        evidence_group_summary["rt_length_max"] > 0, "rt_length_max"
    ].quantile(0.99)
    im_length_max_left_clip = evidence_group_summary.loc[
        evidence_group_summary["mobility_length_max"] > 0, "mobility_length_max"
    ].quantile(0.01)
    im_length_max_right_clip = evidence_group_summary.loc[
        evidence_group_summary["mobility_length_max"] > 0, "mobility_length_max"
    ].quantile(0.99)
    Logger.info(
        f"RT length max left clip: {rt_length_max_left_clip}, RT length max right clip: {rt_length_max_right_clip}"
    )
    Logger.info(
        f"IM length max left clip: {im_length_max_left_clip}, IM length max right clip: {im_length_max_right_clip}"
    )
    evidence_group_summary["RT_search_left"] = (
        evidence_group_summary["rt_apex_min"]
        - evidence_group_summary["rt_apex_std"]
        - np.clip(
            evidence_group_summary["rt_length_max"],
            rt_length_max_left_clip,
            rt_length_max_right_clip,
        )
        - evidence_group_summary["rt_length_std"]
    )
    evidence_group_summary["RT_search_center"] = evidence_group_summary["rt_apex_mean"]
    evidence_group_summary["RT_search_right"] = (
        evidence_group_summary["rt_apex_max"]
        + evidence_group_summary["rt_apex_std"]
        + np.clip(
            evidence_group_summary["rt_length_max"],
            rt_length_max_left_clip,
            rt_length_max_right_clip,
        )
        + evidence_group_summary["rt_length_std"]
    )
    evidence_group_summary["IM_search_left"] = (
        evidence_group_summary["mobility_exp_min"]
        - evidence_group_summary["mobility_exp_std"]
        - np.clip(
            evidence_group_summary["mobility_length_max"],
            im_length_max_left_clip,
            im_length_max_right_clip,
        )
        - evidence_group_summary["mobility_length_std"]
    )
    evidence_group_summary["IM_search_right"] = (
        evidence_group_summary["mobility_exp_max"]
        + evidence_group_summary["mobility_exp_std"]
        + np.clip(
            evidence_group_summary["mobility_length_max"],
            im_length_max_left_clip,
            im_length_max_right_clip,
        )
        + evidence_group_summary["mobility_length_std"]
    )
    evidence_group_summary["IM_search_center"] = evidence_group_summary[
        "mobility_exp_mean"
    ]
    return evidence_group_summary


def pivot_psm_by_mz_rank(
    evidence: pd.DataFrame,
    match_keyword: str = "MATCH",
    msms_keyword: str = "MSMS",
    metric_col: str = "Score",
    intensity_col: str = "Intensity",
    rt_col: str = "Retention time",
    mobility_col: str = "1/K0",
    higher_is_better: bool = True,
    id_cols: List[str] = ["Sequence", "Modifications", "Charge"],
    # summarize_with_match: bool = True,
):
    """
    Pivot the evidence dataframe to have one row per peptide (defined by id_cols) and one column per raw file, with values indicating the status of that peptide in that raw file (Match, Quant_Only, Reference, Other).
    Arguments:
    - evidence: pd.DataFrame, the input evidence dataframe
    - match_keyword: str, the keyword to identify matched PSMs in the "Type" column
    - msms_keyword: str, the keyword to identify MS/MS PSMs in the
    "Type" column
    - metric_col: str, the column name of the metric used to determine global best (e.g., "Score")
    - intensity_col: str, the column name of the intensity used for tie-breaking when determining global best
    - higher_is_better: bool, whether a higher value of the metric indicates a better PSM (default True, e.g., for Score)
    - id_cols: List[str], the columns that define a unique peptide (default ["Sequence", "Modifications", "Charge"])
    Returns:
    - result_df: pd.DataFrame, the pivoted dataframe with one row per peptide and columns for each raw file indicating the status, as well as pivoted RT and mobility values.
    """
    # --- Step 1: Global Best Identification ---
    # Reset index first so that .loc[index] never matches more than one row
    # (duplicate indices arise when the caller passes a filtered/concatenated df).
    evidence = evidence.reset_index(drop=True)
    # Prefer direct MS/MS identifications as global reference candidates.
    # MATCH/MBR rows are still kept for RT/IM metrics (Steps 3-4).
    # Use dropna=False so peptides with NaN in any id column are not silently dropped.
    msms_mask = ~evidence["Type"].str.upper().str.contains(match_keyword.upper(), na=False)
    msms_only = evidence[msms_mask]
    sorted_msms = msms_only.sort_values(
        by=id_cols + [metric_col, intensity_col],
        ascending=[True] * len(id_cols) + [not higher_is_better, False],
    )
    evidence["is_global_best"] = False
    evidence.loc[sorted_msms.groupby(id_cols, dropna=False).head(1).index, "is_global_best"] = True

    # Fallback: for peptides with no MSMS rows at all, use their best row of any type.
    no_best_mask = ~evidence.groupby(id_cols, dropna=False)["is_global_best"].transform("any")
    fallback = evidence[no_best_mask].sort_values(
        by=id_cols + [metric_col, intensity_col],
        ascending=[True] * len(id_cols) + [not higher_is_better, False],
    )
    evidence.loc[fallback.groupby(id_cols, dropna=False).head(1).index, "is_global_best"] = True

    # --- Step 2: Status Labeling ---
    def label_logic(row):
        if row["is_global_best"]:
            return 1
        elif match_keyword.upper() in str(row["Type"]).upper():
            return 3
        elif msms_keyword.upper() in str(row["Type"]).upper():
            return 2
        else:
            return 0

    evidence["Status_Val"] = evidence.apply(label_logic, axis=1)

    # --- Step 3: Pivot Status ---
    # We use min() here as per your logic to prioritize the '1' (Reference) label
    # if a file has multiple entries for the same peptide.
    summary = evidence.groupby(id_cols + ["Raw file"], dropna=False)["Status_Val"].min().reset_index()

    pivot_status = summary.pivot(index=id_cols, columns="Raw file", values="Status_Val")
    mapping = {3: "Match", 2: "Quant_Only", 1: "Reference", 0: "Other"}
    pivot_status = pivot_status.map(lambda x: mapping.get(x, "Not_Match"))

    # --- Step 4: Pivot Metrics (RT and 1/K0) ---
    # We take the mean or first if multiple entries exist per raw file
    metrics_summary = (
        evidence.groupby(id_cols + ["Raw file"])[[rt_col, mobility_col]]
        .mean()
        .reset_index()
    )

    pivot_rt = metrics_summary.pivot(index=id_cols, columns="Raw file", values=rt_col)
    pivot_rt.columns = [f"{col}_RT" for col in pivot_rt.columns]
    pivot_rt.fillna(
        0, inplace=True
    )  # fill missing RT with 0, can be changed to other value if needed
    pivot_mobility = metrics_summary.pivot(
        index=id_cols, columns="Raw file", values=mobility_col
    )
    pivot_mobility.columns = [f"{col}_1K0" for col in pivot_mobility.columns]
    pivot_mobility.fillna(
        0, inplace=True
    )  # fill missing mobility with 0, can be changed to other value if needed
    # --- Step 5: Combine everything ---
    # join=inner ensures we keep only rows consistent across pivots
    result_df = pd.concat([pivot_status, pivot_rt, pivot_mobility], axis=1)

    # --- Step 6: Validation ---
    ref_counts = (pivot_status == "Reference").sum(axis=1)
    if not (ref_counts == 1).all():
        raise AssertionError(
            f"Validation failed: {sum(ref_counts != 1)} rows don't have exactly 1 Reference."
        )

    return result_df


def add_rt_index_to_pivot(
    raw_file_swaps_dir_map: Dict[str, str],
    pivot_df: pd.DataFrame,
    swaps_parent_dir: str,
):
    for raw_file, swaps_dir in raw_file_swaps_dir_map.items():
        ms1scans = pd.read_csv(
            os.path.join(swaps_parent_dir, swaps_dir, "ms1scans.csv")
        )

        pivot_df.sort_values(by=f"{raw_file}_RT", inplace=True)
        pivot_df = pd.merge_asof(
            left=pivot_df,
            right=ms1scans[["Time_minute", "MS1_frame_idx"]],
            left_on=f"{raw_file}_RT",
            right_on="Time_minute",
            direction="nearest",
        )
        pivot_df.rename(
            columns={"MS1_frame_idx": f"{raw_file}_MS1_frame_idx_exp"}, inplace=True
        )
        pivot_df.drop(columns=["Time_minute"], inplace=True)

    return pivot_df


def add_im_index_to_pivot(
    raw_file_swaps_dir_map: Dict[str, str],
    pivot_df: pd.DataFrame,
    swaps_parent_dir: str,
):
    for raw_file, swaps_dir in raw_file_swaps_dir_map.items():
        mobility_values = pd.read_csv(
            os.path.join(swaps_parent_dir, swaps_dir, "mobility_values.csv")
        )
        pivot_df.sort_values(by=f"{raw_file}_1K0", inplace=True)
        pivot_df = pd.merge_asof(
            left=pivot_df,
            right=mobility_values[["mobility_values_index", "mobility_values"]],
            left_on=f"{raw_file}_1K0",
            right_on="mobility_values",
            direction="nearest",
        )
        pivot_df.rename(
            columns={"mobility_values_index": f"{raw_file}_mobility_values_index_exp"},
            inplace=True,
        )
        pivot_df.drop(columns=["mobility_values"], inplace=True)
    return pivot_df
