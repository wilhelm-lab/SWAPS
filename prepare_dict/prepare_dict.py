"""

Prepare dictionary
Usage:
    generate_reference.py --config_path=<config_path>
    generate_reference.py -h | --help

Options:
    -h --help               show this screen help
    --version              show version
"""

from typing import List, Literal
import re
import os
import logging
from typing import Literal
from math import ceil
import pandas as pd
import matplotlib.pyplot as plt
import IsoSpecPy as iso
import torch
from utils.constants import decoy_mutation_rule
from utils.plot import save_plot
from utils.tools import cleanup_maxquant
from utils.metrics import RT_metrics
from .predict_rt import dict_add_rt_pred, update_rt_model
import numpy as np
from typing import Literal
from alphabase.psm_reader import psm_reader_provider
from sklearn.model_selection import train_test_split
from peptdeep.pretrained_models import ModelManager

Logger = logging.getLogger(__name__)


def merge_ref_and_exp(
    maxquant_ref_df: pd.DataFrame,
    maxquant_exp_df: pd.DataFrame,
    save_dir: str,
    ref_type: str = ["MQ", "pred"],
):
    """Merge dictionaries from multiple files using Modified sequence and Charge"""
    # evaluate the elution counts of the experiment file
    exp_elution_counts = maxquant_exp_df.groupby(["Modified sequence", "Charge"]).size()
    plt.bar(
        height=exp_elution_counts.value_counts().values,
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
    maxquant_exp_df = cleanup_maxquant(
        maxquant_exp_df, remove_decoys=True, how_duplicates="keep_highest_int"
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

    # Drop the indicator column
    ref_unique_rows = ref_unique_rows.drop(columns=["_merge"])
    exp_unique_rows = exp_unique_rows.drop(columns=["_merge"])
    both_unique_rows = both_unique_rows.drop(columns=["_merge"])

    # Optionally, you can merge the unique rows back to the original ref dataframe
    ref_unqiue_df = maxquant_ref_df.merge(
        ref_unique_rows, on=join_on_columns, how="inner"
    )
    exp_unique_df = maxquant_exp_df.merge(
        exp_unique_rows, on=join_on_columns, how="inner"
    )
    if ref_type == "MQ":
        ref_spec_columns = [
            # "MS1_frame_idx_left_ref",
            # "MS1_frame_idx_right_ref",
            # "MS1_frame_idx_center_ref",
            # "MS1_frame_idx_left",
            # "MS1_frame_idx_right",
            # "MS1_frame_idx_center",
            "mobility_values_index_left_ref",
            "mobility_values_index_right_ref",
            "mobility_values_index_center_ref",
            # "mobility_values_left_ref",
            # "mobility_values_right_ref",
            "mobility_values_center_ref",
            # "mobility_values_left_ref",
            # "mobility_values_right_ref",
            # "mobility_values_center_ref",
            # "Time_minute_left_ref",
            # "Time_minute_right_ref",
            # "Time_minute_center_ref",
        ]
    elif ref_type == "pred":
        ref_spec_columns = []
    Logger.info("Maxquant_ref_df columns: %s", maxquant_ref_df.columns)
    Logger.info("Maxquant_exp_df columns: %s", maxquant_exp_df.columns)
    both_unique_df = maxquant_exp_df.merge(
        maxquant_ref_df[join_on_columns + ref_spec_columns],
        on=join_on_columns,
        how="inner",
    )
    exp_unique_df["source"] = "exp"
    ref_unqiue_df["source"] = "ref"
    both_unique_df["source"] = "both"
    maxquant_merge_df = pd.concat(
        [ref_unqiue_df, exp_unique_df, both_unique_df],
        ignore_index=True,
        axis=0,
        join="outer",
    )
    # evaluate
    plt.bar(
        height=maxquant_merge_df["source"].value_counts().values,
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

    # Logger.debug(
    #     "Mutated sequence: %s, %s, %s, %s, %s",
    #     seq[0],
    #     decoy_mutation_rule[seq[1]],
    #     seq[2:-2],
    #     decoy_mutation_rule[seq[-2]],
    #     seq[-1],
    # )
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
    logging.info("Number of TD pairs with identical iso mz: %s", result.shape[0])
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
        im_idx_length = maxquant_df_with_im_index["Ion mobility length"] // 2 + 1
        Logger.info(
            "mq_im_left_col or mq_im_right_col not given, \
                IM index length required but not given, \
                    using peptide specific im length values"
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
                ),
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
                ),
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
    mq_rt_left_col: str = "RT_search_left",
    mq_rt_center_col: str = "RT_search_center",
    mq_rt_right_col: str = "RT_search_right",
    idx_suffix: str = "",
):
    """Get RT index of RT_search_left, RT_search_center and RT_search_right/
    as indicated in RT values from .d file for each row in maxquant_dict_df/
    has built-in control of int type for index"""
    Logger.debug("dict_add_rt_index columns: %s", maxquant_df.columns)
    Logger.debug("mq_rt_center_col columns: %s", mq_rt_center_col)
    maxquant_df = maxquant_df.sort_values(mq_rt_center_col)
    maxquant_df = pd.merge_asof(
        left=maxquant_df,
        right=rt_values_df[["Time_minute", "MS1_frame_idx"]],
        left_on=mq_rt_center_col,
        right_on="Time_minute",
        direction="nearest",
        suffixes=("", "_center" + idx_suffix),
    )
    for col in ["Time_minute", "MS1_frame_idx"]:
        if col in maxquant_df.columns:
            maxquant_df.rename(
                {col: col + "_center" + idx_suffix}, axis=1, inplace=True
            )
    maxquant_df = maxquant_df.sort_values(mq_rt_left_col)
    maxquant_df = pd.merge_asof(
        left=maxquant_df,
        right=rt_values_df[["Time_minute", "MS1_frame_idx"]],
        left_on=mq_rt_left_col,
        right_on="Time_minute",
        direction="nearest",
        suffixes=("", "_left" + idx_suffix),
    )
    for col in ["Time_minute", "MS1_frame_idx"]:
        if col in maxquant_df.columns:
            maxquant_df.rename({col: col + "_left" + idx_suffix}, axis=1, inplace=True)

    maxquant_df = maxquant_df.sort_values(mq_rt_right_col)
    maxquant_df = pd.merge_asof(
        left=maxquant_df,
        right=rt_values_df[["Time_minute", "MS1_frame_idx"]],
        left_on=mq_rt_right_col,
        right_on="Time_minute",
        direction="nearest",
        suffixes=("", "_right" + idx_suffix),
    )
    for col in ["Time_minute", "MS1_frame_idx"]:
        if col in maxquant_df.columns:
            maxquant_df.rename({col: col + "_right" + idx_suffix}, axis=1, inplace=True)

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
    rt_ref: Literal["exp", "pred", "mix"],
    rt_range: tuple[float, float],
):
    """Define the search range for the precursor RT."""
    match rt_ref:
        case "exp":
            maxquant_result_dict["Calibrated retention time start"] = (
                maxquant_result_dict["Calibrated retention time start"]
                - maxquant_result_dict["Retention time calibration"]
            )
            maxquant_result_dict["Calibrated retention time finish"] = (
                maxquant_result_dict["Calibrated retention time finish"]
                - maxquant_result_dict["Retention time calibration"]
            )
            maxquant_result_dict["Calibrated retention time start"].fillna(
                maxquant_result_dict["Retention time"]
                - 0.5 * maxquant_result_dict["Retention length"],
                inplace=True,
            )
            maxquant_result_dict["Calibrated retention time finish"].fillna(
                maxquant_result_dict["Retention time"]
                + 0.5 * maxquant_result_dict["Retention length"],
                inplace=True,
            )
            maxquant_result_dict["RT_search_left"] = (
                maxquant_result_dict["Calibrated retention time start"] - rt_tol
            )
            maxquant_result_dict["RT_search_right"] = (
                maxquant_result_dict["Calibrated retention time finish"] + rt_tol
            )
            rt_ref_act_peak = "Calibrated retention time"
        case "pred":
            maxquant_result_dict["RT_search_left"] = (
                maxquant_result_dict["predicted_RT"] - rt_tol
            )
            maxquant_result_dict["RT_search_right"] = (
                maxquant_result_dict["predicted_RT"] + rt_tol
            )
            rt_ref_act_peak = "predicted_RT"
        case "mix":
            maxquant_result_dict["RT_search_left"] = (
                maxquant_result_dict["Calibrated retention time start_ss"] - rt_tol
            )
            maxquant_result_dict["RT_search_right"] = (
                maxquant_result_dict["Calibrated retention time finish_ss"] + rt_tol
            )
            rt_ref_act_peak = "Calibrated retention time_ss"
    maxquant_result_dict["RT_search_center"] = maxquant_result_dict[rt_ref_act_peak]
    maxquant_result_dict[
        ["RT_search_left", "RT_search_center", "RT_search_right"]
    ].clip(rt_range[0], rt_range[1], inplace=True)
    return maxquant_result_dict


def _get_im_idx_span_from_values(mobility_values_df: pd.DataFrame, im_length: float):
    """Get the span of the ion mobility index given im value length."""
    im_idx_span = ceil(im_length / mobility_values_df["mobility_values"].diff().mean())
    return im_idx_span


def _define_im_idx_search_range(
    maxquant_df: pd.DataFrame,
    im_length: int,
    im_ref: Literal["exp", "pred", "mix", "ref"],
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
        case "pred":
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
            Logger.info(
                "Delta IM index 95 left: %s, right: %s",
                delta_im_idx_95_left,
                delta_im_idx_95_right,
            )
            maxquant_df["IM_search_idx_center"] = maxquant_df[
                "mobility_values_index_center_ref"
            ]
            maxquant_df["IM_search_idx_center"].fillna(
                maxquant_df["mobility_values_index_center_exp"], inplace=True
            )
            maxquant_df["IM_search_idx_center"] = maxquant_df[
                "IM_search_idx_center"
            ].astype(int)
            maxquant_df["IM_search_idx_left"] = (
                maxquant_df["IM_search_idx_center"]
                - half_im_length
                - delta_im_idx_95_left
            )
            maxquant_df["IM_search_idx_right"] = (
                maxquant_df["IM_search_idx_center"]
                + half_im_length
                + delta_im_idx_95_right
            )
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
    if pept_property == "rt":
        model = models.rt_model
    elif pept_property == "mobility":
        model = models.ccs_model

    # train
    model.train(train_df, epoch=epoch, verbose=True, verbose_each_epoch=False)

    # eval
    predict_train_df = model.predict(train_df)
    predict_test_df = model.predict(test_df)
    if pept_property == "mobility":
        predict_train_df = model.ccs_to_mobility_pred(train_df)
        predict_test_df = model.ccs_to_mobility_pred(test_df)

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
    # maxquant_dict = pd.read_csv(dict_for_pred_path, sep="\t")
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
        maxquant_dict_new["predicted_RT"] = maxquant_dict_new["rt_pred"]

    return maxquant_dict_new


def construct_dict(
    cfg_prepare_dict,
    filter_exp_by_raw_file: List[str],
    maxquant_ref_df: pd.DataFrame,
    maxquant_exp_path: str,
    ref_type: Literal["MQ", "pred"] = "MQ",
    # maxquant_exp_df: pd.DataFrame,
    result_dir: str = None,
    mobility_values_df: pd.DataFrame = None,
    rt_values_df: pd.DataFrame = None,
    random_seed: int = 42,
    n_blocks_by_pept: int = 1,
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
    # device = str("cuda" if torch.cuda.is_available() else "cpu")
    # Logger.info(f"Device: {device}")
    maxquant_exp_df = pd.read_csv(maxquant_exp_path, sep="\t", low_memory=False)
    Logger.info("maxquant_exp_df size: %s", maxquant_exp_df.shape)
    maxquant_exp_df = maxquant_exp_df.loc[
        maxquant_exp_df["Raw file"].isin(filter_exp_by_raw_file)
    ]
    Logger.info(
        "maxquant_exp_df size after filter by raw file %s: %s",
        filter_exp_by_raw_file,
        maxquant_exp_df.shape,
    )
    rt_range = (
        rt_values_df["Time_minute"].min(),
        rt_values_df["Time_minute"].max(),
    )
    Logger.info("RT index range: %s", rt_range)
    im_range = (
        mobility_values_df["mobility_values"].min(),
        mobility_values_df["mobility_values"].max(),
    )
    im_idx_range = (
        mobility_values_df["mobility_values_index"].min(),
        mobility_values_df["mobility_values_index"].max(),
    )
    Logger.info("IM index range: %s", im_range)
    construct_dict_dir = os.path.join(result_dir, "construct_dict")
    rt_transfer_dir = os.path.join(construct_dict_dir, "RT_transfer_learn")
    im_transfer_dir = os.path.join(construct_dict_dir, "IM_transfer_learn")
    os.makedirs(construct_dict_dir, exist_ok=True)
    os.makedirs(rt_transfer_dir, exist_ok=True)
    os.makedirs(im_transfer_dir, exist_ok=True)

    # # retrain model
    # if len(cfg_prepare_dict.UPDATED_MODEL_PATH) == 0:
    #     Logger.info("Retraining RT model")
    #     delta_rt_95, model_path = update_rt_model(
    #         train_maxquant_df=maxquant_exp_df,
    #         train_dir=rt_transfer_dir,
    #         train_frac=cfg_prepare_dict.TRAIN_FRAC,
    #         seed=random_seed,
    #         keep_matched_precursors=cfg_prepare_dict.KEEP_MATCHED_PRECURSORS,
    #         save_model_name="updated",
    #     )  # delta rt 95 is one side
    #     cfg_prepare_dict.UPDATED_MODEL_PATH = model_path
    #     if cfg_prepare_dict.RT_TOL < 0:
    #         cfg_prepare_dict.RT_TOL = delta_rt_95.item()
    # else:
    #     Logger.info("Using existing RT model")
    #     delta_rt_95 = cfg_prepare_dict.RT_TOL

    # retrain model w/ AlphaPeptDeep
    # prepare train and test df: from maxquant_exp_df
    _LOADED_ALPHA_DATASET = False
    # RT
    if cfg_prepare_dict.RT_REF == "pred":
        if cfg_prepare_dict.UPDATED_RT_MODEL_PATH == "":
            if not _LOADED_ALPHA_DATASET:
                train_df, test_df, rt_max = prepare_alpha_train_test_df(
                    maxquant_exp_path,
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
    if cfg_prepare_dict.IM_REF == "pred":
        if cfg_prepare_dict.UPDATED_IM_MODEL_PATH == "":
            Logger.info("Retraining IM model with AlphaPeptDeep")
            if not _LOADED_ALPHA_DATASET:
                train_df, test_df, rt_max = prepare_alpha_train_test_df(
                    maxquant_exp_path,
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

    # get idx of exp RT and IM values
    maxquant_exp_df = dict_add_rt_index(
        maxquant_df=maxquant_exp_df,
        rt_values_df=rt_values_df,
        mq_rt_left_col="Calibrated retention time start",
        mq_rt_center_col="Calibrated retention time",
        mq_rt_right_col="Calibrated retention time finish",
        idx_suffix="_exp",
    )  # exp values are based on the calibration
    maxquant_exp_df = dict_add_im_index(
        maxquant_df=maxquant_exp_df,
        mobility_values_df=mobility_values_df,
        mq_im_center_col="1/K0",
        idx_suffix="_exp",
    )
    if ref_type == "MQ":
        maxquant_ref_df = dict_add_im_index(
            maxquant_df=maxquant_ref_df,
            mobility_values_df=mobility_values_df,
            mq_im_center_col="1/K0",
            idx_suffix="_ref",
            im_idx_length=None,
        )

    # merge reference and experiment
    maxquant_dict = merge_ref_and_exp(
        maxquant_ref_df=maxquant_ref_df,
        maxquant_exp_df=maxquant_exp_df,
        save_dir=construct_dict_dir,
        ref_type=ref_type,
    )

    # maxquant_dict = dict_add_rt_pred(
    #     updated_models=cfg_prepare_dict.UPDATED_MODEL_PATH,
    #     deeplc_train_path=os.path.join(rt_transfer_dir, "deeplc_train.csv"),
    #     maxquant_df=maxquant_dict,
    #     save_dir=construct_dict_dir,
    #     keep_matched_precursors=cfg_prepare_dict.KEEP_MATCHED_PRECURSORS,
    #     filter_by_rt_diff=None,  # TODO: no filter atm
    # )

    # generate decoy first and then predict RT and IM
    if cfg_prepare_dict.GENERATE_DECOY:
        Logger.info("Generating decoy")
        maxquant_dict = concat_decoy_and_target(maxquant_dict)

    # Do prediction first and then concat target and decoys
    # IM/RT prediction for full dictionary, define RT and IM search range
    dict_path = os.path.join(construct_dict_dir, "maxquant_dict_for_pred.txt")
    maxquant_dict_to_pred = maxquant_dict.copy()
    maxquant_dict_to_pred["Retention time"] = maxquant_dict["Retention time"].fillna(
        maxquant_dict["Retention time"].mean()
    )
    maxquant_dict_to_pred.to_csv(
        dict_path,
        sep="\t",
        index=False,
    )
    # add rt pred
    if cfg_prepare_dict.RT_REF == "pred":
        maxquant_dict = dict_add_alpha_pept_pred(
            model_path=cfg_prepare_dict.UPDATED_RT_MODEL_PATH,
            pept_property="rt",
            dict_for_pred_path=dict_path,
            maxquant_dict=maxquant_dict,
            lc_grad=cfg_prepare_dict.RT_MAX,
            device=device,
        )

    # add im pred
    if cfg_prepare_dict.IM_REF == "pred":
        maxquant_dict = dict_add_alpha_pept_pred(
            model_path=cfg_prepare_dict.UPDATED_IM_MODEL_PATH,
            pept_property="mobility",
            dict_for_pred_path=dict_path,
            maxquant_dict=maxquant_dict,
            lc_grad=cfg_prepare_dict.RT_MAX,
            device=device,
        )

    # maxquant_dict = maxquant_dict.sort_values("mobility_pred")
    # maxquant_dict = pd.merge_asof(
    #     left=maxquant_dict,
    #     right=mobility_values_df[["mobility_values_index", "mobility_values"]],
    #     left_on="mobility_pred",
    #     right_on="mobility_values",
    #     direction="nearest",
    # )
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
    maxquant_dict = dict_add_mz_bin(
        maxquant_dict_df=maxquant_dict,
        mz_bin_digits=cfg_prepare_dict.MZ_BIN_DIGITS,
    )
    maxquant_dict = dict_add_mz_len(maxquant_dict_df=maxquant_dict)

    pept_batch_size = ceil(maxquant_dict.shape[0] / n_blocks_by_pept) + 1
    maxquant_dict["pept_batch_idx"] = (
        maxquant_dict["mz_rank"] // pept_batch_size
    ).astype(int)
    # save results
    dict_pickle_path = os.path.join(result_dir, "maxquant_result_ref.pkl")
    maxquant_dict.to_pickle(dict_pickle_path)
    logging.info(
        "Finish. Filtered prediction dataframe dimension: %s, columns: %s",
        maxquant_dict.shape,
        maxquant_dict.columns,
    )
    return maxquant_dict, dict_pickle_path, cfg_prepare_dict


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


# if __name__ == "__main__":
#     logging.basicConfig(
#         format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
#         level=logging.INFO,
#     )
#     try:
#         arguments = docopt(
#             __doc__, argv=None, help=False, version=None, options_first=False
#         )
#         print("Arguments parsed:")
#         print(arguments)

#         # Load config
#         cfg_path = arguments["--config_path"]
#         cfg = get_cfg_defaults(swaps_optimization_cfg)
#         if cfg_path is not None:
#             cfg.merge_from_file(cfg_path)
#             print(f"Config file loaded: {cfg_path}")

#         # Prepare dictionary
#         maxquant_dict_df, delta_rt_95, dict_pickle_path = construct_dict(
#             cfg.PREPARE_DICT
#         )
#     except Exception as e:
#         print(f"Error: {e}")
#         print(__doc__)
#         raise
