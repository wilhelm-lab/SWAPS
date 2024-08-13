""" Module for comparing with maxquant results """

import logging
from typing import Literal, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.plot import save_plot
from utils.tools import _perc_fmt

Logger = logging.getLogger(__name__)


def merge_with_maxquant_exp(
    maxquant_ref_df: pd.DataFrame,
    maxquant_exp_df: pd.DataFrame,
    ref_cols: List[str] = None,
    exp_cols: List[str] = None,
    other_cols: List[str] = None,
    how_join: Literal["right", "left", "outer"] = "right",
):
    """compare the inferred intensities from maxquant and SBS,
        when a different dictionary then experiment MQ result is used

    :maxquant_ref_df: the maxquant result of reference dictionary, usually deeper than the experiment
    :maxquant_exp_df: the maxquant result of the experiment MS2 data, usually shallower than the reference
    :ref_cols: columns to keep from the maxquant_ref_df
    :exp_cols: columns to keep from the maxquant_exp_df
    :other_cols: columns to keep from reference
    :how_join: how to join the two dataframes, "left" means keeping all entries from reference, /
        right means keeping all entries from experiment, usually used in result evaluation, /
        outer means keeping all entries from both dataframes, usually used in constructing complete dictionary
    """
    for cols in [ref_cols, exp_cols, other_cols]:  # fix [dangerous-default-value]
        if cols is None:
            cols = []
    ref_fix_cols = [
        "Sequence",
        "Modified sequence",
        "Charge",
        "predicted_RT",
        "m/z",
        "Mass",
        "Length",
        "id",
        "RT_search_left",
        "RT_search_right",
        "RT_search_center",
        "mz_rank",
        "Reverse",
    ]
    exp_fix_cols = [
        "Modified sequence",
        "Charge",
        "Calibrated retention time start",
        "Calibrated retention time finish",
        "Calibrated retention time",
        "Retention time",
        "Intensity",
    ]
    if "mz_rank" in maxquant_exp_df.columns:
        maxquant_ref_and_exp = pd.merge(
            left=maxquant_ref_df[set(ref_fix_cols + ref_cols + other_cols)],
            right=maxquant_exp_df[set(exp_fix_cols + exp_cols + ["mz_rank"])],
            on=["mz_rank", "Modified sequence", "Charge"],
            how=how_join,
            indicator=True,
        )
    else:
        maxquant_ref_and_exp = pd.merge(
            left=maxquant_ref_df[set(ref_fix_cols + ref_cols + other_cols)],
            right=maxquant_exp_df[set(exp_fix_cols + exp_cols)],
            on=["Modified sequence", "Charge"],
            how=how_join,
            indicator=True,
        )
    Logger.debug("Experiment file has %s entries.", maxquant_exp_df.shape[0])
    Logger.debug("Merged file has %s entries.", maxquant_ref_and_exp.shape[0])
    return maxquant_ref_and_exp


def evaluate_rt_overlap(
    maxquant_ref_and_exp: pd.DataFrame, save_dir: str | None = None
):
    """evaluate the RT overlap between the experiment and reference file"""

    def _categorize_ranges(row):
        if (
            row["RT_search_left"] <= row["Calibrated retention time start"]
            and row["RT_search_right"] >= row["Calibrated retention time finish"]
        ):
            return "full_overlap"
        elif (
            row["RT_search_right"] <= row["Calibrated retention time start"]
            or row["RT_search_left"] >= row["Calibrated retention time finish"]
        ):
            return "no_overlap"
        elif np.isnan(row["Calibrated retention time start"]):
            return "no_entry_in_exp"
        elif np.isnan(row["RT_search_left"]):
            return "no_entry_in_ref"
        else:
            return "partial_overlap"

    maxquant_ref_and_exp["RT_overlap"] = maxquant_ref_and_exp.apply(
        _categorize_ranges, axis=1
    )
    Logger.info(
        "RT overlap counts: %s", maxquant_ref_and_exp["RT_overlap"].value_counts()
    )
    plt.pie(
        maxquant_ref_and_exp["RT_overlap"].value_counts().values,
        labels=maxquant_ref_and_exp["RT_overlap"].value_counts().index,
        autopct=lambda x: _perc_fmt(x, maxquant_ref_and_exp.shape[0]),
    )
    save_plot(save_dir=save_dir, fig_type_name="PieChart", fig_spec_name="RT_overlap")
    return maxquant_ref_and_exp


def evaluate_im_overlap(
    maxquant_ref_and_exp: pd.DataFrame,
    save_dir: str | None = None,
    delta_im: float = 0.04,
):
    """evaluate the IM overlap between the experiment and reference file"""
    maxquant_ref_and_exp["IM_search_left"] = (
        maxquant_ref_and_exp["mobility_values"] - delta_im
    )
    maxquant_ref_and_exp["IM_search_right"] = (
        maxquant_ref_and_exp["mobility_values"] + delta_im
    )

    def _categorize_ranges(row):
        if (
            row["IM_search_left"] <= row["1/K0"] - row["1/K0 length"] / 2
            and row["IM_search_right"] >= row["1/K0"] + row["1/K0 length"] / 2
        ):
            return "full_overlap"
        elif (
            row["IM_search_left"] >= row["1/K0"] + row["1/K0 length"] / 2
            or row["IM_search_right"] <= row["1/K0"] - row["1/K0 length"] / 2
        ):
            return "no_overlap"

        elif np.isnan(row["1/K0"]):
            return "no_entry_in_exp"
        elif np.isnan(row["IM_search_left"]):
            return "no_entry_in_ref"
        else:
            return "partial_overlap"

    maxquant_ref_and_exp["IM_overlap"] = maxquant_ref_and_exp.apply(
        _categorize_ranges, axis=1
    )
    Logger.info(
        "IM overlap counts: %s", maxquant_ref_and_exp["IM_overlap"].value_counts()
    )
    plt.pie(
        maxquant_ref_and_exp["IM_overlap"].value_counts().values,
        labels=maxquant_ref_and_exp["IM_overlap"].value_counts().index,
        autopct=lambda x: _perc_fmt(x, maxquant_ref_and_exp.shape[0]),
    )
    save_plot(save_dir=save_dir, fig_type_name="PieChart_", fig_spec_name="IM_overlap")
    return maxquant_ref_and_exp


def filter_merged_by_rt_overlap(
    maxquant_ref_and_exp: pd.DataFrame,
    keep_condition: (
        List[Literal["full_overlap", "partial_overlap", "no_overlap"]] | None
    ) = None,
):
    """filter the merged maxquant result by RT_overlap condition"""
    if keep_condition is None:
        keep_condition = ["full_overlap", "partial_overlap"]
        Logger.info("No RT_overlap condition given, using %s", keep_condition)
    n_pre_filter = maxquant_ref_and_exp.shape[0]
    filtered = maxquant_ref_and_exp.loc[
        maxquant_ref_and_exp["RT_overlap"].isin(keep_condition), :
    ]
    n_post_filter = filtered.shape[0]
    Logger.info(
        "Removing %s entries with RT_overlap not in %s, %s entries left.",
        n_pre_filter - n_post_filter,
        keep_condition,
        n_post_filter,
    )
    Logger.debug("columns after filter by RT %s", n_post_filter)
    return filtered


def filter_merged_by_im_overlap(
    maxquant_ref_and_exp: pd.DataFrame,
    keep_condition: (
        List[Literal["full_overlap", "partial_overlap", "no_overlap"]] | None
    ) = None,
):
    """filter the merged maxquant result by IM_overlap condition"""
    if keep_condition is None:
        keep_condition = ["full_overlap", "partial_overlap"]
        Logger.info("No IM_overlap condition given, using %s", keep_condition)
    n_pre_filter = maxquant_ref_and_exp.shape[0]
    filtered = maxquant_ref_and_exp.loc[
        maxquant_ref_and_exp["IM_overlap"].isin(keep_condition), :
    ]
    n_post_filter = filtered.shape[0]
    Logger.info(
        "Removing %s entries with IM_overlap not in %s, %s entries left.",
        n_pre_filter - n_post_filter,
        keep_condition,
        n_post_filter,
    )
    Logger.debug("columns after filter by IM %s", n_post_filter)
    return filtered


def sum_pcm_intensity_from_exp(maxquant_exp: pd.DataFrame):
    """sum the intensity of the precursors from the experiment file

    In case of multiple PCM start and finish are the RT range of the precursor
    """
    n_pre_agg = maxquant_exp.shape[0]
    maxquant_ref_and_exp_sum_intensity = (
        maxquant_exp.groupby(["Modified sequence", "Charge"])
        .agg(
            {
                "Calibrated retention time start": "min",
                "Calibrated retention time finish": "max",
                "Calibrated retention time": "median",
                "Retention time": "median",
                "Intensity": "sum",
                "id": "first",
                "Mass": "first",
                "m/z": "first",
                "Length": "first",
                "Reverse": "first",
            }
        )
        .reset_index()
    )
    n_post_agg = maxquant_ref_and_exp_sum_intensity.shape[0]
    Logger.info(
        "Experiment data: removing %s entries with aggregation over PCM, %s entries"
        " left.",
        n_pre_agg - n_post_agg,
        n_post_agg,
    )
    # Logger.debug("columns after agg %s", maxquant_ref_and_exp_sum_intensity.columns)
    return maxquant_ref_and_exp_sum_intensity


def add_sum_act_cols(
    maxquant_ref_and_exp_sum_intensity: pd.DataFrame,
    maxquant_ref_sum_act_col: list,
    maxquant_ref_df: pd.DataFrame,
):
    """add the sum activation columns from the activation columns"""
    maxquant_ref_and_exp_sum_intensity_act = pd.merge(
        left=maxquant_ref_df[
            ["Modified sequence", "Charge", "predicted_RT"] + maxquant_ref_sum_act_col
        ],
        right=maxquant_ref_and_exp_sum_intensity,
        on=["Modified sequence", "Charge"],
        how="right",
    )
    # empty field in 'id' because some entries are in ss but not in exp.

    return maxquant_ref_and_exp_sum_intensity_act
