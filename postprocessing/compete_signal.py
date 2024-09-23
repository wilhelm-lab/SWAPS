import logging
import numpy as np
import pandas as pd

Logger = logging.getLogger(__name__)


# compete target decoy score
# Function to determine the 'loser' based on the lower target_decoy_score
# Function to determine the 'loser' and 'winner' based on target_decoy_score
def determine_winner_loser(row):
    if row["target_decoy_score_entry1"] < row["target_decoy_score_entry2"]:
        return pd.Series(
            [row["mz_rank_entry1"], row["mz_rank_entry2"]], index=["loser", "winner"]
        )
    else:
        return pd.Series(
            [row["mz_rank_entry2"], row["mz_rank_entry1"]], index=["loser", "winner"]
        )


def get_target_mz_rank(row):
    if row["Decoy_entry1"]:
        return row["mz_rank_entry2"]
    else:
        return row["mz_rank_entry1"]


def compete_candidates_for_signal(
    result,
    pept_act_sum_ps: pd.DataFrame,
    delta_rt_95=0.86,
    delta_im_idx_95=290,
    log_sum_intensity_thres=2,
    delta_log_sum_intensity_thres=0.01,
):
    # Step 1: Calculate the distance between the center of the RT and IM ranges
    result["rt_center_dist"] = abs(
        result["RT_search_right_entry1"] - result["RT_search_left_entry2"]
    )
    result["im_center_dist"] = abs(
        result["IM_search_idx_right_entry1"] - result["IM_search_idx_left_entry2"]
    )

    # Step 2: filter only the pairs that are close in RT and IM
    result_filtered = result.loc[
        (result["rt_center_dist"] < delta_rt_95 * 2)
        & (result["im_center_dist"] < delta_im_idx_95 * 2)
    ]
    Logger.info(
        "Number of pairs after filtering rt and im distance: %d",
        result_filtered.shape[0],
    )

    # Step 3: merge for model inference
    result_filtered = pd.merge(
        left=result_filtered,
        right=pept_act_sum_ps[["mz_rank", "target_decoy_score", "sum_intensity"]],
        left_on="mz_rank_entry1",
        how="inner",
        right_on=["mz_rank"],
    )
    result_filtered.drop("mz_rank", axis=1, inplace=True)
    result_filtered = pd.merge(
        left=result_filtered,
        right=pept_act_sum_ps[["mz_rank", "target_decoy_score", "sum_intensity"]],
        left_on="mz_rank_entry2",
        how="inner",
        right_on=["mz_rank"],
        suffixes=("_entry1", "_entry2"),
    )

    # get log sum
    result_filtered["log_sum_intensity_entry1"] = np.log10(
        result_filtered["sum_intensity_entry1"] + 1
    )
    result_filtered["log_sum_intensity_entry2"] = np.log10(
        result_filtered["sum_intensity_entry2"] + 1
    )
    result_filtered["delta_log_sum_intensity"] = abs(
        result_filtered["log_sum_intensity_entry1"]
        - result_filtered["log_sum_intensity_entry2"]
    )
    result_filtered["delta_log_sum_intensity"].describe()

    # filtered out both zero or one zero column --> no compepetition needed
    result_filtered_no_low_int = result_filtered.loc[
        (result_filtered["log_sum_intensity_entry1"] > log_sum_intensity_thres)
        & (result_filtered["log_sum_intensity_entry2"] > log_sum_intensity_thres)
    ]
    Logger.info(
        "Number of pairs after filtering by log sum intensity: %d",
        result_filtered_no_low_int.shape[0],
    )

    # if log intensity is more than 1 --> no competition needed
    result_filtered_no_low_int_and_only_close_int = result_filtered_no_low_int.loc[
        result_filtered_no_low_int["delta_log_sum_intensity"]
        < delta_log_sum_intensity_thres
    ]
    Logger.info(
        "Number of pairs with delta log intensity < 0.5: %d",
        result_filtered_no_low_int_and_only_close_int.shape[0],
    )

    result_filtered_no_low_int_and_only_close_int[["loser", "winner"]] = (
        result_filtered_no_low_int_and_only_close_int.apply(
            determine_winner_loser, axis=1
        )
    )

    pept_act_sum_ps.loc[
        pept_act_sum_ps["mz_rank"].isin(
            result_filtered_no_low_int_and_only_close_int["winner"]
        ),
        "competition",
    ] = "winner"
    pept_act_sum_ps.loc[
        pept_act_sum_ps["mz_rank"].isin(
            result_filtered_no_low_int_and_only_close_int["loser"]
        ),
        "competition",
    ] = "loser"
    pept_act_sum_ps["competition"].fillna("no_competition", inplace=True)
    Logger.info(
        "Number of winners, losers and no competition: %s",
        pept_act_sum_ps["competition"].value_counts(),
    )
    return (
        pept_act_sum_ps,
        result_filtered_no_low_int_and_only_close_int,
        result_filtered,
    )
