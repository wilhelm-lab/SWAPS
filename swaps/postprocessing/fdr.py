from typing import List
import itertools
from tqdm import tqdm
import pandas as pd
import logging

Logger = logging.getLogger(__name__)


def generate_signal_compete_pairs(
    maxquant_dict, groupby_columns="mz_bin", col_to_keep=None
):
    mz_bin_groups = (
        maxquant_dict.groupby(groupby_columns)
        .filter(lambda x: len(x) > 1)
        .groupby("mz_bin")
    )
    # Step 3: Apply the function to each group and concatenate the results
    result = pd.concat(
        [
            _generate_signal_compete_pairs_within_group(group, col_to_keep)
            for name, group in tqdm(mz_bin_groups, desc="Processing groups")
        ]
    ).reset_index(drop=True)
    return result


# Step 2: Define a function to generate pairs and retain specific columns
def _generate_signal_compete_pairs_within_group(
    group,
    col_to_keep: List[str] | None = None,
):
    if col_to_keep is None:
        col_to_keep = [
            "mz_rank",
            "RT_search_left",
            "RT_search_center",
            "RT_search_right",
            "IM_search_idx_left",
            "IM_search_idx_center",
            "IM_search_idx_right",
            "mz_bin",
            "Decoy",
        ]
    if len(group) > 1:
        pairs = list(itertools.combinations(group.index, 2))
        entry1 = group.loc[
            [p[0] for p in pairs],
            col_to_keep,
        ].reset_index(drop=True)
        entry2 = group.loc[
            [p[1] for p in pairs],
            col_to_keep,
        ].reset_index(drop=True)

        # Combine entry1 and entry2 columns into a single DataFrame
        result_pairs = pd.concat(
            [entry1.add_suffix("_entry1"), entry2.add_suffix("_entry2")], axis=1
        )
        return result_pairs
    return pd.DataFrame()  # Return empty DataFrame if group has only one entry


def get_isolated_decoys_from_pairs(
    result: pd.DataFrame,
    decoy_mz_ranks: List,
    delta_rt_95: float = 0.86,
    delta_im_index_95: int = 290,
):
    # Step 1: Calculate the distance between the center of the RT and IM ranges
    result["rt_center_dist"] = abs(
        result["RT_search_right_entry1"] - result["RT_search_left_entry2"]
    )
    result["im_center_dist"] = abs(
        result["IM_search_idx_right_entry1"] - result["IM_search_idx_left_entry2"]
    )
    result["Decoy_count"] = result["Decoy_entry1"].astype(int) + result[
        "Decoy_entry2"
    ].astype(int)

    # Step 2: classsify the pairs as close or far based on the distance
    result["dist"] = "far"
    result.loc[
        (result["rt_center_dist"] < delta_rt_95 * 2)
        & (result["im_center_dist"] < delta_im_index_95 * 2),
        "dist",
    ] = "close"

    # Step 3: get sets of isolated decoys (1, far), (2, far) and (2, close)
    result_1_far = result.loc[(result["dist"] == "far") & (result["Decoy_count"] == 1)]
    result_2_far = result.loc[(result["dist"] == "far") & (result["Decoy_count"] == 2)]
    result_2_close = result.loc[
        (result["dist"] == "close") & (result["Decoy_count"] == 2)
    ]
    isolated_decoys_set = set.union(
        set(result_1_far["mz_rank_entry1"]),
        set(result_1_far["mz_rank_entry2"]),
        set(result_2_far["mz_rank_entry1"]),
        set(result_2_far["mz_rank_entry2"]),
        set(result_2_close["mz_rank_entry1"]),
        set(result_2_close["mz_rank_entry2"]),
    )
    result_1_close = result.loc[
        (result["dist"] == "close") & (result["Decoy_count"] == 1)
    ]
    non_isolated_mz_ranks = set.union(
        set(result_1_close["mz_rank_entry1"]), set(result_1_close["mz_rank_entry2"])
    )
    isolated_decoys_set = isolated_decoys_set - non_isolated_mz_ranks
    isolated_decoys_set = isolated_decoys_set.intersection(decoy_mz_ranks)
    return isolated_decoys_set


def get_isolated_decoy_from_mzbins(maxquant_result_ref: pd.DataFrame):
    mz_bin_groups_tdc = (
        maxquant_result_ref.groupby(["mz_bin"])["Decoy"]
        .value_counts()
        .unstack()
        .fillna(0)
    )
    decoy_only_bins = mz_bin_groups_tdc.loc[
        (mz_bin_groups_tdc[True] > 0) & (mz_bin_groups_tdc[False] == 0)
    ]
    Logger.info("Number of decoy-only bins: %s", decoy_only_bins.shape[0])
    Logger.info("Number of decoys in decoy-only bins: %s", decoy_only_bins[True].sum())
    decoy_mz_bin = mz_bin_groups_tdc.loc[
        (mz_bin_groups_tdc[True] == 1) & (mz_bin_groups_tdc[False] == 0)
    ].index
    isolated_decoy_mz_bin_rank = set(
        maxquant_result_ref.loc[maxquant_result_ref["mz_bin"].isin(decoy_mz_bin)][
            "mz_rank"
        ].values
    )
    return isolated_decoy_mz_bin_rank
