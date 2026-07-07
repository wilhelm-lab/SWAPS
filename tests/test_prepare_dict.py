import numpy as np
import pandas as pd
import pytest

from swaps.prepare_dict.prepare_dict import (
    calculate_modpept_isopattern,
    dict_add_confounding_groups,
    dict_add_confounder_group_id,
    dict_add_merged_confounder_pattern,
)


def _make_df(rows):
    """Build a minimal dict_ref-like DataFrame from a list of dicts."""
    df = pd.DataFrame(rows)
    df["mz_rank"] = range(1, len(df) + 1)
    return df


@pytest.fixture
def confound_df():
    """
    Five candidates designed to test all confounding conditions.

    Row 0 (mz_bin=500.00, RT=[1.0,2.0], IM=[0.5,0.7]):
        confounders: row 1 (same mz_bin, overlapping RT+IM)
                     row 3 (mz_bin=500.01, same RT+IM windows)
    Row 1 (mz_bin=500.00, RT=[1.5,2.5], IM=[0.6,0.8]):
        confounders: row 0, row 3
    Row 2 (mz_bin=500.00, RT=[1.5,2.5], IM=[0.9,1.1]):
        IM does not overlap row 0 (0.9 > 0.7) → row 0 is NOT a confounder
        confounders: row 1 (IM [0.6,0.8] vs [0.9,1.1] → no overlap), row 3
        Actually row 1 IM=[0.6,0.8] vs row 2 IM=[0.9,1.1] → 0.9 > 0.8 → no overlap
        Actually row 3 IM=[0.5,0.7] vs row 2 IM=[0.9,1.1] → 0.9 > 0.7 → no overlap
        → row 2 has NO confounders
    Row 3 (mz_bin=500.01, RT=[1.0,2.0], IM=[0.5,0.7]):
        confounders: row 0, row 1
    Row 4 (mz_bin=500.02, RT=[1.0,2.0], IM=[0.5,0.7]):
        |500.02 - 500.00| = 0.02 > 0.01 → outside mz tolerance
        → row 4 has NO confounders
    """
    return _make_df(
        [
            {
                "mz_bin": 500.00,
                "RT_search_left": 1.0,
                "RT_search_right": 2.0,
                "IM_search_left": 0.5,
                "IM_search_right": 0.7,
                "Charge": 2,
            },
            {
                "mz_bin": 500.00,
                "RT_search_left": 1.5,
                "RT_search_right": 2.5,
                "IM_search_left": 0.6,
                "IM_search_right": 0.8,
                "Charge": 2,
            },
            {
                "mz_bin": 500.00,
                "RT_search_left": 1.5,
                "RT_search_right": 2.5,
                "IM_search_left": 0.9,
                "IM_search_right": 1.1,
                "Charge": 2,
            },
            {
                "mz_bin": 500.01,
                "RT_search_left": 1.0,
                "RT_search_right": 2.0,
                "IM_search_left": 0.5,
                "IM_search_right": 0.7,
                "Charge": 2,
            },
            {
                "mz_bin": 500.02,
                "RT_search_left": 1.0,
                "RT_search_right": 2.0,
                "IM_search_left": 0.5,
                "IM_search_right": 0.7,
                "Charge": 2,
            },
        ]
    )


def test_confounders_column_added(confound_df):
    result = dict_add_confounding_groups(confound_df)
    assert "confounders" in result.columns


def test_confounders_are_numpy_arrays(confound_df):
    result = dict_add_confounding_groups(confound_df)
    for arr in result["confounders"]:
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == int or np.issubdtype(arr.dtype, np.integer)


def test_self_not_in_confounders(confound_df):
    result = dict_add_confounding_groups(confound_df)
    for _, row in result.iterrows():
        assert row["mz_rank"] not in row["confounders"]


def test_overlapping_same_bin(confound_df):
    result = dict_add_confounding_groups(confound_df)
    # row 0 and row 1 share mz_bin and overlap in both RT and IM
    row0_conf = set(result.loc[result["mz_rank"] == 1, "confounders"].iloc[0])
    row1_conf = set(result.loc[result["mz_rank"] == 2, "confounders"].iloc[0])
    assert 2 in row0_conf  # mz_rank of row 1
    assert 1 in row1_conf  # mz_rank of row 0


def test_adjacent_bin_included(confound_df):
    result = dict_add_confounding_groups(confound_df)
    # row 3 is at mz_bin=500.01 (Δ=0.01 from row 0 at 500.00) and overlaps RT+IM
    row0_conf = set(result.loc[result["mz_rank"] == 1, "confounders"].iloc[0])
    assert 4 in row0_conf  # mz_rank of row 3


def test_im_no_overlap_excluded(confound_df):
    result = dict_add_confounding_groups(confound_df)
    # row 2 has IM=[0.9,1.1] — no overlap with rows 0,1,3 (all ≤0.8)
    row2_conf = result.loc[result["mz_rank"] == 3, "confounders"].iloc[0]
    assert len(row2_conf) == 0


def test_beyond_mz_tolerance_excluded(confound_df):
    result = dict_add_confounding_groups(confound_df)
    # row 4 (500.02) is 0.02 from rows 0-2 (500.00) → they must NOT be confounders
    # row 4 is 0.01 from row 3 (500.01) → row 3 IS a confounder
    row4_conf = set(result.loc[result["mz_rank"] == 5, "confounders"].iloc[0])
    assert 1 not in row4_conf  # mz_rank of row 0, |Δmz_bin|=0.02
    assert 2 not in row4_conf  # mz_rank of row 1
    assert 3 not in row4_conf  # mz_rank of row 2
    assert 4 in row4_conf  # mz_rank of row 3, |Δmz_bin|=0.01 → included
    # row 0 should not see row 4 either
    row0_conf = set(result.loc[result["mz_rank"] == 1, "confounders"].iloc[0])
    assert 5 not in row0_conf


def test_original_row_order_preserved(confound_df):
    result = dict_add_confounding_groups(confound_df)
    assert list(result.index) == list(confound_df.index)
    assert list(result["mz_rank"]) == list(confound_df["mz_rank"])


def test_input_df_not_mutated(confound_df):
    original_cols = list(confound_df.columns)
    dict_add_confounding_groups(confound_df)
    assert list(confound_df.columns) == original_cols
    assert "confounders" not in confound_df.columns


def test_charge_mismatch_excluded(confound_df):
    # give row 1 (mz_rank=2) a different charge than row 0 (mz_rank=1);
    # they otherwise overlap in mz_bin/RT/IM and would be confounders
    confound_df.loc[confound_df["mz_rank"] == 2, "Charge"] = 3
    result = dict_add_confounding_groups(confound_df)
    row0_conf = set(result.loc[result["mz_rank"] == 1, "confounders"].iloc[0])
    row1_conf = set(result.loc[result["mz_rank"] == 2, "confounders"].iloc[0])
    assert 2 not in row0_conf
    assert 1 not in row1_conf


# --- calculate_modpept_isopattern (modification token handling) ---


def test_isopattern_nterm_acetyl_token_stripped():
    # 'n[43]' N-term acetyl: the leading lowercase 'n' must be stripped before
    # ParseFASTA, else it is misread as an Asparagine residue (+114.0429 Da).
    # Recorded precursor m/z for this peptide (charge 2) is 620.8236.
    mz, _ = calculate_modpept_isopattern("n[43]AAAAAAAAAAGAAGGR", 2)
    assert abs(float(mz[0]) - 620.8236) < 0.01


def test_isopattern_oxmet_token_adds_one_oxygen():
    # '[147]' ox-Met stays correct: exactly one oxygen (~15.9949 Da) heavier
    # than the unmodified peptide (regression for the shared token stripper).
    base, _ = calculate_modpept_isopattern("MPEPTIDEK", 1)
    ox, _ = calculate_modpept_isopattern("M[147]PEPTIDEK", 1)
    assert abs((float(ox[0]) - float(base[0])) - 15.9949) < 0.01


# --- dict_add_confounder_group_id ---


def _assert_clique_partition(result):
    """Every pair sharing a (non -1) group id must mutually confound."""
    rank_to_conf = {
        int(r): {int(c) for c in conf}
        for r, conf in zip(result["mz_rank"], result["confounders"])
    }
    by_rank = result.set_index("mz_rank")["confounder_group_id"]
    for gid in set(by_rank) - {-1}:
        members = list(by_rank[by_rank == gid].index)
        for i in members:
            for j in members:
                if i != j:
                    assert (
                        j in rank_to_conf[i]
                    ), f"ranks {i},{j} share group {gid} but do not confound"


def test_group_id_clique_no_chaining(confound_df):
    # Confounder edges: 1-2, 1-4, 2-4 (triangle) and 4-5; rank 3 isolated.
    # Connected components would chain {1,2,4,5} into ONE group. Complete-linkage
    # (clique) must not: rank 5 confounds only rank 4, so it can never join the
    # {1,2,4} clique. Greedy (ascending-degree seeds) yields pairs {1,2} & {4,5}.
    grouped = dict_add_confounding_groups(confound_df)
    result = dict_add_confounder_group_id(grouped, group_id_offset=1000)
    assert "confounder_group_id" in result.columns
    by_rank = result.set_index("mz_rank")["confounder_group_id"]
    _assert_clique_partition(result)
    assert by_rank[3] == -1  # no confounders -> solo
    # rank 5 must never be grouped with the non-adjacent ranks 1 or 2
    assert by_rank[5] != by_rank[1]
    assert by_rank[5] != by_rank[2]
    # deterministic partition for this fixture
    assert by_rank[1] == by_rank[2] == 1001  # 1000 + min(1, 2)
    assert by_rank[4] == by_rank[5] == 1004  # 1000 + min(4, 5)


def test_group_id_chain_a_b_c_not_one_group():
    # A-B and B-C overlap in RT but A and C do NOT (non-transitive). A clique
    # partition must never put all three in one group.
    df = _make_df(
        [
            {"mz_bin": 500.00, "RT_search_left": 1.0, "RT_search_right": 2.0, "IM_search_left": 0.5, "IM_search_right": 0.7, "Charge": 2},
            {"mz_bin": 500.00, "RT_search_left": 1.8, "RT_search_right": 2.8, "IM_search_left": 0.5, "IM_search_right": 0.7, "Charge": 2},
            {"mz_bin": 500.00, "RT_search_left": 2.6, "RT_search_right": 3.6, "IM_search_left": 0.5, "IM_search_right": 0.7, "Charge": 2},
        ]
    )
    grouped = dict_add_confounding_groups(df)
    result = dict_add_confounder_group_id(grouped, group_id_offset=1000)
    by_rank = result.set_index("mz_rank")["confounder_group_id"]
    _assert_clique_partition(result)
    # A (rank 1) and C (rank 3) never share a group
    assert not (by_rank[1] != -1 and by_rank[1] == by_rank[3])


def test_group_id_disjoint_pairs_get_separate_ids():
    df = _make_df(
        [
            {"mz_bin": 500.00, "RT_search_left": 1.0, "RT_search_right": 2.0, "IM_search_left": 0.5, "IM_search_right": 0.7, "Charge": 2},
            {"mz_bin": 500.00, "RT_search_left": 1.0, "RT_search_right": 2.0, "IM_search_left": 0.5, "IM_search_right": 0.7, "Charge": 2},
            {"mz_bin": 600.00, "RT_search_left": 1.0, "RT_search_right": 2.0, "IM_search_left": 0.5, "IM_search_right": 0.7, "Charge": 2},
            {"mz_bin": 600.00, "RT_search_left": 1.0, "RT_search_right": 2.0, "IM_search_left": 0.5, "IM_search_right": 0.7, "Charge": 2},
        ]
    )
    grouped = dict_add_confounding_groups(df)
    result = dict_add_confounder_group_id(grouped, group_id_offset=1000)
    by_rank = result.set_index("mz_rank")["confounder_group_id"]
    assert by_rank[1] == by_rank[2] == 1001
    assert by_rank[3] == by_rank[4] == 1003
    assert by_rank[1] != by_rank[3]


def test_group_id_all_solo_when_no_confounders():
    df = _make_df(
        [
            {"mz_bin": 500.00, "RT_search_left": 1.0, "RT_search_right": 2.0, "IM_search_left": 0.5, "IM_search_right": 0.7, "Charge": 2},
            {"mz_bin": 600.00, "RT_search_left": 1.0, "RT_search_right": 2.0, "IM_search_left": 0.5, "IM_search_right": 0.7, "Charge": 2},
        ]
    )
    grouped = dict_add_confounding_groups(df)
    result = dict_add_confounder_group_id(grouped, group_id_offset=1000)
    assert (result["confounder_group_id"] == -1).all()


def test_group_id_auto_offset_disjoint_from_mz_rank():
    df = _make_df(
        [
            {"mz_bin": 500.00, "RT_search_left": 1.0, "RT_search_right": 2.0, "IM_search_left": 0.5, "IM_search_right": 0.7, "Charge": 2},
            {"mz_bin": 500.00, "RT_search_left": 1.0, "RT_search_right": 2.0, "IM_search_left": 0.5, "IM_search_right": 0.7, "Charge": 2},
        ]
    )
    grouped = dict_add_confounding_groups(df)
    result = dict_add_confounder_group_id(grouped, group_id_offset=-1)
    group_ids = result.loc[result["confounder_group_id"] != -1, "confounder_group_id"]
    assert (group_ids > df["mz_rank"].max()).all()


def test_group_id_excludes_cross_target_decoy():
    df = _make_df(
        [
            {"mz_bin": 500.00, "RT_search_left": 1.0, "RT_search_right": 2.0, "IM_search_left": 0.5, "IM_search_right": 0.7, "Charge": 2, "Decoy": False},
            {"mz_bin": 500.00, "RT_search_left": 1.0, "RT_search_right": 2.0, "IM_search_left": 0.5, "IM_search_right": 0.7, "Charge": 2, "Decoy": True},
        ]
    )
    grouped = dict_add_confounding_groups(df)
    # sanity: pairwise confounders logic (unaware of Decoy) still links them
    assert 2 in set(grouped.loc[grouped["mz_rank"] == 1, "confounders"].iloc[0])
    result = dict_add_confounder_group_id(
        grouped, group_id_offset=1000, exclude_cross_target_decoy=True
    )
    assert (result["confounder_group_id"] == -1).all()


def test_group_id_allows_cross_target_decoy_when_disabled():
    df = _make_df(
        [
            {"mz_bin": 500.00, "RT_search_left": 1.0, "RT_search_right": 2.0, "IM_search_left": 0.5, "IM_search_right": 0.7, "Charge": 2, "Decoy": False},
            {"mz_bin": 500.00, "RT_search_left": 1.0, "RT_search_right": 2.0, "IM_search_left": 0.5, "IM_search_right": 0.7, "Charge": 2, "Decoy": True},
        ]
    )
    grouped = dict_add_confounding_groups(df)
    result = dict_add_confounder_group_id(
        grouped, group_id_offset=1000, exclude_cross_target_decoy=False
    )
    by_rank = result.set_index("mz_rank")["confounder_group_id"]
    assert by_rank[1] == by_rank[2] == 1001


# --- dict_add_merged_confounder_pattern ---


@pytest.fixture
def iso_group_df():
    """Two confounding candidates (same group) with overlapping isotope
    patterns, plus one solo candidate."""
    df = _make_df(
        [
            {
                "mz_bin": 500.00,
                "RT_search_left": 1.0,
                "RT_search_right": 2.0,
                "IM_search_left": 0.5,
                "IM_search_right": 0.7,
                "Charge": 2,
                "IsoMZ": np.array([500.00, 500.50, 501.00]),
                "IsoAbundance": np.array([0.6, 0.3, 0.05]),
            },
            {
                "mz_bin": 500.00,
                "RT_search_left": 1.0,
                "RT_search_right": 2.0,
                "IM_search_left": 0.5,
                "IM_search_right": 0.7,
                "Charge": 2,
                "IsoMZ": np.array([500.00, 500.50, 501.50]),
                "IsoAbundance": np.array([0.5, 0.4, 0.1]),
            },
            {
                "mz_bin": 700.00,
                "RT_search_left": 1.0,
                "RT_search_right": 2.0,
                "IM_search_left": 0.5,
                "IM_search_right": 0.7,
                "Charge": 2,
                "IsoMZ": np.array([700.00, 700.50]),
                "IsoAbundance": np.array([0.7, 0.2]),
            },
        ]
    )
    grouped = dict_add_confounding_groups(df, mz_bin_digits=2)
    return dict_add_confounder_group_id(grouped, group_id_offset=1000)


def test_merged_pattern_columns_added(iso_group_df):
    result = dict_add_merged_confounder_pattern(iso_group_df, mz_bin_digits=2)
    for col in (
        "GroupIsoMZ",
        "GroupIsoAbundance",
        "GroupRT_search_left",
        "GroupRT_search_right",
        "GroupIM_search_left",
        "GroupIM_search_center",
        "GroupIM_search_right",
        "GroupMzLength",
    ):
        assert col in result.columns


def test_merged_pattern_solo_passthrough(iso_group_df):
    result = dict_add_merged_confounder_pattern(iso_group_df, mz_bin_digits=2)
    solo = result.loc[result["mz_rank"] == 3].iloc[0]
    assert np.array_equal(solo["GroupIsoMZ"], solo["IsoMZ"])
    assert np.array_equal(solo["GroupIsoAbundance"], solo["IsoAbundance"])
    assert solo["GroupRT_search_left"] == solo["RT_search_left"]
    assert solo["GroupRT_search_right"] == solo["RT_search_right"]
    assert solo["GroupIM_search_left"] == solo["IM_search_left"]
    assert solo["GroupIM_search_right"] == solo["IM_search_right"]
    assert solo["GroupIM_search_center"] == pytest.approx(
        (solo["IM_search_left"] + solo["IM_search_right"]) / 2
    )
    assert solo["GroupMzLength"] == len(solo["IsoMZ"])


def test_merged_pattern_is_shared_across_members(iso_group_df):
    result = dict_add_merged_confounder_pattern(iso_group_df, mz_bin_digits=2)
    m1 = result.loc[result["mz_rank"] == 1].iloc[0]
    m2 = result.loc[result["mz_rank"] == 2].iloc[0]
    assert np.array_equal(m1["GroupIsoMZ"], m2["GroupIsoMZ"])
    assert np.array_equal(m1["GroupIsoAbundance"], m2["GroupIsoAbundance"])


def test_merged_pattern_max_abundance_per_bin(iso_group_df):
    # union bins: 500.00 (max(0.6,0.5)=0.6), 500.50 (max(0.3,0.4)=0.4),
    # 501.00 (0.05, only member 1), 501.50 (0.1, only member 2)
    result = dict_add_merged_confounder_pattern(iso_group_df, mz_bin_digits=2)
    m1 = result.loc[result["mz_rank"] == 1].iloc[0]
    merged = dict(zip(np.round(m1["GroupIsoMZ"], 2), m1["GroupIsoAbundance"]))
    raw_max = {500.00: 0.6, 500.50: 0.4, 501.00: 0.05, 501.50: 0.1}
    expected_total = sum(raw_max.values())
    for mz, raw in raw_max.items():
        assert merged[mz] == pytest.approx(raw / expected_total)


def test_merged_pattern_sums_to_one(iso_group_df):
    result = dict_add_merged_confounder_pattern(iso_group_df, mz_bin_digits=2)
    m1 = result.loc[result["mz_rank"] == 1].iloc[0]
    assert m1["GroupIsoAbundance"].sum() == pytest.approx(1.0)


def test_merged_pattern_rt_window_is_union():
    df = _make_df(
        [
            {
                "mz_bin": 500.00,
                "RT_search_left": 1.0,
                "RT_search_right": 2.0,
                "IM_search_left": 0.5,
                "IM_search_right": 0.7,
                "Charge": 2,
                "IsoMZ": np.array([500.00]),
                "IsoAbundance": np.array([1.0]),
            },
            {
                "mz_bin": 500.00,
                "RT_search_left": 1.5,
                "RT_search_right": 2.5,
                "IM_search_left": 0.5,
                "IM_search_right": 0.7,
                "Charge": 2,
                "IsoMZ": np.array([500.00]),
                "IsoAbundance": np.array([1.0]),
            },
        ]
    )
    grouped = dict_add_confounder_group_id(
        dict_add_confounding_groups(df, mz_bin_digits=2), group_id_offset=1000
    )
    result = dict_add_merged_confounder_pattern(grouped, mz_bin_digits=2)
    assert (result["GroupRT_search_left"] == 1.0).all()
    assert (result["GroupRT_search_right"] == 2.5).all()


def test_merged_pattern_im_window_is_union():
    df = _make_df(
        [
            {
                "mz_bin": 500.00,
                "RT_search_left": 1.0,
                "RT_search_right": 2.0,
                "IM_search_left": 0.5,
                "IM_search_right": 0.7,
                "Charge": 2,
                "IsoMZ": np.array([500.00]),
                "IsoAbundance": np.array([1.0]),
            },
            {
                "mz_bin": 500.00,
                "RT_search_left": 1.0,
                "RT_search_right": 2.0,
                "IM_search_left": 0.6,
                "IM_search_right": 0.8,
                "Charge": 2,
                "IsoMZ": np.array([500.00]),
                "IsoAbundance": np.array([1.0]),
            },
        ]
    )
    grouped = dict_add_confounder_group_id(
        dict_add_confounding_groups(df, mz_bin_digits=2), group_id_offset=1000
    )
    result = dict_add_merged_confounder_pattern(grouped, mz_bin_digits=2)
    assert (result["GroupIM_search_left"] == 0.5).all()
    assert (result["GroupIM_search_right"] == 0.8).all()
    assert result["GroupIM_search_center"].tolist() == pytest.approx([0.65, 0.65])


def test_merged_pattern_does_not_require_mz_length_column(iso_group_df):
    assert "mz_length" not in iso_group_df.columns
    result = dict_add_merged_confounder_pattern(iso_group_df, mz_bin_digits=2)
    assert result["GroupMzLength"].notna().all()
