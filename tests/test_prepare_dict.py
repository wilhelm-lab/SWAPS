import numpy as np
import pandas as pd
import pytest

from swaps.prepare_dict.prepare_dict import (
    calculate_modpept_isopattern,
    dict_add_confounding_groups,
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
