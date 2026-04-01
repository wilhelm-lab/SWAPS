"""
Unit tests for postprocessing.helper.build_pivot.

build_pivot takes a long-form pp_all DataFrame and dict_ref, and returns a
wide pivot with one row per mz_rank and columns for each run's Match Type
and Intensity.
"""

import numpy as np
import pandas as pd
import pytest

from postprocessing.helper import build_pivot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pp_all(mz_ranks, runs, match_types, intensities):
    return pd.DataFrame(
        {
            "mz_rank": mz_ranks,
            "Run_name": runs,
            "Match Type": match_types,
            "intensity_sum": intensities,
        }
    )


def _make_dict_ref(mz_ranks):
    return pd.DataFrame({"mz_rank": list(mz_ranks)})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBuildPivot:
    def test_returns_dataframe(self):
        pp = _make_pp_all([0, 1], ["run_A", "run_A"], ["MS/MS Ref", "MBR"], [1000.0, 500.0])
        dr = _make_dict_ref([0, 1])
        result = build_pivot(pp, dr)
        assert isinstance(result, pd.DataFrame)

    def test_index_is_mz_rank(self):
        pp = _make_pp_all([0, 1], ["run_A", "run_A"], ["MS/MS Ref", "MBR"], [1000.0, 500.0])
        dr = _make_dict_ref([0, 1])
        result = build_pivot(pp, dr)
        assert result.index.name == "mz_rank" or set(result.index) == {0, 1}

    def test_intensity_columns_present(self):
        pp = _make_pp_all([0, 1], ["run_A", "run_B"], ["MS/MS Ref", "MBR"], [1000.0, 500.0])
        dr = _make_dict_ref([0, 1])
        result = build_pivot(pp, dr)
        intensity_cols = [c for c in result.columns if "Intensity" in c]
        assert len(intensity_cols) > 0

    def test_match_type_columns_present(self):
        pp = _make_pp_all([0, 1], ["run_A", "run_B"], ["MS/MS Ref", "MBR"], [1000.0, 500.0])
        dr = _make_dict_ref([0, 1])
        result = build_pivot(pp, dr)
        match_cols = [c for c in result.columns if "Match Type" in c]
        assert len(match_cols) > 0

    def test_all_dict_ref_ranks_in_output(self):
        """All mz_ranks from dict_ref should appear as rows, even if no match."""
        pp = _make_pp_all([0], ["run_A"], ["MS/MS Ref"], [1000.0])
        dr = _make_dict_ref([0, 1, 2])  # ranks 1 and 2 have no match
        result = build_pivot(pp, dr)
        assert set(result.index) >= {0, 1, 2}

    def test_unmatched_rows_have_unmatched_match_type(self):
        pp = _make_pp_all([0], ["run_A"], ["MS/MS Ref"], [1000.0])
        dr = _make_dict_ref([0, 1])
        result = build_pivot(pp, dr)
        match_cols = [c for c in result.columns if "Match Type" in c]
        # mz_rank=1 has no entry in pp_all → should be "unmatched"
        assert result.loc[1, match_cols[0]] == "unmatched"

    def test_two_runs_produce_separate_columns(self):
        pp = _make_pp_all(
            [0, 0],
            ["run_A", "run_B"],
            ["MS/MS Ref", "MBR"],
            [1000.0, 800.0],
        )
        dr = _make_dict_ref([0])
        result = build_pivot(pp, dr)
        run_a_int = [c for c in result.columns if "run_A" in c and "Intensity" in c]
        run_b_int = [c for c in result.columns if "run_B" in c and "Intensity" in c]
        assert len(run_a_int) == 1
        assert len(run_b_int) == 1

    def test_intensity_value_correct(self):
        pp = _make_pp_all([0], ["run_A"], ["MS/MS Ref"], [12345.0])
        dr = _make_dict_ref([0])
        result = build_pivot(pp, dr)
        intensity_cols = [c for c in result.columns if "run_A" in c and "Intensity" in c]
        assert result.loc[0, intensity_cols[0]] == 12345.0

    def test_match_type_value_correct(self):
        pp = _make_pp_all([0], ["run_A"], ["MBR"], [500.0])
        dr = _make_dict_ref([0])
        result = build_pivot(pp, dr)
        match_cols = [c for c in result.columns if "run_A" in c and "Match Type" in c]
        assert result.loc[0, match_cols[0]] == "MBR"

    def test_empty_pp_all_raises_or_returns_empty(self):
        # build_pivot requires at least one row to infer run columns from
        # pivot_table — empty input produces no columns, which is acceptable
        pp = _make_pp_all([], [], [], [])
        dr = _make_dict_ref([0, 1, 2])
        try:
            result = build_pivot(pp, dr)
            # If it returns, the result should at least have the right index length
            assert len(result) == 3
        except (KeyError, ValueError):
            pass  # expected when pivot_table has nothing to work with
