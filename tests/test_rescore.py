"""
Unit tests for postprocessing.rescore:
  - normalize_shift_by_runs
  - combine_matches_target_decoy
"""

import numpy as np
import pandas as pd
import pytest

from postprocessing.rescore import combine_matches_target_decoy, normalize_shift_by_runs


# ---------------------------------------------------------------------------
# normalize_shift_by_runs
# ---------------------------------------------------------------------------


class TestNormalizeShiftByRuns:
    def test_returns_two_dataframes(self, minimal_matches_target, minimal_matches_decoy):
        t_norm, d_norm = normalize_shift_by_runs(
            minimal_matches_target, minimal_matches_decoy
        )
        assert isinstance(t_norm, pd.DataFrame)
        assert isinstance(d_norm, pd.DataFrame)

    def test_does_not_mutate_inputs(self, minimal_matches_target, minimal_matches_decoy):
        t_copy = minimal_matches_target.copy()
        d_copy = minimal_matches_decoy.copy()
        normalize_shift_by_runs(minimal_matches_target, minimal_matches_decoy)
        pd.testing.assert_frame_equal(minimal_matches_target, t_copy)
        pd.testing.assert_frame_equal(minimal_matches_decoy, d_copy)

    def test_scaled_columns_added_to_target(self, minimal_matches_target, minimal_matches_decoy):
        t_norm, _ = normalize_shift_by_runs(minimal_matches_target, minimal_matches_decoy)
        for col in ["rt_shift_scaled", "rt_shift_abs_scaled", "im_shift_scaled", "im_shift_abs_scaled"]:
            assert col in t_norm.columns, f"Missing {col} in normalized target"

    def test_scaled_columns_added_to_decoy(self, minimal_matches_target, minimal_matches_decoy):
        _, d_norm = normalize_shift_by_runs(minimal_matches_target, minimal_matches_decoy)
        for col in ["rt_shift_scaled", "rt_shift_abs_scaled", "im_shift_scaled", "im_shift_abs_scaled"]:
            assert col in d_norm.columns, f"Missing {col} in normalized decoy"

    def test_abs_scaled_is_nonnegative(self, minimal_matches_target, minimal_matches_decoy):
        t_norm, d_norm = normalize_shift_by_runs(minimal_matches_target, minimal_matches_decoy)
        assert (t_norm["rt_shift_abs_scaled"] >= 0).all()
        assert (t_norm["im_shift_abs_scaled"] >= 0).all()
        assert (d_norm["rt_shift_abs_scaled"] >= 0).all()
        assert (d_norm["im_shift_abs_scaled"] >= 0).all()

    def test_target_scaled_approximately_standardised(self, minimal_matches_target, minimal_matches_decoy):
        """Within each run-pair, target scaled values should have ~zero mean."""
        t_norm, _ = normalize_shift_by_runs(minimal_matches_target, minimal_matches_decoy)
        for (ref, matched), grp in t_norm.groupby(["reference_run", "matched_run"]):
            mean_scaled = grp["rt_shift_scaled"].mean()
            assert abs(mean_scaled) < 0.5, (
                f"run-pair ({ref},{matched}) scaled rt_shift mean {mean_scaled:.3f} too far from 0"
            )

    def test_row_count_preserved(self, minimal_matches_target, minimal_matches_decoy):
        t_norm, d_norm = normalize_shift_by_runs(minimal_matches_target, minimal_matches_decoy)
        assert len(t_norm) == len(minimal_matches_target)
        assert len(d_norm) == len(minimal_matches_decoy)

    def test_original_shift_columns_still_present(self, minimal_matches_target, minimal_matches_decoy):
        t_norm, _ = normalize_shift_by_runs(minimal_matches_target, minimal_matches_decoy)
        assert "rt_shift" in t_norm.columns
        assert "im_shift" in t_norm.columns

    def test_custom_cols_to_scale(self, minimal_matches_target, minimal_matches_decoy):
        t_norm, d_norm = normalize_shift_by_runs(
            minimal_matches_target, minimal_matches_decoy, cols_to_scale=["rt_shift"]
        )
        assert "rt_shift_scaled" in t_norm.columns
        assert "im_shift_scaled" not in t_norm.columns


# ---------------------------------------------------------------------------
# combine_matches_target_decoy
# ---------------------------------------------------------------------------


class TestCombineMatchesTargetDecoy:
    def test_returns_dataframe(self, minimal_matches_target, minimal_matches_decoy, minimal_dict_ref):
        result = combine_matches_target_decoy(
            minimal_matches_target, minimal_matches_decoy, minimal_dict_ref
        )
        assert isinstance(result, pd.DataFrame)

    def test_row_count_is_sum_of_inputs(self, minimal_matches_target, minimal_matches_decoy, minimal_dict_ref):
        result = combine_matches_target_decoy(
            minimal_matches_target, minimal_matches_decoy, minimal_dict_ref
        )
        assert len(result) == len(minimal_matches_target) + len(minimal_matches_decoy)

    def test_label_column_matches_is_target(self, minimal_matches_target, minimal_matches_decoy, minimal_dict_ref):
        result = combine_matches_target_decoy(
            minimal_matches_target, minimal_matches_decoy, minimal_dict_ref
        )
        assert "label" in result.columns
        assert "Decoy" in result.columns
        # label == True  ↔  Decoy == False
        pd.testing.assert_series_equal(
            result["label"],
            ~result["Decoy"],
            check_names=False,
        )

    def test_sequence_with_runs_column_exists(self, minimal_matches_target, minimal_matches_decoy, minimal_dict_ref):
        result = combine_matches_target_decoy(
            minimal_matches_target, minimal_matches_decoy, minimal_dict_ref
        )
        assert "Sequence_with_runs" in result.columns

    def test_sequence_with_runs_format(self, minimal_matches_target, minimal_matches_decoy, minimal_dict_ref):
        matches_target = minimal_matches_target.copy()
        matches_target["matched_run"] = "run_X"
        result = combine_matches_target_decoy(
            matches_target, minimal_matches_decoy, minimal_dict_ref
        )
        target_rows = result[result["label"] == True]
        assert (target_rows["Sequence_with_runs"].str.contains("_run_X")).all()

    def test_sequence_and_proteins_joined_from_dict_ref(
        self, minimal_matches_target, minimal_matches_decoy, minimal_dict_ref
    ):
        result = combine_matches_target_decoy(
            minimal_matches_target, minimal_matches_decoy, minimal_dict_ref
        )
        assert "Sequence" in result.columns
        assert "Proteins" in result.columns

    def test_decoy_mz_rank_set_to_minus_one_in_target(
        self, minimal_matches_target, minimal_matches_decoy, minimal_dict_ref
    ):
        result = combine_matches_target_decoy(
            minimal_matches_target, minimal_matches_decoy, minimal_dict_ref
        )
        target_rows = result[result["label"] == True]
        assert (target_rows["decoy_mz_rank"] == -1).all()
