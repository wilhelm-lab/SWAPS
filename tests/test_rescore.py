"""
Unit tests for postprocessing.rescore:
  - normalize_shift_by_runs
  - combine_matches_target_decoy
  - brew_with_percolator
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from postprocessing.rescore import (
    combine_matches_target_decoy,
    normalize_shift_by_runs,
    brew_with_percolator,
    split_pp_by_match_status,
)


# ---------------------------------------------------------------------------
# split_pp_by_match_status
# ---------------------------------------------------------------------------


@pytest.fixture
def dict_ref_with_match_status():
    """dict_ref with two per-run match-type columns and three mz_ranks."""
    return pd.DataFrame(
        {
            "mz_rank": [0, 1, 2],
            "Sequence": ["PEPTIDEK", "SEQR", "ACDEFK"],
            "Proteins": ["PROT0", "PROT1", "PROT2"],
            "run_A": ["Reference", "Not_Match", "Quant_Only"],
            "run_B": ["Not_Match", "Quant_Only", "Not_Match"],
        }
    )


@pytest.fixture
def pp_target_for_split():
    return pd.DataFrame(
        {
            "mz_rank": [0, 0, 1, 1, 2, 2],
            "Run_name": ["run_A", "run_B", "run_A", "run_B", "run_A", "run_B"],
            "feature_instance_id": [f"0_A", "0_B", "1_A", "1_B", "2_A", "2_B"],
            "intensity_sum": [100.0] * 6,
        }
    )


@pytest.fixture
def pp_decoy_for_split(pp_target_for_split):
    df = pp_target_for_split.copy()
    df["intensity_sum"] = 50.0
    return df


class TestSplitPpByMatchStatus:
    def test_returns_three_dataframes(
        self, dict_ref_with_match_status, pp_target_for_split, pp_decoy_for_split
    ):
        result = split_pp_by_match_status(
            dict_ref_with_match_status, pp_target_for_split, pp_decoy_for_split
        )
        assert len(result) == 3
        for df in result:
            assert isinstance(df, pd.DataFrame)

    def test_not_match_entries_in_pp_not_match(
        self, dict_ref_with_match_status, pp_target_for_split, pp_decoy_for_split
    ):
        pp_not_match, _, _ = split_pp_by_match_status(
            dict_ref_with_match_status, pp_target_for_split, pp_decoy_for_split
        )
        # (mz_rank=0, run_B), (mz_rank=1, run_A), (mz_rank=2, run_B) are Not_Match
        keys = set(zip(pp_not_match["mz_rank"], pp_not_match["Run_name"]))
        assert (0, "run_B") in keys
        assert (1, "run_A") in keys
        assert (2, "run_B") in keys

    def test_msms_entries_in_pp_msms(
        self, dict_ref_with_match_status, pp_target_for_split, pp_decoy_for_split
    ):
        _, pp_msms, _ = split_pp_by_match_status(
            dict_ref_with_match_status, pp_target_for_split, pp_decoy_for_split
        )
        # (mz_rank=0, run_A)=Reference, (mz_rank=1, run_B)=Quant_Only, (mz_rank=2, run_A)=Quant_Only
        keys = set(zip(pp_msms["mz_rank"], pp_msms["Run_name"]))
        assert (0, "run_A") in keys
        assert (1, "run_B") in keys
        assert (2, "run_A") in keys

    def test_not_match_and_msms_are_disjoint(
        self, dict_ref_with_match_status, pp_target_for_split, pp_decoy_for_split
    ):
        pp_not_match, pp_msms, _ = split_pp_by_match_status(
            dict_ref_with_match_status, pp_target_for_split, pp_decoy_for_split
        )
        keys_not_match = set(zip(pp_not_match["mz_rank"], pp_not_match["Run_name"]))
        keys_msms = set(zip(pp_msms["mz_rank"], pp_msms["Run_name"]))
        assert keys_not_match.isdisjoint(keys_msms)

    def test_decoy_filtered_to_not_match(
        self, dict_ref_with_match_status, pp_target_for_split, pp_decoy_for_split
    ):
        pp_not_match, _, pp_decoy_not_match = split_pp_by_match_status(
            dict_ref_with_match_status, pp_target_for_split, pp_decoy_for_split
        )
        assert set(zip(pp_decoy_not_match["mz_rank"], pp_decoy_not_match["Run_name"])) == set(
            zip(pp_not_match["mz_rank"], pp_not_match["Run_name"])
        )

    def test_no_file_cols_returns_empty(self, pp_target_for_split, pp_decoy_for_split):
        dict_ref_no_cols = pd.DataFrame(
            {"mz_rank": [0, 1, 2], "Sequence": ["A", "B", "C"], "Proteins": ["P0", "P1", "P2"]}
        )
        pp_not_match, pp_msms, pp_decoy_not_match = split_pp_by_match_status(
            dict_ref_no_cols, pp_target_for_split, pp_decoy_for_split
        )
        assert len(pp_not_match) == 0
        assert len(pp_msms) == 0
        assert len(pp_decoy_not_match) == 0


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


# ---------------------------------------------------------------------------
# brew_with_percolator
# ---------------------------------------------------------------------------

_PERC_PSM_COLS = ["PSMId", "score", "q-value", "posterior_error_prob", "peptide", "proteinIds"]


@pytest.fixture
def percolator_input_df():
    rng = np.random.default_rng(42)
    n = 20
    return pd.DataFrame(
        {
            "mz_rank": [f"rank_{i}" for i in range(n)],
            "Run_name": ["run_A"] * n,
            "decoy": [False] * (n // 2) + [True] * (n // 2),
            "modified_sequence": [f"PEP{i}K" for i in range(n)],
            "proteins": [f"PROT{i}" for i in range(n)],
            "feature1": rng.normal(0, 1, n),
            "feature2": rng.uniform(0, 1, n),
        }
    )


def _make_mock_run(tmp_path):
    """Returns a subprocess.run side_effect that writes fake percolator output files."""
    fake_target = pd.DataFrame(
        {
            "PSMId": ["id0", "id1"],
            "score": [0.9, 0.5],
            "q-value": [0.01, 0.05],
            "posterior_error_prob": [0.01, 0.05],
            "peptide": ["PEP0K", "PEP1K"],
            "proteinIds": ["PROT0", "PROT1"],
        }
    )
    fake_decoy = pd.DataFrame(
        {
            "PSMId": ["id10", "id11"],
            "score": [-0.3, -0.8],
            "q-value": [0.5, 0.9],
            "posterior_error_prob": [0.5, 0.9],
            "peptide": ["PEP10K", "PEP11K"],
            "proteinIds": ["PROT10", "PROT11"],
        }
    )
    def side_effect(cmd, **kwargs):
        psms_path = cmd[cmd.index("-m") + 1]
        decoy_psms_path = cmd[cmd.index("-M") + 1]
        fake_target.to_csv(psms_path, sep="\t", index=False)
        fake_decoy.to_csv(decoy_psms_path, sep="\t", index=False)
        return MagicMock(stdout="", stderr="", returncode=0)

    return side_effect


class TestBrewWithPercolator:
    def test_returns_three_elements(self, percolator_input_df, tmp_path):
        with patch("postprocessing.rescore.subprocess.run", side_effect=_make_mock_run(tmp_path)):
            psms_df, peptides_df, all_psms = brew_with_percolator(
                percolator_input_df,
                feature_cols=["feature1", "feature2"],
                work_dir=str(tmp_path),
            )
        assert isinstance(psms_df, pd.DataFrame)
        assert peptides_df is None  # percolator wrapper returns None for peptides
        assert isinstance(all_psms, pd.DataFrame)

    def test_all_psms_contains_target_and_decoy(self, percolator_input_df, tmp_path):
        with patch("postprocessing.rescore.subprocess.run", side_effect=_make_mock_run(tmp_path)):
            psms_df, _, all_psms = brew_with_percolator(
                percolator_input_df,
                feature_cols=["feature1", "feature2"],
                work_dir=str(tmp_path),
            )
        assert set(all_psms["label"].unique()) == {1, -1}
        assert len(all_psms) == len(psms_df) + len(all_psms[all_psms["label"] == -1])

    def test_percolator_called_with_decoy_flag(self, percolator_input_df, tmp_path):
        captured = {}

        def capturing_run(cmd, **kwargs):
            captured["cmd"] = cmd
            return _make_mock_run(tmp_path)(cmd, **kwargs)

        with patch("postprocessing.rescore.subprocess.run", side_effect=capturing_run):
            brew_with_percolator(
                percolator_input_df,
                feature_cols=["feature1", "feature2"],
                work_dir=str(tmp_path),
            )
        assert "-M" in captured["cmd"]
        assert "-m" in captured["cmd"]

    def test_input_tsv_written(self, percolator_input_df, tmp_path):
        with patch("postprocessing.rescore.subprocess.run", side_effect=_make_mock_run(tmp_path)):
            brew_with_percolator(
                percolator_input_df,
                feature_cols=["feature1", "feature2"],
                work_dir=str(tmp_path),
            )
        assert (tmp_path / "percolator_input.tsv").exists()
