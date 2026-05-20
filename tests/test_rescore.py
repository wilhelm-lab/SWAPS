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
)


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
    fake_peptides = pd.DataFrame(
        {
            "PSMId": ["id0"],
            "score": [0.9],
            "q-value": [0.01],
            "posterior_error_prob": [0.01],
            "peptide": ["PEP0K"],
            "proteinIds": ["PROT0"],
        }
    )

    def side_effect(cmd, **kwargs):
        psms_path = cmd[cmd.index("-m") + 1]
        decoy_psms_path = cmd[cmd.index("-M") + 1]
        peptides_path = cmd[cmd.index("-r") + 1]
        fake_target.to_csv(psms_path, sep="\t", index=False)
        fake_decoy.to_csv(decoy_psms_path, sep="\t", index=False)
        fake_peptides.to_csv(peptides_path, sep="\t", index=False)
        return MagicMock(stdout="", stderr="", returncode=0)

    return side_effect


class TestBrewWithPercolator:
    def test_returns_three_dataframes(self, percolator_input_df, tmp_path):
        with patch("postprocessing.rescore.subprocess.run", side_effect=_make_mock_run(tmp_path)):
            psms_df, peptides_df, all_psms = brew_with_percolator(
                percolator_input_df,
                feature_cols=["feature1", "feature2"],
                work_dir=str(tmp_path),
            )
        assert isinstance(psms_df, pd.DataFrame)
        assert isinstance(peptides_df, pd.DataFrame)
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
        assert "-r" in captured["cmd"]

    def test_train_test_fdr_passed_to_command(self, percolator_input_df, tmp_path):
        captured = {}

        def capturing_run(cmd, **kwargs):
            captured["cmd"] = cmd
            return _make_mock_run(tmp_path)(cmd, **kwargs)

        with patch("postprocessing.rescore.subprocess.run", side_effect=capturing_run):
            brew_with_percolator(
                percolator_input_df,
                train_fdr=0.05,
                test_fdr=0.01,
                feature_cols=["feature1", "feature2"],
                work_dir=str(tmp_path),
            )
        cmd = captured["cmd"]
        assert cmd[cmd.index("-F") + 1] == "0.05"
        assert cmd[cmd.index("-f") + 1] == "0.01"

    def test_input_tsv_written(self, percolator_input_df, tmp_path):
        with patch("postprocessing.rescore.subprocess.run", side_effect=_make_mock_run(tmp_path)):
            brew_with_percolator(
                percolator_input_df,
                feature_cols=["feature1", "feature2"],
                work_dir=str(tmp_path),
            )
        assert (tmp_path / "percolator_input.tsv").exists()
