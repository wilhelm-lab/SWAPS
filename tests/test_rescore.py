"""
Unit tests for postprocessing.rescore:
  - normalize_shift_by_runs
  - combine_matches_target_decoy
  - brew_with_percolator
  - select_trusted_training_rows
  - brew_trusted_target_model
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
    select_trusted_training_rows,
    brew_trusted_target_model,
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


# ---------------------------------------------------------------------------
# select_trusted_training_rows / brew_trusted_target_model
# ---------------------------------------------------------------------------


def _make_tdc_rows(n, is_target, feature_mean, seed):
    """tdc_df-shaped rows: mz_rank/matched_run/IsTarget/Decoy/Sequence/
    Proteins/feature_instance_id/sequence_variant/Sequence_with_runs/feat1/
    feat2, mirroring combine_matches_target_decoy's output schema."""
    rng = np.random.default_rng(seed)
    variant = "target_0" if is_target else "decoy_0"
    return pd.DataFrame(
        {
            "matched_run": ["run_A"] * n,
            "IsTarget": is_target,
            "Decoy": not is_target,
            "Sequence": [f"PEP{seed}_{i}K" for i in range(n)],
            "Proteins": [f"PROT{seed}_{i}" for i in range(n)],
            "feature_instance_id": [f"{seed}_{i}" for i in range(n)],
            "sequence_variant": variant,
            "Sequence_with_runs": [
                f"PEP{seed}_{i}K_run_A_{seed}_{i}_{variant}" for i in range(n)
            ],
            "feat1": rng.normal(feature_mean, 1.0, n),
            "feat2": rng.normal(feature_mean, 1.0, n),
        }
    )


@pytest.fixture
def trusted_training_tdc_df():
    """40 MS/MS-confirmed targets (well-separated), 60 Not_Match targets
    (indistinguishable from decoys), 80 decoys — all in run_A, unique
    mz_rank per row."""
    msms_targets = _make_tdc_rows(40, True, 2.0, seed=1)
    msms_targets["mz_rank"] = np.arange(1000, 1040)
    notmatch_targets = _make_tdc_rows(60, True, 0.0, seed=2)
    notmatch_targets["mz_rank"] = np.arange(2000, 2060)
    decoys = _make_tdc_rows(80, False, 0.0, seed=3)
    decoys["mz_rank"] = np.arange(3000, 3080)
    return pd.concat([msms_targets, notmatch_targets, decoys], ignore_index=True)


@pytest.fixture
def trusted_training_dict_ref(trusted_training_tdc_df):
    """dict_ref labelling mz_rank 1000-1039 Reference, everything else Not_Match."""
    mz_ranks = trusted_training_tdc_df["mz_rank"].unique()
    status = np.where((mz_ranks >= 1000) & (mz_ranks < 1040), "Reference", "Not_Match")
    return pd.DataFrame({"mz_rank": mz_ranks, "run_A": status})


class TestSelectTrustedTrainingRows:
    def test_only_msms_targets_selected(
        self, trusted_training_tdc_df, trusted_training_dict_ref
    ):
        train_df = select_trusted_training_rows(
            trusted_training_tdc_df, trusted_training_dict_ref, rng=0
        )
        target_rows = train_df.loc[train_df["IsTarget"]]
        assert set(target_rows["mz_rank"]) == set(range(1000, 1040))

    def test_decoy_count_matches_ratio(
        self, trusted_training_tdc_df, trusted_training_dict_ref
    ):
        train_df = select_trusted_training_rows(
            trusted_training_tdc_df,
            trusted_training_dict_ref,
            decoy_target_ratio=0.5,
            rng=0,
        )
        n_targets = train_df["IsTarget"].sum()
        n_decoys = (~train_df["IsTarget"]).sum()
        assert n_targets == 40
        assert n_decoys == 20

    def test_falls_back_to_all_decoys_when_pool_too_small(
        self, trusted_training_tdc_df, trusted_training_dict_ref
    ):
        train_df = select_trusted_training_rows(
            trusted_training_tdc_df,
            trusted_training_dict_ref,
            decoy_target_ratio=10.0,  # would need 400 decoys, only 80 exist
            rng=0,
        )
        assert (~train_df["IsTarget"]).sum() == 80

    def test_raises_without_msms_targets(self, trusted_training_tdc_df):
        dict_ref_no_msms = pd.DataFrame(
            {
                "mz_rank": trusted_training_tdc_df["mz_rank"].unique(),
                "run_A": "Not_Match",
            }
        )
        with pytest.raises(ValueError):
            select_trusted_training_rows(
                trusted_training_tdc_df, dict_ref_no_msms, rng=0
            )

    def test_seed_is_reproducible(
        self, trusted_training_tdc_df, trusted_training_dict_ref
    ):
        train_df_1 = select_trusted_training_rows(
            trusted_training_tdc_df, trusted_training_dict_ref, rng=42
        )
        train_df_2 = select_trusted_training_rows(
            trusted_training_tdc_df, trusted_training_dict_ref, rng=42
        )
        assert set(train_df_1["mz_rank"]) == set(train_df_2["mz_rank"])


class TestBrewTrustedTargetModel:
    @pytest.mark.parametrize("model_type", ["percolator", "supervised"])
    def test_scores_full_population(
        self, trusted_training_tdc_df, trusted_training_dict_ref, model_type, tmp_path
    ):
        train_df = select_trusted_training_rows(
            trusted_training_tdc_df, trusted_training_dict_ref, rng=0
        )
        targets_scored, full_scored, model = brew_trusted_target_model(
            train_df,
            trusted_training_tdc_df,
            feature_cols=["feat1", "feat2"],
            model_type=model_type,
            train_fdr=0.1,
            work_dir=str(tmp_path / model_type),
        )
        assert len(full_scored) == len(trusted_training_tdc_df)
        assert {"score", "q-value"}.issubset(full_scored.columns)
        assert targets_scored["label"].all()
        assert set(targets_scored["mz_rank"]) == set(
            trusted_training_tdc_df.loc[
                trusted_training_tdc_df["IsTarget"], "mz_rank"
            ]
        )

    @pytest.mark.parametrize("model_type", ["percolator", "supervised"])
    def test_msms_targets_score_above_decoys(
        self, trusted_training_tdc_df, trusted_training_dict_ref, model_type, tmp_path
    ):
        """Sanity check the model learned something: the well-separated
        MS/MS-confirmed targets should score higher on average than decoys."""
        train_df = select_trusted_training_rows(
            trusted_training_tdc_df, trusted_training_dict_ref, rng=0
        )
        _, full_scored, _ = brew_trusted_target_model(
            train_df,
            trusted_training_tdc_df,
            feature_cols=["feat1", "feat2"],
            model_type=model_type,
            train_fdr=0.1,
            work_dir=str(tmp_path / model_type),
        )
        msms_scores = full_scored.loc[
            full_scored["scannr"].isin(range(1000, 1040)), "score"
        ]
        decoy_scores = full_scored.loc[~full_scored["label"], "score"]
        assert msms_scores.mean() > decoy_scores.mean()

    @pytest.mark.parametrize("model_type", ["percolator", "supervised"])
    def test_weights_file_written(
        self, trusted_training_tdc_df, trusted_training_dict_ref, model_type, tmp_path
    ):
        train_df = select_trusted_training_rows(
            trusted_training_tdc_df, trusted_training_dict_ref, rng=0
        )
        work_dir = tmp_path / model_type
        brew_trusted_target_model(
            train_df,
            trusted_training_tdc_df,
            feature_cols=["feat1", "feat2"],
            model_type=model_type,
            train_fdr=0.1,
            work_dir=str(work_dir),
        )
        assert (work_dir / f"mokapot_trusted_{model_type}_weights.txt").exists()

    @pytest.mark.parametrize("model_type", ["percolator", "supervised"])
    def test_psms_tsv_is_percolator_compatible(
        self, trusted_training_tdc_df, trusted_training_dict_ref, model_type, tmp_path
    ):
        """mokapot_trusted_<model_type>_psms.tsv must carry the same columns
        FDR_benchmark_with_HeLa_HYE.build_filtered_combined_ions reads from
        percolator_psms.tsv (PSMId/score/q-value/filename), and PSMId's
        leading "<mz_rank>_" must round-trip back to the real mz_rank."""
        train_df = select_trusted_training_rows(
            trusted_training_tdc_df, trusted_training_dict_ref, rng=0
        )
        work_dir = tmp_path / model_type
        targets_scored, _, _ = brew_trusted_target_model(
            train_df,
            trusted_training_tdc_df,
            feature_cols=["feat1", "feat2"],
            model_type=model_type,
            train_fdr=0.1,
            work_dir=str(work_dir),
        )
        psms_path = work_dir / f"mokapot_trusted_{model_type}_psms.tsv"
        assert psms_path.exists()
        psms_tsv = pd.read_csv(psms_path, sep="\t")
        assert {"PSMId", "score", "q-value", "peptide", "proteinIds", "filename"}.issubset(
            psms_tsv.columns
        )
        assert len(psms_tsv) == len(targets_scored)
        recovered_mz_rank = psms_tsv["PSMId"].str.split("_").str[0].astype(int)
        assert set(recovered_mz_rank) == set(targets_scored["mz_rank"])

    def test_invalid_model_type_raises(
        self, trusted_training_tdc_df, trusted_training_dict_ref, tmp_path
    ):
        train_df = select_trusted_training_rows(
            trusted_training_tdc_df, trusted_training_dict_ref, rng=0
        )
        with pytest.raises(ValueError):
            brew_trusted_target_model(
                train_df,
                trusted_training_tdc_df,
                feature_cols=["feat1", "feat2"],
                model_type="not_a_real_model_type",
                work_dir=str(tmp_path),
            )


# ---------------------------------------------------------------------------
# brew_trusted_target_model -- target-decoy competition (paired target/decoy
# sharing an (mz_rank, matched_run) slot)
# ---------------------------------------------------------------------------


@pytest.fixture
def tdc_df_with_paired_slots():
    """Trusted-training pool (40 well-separated MS/MS targets + matching
    decoys, disjoint mz_ranks -- same shape as trusted_training_tdc_df) plus
    two special Not_Match slots where a target and its own decoy share
    (mz_rank, matched_run):
      - mz_rank 500: decoy outscores its target -> target must be dropped.
      - mz_rank 600: target outscores its decoy -> target must survive.
    """
    msms_targets = _make_tdc_rows(40, True, 2.0, seed=1)
    msms_targets["mz_rank"] = np.arange(1000, 1040)
    decoys = _make_tdc_rows(40, False, 0.0, seed=3)
    decoys["mz_rank"] = np.arange(3000, 3040)

    def _paired_row(mz_rank, is_target, feat_val, seed):
        row = _make_tdc_rows(1, is_target, 0.0, seed=seed)
        row["mz_rank"] = mz_rank
        row["feat1"] = feat_val
        row["feat2"] = feat_val
        return row

    decoy_wins_target = _paired_row(500, True, -5.0, seed=10)
    decoy_wins_decoy = _paired_row(500, False, 5.0, seed=11)
    target_wins_target = _paired_row(600, True, 5.0, seed=12)
    target_wins_decoy = _paired_row(600, False, -5.0, seed=13)

    return pd.concat(
        [
            msms_targets,
            decoys,
            decoy_wins_target,
            decoy_wins_decoy,
            target_wins_target,
            target_wins_decoy,
        ],
        ignore_index=True,
    )


@pytest.fixture
def dict_ref_with_paired_slots(tdc_df_with_paired_slots):
    mz_ranks = tdc_df_with_paired_slots["mz_rank"].unique()
    status = np.where((mz_ranks >= 1000) & (mz_ranks < 1040), "Reference", "Not_Match")
    return pd.DataFrame({"mz_rank": mz_ranks, "run_A": status})


class TestBrewTrustedTargetModelCompetition:
    @pytest.mark.parametrize("model_type", ["percolator", "supervised"])
    def test_outscored_target_is_dropped_outscoring_target_survives(
        self, tdc_df_with_paired_slots, dict_ref_with_paired_slots, model_type, tmp_path
    ):
        train_df = select_trusted_training_rows(
            tdc_df_with_paired_slots, dict_ref_with_paired_slots, rng=0
        )
        targets_scored, full_scored, _ = brew_trusted_target_model(
            train_df,
            tdc_df_with_paired_slots,
            feature_cols=["feat1", "feat2"],
            model_type=model_type,
            train_fdr=0.1,
            work_dir=str(tmp_path / model_type),
            rng=0,
        )
        # mz_rank 500: decoy outscored its target -> target dropped from
        # targets_scored_df, and full_scored_df keeps only the decoy row.
        assert 500 not in set(targets_scored["mz_rank"])
        slot_500 = full_scored.loc[full_scored["scannr"] == 500]
        assert len(slot_500) == 1
        assert not slot_500["label"].iloc[0]

        # mz_rank 600: target outscored its decoy -> target survives, and
        # full_scored_df keeps only the target row.
        assert 600 in set(targets_scored["mz_rank"])
        slot_600 = full_scored.loc[full_scored["scannr"] == 600]
        assert len(slot_600) == 1
        assert slot_600["label"].iloc[0]

    @pytest.mark.parametrize("model_type", ["percolator", "supervised"])
    def test_full_scored_df_has_one_row_per_slot(
        self, tdc_df_with_paired_slots, dict_ref_with_paired_slots, model_type, tmp_path
    ):
        train_df = select_trusted_training_rows(
            tdc_df_with_paired_slots, dict_ref_with_paired_slots, rng=0
        )
        _, full_scored, _ = brew_trusted_target_model(
            train_df,
            tdc_df_with_paired_slots,
            feature_cols=["feat1", "feat2"],
            model_type=model_type,
            train_fdr=0.1,
            work_dir=str(tmp_path / model_type),
            rng=0,
        )
        n_slots = tdc_df_with_paired_slots.drop_duplicates(
            ["mz_rank", "matched_run"]
        ).shape[0]
        assert len(full_scored) == n_slots
