"""Unit tests for postprocessing.sage_baseline (Sage lfq.tsv -> combined_ion table) and the
Sage baseline redirect in postprocessing.mq_baseline. Synthetic tables only."""

import pandas as pd
import pytest

from postprocessing.mq_baseline import (
    SAGE_BASELINE_ROOT_ENV,
    _resolve_search_output_file,
    baseline_label,
)
from postprocessing.sage_baseline import lfq_q_cutoff_grid, sage_to_combined_ion

RUNS = ["runA", "runB"]


def _write_lfq(path, rows):
    """rows: (peptide, charge, proteins, q_value, intensity_A, intensity_B)"""
    df = pd.DataFrame(rows, columns=["peptide", "charge", "proteins", "q_value", "runA.d", "runB.d"])
    df.insert(4, "score", 1.0)
    df.insert(5, "spectral_angle", 0.9)
    df.to_csv(path / "lfq.tsv", sep="\t", index=False)
    return path / "results.sage.tsv"  # only its dir is used when dict_ref is given


def _dict_ref():
    """PEPTIDEK/2 identified in runA, PEPTIDEK/3 in runB; OTHERK/2 in neither."""
    return pd.DataFrame(
        {
            "Modified sequence": ["PEPTIDEK", "PEPTIDEK", "OTHERK"],
            "Charge": [2, 3, 2],
            "Proteins": ["sp|P1|A_HUMAN"] * 3,
            "runA": ["Reference", "Match", "Match"],
            "runB": ["Match", "Quant_Only", "Match"],
        }
    )


def test_per_charge_lfq_gives_one_row_per_ion_with_ion_level_identifications(tmp_path):
    results = _write_lfq(
        tmp_path,
        [
            ("PEPTIDEK", 2, "sp|P1|A_HUMAN", 0.01, 10.0, 0.0),
            ("PEPTIDEK", 3, "sp|P1|A_HUMAN", 0.02, 0.0, 5.0),
            ("OTHERK", 2, "sp|P1|A_HUMAN", 0.03, 7.0, 7.0),
        ],
    )
    out = sage_to_combined_ion(results, dict_ref=_dict_ref()).set_index(["Modified Sequence", "Charge"])

    assert set(out.index) == {("PEPTIDEK", 2), ("PEPTIDEK", 3), ("OTHERK", 2)}
    # identification is per charge state, not "any charge state of the peptide"
    assert out.loc[("PEPTIDEK", 2), "runA Match Type"] == "MS/MS"
    assert out.loc[("PEPTIDEK", 2), "runB Match Type"] == "unmatched"
    assert out.loc[("PEPTIDEK", 3), "runA Match Type"] == "unmatched"
    assert out.loc[("PEPTIDEK", 3), "runB Match Type"] == "MS/MS"
    # quantified but never identified -> MBR
    assert (out.loc[("OTHERK", 2), ["runA Match Type", "runB Match Type"]] == "MBR").all()
    assert out.loc[("OTHERK", 2), "runA Intensity"] == 7.0
    assert out["In LFQ"].all()


def test_peptide_level_lfq_still_combines_charge_states(tmp_path):
    results = _write_lfq(tmp_path, [("PEPTIDEK", -1, "sp|P1|A_HUMAN", 0.01, 10.0, 5.0)])
    out = sage_to_combined_ion(results, dict_ref=_dict_ref())

    assert len(out) == 2  # PEPTIDEK (charge -1) + OTHERK (identified only, no lfq row)
    row = out.set_index("Modified Sequence").loc["PEPTIDEK"]
    assert row["Charge"] == -1
    # identified in a run if any charge state is
    assert row["runA Match Type"] == "MS/MS" and row["runB Match Type"] == "MS/MS"
    assert row["In LFQ"] and not out.set_index("Modified Sequence").loc["OTHERK", "In LFQ"]


def test_table_covers_exactly_the_dict_ref_ions(tmp_path):
    results = _write_lfq(
        tmp_path,
        [
            ("PEPTIDEK", 2, "sp|P1|A_HUMAN", 0.01, 10.0, 4.0),
            ("STRAYK", 2, "sp|P9|Z_HUMAN", 0.01, 3.0, 3.0),  # quantified, but not in the dictionary
        ],
    )
    out = sage_to_combined_ion(results, dict_ref=_dict_ref()).set_index(["Modified Sequence", "Charge"])

    assert set(out.index) == {("PEPTIDEK", 2), ("PEPTIDEK", 3), ("OTHERK", 2)}  # STRAYK discarded
    assert bool(out.loc[("PEPTIDEK", 2), "In LFQ"])
    # dictionary ions with no lfq.tsv row: kept, flagged, all-zero intensities
    not_in_lfq = out.loc[[("PEPTIDEK", 3), ("OTHERK", 2)]]
    assert not not_in_lfq["In LFQ"].any()
    assert (not_in_lfq[["runA Intensity", "runB Intensity"]] == 0).all().all()


def test_lfq_q_cutoff_zeroes_only_unidentified_quantifications(tmp_path):
    results = _write_lfq(tmp_path, [("PEPTIDEK", 2, "sp|P1|A_HUMAN", 0.5, 10.0, 6.0)])
    out = sage_to_combined_ion(results, dict_ref=_dict_ref(), lfq_q_cutoff=0.1)
    row = out.set_index(["Modified Sequence", "Charge"]).loc[("PEPTIDEK", 2)]

    assert row["runA Intensity"] == 10.0  # identified: kept regardless of the lfq q-value
    assert row["runB Intensity"] == 0.0  # MBR-only: gated out


class TestSageBaselineRoot:
    def test_redirects_sage_results_under_the_root(self, monkeypatch, tmp_path):
        monkeypatch.setenv(SAGE_BASELINE_ROOT_ENV, str(tmp_path / "new_search"))
        p = _resolve_search_output_file("/data/old_search/HT_HeLa/results.sage.tsv")
        assert p == tmp_path / "new_search" / "HT_HeLa" / "results.sage.tsv"
        assert baseline_label("/data/old_search/HT_HeLa/results.sage.tsv") == "Sage"

    def test_other_baselines_are_left_alone(self, monkeypatch, tmp_path):
        monkeypatch.setenv(SAGE_BASELINE_ROOT_ENV, str(tmp_path))
        assert str(_resolve_search_output_file("/data/mq/txt/evidence.txt")) == "/data/mq/txt/evidence.txt"

    def test_unset_env_keeps_the_recorded_path(self, monkeypatch):
        monkeypatch.delenv(SAGE_BASELINE_ROOT_ENV, raising=False)
        assert str(_resolve_search_output_file("/data/old/X/results.sage.tsv")) == "/data/old/X/results.sage.tsv"


class TestLfqQCutoffGrid:
    def _grid(self, tmp_path, qs, decoy_q=None):
        rows = [(f"PEP{i}K", 2, "sp|P1|A_HUMAN", q, 1.0, 1.0) for i, q in enumerate(qs)]
        if decoy_q is not None:
            rows.append(("DECOYK", 2, "rev_sp|P1|A_HUMAN", decoy_q, 1.0, 1.0))
        results = _write_lfq(tmp_path, rows)
        return lfq_q_cutoff_grid(results)

    def test_spans_the_table_range_in_0p01_steps(self, tmp_path):
        assert self._grid(tmp_path, [0.0028, 0.02, 0.0343]).tolist() == [0.01, 0.02, 0.03, 0.04]

    def test_starts_above_the_minimum_when_nothing_passes_at_0p01(self, tmp_path):
        assert self._grid(tmp_path, [0.0135, 0.031, 0.0442]).tolist() == [0.02, 0.03, 0.04, 0.05]

    def test_exact_multiples_are_not_bumped_to_the_next_cutoff(self, tmp_path):
        assert self._grid(tmp_path, [0.01, 0.03]).tolist() == [0.01, 0.02, 0.03]

    def test_decoy_rows_do_not_widen_the_range(self, tmp_path):
        assert self._grid(tmp_path, [0.005, 0.015], decoy_q=0.5).tolist() == [0.01, 0.02]
