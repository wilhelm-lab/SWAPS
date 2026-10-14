"""plot_match_type_comparison's "Not in LFQ" category (baseline tables with an ``In LFQ`` column)."""

import matplotlib

matplotlib.use("Agg")

import pandas as pd

from utils.plot import plot_match_type_comparison


def _table(match_types, intensities, in_lfq=None):
    df = pd.DataFrame({"run10001 Match Type": match_types, "run10001 Intensity": intensities})
    if in_lfq is not None:
        df["In LFQ"] = in_lfq
    return df


def _heights(ax, label):
    """Bar heights of the first (= baseline, drawn before SWAPS) container labelled *label*."""
    for container in ax.containers:
        if container.get_label() == label:
            return [p.get_height() for p in container]
    return None


def test_ions_without_an_lfq_row_are_their_own_category(tmp_path):
    swaps = _table(["MS/MS"] * 4, [5.0] * 4)
    baseline = _table(
        ["MS/MS", "MS/MS", "MBR", "unmatched"], [5.0, 0.0, 3.0, 0.0], in_lfq=[True, False, True, False]
    )
    _, ax = plot_match_type_comparison(swaps, baseline, int_col_keyword="Intensity", fig_dir=str(tmp_path))

    assert _heights(ax, "Not in LFQ") == [2]
    # the two ions without an LFQ row no longer count as Zero Quant / unmatched
    assert _heights(ax, "Zero Quant") is None
    assert _heights(ax, "unmatched") is None
    assert _heights(ax, "MS/MS") == [1]
    assert _heights(ax, "MBR") == [1]


def test_tables_without_the_flag_are_unchanged(tmp_path):
    swaps = _table(["MS/MS"] * 3, [5.0] * 3)
    baseline = _table(["MS/MS", "MS/MS", "unmatched"], [5.0, 0.0, 0.0])
    _, ax = plot_match_type_comparison(swaps, baseline, int_col_keyword="Intensity", fig_dir=str(tmp_path))

    assert _heights(ax, "Not in LFQ") is None
    assert _heights(ax, "Zero Quant") == [1]
    assert _heights(ax, "unmatched") == [1]
