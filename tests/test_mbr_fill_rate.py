"""dilution_series_analysis._mbr_fill_rates_per_run: the dict_ref-based MBR fill rate used when the
baseline is Sage (slots = dictionary ions not MS/MS-identified in a run; filled = quantified there)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO = Path(__file__).resolve().parents[1]
for p in (str(_REPO), str(_REPO / "swaps")):
    if p not in sys.path:
        sys.path.insert(0, p)

_spec = importlib.util.spec_from_file_location(
    "dilution_series_analysis", _REPO / "notebooks/papers/analysis_scripts/dilution_series_analysis.py"
)
ds = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ds)


@pytest.fixture
def dict_ref():
    return pd.DataFrame(
        {
            "mz_rank": [0, 1, 2, 3],
            "Modified sequence": ["A", "B", "C", "D"],
            "Charge": [2, 2, 2, 2],
            "r1": ["Reference", "Match", "Match", "Quant_Only"],
            "r2": ["Match", "Match", "Reference", "Match"],
        }
    )


def test_swaps_slots_come_from_dict_ref_not_from_swaps_labels(dict_ref):
    swaps = pd.DataFrame(
        {
            "mz_rank": [0, 1, 2, 3],
            # ion 3 is MS/MS-identified in r1 per dict_ref, yet SWAPS labels it unmatched
            "r1 Match Type": ["MS/MS Ref", "MBR", "unmatched", "unmatched"],
            "r1 Intensity": [5.0, 3.0, 0.0, 0.0],
            # ion 3 is an MBR hit in r2 but with zero intensity: not a filled slot
            "r2 Match Type": ["MBR", "unmatched", "MS/MS", "MBR"],
            "r2 Intensity": [2.0, 0.0, 5.0, 0.0],
        }
    )
    with_dict_ref = ds._mbr_fill_rates_per_run(swaps, dict_ref)
    # r1: slots = ions 1, 2 -> ion 1 filled -> 50%; r2: slots = ions 0, 1, 3 -> ion 0 filled -> 33.3%
    assert with_dict_ref == pytest.approx([50.0, 100 / 3])
    # the label-based definition counts the identified-but-unmatched ion 3 as an open slot in r1
    assert ds._mbr_fill_rates_per_run(swaps)[0] == pytest.approx(100 / 3)


def test_sage_needs_an_lfq_row_and_a_nonzero_quantification(dict_ref):
    sage = pd.DataFrame(
        {
            "Modified Sequence": ["A", "B", "C"],  # ion D is not in the table at all
            "Charge": [2, 2, 2],
            "In LFQ": [True, True, False],
            "r1 Match Type": ["MS/MS", "MBR", "unmatched"],
            "r1 Intensity": [4.0, 3.0, 0.0],
            "r2 Match Type": ["MBR", "unmatched", "MS/MS"],
            "r2 Intensity": [2.0, 0.0, 0.0],
        }
    )
    rates = ds._mbr_fill_rates_per_run(sage, dict_ref)
    # r1: slots = B, C -> only B quantified -> 50%; r2: slots = A, B, D -> only A -> 33.3%
    assert rates == pytest.approx([50.0, 100 / 3])


def test_without_dict_ref_the_label_based_definition_is_unchanged():
    df = pd.DataFrame(
        {"r1 Match Type": ["MS/MS Ref", "MBR", "MBR_undistinguished", "unmatched"], "r1 Intensity": [1, 1, 1, 0]}
    )
    assert ds._mbr_fill_rates_per_run(df) == pytest.approx([2 / 3 * 100])
