"""
Shared pytest fixtures and configuration for the SWAPS test suite.

Environment variables (set these before running integration tests):
  SWAPS_TEST_RAW_DIR      – directory containing *.d raw files
  SWAPS_TEST_FRAGPIPE_DIR – FragPipe output root (contains per-run sub-dirs with psm.tsv)
  SWAPS_TEST_SAGE_DIR     – SAGE output root (placeholder, not yet available)
  SWAPS_TEST_MAXQUANT_DIR – MaxQuant output root (placeholder, not yet available)

Example:
  export SWAPS_TEST_RAW_DIR=/cmnfs/proj/ORIGINS/data/.../Data/test_hela/500SPD/jigsawPASEF_a000_IntThres1600_2R_150ms
  export SWAPS_TEST_FRAGPIPE_DIR=/cmnfs/proj/ORIGINS/data/.../fragPipeOutput
  pytest tests/ -m "not slow"
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Make the swaps package importable when pytest is run from the repo root
# ---------------------------------------------------------------------------
_SWAPS_DIR = os.path.join(os.path.dirname(__file__), "..", "swaps")
if _SWAPS_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(_SWAPS_DIR))


# ---------------------------------------------------------------------------
# Custom pytest marks
# ---------------------------------------------------------------------------
def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: tests that require real data (set env vars to enable)",
    )
    config.addinivalue_line(
        "markers",
        "slow: tests that run the full pipeline (very slow, opt-in only)",
    )


@pytest.fixture(scope="session", autouse=True)
def _suppress_alphatims_atexit_logging_error():
    """Alphatims registers an atexit handler that calls logging.warning() to
    report temp mmap cleanup. By the time atexit runs, pytest has already
    closed its log streams, causing 'ValueError: I/O operation on closed file'.
    Setting logging.raiseExceptions = False tells the logging system to silently
    discard handler errors instead of printing a traceback."""
    yield
    import logging
    logging.raiseExceptions = False


# ---------------------------------------------------------------------------
# Data-path fixtures (skip when env vars are absent)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def raw_dir():
    path = os.environ.get(
        "SWAPS_TEST_RAW_DIR",
        "/cmnfs/proj/ORIGINS/data/random_precursor_sampling/"
        "jigsaw_bruker_single_cell/Data/test_hela/500SPD/"
        "jigsawPASEF_a000_IntThres1600_2R_150ms/",
    )
    if not path or not os.path.isdir(path):
        pytest.skip("SWAPS_TEST_RAW_DIR not set or not a directory")
    return path


@pytest.fixture(scope="session")
def fragpipe_dir():
    path = os.environ.get(
        "SWAPS_TEST_FRAGPIPE_DIR",
        "/cmnfs/proj/ORIGINS/data/random_precursor_sampling/"
        "jigsaw_bruker_single_cell/fragpipe_diann/test_hela/500SPD/"
        "jigsawPASEF_a000_IntThres1600_2R_150ms/fragPipeOutput",
    )
    if not path or not os.path.isdir(path):
        pytest.skip(f"FragPipe dir not found: {path}")
    return path


@pytest.fixture(scope="session")
def sage_dir():
    path = os.environ.get(
        "SWAPS_TEST_SAGE_DIR",
        "/cmnfs/proj/ORIGINS/data/random_precursor_sampling/"
        "jigsaw_bruker_single_cell/sage/test_hela/500SPD/"
        "jigsawPASEF_a000_IntThres1600_2R_150ms",
    )
    if not path or not os.path.isdir(path):
        pytest.skip(f"SAGE dir not found: {path}")
    return path


@pytest.fixture(scope="session")
def maxquant_dir():
    path = os.environ.get(
        "SWAPS_TEST_MAXQUANT_DIR",
        "/cmnfs/proj/ORIGINS/data/random_precursor_sampling/"
        "jigsaw_bruker_single_cell/MaxQuant/test_hela/500SPD/"
        "jigsawPASEF_a000_IntThres1600_2R_150ms",
    )
    if not path or not os.path.isdir(path):
        pytest.skip(f"MaxQuant dir not found: {path}")
    return path


# ---------------------------------------------------------------------------
# Minimal synthetic DataFrames reused across unit-test modules
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_fragpipe_psm_df():
    """Minimal FragPipe psm.tsv-like DataFrame (2 rows, all required columns)."""
    return pd.DataFrame(
        {
            "Spectrum": [
                "sample_A.00100.00100.2",
                "sample_A.00200.00200.3",
            ],
            "Peptide": ["PEPTIDEK", "SEQUENCER"],
            "Modified Peptide": ["PEPTIDEK", np.nan],
            "Protein": ["sp|P1|GENE1_HUMAN", "sp|P2|GENE2_HUMAN"],
            "Observed M/Z": [500.25, 400.10],
            "Retention": [3086.4, 2700.0],  # seconds → will be /60
            "Apex Retention Time": [3100.0, 2710.0],
            "Retention Time Start": [3080.0, 2690.0],
            "Retention Time End": [3120.0, 2730.0],
            "Ion Mobility": [0.92, 1.05],
            "Ion Mobility Start": [0.90, 1.03],
            "Ion Mobility End": [0.94, 1.07],
            "Assigned Modifications": ["15.9949M2", np.nan],
            "Hyperscore": [25.4, 18.9],
            "Is Decoy": [False, False],
            "Charge": [2, 3],
            "Peptide Length": [8, 8],
        }
    )


@pytest.fixture
def minimal_sage_df():
    """Minimal SAGE results-like DataFrame (3 rows, some pass FDR filter)."""
    return pd.DataFrame(
        {
            "psm_id": ["id1", "id2", "id3"],
            "peptide": ["_PEPTIDEK_", "_SEQUENCER_", "_DECOYYYY_"],
            "proteins": ["sp|P1|GENE1", "sp|P2|GENE2", "rev_sp|P3|GENE3"],
            "filename": ["sample_A.d", "sample_A.d", "sample_A.d"],
            "scannr": [100, 200, 300],
            "rank": [1, 1, 1],
            "expmass": [1001.5, 1200.3, 999.9],
            "calcmass": [1001.4, 1200.2, 999.8],
            "charge": [2, 3, 2],
            "peptide_len": [8, 8, 8],
            "missed_cleavages": [0, 0, 0],
            "precursor_ppm": [1.2, 0.8, 5.0],
            "hyperscore": [30.1, 22.5, 10.0],
            "delta_next": [5.0, 3.0, 1.0],
            "rt": [51.4, 45.0, 30.0],
            "ion_mobility": [0.92, 1.05, 0.80],
            "delta_mobility": [0.01, 0.02, 0.05],
            "matched_peaks": [8, 6, 3],
            "matched_intensity_pct": [0.85, 0.70, 0.30],
            "sage_discriminant_score": [0.95, 0.80, 0.20],
            "posterior_error": [0.001, 0.005, 0.4],
            "spectrum_q": [0.001, 0.005, 0.3],
            "peptide_q": [0.001, 0.008, 0.4],
            "ms2_intensity": [50000.0, 30000.0, 5000.0],
        }
    )


@pytest.fixture
def minimal_matches_target():
    """Synthetic matches_target DataFrame for rescore unit tests."""
    rng = np.random.default_rng(0)
    n = 40
    return pd.DataFrame(
        {
            "mz_rank": np.arange(n),
            "reference_run": ["run_A"] * 20 + ["run_B"] * 20,
            "matched_run": ["run_B"] * 20 + ["run_A"] * 20,
            "rt_shift": rng.normal(0.5, 0.2, n),
            "im_shift": rng.normal(0.02, 0.01, n),
            "template_matching_score": rng.uniform(0.5, 1.0, n),
            "sift_similarities": rng.uniform(0.4, 1.0, n),
            "hu_similarities": rng.uniform(0.4, 1.0, n),
            "zernike_similarities": rng.uniform(0.4, 1.0, n),
            "sift_distance": rng.uniform(0.0, 0.5, n),
            "hu_distance": rng.uniform(0.0, 0.5, n),
            "zernike_distance": rng.uniform(0.0, 0.5, n),
        }
    )


@pytest.fixture
def minimal_matches_decoy(minimal_matches_target):
    """Synthetic matches_decoy – same schema as target, shifted distributions."""
    rng = np.random.default_rng(1)
    df = minimal_matches_target.copy()
    n = len(df)
    df["rt_shift"] = rng.normal(1.5, 0.5, n)
    df["im_shift"] = rng.normal(0.08, 0.03, n)
    df["template_matching_score"] = rng.uniform(0.0, 0.5, n)
    df["decoy_mz_rank"] = df["mz_rank"] + 10000
    return df


@pytest.fixture
def minimal_dict_ref():
    """Minimal dict_ref DataFrame for rescore / helper unit tests."""
    n = 40
    return pd.DataFrame(
        {
            "mz_rank": np.arange(n),
            "Sequence": [f"PEP{i:04d}K" for i in range(n)],
            "Proteins": [f"sp|P{i:04d}|GENE{i}_HUMAN" for i in range(n)],
        }
    )
