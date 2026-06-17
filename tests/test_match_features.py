import numpy as np
import pandas as pd
import pytest

from swaps.postprocessing.match_features import _confounder_pool


def _dict_ref_by_mz(rows):
    """Minimal dict_ref_by_mz (indexed by mz_rank) for testing."""
    return pd.DataFrame(rows).set_index("mz_rank")


def test_returns_in_batch_confounders():
    batch = np.array([1, 2, 3, 4, 5])
    conf = np.array([2, 3, 10])  # 10 is outside batch
    df = _dict_ref_by_mz([{"mz_rank": 1, "confounders": conf}])
    pool = _confounder_pool(1, batch, df)
    assert set(pool) == {2, 3}


def test_excludes_self():
    batch = np.array([1, 2, 3])
    conf = np.array([1, 2, 3])  # includes self
    df = _dict_ref_by_mz([{"mz_rank": 1, "confounders": conf}])
    pool = _confounder_pool(1, batch, df)
    assert 1 not in pool


def test_empty_when_column_absent():
    batch = np.array([1, 2, 3])
    df = _dict_ref_by_mz([{"mz_rank": 1}])  # no confounders column
    pool = _confounder_pool(1, batch, df)
    assert pool.size == 0


def test_empty_when_all_confounders_outside_batch():
    batch = np.array([1, 2, 3])
    conf = np.array([10, 20, 30])
    df = _dict_ref_by_mz([{"mz_rank": 1, "confounders": conf}])
    pool = _confounder_pool(1, batch, df)
    assert pool.size == 0


def test_empty_when_confounder_array_is_empty():
    batch = np.array([1, 2, 3])
    df = _dict_ref_by_mz([{"mz_rank": 1, "confounders": np.empty(0, dtype=int)}])
    pool = _confounder_pool(1, batch, df)
    assert pool.size == 0


def test_dtype_matches_batch():
    batch = np.array([1, 2, 3], dtype=np.int64)
    conf = np.array([2, 3])
    df = _dict_ref_by_mz([{"mz_rank": 1, "confounders": conf}])
    pool = _confounder_pool(1, batch, df)
    assert pool.dtype == batch.dtype
