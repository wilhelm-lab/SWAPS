import numpy as np
import pandas as pd
import pytest

from swaps.postprocessing.match_features import (
    _confounder_pool,
    align_images_to_reference,
    build_consensus_feature_bundle,
)


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


# ---------------------------------------------------------------------------
# align_images=False
# ---------------------------------------------------------------------------


def _make_test_images(n=3, shape=(20, 20), seed=42):
    rng = np.random.default_rng(seed)
    return [rng.uniform(0, 1, shape).astype(np.float32) for _ in range(n)]


class TestAlignImagesDisabled:
    def test_shifts_are_zero(self):
        images = _make_test_images(3)
        state = align_images_to_reference(images, align_images=False)
        assert all(s == (0, 0) for s in state.shifts)

    def test_non_reference_max_scores_are_zero(self):
        images = _make_test_images(3)
        state = align_images_to_reference(images, reference_idx=0, align_images=False)
        assert state.max_scores[0] == 1.0  # reference unchanged
        assert all(s == 0.0 for s in state.max_scores[1:])

    def test_aligned_images_equal_resized(self):
        images = _make_test_images(3)
        state = align_images_to_reference(images, align_images=False)
        for aligned, resized in zip(state.aligned_images, state.resized_images):
            np.testing.assert_array_equal(aligned, resized)

    def test_build_bundle_zero_shifts_via_processing_kwargs(self):
        """build_consensus_feature_bundle propagates align_images=False correctly."""
        images = _make_test_images(2, shape=(30, 30))
        bundle = build_consensus_feature_bundle(images, align_images=False)
        assert all(s == (0, 0) for s in bundle.alignment.shifts)
        assert bundle.alignment.max_scores[1] == 0.0
