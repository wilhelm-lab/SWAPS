"""
Unit tests for postprocessing.image_processing.smooth_and_denoise_image
and match_features denoise helper functions.
"""

import numpy as np
import pytest
from unittest.mock import patch

from postprocessing.image_processing import smooth_and_denoise_image
from postprocessing.match_features import (
    _denoise_kwargs_for_stage,
    _denoise_kwargs_all,
    ConsensusAlignmentState,
    segment_consensus_from_aligned,
)


# ---------------------------------------------------------------------------
# smooth_and_denoise_image
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_image():
    rng = np.random.default_rng(0)
    img = rng.uniform(0, 100, (20, 20)).astype(float)
    img[5:10, 5:10] = 500.0  # bright region
    return img


class TestSmoothAndDenoiseImageNoOps:
    def test_no_ops_returns_original(self, sample_image):
        result = smooth_and_denoise_image(sample_image)
        np.testing.assert_array_equal(result, sample_image)

    def test_all_none_returns_original(self, sample_image):
        result = smooth_and_denoise_image(sample_image, smooth=None, clean=None, log_transform=False)
        np.testing.assert_array_equal(result, sample_image)


class TestSmoothAndDenoiseImageSmooth:
    def test_gaussian_reduces_peak(self, sample_image):
        result = smooth_and_denoise_image(sample_image, smooth={"filter": "gaussian"})
        assert result.max() < sample_image.max()

    def test_uniform_increases_neighbourhood(self, sample_image):
        flat = np.zeros((20, 20), dtype=float)
        flat[10, 10] = 100.0
        result = smooth_and_denoise_image(flat, smooth={"filter": "uniform"})
        # uniform spreads value, so more pixels should be nonzero
        assert (result > 0).sum() > 1

    def test_gaussian_with_custom_sigma(self, sample_image):
        r1 = smooth_and_denoise_image(sample_image, smooth={"filter": "gaussian", "gaussian_kwargs": {"sigma": 1}})
        r2 = smooth_and_denoise_image(sample_image, smooth={"filter": "gaussian", "gaussian_kwargs": {"sigma": 4}})
        # larger sigma → more blurring → lower max
        assert r2.max() < r1.max()

    def test_unknown_filter_passthrough(self, sample_image):
        result = smooth_and_denoise_image(sample_image, smooth={"filter": "none_of_the_above"})
        np.testing.assert_array_equal(result, sample_image)


class TestSmoothAndDenoiseImageClean:
    def test_clean_zeroes_below_threshold(self):
        img = np.array([[0.0, 5.0], [3.0, 200.0]])
        result = smooth_and_denoise_image(img, clean={"threshold": 10.0, "remove_kwargs": {"min_size": 1}})
        assert result[0, 0] == 0.0
        assert result[0, 1] == 0.0
        assert result[1, 0] == 0.0
        assert result[1, 1] == 200.0

    def test_clean_preserves_large_region(self, sample_image):
        result = smooth_and_denoise_image(sample_image, clean={"threshold": 0.0, "remove_kwargs": {"min_size": 1}})
        # every non-zero pixel kept when threshold=0 and min_size=1
        assert result.max() == sample_image.max()


class TestSmoothAndDenoiseImageLogTransform:
    def test_log_transform_applied(self, sample_image):
        result = smooth_and_denoise_image(sample_image, log_transform=True)
        expected = np.log2(1 + sample_image)
        np.testing.assert_allclose(result, expected)

    def test_log_transform_false_no_change(self, sample_image):
        result = smooth_and_denoise_image(sample_image, log_transform=False)
        np.testing.assert_array_equal(result, sample_image)


class TestSmoothAndDenoiseImageOrder:
    def test_smooth_then_clean_then_log(self):
        """Steps run in order: smooth → clean → log_transform."""
        img = np.ones((10, 10)) * 50.0
        result = smooth_and_denoise_image(
            img,
            smooth={"filter": "gaussian"},
            clean={"threshold": 0.0, "remove_kwargs": {"min_size": 1}},
            log_transform=True,
        )
        # After gaussian on uniform image, value should be ~50; after clean (threshold 0)
        # all kept; after log2(1+~50) should be ~5.7
        assert result.min() > 0
        assert result.max() < 10  # log-scale


# ---------------------------------------------------------------------------
# _denoise_kwargs_for_stage
# ---------------------------------------------------------------------------


_SAMPLE_DENOISE_CFG = {
    "smooth": {"at": "consensus", "filter": "uniform", "uniform_kwargs": {"size": [3, 5]}},
    "clean": {"at": "consensus", "threshold": 0, "remove_kwargs": {"min_size": 3}},
    "log_transform": {"at": "raw", "enabled": True},
}


class TestDenoiseKwargsForStage:
    def test_raw_stage_returns_log_transform(self):
        kw = _denoise_kwargs_for_stage(_SAMPLE_DENOISE_CFG, "raw")
        assert "log_transform" in kw
        assert kw["log_transform"] is True
        assert "smooth" not in kw
        assert "clean" not in kw

    def test_consensus_stage_returns_smooth_and_clean(self):
        kw = _denoise_kwargs_for_stage(_SAMPLE_DENOISE_CFG, "consensus")
        assert "smooth" in kw
        assert "clean" in kw
        assert "log_transform" not in kw

    def test_smooth_at_field_excluded_from_kwargs(self):
        kw = _denoise_kwargs_for_stage(_SAMPLE_DENOISE_CFG, "consensus")
        assert "at" not in kw["smooth"]
        assert "at" not in kw["clean"]

    def test_unknown_stage_returns_empty(self):
        kw = _denoise_kwargs_for_stage(_SAMPLE_DENOISE_CFG, "other")
        assert kw == {}

    def test_empty_cfg_returns_empty(self):
        kw = _denoise_kwargs_for_stage({}, "raw")
        assert kw == {}

    def test_log_transform_disabled(self):
        cfg = {"log_transform": {"at": "raw", "enabled": False}}
        kw = _denoise_kwargs_for_stage(cfg, "raw")
        assert kw["log_transform"] is False


class TestDenoiseKwargsAll:
    def test_all_returns_both_stages(self):
        kw = _denoise_kwargs_all(_SAMPLE_DENOISE_CFG)
        assert "log_transform" in kw
        assert "smooth" in kw
        assert "clean" in kw

    def test_all_on_empty_cfg(self):
        kw = _denoise_kwargs_all({})
        assert kw == {}

    def test_all_values_match_per_stage(self):
        raw_kw = _denoise_kwargs_for_stage(_SAMPLE_DENOISE_CFG, "raw")
        consensus_kw = _denoise_kwargs_for_stage(_SAMPLE_DENOISE_CFG, "consensus")
        all_kw = _denoise_kwargs_all(_SAMPLE_DENOISE_CFG)
        assert all_kw == {**raw_kw, **consensus_kw}


# ---------------------------------------------------------------------------
# Singleton denoise config shape
# ---------------------------------------------------------------------------


class TestSingletonDenoiseConfig:
    def test_denoise_node_present(self):
        from utils.config import get_cfg_defaults
        from utils.singleton_swaps_optimization import swaps_optimization_cfg
        cfg = get_cfg_defaults(swaps_optimization_cfg)
        assert hasattr(cfg.MATCH_FEATURES_KWARGS, "denoise")

    def test_smooth_subnode(self):
        from utils.config import get_cfg_defaults
        from utils.singleton_swaps_optimization import swaps_optimization_cfg
        cfg = get_cfg_defaults(swaps_optimization_cfg)
        s = cfg.MATCH_FEATURES_KWARGS.denoise.smooth
        assert s.at == "consensus"
        assert s.filter in ("gaussian", "uniform")
        assert hasattr(s, "gaussian_kwargs")
        assert hasattr(s, "uniform_kwargs")

    def test_clean_subnode(self):
        from utils.config import get_cfg_defaults
        from utils.singleton_swaps_optimization import swaps_optimization_cfg
        cfg = get_cfg_defaults(swaps_optimization_cfg)
        c = cfg.MATCH_FEATURES_KWARGS.denoise.clean
        assert c.at == "consensus"
        assert hasattr(c, "threshold")
        assert hasattr(c, "remove_kwargs")

    def test_log_transform_subnode(self):
        from utils.config import get_cfg_defaults
        from utils.singleton_swaps_optimization import swaps_optimization_cfg
        cfg = get_cfg_defaults(swaps_optimization_cfg)
        lt = cfg.MATCH_FEATURES_KWARGS.denoise.log_transform
        assert lt.at == "raw"
        assert isinstance(lt.enabled, bool)

    def test_old_keys_absent(self):
        from utils.config import get_cfg_defaults
        from utils.singleton_swaps_optimization import swaps_optimization_cfg
        cfg = get_cfg_defaults(swaps_optimization_cfg)
        mfk = cfg.MATCH_FEATURES_KWARGS
        assert not hasattr(mfk, "smooth_kwargs")
        assert not hasattr(mfk, "smooth_consensus_kwargs")
        assert not hasattr(mfk, "peak_kwargs")
        assert not hasattr(mfk, "filter_kwargs")


# ---------------------------------------------------------------------------
# segment_consensus_from_aligned — seg_mask_thres rollback
# ---------------------------------------------------------------------------


def _make_alignment_state(shape=(20, 20), anchor=(10, 10)) -> ConsensusAlignmentState:
    img = np.zeros(shape, dtype=float)
    r, c = anchor
    img[r - 1 : r + 2, c - 1 : c + 2] = 1.0
    return ConsensusAlignmentState(
        reference_idx=0,
        target_shape=shape,
        anchor_row=r,
        anchor_col=c,
        template_bounds=(r - 2, c - 2, r + 2, c + 2),
        template=img[r - 2 : r + 2, c - 2 : c + 2].copy(),
        resized_images=[img],
        aligned_images=[img],
        matched_boxes=[(r - 2, c - 2, r + 2, c + 2)],
        aligned_anchors=[(float(r), float(c))],
        scaled_anchors=[(float(r), float(c))],
        shifts=[(0, 0)],
        max_scores=[1.0],
    )


def _tiny_watershed_return(shape=(20, 20), anchor=(10, 10)):
    """Return value for detect_2d_peak_with_watershed with a 4-pixel label."""
    labels = np.zeros(shape, dtype=int)
    r, c = anchor
    labels[r, c] = 1
    labels[r, c + 1] = 1
    labels[r + 1, c] = 1
    labels[r + 1, c + 1] = 1  # 4 pixels — below default threshold of 9
    peaks = np.array([[r, c]])
    return (peaks, None, None, labels, None)


class TestSegmentConsensusSegMaskThres:
    def test_small_label_triggers_bbox_rollback(self):
        alignment = _make_alignment_state()
        with patch(
            "postprocessing.match_features.detect_2d_peak_with_watershed",
            return_value=_tiny_watershed_return(),
        ):
            state = segment_consensus_from_aligned(alignment, seg_mask_thres=9)

        assert state.snap_log["no_seg_log"] is not None

    def test_small_label_kept_when_thres_disabled(self):
        alignment = _make_alignment_state()
        with patch(
            "postprocessing.match_features.detect_2d_peak_with_watershed",
            return_value=_tiny_watershed_return(),
        ):
            state = segment_consensus_from_aligned(alignment, seg_mask_thres=0)

        assert state.snap_log["no_seg_log"] is None
        assert 1 in state.target_label_ids

    def test_large_label_not_rolled_back(self):
        shape = (20, 20)
        alignment = _make_alignment_state(shape)
        labels = np.zeros(shape, dtype=int)
        labels[8:14, 8:14] = 1  # 36 pixels — above any reasonable threshold
        peaks = np.array([[10, 10]])
        with patch(
            "postprocessing.match_features.detect_2d_peak_with_watershed",
            return_value=(peaks, None, None, labels, None),
        ):
            state = segment_consensus_from_aligned(alignment, seg_mask_thres=9)

        assert state.snap_log["no_seg_log"] is None
        assert 1 in state.target_label_ids


class TestSegmentConsensusJumpDistThres:
    """Tests for jump_dist_thres: anchors in background whose nearest labeled pixel
    exceeds the RT or IM threshold are discarded (snapped_per_anchor stays None)."""

    def _make_background_alignment(self, shape=(30, 30), anchor=(5, 5)):
        """Anchor at (5,5) is in background; peak/label cluster is far away at (20,20)."""
        img = np.zeros(shape, dtype=float)
        r, c = (20, 20)
        img[r - 1 : r + 2, c - 1 : c + 2] = 1.0
        return ConsensusAlignmentState(
            reference_idx=0,
            target_shape=shape,
            anchor_row=anchor[0],
            anchor_col=anchor[1],
            template_bounds=(anchor[0] - 2, anchor[1] - 2, anchor[0] + 2, anchor[1] + 2),
            template=img[anchor[0] - 2 : anchor[0] + 2, anchor[1] - 2 : anchor[1] + 2].copy(),
            resized_images=[img],
            aligned_images=[img],
            matched_boxes=[(anchor[0] - 2, anchor[1] - 2, anchor[0] + 2, anchor[1] + 2)],
            aligned_anchors=[(float(anchor[0]), float(anchor[1]))],
            scaled_anchors=[(float(anchor[0]), float(anchor[1]))],
            shifts=[(0, 0)],
            max_scores=[1.0],
        )

    def _far_label_watershed(self, shape=(30, 30), label_at=(20, 20)):
        """Watershed with a single-pixel label; nearest_labeled_rc == label_at exactly."""
        labels = np.zeros(shape, dtype=int)
        r, c = label_at
        labels[r, c] = 1
        peaks = np.array([[r, c]])
        return (peaks, None, None, labels, None)

    def test_jump_within_threshold_is_kept(self):
        """Anchor in background but close to a label → jump proceeds normally."""
        shape = (30, 30)
        # anchor at (18,18), single-pixel label at (20,20): rt_dist=2, im_dist=2 — within thres=5
        # seg_mask_thres disabled so rollback doesn't wipe jump_anchor_log
        alignment = self._make_background_alignment(shape, anchor=(18, 18))
        with patch(
            "postprocessing.match_features.detect_2d_peak_with_watershed",
            return_value=self._far_label_watershed(shape, label_at=(20, 20)),
        ):
            state = segment_consensus_from_aligned(
                alignment, jump_dist_thres=(5, 5), seg_mask_thres=(0, 0)
            )
        assert state.snapped_per_anchor[0] is not None
        assert state.snap_log["discard_record"] == {}
        assert 0 in state.snap_log["jump_anchor_log"]

    def test_rt_jump_exceeds_threshold_discards_anchor(self):
        """Anchor in background with rt_dist > threshold → anchor discarded."""
        shape = (30, 30)
        # anchor at (5,5), single-pixel label at (20,5): rt_dist=15, im_dist=0 — rt exceeds thres=10
        alignment = self._make_background_alignment(shape, anchor=(5, 5))
        with patch(
            "postprocessing.match_features.detect_2d_peak_with_watershed",
            return_value=self._far_label_watershed(shape, label_at=(20, 5)),
        ):
            state = segment_consensus_from_aligned(
                alignment, jump_dist_thres=(10, 10)
            )
        assert state.snapped_per_anchor[0] is None
        assert 0 in state.snap_log["discard_record"]
        assert state.snap_log["discard_record"][0]["rt_dist"] == 15
        assert state.snap_log["jump_anchor_log"] == {}

    def test_im_jump_exceeds_threshold_discards_anchor(self):
        """Anchor in background with im_dist > threshold → anchor discarded."""
        shape = (30, 30)
        # anchor at (5,5), single-pixel label at (5,20): rt_dist=0, im_dist=15 — im exceeds thres=10
        alignment = self._make_background_alignment(shape, anchor=(5, 5))
        with patch(
            "postprocessing.match_features.detect_2d_peak_with_watershed",
            return_value=self._far_label_watershed(shape, label_at=(5, 20)),
        ):
            state = segment_consensus_from_aligned(
                alignment, jump_dist_thres=(10, 10)
            )
        assert state.snapped_per_anchor[0] is None
        assert 0 in state.snap_log["discard_record"]
        assert state.snap_log["discard_record"][0]["im_dist"] == 15

    def test_zero_threshold_disables_filter(self):
        """jump_dist_thres=(0,0) means no filtering — all jumps are kept."""
        shape = (30, 30)
        alignment = self._make_background_alignment(shape, anchor=(5, 5))
        with patch(
            "postprocessing.match_features.detect_2d_peak_with_watershed",
            return_value=self._far_label_watershed(shape, label_at=(20, 20)),
        ):
            state = segment_consensus_from_aligned(
                alignment, jump_dist_thres=(0, 0), seg_mask_thres=(0, 0)
            )
        assert state.snapped_per_anchor[0] is not None
        assert state.snap_log["discard_record"] == {}

    def test_discard_record_preserved_through_seg_mask_rollback(self):
        """When seg_mask_thres rollback fires, discard_record is NOT cleared."""
        shape = (30, 30)
        # Two anchors: anchor[0] at (5,5) in background far from label → discard
        #              anchor[1] at (20,20) inside the label → snap normally
        # But the label is tiny (below seg_mask_thres) → rollback fires
        img = np.zeros(shape, dtype=float)
        img[19:22, 19:22] = 1.0
        alignment = ConsensusAlignmentState(
            reference_idx=0,
            target_shape=shape,
            anchor_row=20,
            anchor_col=20,
            template_bounds=(18, 18, 22, 22),
            template=img[18:22, 18:22].copy(),
            resized_images=[img, img],
            aligned_images=[img, img],
            matched_boxes=[(18, 18, 22, 22), (18, 18, 22, 22)],
            aligned_anchors=[(5.0, 5.0), (20.0, 20.0)],
            scaled_anchors=[(5.0, 5.0), (20.0, 20.0)],
            shifts=[(0, 0), (0, 0)],
            max_scores=[1.0, 1.0],
        )
        tiny_labels = np.zeros(shape, dtype=int)
        tiny_labels[20, 20] = 1  # single pixel — tiny, triggers rollback
        peaks = np.array([[20, 20]])
        with patch(
            "postprocessing.match_features.detect_2d_peak_with_watershed",
            return_value=(peaks, None, None, tiny_labels, None),
        ):
            state = segment_consensus_from_aligned(
                alignment, seg_mask_thres=9, jump_dist_thres=(10, 10)
            )
        # rollback should have fired (no_seg_log is not None)
        assert state.snap_log["no_seg_log"] is not None
        # discard_record should survive the rollback
        assert 0 in state.snap_log["discard_record"]


# ---------------------------------------------------------------------------
# segment_consensus_from_aligned — consensus_image_indices
# ---------------------------------------------------------------------------


class TestSegmentConsensusImageIndices:
    """Only the images at consensus_image_indices contribute to the mean used
    for watershed segmentation.  All images still receive per-image peak
    properties via snapped_per_anchor / label_to_snap."""

    def _make_two_image_state(self, shape=(20, 20), anchor=(10, 10)):
        """Reference (index 0) has a bright blob; index 1 is all zeros (no signal)."""
        img_signal = np.zeros(shape, dtype=float)
        r, c = anchor
        img_signal[r - 1 : r + 2, c - 1 : c + 2] = 1.0
        img_noise = np.zeros(shape, dtype=float)
        return ConsensusAlignmentState(
            reference_idx=0,
            target_shape=shape,
            anchor_row=r,
            anchor_col=c,
            template_bounds=(r - 2, c - 2, r + 2, c + 2),
            template=img_signal[r - 2 : r + 2, c - 2 : c + 2].copy(),
            resized_images=[img_signal, img_noise],
            aligned_images=[img_signal, img_noise],
            matched_boxes=[(r - 2, c - 2, r + 2, c + 2)] * 2,
            aligned_anchors=[(float(r), float(c)), None],
            scaled_anchors=[(float(r), float(c)), None],
            shifts=[(0, 0), (0, 0)],
            max_scores=[1.0, 0.5],
        )

    def test_consensus_uses_only_specified_indices(self):
        """With consensus_image_indices=[0], the mean is the reference image only,
        not diluted by the zero second image."""
        shape = (20, 20)
        anchor = (10, 10)
        state = self._make_two_image_state(shape, anchor)
        labels = np.zeros(shape, dtype=int)
        r, c = anchor
        labels[r - 1 : r + 2, c - 1 : c + 2] = 1
        peaks = np.array([[r, c]])

        with patch(
            "postprocessing.match_features.detect_2d_peak_with_watershed",
            return_value=(peaks, None, None, labels, None),
        ) as mock_ws:
            seg = segment_consensus_from_aligned(
                state,
                seg_mask_thres=0,
                consensus_image_indices=[0],
            )
            called_image = mock_ws.call_args[0][0]

        # The image passed to watershed should equal the reference image alone
        img_signal = state.aligned_images[0]
        np.testing.assert_array_equal(called_image, img_signal)
        assert 1 in seg.target_label_ids

    def test_without_consensus_image_indices_uses_all(self):
        """Default behaviour (no consensus_image_indices) averages all images."""
        shape = (20, 20)
        anchor = (10, 10)
        state = self._make_two_image_state(shape, anchor)
        labels = np.zeros(shape, dtype=int)
        r, c = anchor
        labels[r - 1 : r + 2, c - 1 : c + 2] = 1
        peaks = np.array([[r, c]])

        with patch(
            "postprocessing.match_features.detect_2d_peak_with_watershed",
            return_value=(peaks, None, None, labels, None),
        ) as mock_ws:
            segment_consensus_from_aligned(state, seg_mask_thres=0)
            called_image = mock_ws.call_args[0][0]

        # Mean of signal + zeros → half the signal
        expected_mean = state.aligned_images[0] / 2.0
        np.testing.assert_allclose(called_image, expected_mean)
