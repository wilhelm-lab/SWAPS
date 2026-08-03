import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from swaps.postprocessing.match_features import (
    _build_consensus_peptide_swap_decoy,
    _confounder_pool,
    _crop_consensus_feature_bundle_to_window,
    _find_shift_via_template_match,
    _select_group_reference_run,
    _shift_and_fit,
    align_images_to_reference,
    build_consensus_feature_bundle,
    match_features_batch,
    _snap_anchor_to_watershed_label,
    _project_anchor_into_aligned_space,
    ConsensusAlignmentState,
)


def _two_blob_image(shape=(40, 40), centers=((10, 10), (10, 30)), amp=10.0, sigma=3.0):
    """Two well-separated Gaussian blobs -> detect_2d_peak_with_watershed
    resolves them as two distinct watershed labels/peaks."""
    img = np.zeros(shape, dtype=np.float64)
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    for cy, cx in centers:
        img += amp * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2)))
    return img


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


# ---------------------------------------------------------------------------
# broad_alignment: forced_shifts center (rather than replace) template-match
# discovery on a small window around the given shift -- a real
# (non-NaN) template_matching_score comes out either way, distinct from the
# 0.0 sentinel align_images=False uses ("bounded search" vs "not attempted").
# With no explicit max_deviation, the window collapses to the forced shift
# itself (an exact rescore there), as long as that position is actually
# reachable by valid-mode template matching -- template_anchor=(15, 15) below
# keeps the template away from the image edge so it always is; see
# TestFindShiftViaTemplateMatchConstrained for the near-edge/clamped case.
# ---------------------------------------------------------------------------


class TestBroadAlignmentForcedShift:
    def test_resize_mode_applies_forced_shift(self):
        images = _make_test_images(2, shape=(30, 30))
        state = align_images_to_reference(
            images, forced_shifts=[None, (2, -3)], template_anchor=(15, 15)
        )
        assert state.shifts[1] == (2, -3)
        assert not np.isnan(state.max_scores[1])

    def test_crop_pad_mode_applies_forced_shift(self):
        images = _make_test_images(2, shape=(30, 30))
        state = align_images_to_reference(
            images,
            forced_shifts=[None, (2, -3)],
            use_shift_crop_pad=True,
            template_anchor=(15, 15),
        )
        assert state.shifts[1] == (2, -3)
        assert not np.isnan(state.max_scores[1])

    def test_crop_pad_mode_forced_shift_matches_shift_and_fit(self):
        images = _make_test_images(2, shape=(30, 30))
        state = align_images_to_reference(
            images,
            forced_shifts=[None, (2, -3)],
            use_shift_crop_pad=True,
            template_anchor=(15, 15),
        )
        expected = _shift_and_fit(images[1], (30, 30), (2, -3))
        np.testing.assert_array_equal(state.aligned_images[1], expected)

    def test_reference_image_ignores_forced_shifts(self):
        images = _make_test_images(2, shape=(30, 30))
        state = align_images_to_reference(images, reference_idx=0, forced_shifts=[(9, 9), None])
        assert state.shifts[0] == (0, 0)
        assert state.max_scores[0] == 1.0

    def test_none_entries_still_discover_normally(self):
        images = _make_test_images(3, shape=(30, 30))
        forced = align_images_to_reference(images, forced_shifts=[None, (2, -3), None])
        discovered = align_images_to_reference(images)
        assert forced.shifts[2] == discovered.shifts[2]
        assert forced.max_scores[2] == discovered.max_scores[2]

    def test_build_bundle_propagates_forced_shifts(self):
        images = _make_test_images(2, shape=(30, 30))
        bundle = build_consensus_feature_bundle(
            images, forced_shifts=[None, (4, 1)], template_anchor=(15, 15)
        )
        assert bundle.alignment.shifts[1] == (4, 1)
        assert not np.isnan(bundle.alignment.max_scores[1])

    def test_bundle_forced_shift_ignores_broad_alignment_max_deviation_when_none(self):
        # No max_deviation given at all (neither forced_shifts-adjacent kwarg) ->
        # still defaults to an exact rescore (deviation 0), not an unconstrained
        # search that would silently discard the forced shift.
        images = _make_test_images(2, shape=(30, 30))
        bundle = build_consensus_feature_bundle(
            images,
            forced_shifts=[None, (4, 1)],
            template_anchor=(15, 15),
            broad_alignment_max_deviation=None,
        )
        assert bundle.alignment.shifts[1] == (4, 1)

    def test_peptide_swap_decoy_rescores_at_forced_shift(self):
        # Real blobs (not random noise), plus anchors/raw_images/labels, so
        # watershed finds a non-empty target_label_ids -- mirrors
        # TestBuildConsensusFeatureBundleReuse's fixture requirements.
        img = _two_blob_image()
        bundle = build_consensus_feature_bundle(
            images=[img, img],
            anchors=[(10, 10), (10, 10)],
            raw_images=[img, img],
            labels=["R1", "R2"],
        )
        decoy_image = _two_blob_image(centers=((25, 5), (25, 25)))
        # (-2, -2), not (5, -2): the target's own template touches the image's
        # top/left edge (anchor (10, 10), template_frac 0.3 on a 40x40 image),
        # so only non-positive rt shifts are reachable by valid-mode
        # template matching without clamping -- see the module-level note
        # above and TestFindShiftViaTemplateMatchConstrained for that clamp.
        _decoy_pp, shift, max_score = _build_consensus_peptide_swap_decoy(
            bundle, decoy_image, "R2", forced_shift=(-2, -2)
        )
        assert shift == (-2, -2)
        assert not np.isnan(max_score)

    def test_peptide_swap_decoy_constrained_search_stays_within_window(self):
        img = _two_blob_image()
        bundle = build_consensus_feature_bundle(
            images=[img, img],
            anchors=[(10, 10), (10, 10)],
            raw_images=[img, img],
            labels=["R1", "R2"],
        )
        decoy_image = _two_blob_image(centers=((25, 5), (25, 25)))
        _decoy_pp, shift, _max_score = _build_consensus_peptide_swap_decoy(
            bundle, decoy_image, "R2", forced_shift=(-2, -2), max_deviation=3
        )
        assert abs(shift[0] - (-2)) <= 3
        assert abs(shift[1] - (-2)) <= 3


# ---------------------------------------------------------------------------
# _find_shift_via_template_match's search_center/max_deviation constraint --
# the core mechanism broad_alignment uses to bound per-candidate discovery to
# a small neighborhood of a precalibrated shift instead of either fixing it
# outright (old behavior) or searching the whole image (the original
# unconstrained per-candidate discovery, which is what let low-S/N peptides
# lock onto a spurious, far-away, higher-scoring correlation peak).
# ---------------------------------------------------------------------------


def _bump(img, center, amp=5.0, sigma=1.5):
    yy, xx = np.mgrid[0 : img.shape[0], 0 : img.shape[1]]
    img += amp * np.exp(-(((yy - center[0]) ** 2 + (xx - center[1]) ** 2) / (2 * sigma**2)))
    return img


class TestFindShiftViaTemplateMatchConstrained:
    """search_image has two bumps: a nearby, shape-mismatched one (imperfect
    correlation) at the position search_center=(0, 0) implies, and a distant
    one built from the exact same gaussian as the template (near-perfect
    correlation) 9 pixels away in each axis -- so unconstrained discovery
    reliably prefers the distant, better-scoring, but wrong bump."""

    def _search_image_and_template(self):
        search_image = np.random.default_rng(0).normal(0, 0.05, (20, 20))
        _bump(search_image, (5, 5), amp=5.0, sigma=2.5)  # near, shape-mismatched
        _bump(search_image, (14, 14), amp=3.0, sigma=1.5)  # far, shape-matched
        template = _bump(np.zeros((5, 5)), (2, 2), amp=1.0, sigma=1.5)
        template_bounds = (3, 3, 8, 8)
        return search_image, template, template_bounds

    def test_unconstrained_prefers_the_far_better_match(self):
        search_image, template, template_bounds = self._search_image_and_template()
        shift, score, _, _ = _find_shift_via_template_match(
            search_image, template, template_bounds
        )
        assert shift == (-9, -9)
        assert score > 0.99

    def test_zero_deviation_forces_exact_center_even_if_worse(self):
        search_image, template, template_bounds = self._search_image_and_template()
        shift, score, _, _ = _find_shift_via_template_match(
            search_image, template, template_bounds, search_center=(0, 0), max_deviation=0
        )
        assert shift == (0, 0)
        assert score < 0.99  # the near bump's imperfect match, not the far one's

    def test_small_deviation_window_excludes_the_far_optimum(self):
        search_image, template, template_bounds = self._search_image_and_template()
        shift, score, _, _ = _find_shift_via_template_match(
            search_image, template, template_bounds, search_center=(0, 0), max_deviation=2
        )
        assert abs(shift[0]) <= 2 and abs(shift[1]) <= 2
        assert shift != (-9, -9)

    def test_large_enough_deviation_recovers_the_far_optimum(self):
        search_image, template, template_bounds = self._search_image_and_template()
        shift, score, _, _ = _find_shift_via_template_match(
            search_image, template, template_bounds, search_center=(0, 0), max_deviation=9
        )
        assert shift == (-9, -9)
        assert score > 0.99

    def test_search_center_beyond_valid_range_is_clamped_not_erroring(self):
        # template_bounds' top-left is (3, 3); a search_center that would
        # imply a negative match_score index (e.g. (10, 10), far outside the
        # image) must clamp into the valid range rather than raising/
        # wrapping -- the exact returned shift isn't asserted, only that it
        # stays finite and near the clamped boundary.
        search_image, template, template_bounds = self._search_image_and_template()
        shift, score, _, _ = _find_shift_via_template_match(
            search_image, template, template_bounds, search_center=(10, 10), max_deviation=0
        )
        assert np.isfinite(score)
        assert shift[0] <= 3 and shift[1] <= 3


# ---------------------------------------------------------------------------
# _snap_anchor_to_watershed_label (extracted from segment_consensus_from_aligned,
# reused by coSWA to snap other confounder-group members against one shared
# watershed segmentation)
# ---------------------------------------------------------------------------


def _two_label_watershed():
    """8x8 label map: label 1 occupies rows 0-2/cols 0-2, label 2 occupies
    rows 5-7/cols 5-7, background (0) everywhere else. Peaks sit at the
    corner of each label."""
    labels = np.zeros((8, 8), dtype=int)
    labels[0:3, 0:3] = 1
    labels[5:8, 5:8] = 2
    all_peaks = np.array([[1, 1], [6, 6]])
    labeled_coords = np.argwhere(labels > 0)
    return labels, all_peaks, labeled_coords


class TestSnapAnchorToWatershedLabel:
    def test_anchor_inside_label_snaps_to_nearest_peak(self):
        labels, all_peaks, labeled_coords = _two_label_watershed()
        snapped_rc, label_id, jump_info = _snap_anchor_to_watershed_label(
            1, 1, labels, all_peaks, labeled_coords, jump_dist_thres=(0, 0)
        )
        assert snapped_rc == (1, 1)
        assert label_id == 1
        assert jump_info is None

    def test_anchor_inside_label_but_off_peak_still_finds_same_label(self):
        labels, all_peaks, labeled_coords = _two_label_watershed()
        snapped_rc, label_id, jump_info = _snap_anchor_to_watershed_label(
            2, 2, labels, all_peaks, labeled_coords, jump_dist_thres=(0, 0)
        )
        assert snapped_rc == (1, 1)  # nearest peak within label 1
        assert label_id == 1
        assert jump_info is None

    def test_background_anchor_jumps_to_nearest_label_within_threshold(self):
        labels, all_peaks, labeled_coords = _two_label_watershed()
        # (3, 1) is background, one row below label 1's bottom edge (2, 1)
        snapped_rc, label_id, jump_info = _snap_anchor_to_watershed_label(
            3, 1, labels, all_peaks, labeled_coords, jump_dist_thres=(2, 2)
        )
        assert label_id == 1
        assert snapped_rc == (1, 1)
        assert jump_info is not None
        assert jump_info["jumped_label"] == 1
        assert jump_info["snapped_peak"] == (1, 1)

    def test_background_anchor_beyond_threshold_is_discarded(self):
        labels, all_peaks, labeled_coords = _two_label_watershed()
        # (3, 1) is 1 pixel from label 1 -- a threshold of (0, 0) discards it
        # ((0, 0) is treated as "no limit" by the rt/im > 0 gating, so use a
        # threshold that's actually enforced: (1, 1) still allows dist=1, so
        # push further away to guarantee a discard.
        snapped_rc, label_id, jump_info = _snap_anchor_to_watershed_label(
            0, 7, labels, all_peaks, labeled_coords, jump_dist_thres=(1, 1)
        )
        assert label_id is None
        assert snapped_rc is None
        assert jump_info is not None
        assert "jumped_label" not in jump_info  # discarded before jumping

    def test_no_labeled_pixels_returns_none(self):
        labels = np.zeros((8, 8), dtype=int)
        all_peaks = np.empty((0, 2), dtype=int)
        labeled_coords = np.empty((0, 2), dtype=int)
        snapped_rc, label_id, jump_info = _snap_anchor_to_watershed_label(
            3, 3, labels, all_peaks, labeled_coords, jump_dist_thres=(0, 0)
        )
        assert (snapped_rc, label_id, jump_info) == (None, None, None)


# ---------------------------------------------------------------------------
# _project_anchor_into_aligned_space
# ---------------------------------------------------------------------------


def _minimal_alignment_state(target_shape, shifts):
    """A ConsensusAlignmentState with only the fields _project_anchor_into_aligned_space reads populated."""
    return ConsensusAlignmentState(
        reference_idx=0,
        target_shape=target_shape,
        anchor_row=0,
        anchor_col=0,
        template_bounds=(0, 0, 0, 0),
        template=np.zeros((1, 1)),
        resized_images=[],
        aligned_images=[],
        matched_boxes=[],
        aligned_anchors=[],
        scaled_anchors=[],
        shifts=shifts,
        max_scores=[],
    )


class TestProjectAnchorIntoAlignedSpace:
    def test_same_shape_no_shift_is_identity(self):
        state = _minimal_alignment_state(target_shape=(20, 20), shifts=[(0, 0)])
        result = _project_anchor_into_aligned_space((5, 7), (20, 20), state, 0)
        assert result == (5.0, 7.0)

    def test_applies_shift(self):
        state = _minimal_alignment_state(target_shape=(20, 20), shifts=[(3, -2)])
        result = _project_anchor_into_aligned_space((5, 7), (20, 20), state, 0)
        assert result == (8.0, 5.0)

    def test_scales_before_shifting(self):
        # source image half the target size -> anchor coordinates double
        state = _minimal_alignment_state(target_shape=(20, 20), shifts=[(1, 1)])
        result = _project_anchor_into_aligned_space((5, 5), (10, 10), state, 0)
        assert result == (11.0, 11.0)  # (5*2)+1, (5*2)+1

    def test_none_anchor_returns_none(self):
        state = _minimal_alignment_state(target_shape=(20, 20), shifts=[(0, 0)])
        assert _project_anchor_into_aligned_space(None, (20, 20), state, 0) is None

    def test_matches_representative_own_aligned_anchor(self):
        """Sanity check against the real pipeline: projecting the SAME anchor
        that align_images_to_reference already aligned should reproduce its
        own aligned_anchors entry exactly."""
        images = _make_test_images(2, shape=(20, 20))
        anchors = [(5, 5), (8, 3)]
        state = align_images_to_reference(images, anchors=anchors)
        projected = _project_anchor_into_aligned_space(
            anchors[1], images[1].shape, state, 1
        )
        assert projected == state.aligned_anchors[1]


# ---------------------------------------------------------------------------
# build_consensus_feature_bundle(reuse_from=...) -- coSWA confounder-group
# member reuse: skip alignment/watershed, only re-snap + re-extract.
# ---------------------------------------------------------------------------


class TestBuildConsensusFeatureBundleReuse:
    def test_reused_bundle_snaps_to_its_own_anchor_independently(self):
        img = _two_blob_image()
        representative = build_consensus_feature_bundle(
            images=[img],
            anchors=[(10, 10)],
            raw_images=[img],
            labels=["run1"],
        )
        assert representative.segmentation.snapped_per_anchor[0] == (10, 10)
        assert representative.consensus_pp is not None

        other_member = build_consensus_feature_bundle(
            images=[img],  # ignored when reuse_from is set
            anchors=[(10, 30)],  # this member's OWN, different, anchor
            raw_images=[img],
            labels=["run1"],
            reuse_from=representative,
        )
        assert other_member.segmentation.snapped_per_anchor[0] == (10, 30)
        assert other_member.consensus_pp is not None
        # the two members land on DIFFERENT watershed labels (correctly
        # told apart), even though they share the identical underlying image
        assert (
            representative.segmentation.target_label_ids
            != other_member.segmentation.target_label_ids
        )

    def test_reused_bundle_shares_expensive_state_not_recomputed(self):
        img = _two_blob_image()
        representative = build_consensus_feature_bundle(
            images=[img], anchors=[(10, 10)], raw_images=[img], labels=["run1"]
        )
        other_member = build_consensus_feature_bundle(
            images=[img],
            anchors=[(10, 30)],
            raw_images=[img],
            labels=["run1"],
            reuse_from=representative,
        )
        # consensus/consensus_denoised/all_peaks/aligned_images are reused
        # verbatim (same object), not recomputed -- watershed_labels is
        # defensively copied (see _snap_all_anchors_to_watershed) but must
        # still be content-equal.
        assert other_member.segmentation.consensus is representative.segmentation.consensus
        assert (
            other_member.segmentation.consensus_denoised
            is representative.segmentation.consensus_denoised
        )
        assert other_member.segmentation.all_peaks is representative.segmentation.all_peaks
        assert other_member.alignment.aligned_images is representative.alignment.aligned_images
        np.testing.assert_array_equal(
            other_member.segmentation.watershed_labels,
            representative.segmentation.watershed_labels,
        )

    def test_two_members_colliding_on_same_label_get_identical_quantification(self):
        """If two confounder-group members' anchors happen to land on the
        SAME watershed label, their extracted quantification comes out
        identical by construction (same mask, same raw image) -- this is
        the property the collision-flagging logic in match_features_batch
        relies on."""
        img = _two_blob_image()
        representative = build_consensus_feature_bundle(
            images=[img], anchors=[(10, 10)], raw_images=[img], labels=["run1"]
        )
        colliding_member = build_consensus_feature_bundle(
            images=[img],
            anchors=[(11, 11)],  # close enough to snap to the SAME label
            raw_images=[img],
            labels=["run1"],
            reuse_from=representative,
        )
        assert (
            representative.segmentation.target_label_ids
            == colliding_member.segmentation.target_label_ids
        )
        pd.testing.assert_frame_equal(
            representative.consensus_pp, colliding_member.consensus_pp
        )

    def test_collapse_to_single_label_tie_broken_by_priority_anchor(self):
        """coSWA per-member assignment: a member whose anchors span BOTH blobs
        (one each -> a tie) collapses to the single segment chosen by
        priority_anchor_index (its reference-run anchor)."""
        img = _two_blob_image()
        rep = build_consensus_feature_bundle(
            images=[img, img],
            anchors=[(10, 10), (10, 10)],
            raw_images=[img, img],
            labels=["r1", "r2"],
        )

        def _reuse(anchors, **kw):
            return build_consensus_feature_bundle(
                images=[img, img],
                anchors=anchors,
                raw_images=[img, img],
                labels=["r1", "r2"],
                reuse_from=rep,
                **kw,
            )

        label_a = _reuse([(10, 10), (10, 10)]).segmentation.target_label_ids
        label_b = _reuse([(10, 30), (10, 30)]).segmentation.target_label_ids
        assert label_a != label_b

        # Without collapse, both blobs' labels are kept.
        spanning = _reuse([(10, 10), (10, 30)])
        assert len(spanning.segmentation.target_label_ids) == 2

        # Tie -> priority anchor decides which single label survives.
        prio0 = _reuse(
            [(10, 10), (10, 30)],
            collapse_to_single_label=True,
            priority_anchor_index=0,
        )
        prio1 = _reuse(
            [(10, 10), (10, 30)],
            collapse_to_single_label=True,
            priority_anchor_index=1,
        )
        assert prio0.segmentation.target_label_ids == label_a
        assert prio1.segmentation.target_label_ids == label_b
        # The discarded anchor is recorded (renders as 'x' in the overlay).
        assert 1 in prio0.segmentation.snap_log["discard_record"]

    def test_collapse_to_single_label_majority_over_priority(self):
        """Majority vote wins even against the priority anchor: two anchors on
        blob B, one (the priority) on blob A -> B survives."""
        img = _two_blob_image()
        rep = build_consensus_feature_bundle(
            images=[img, img, img],
            anchors=[(10, 30), (10, 30), (10, 30)],
            raw_images=[img, img, img],
            labels=["r1", "r2", "r3"],
        )
        label_b = rep.segmentation.target_label_ids
        member = build_consensus_feature_bundle(
            images=[img, img, img],
            anchors=[(10, 10), (10, 30), (10, 30)],
            raw_images=[img, img, img],
            labels=["r1", "r2", "r3"],
            reuse_from=rep,
            collapse_to_single_label=True,
            priority_anchor_index=0,  # points at the minority blob A
        )
        assert member.segmentation.target_label_ids == label_b


# ---------------------------------------------------------------------------
# match_features_batch end-to-end: coSWA confounder-group orchestration (ONE
# shared alignment + watershed segmentation per group; each member snaps its
# own anchors onto it, without forcing a single-label collapse; overlap
# between members' own assigned label sets is flagged via
# undistinguishable_group_id, plus continuous pixel/intensity overlap-
# fraction diagnostics -- see _mark_overlapping_group_members)
# ---------------------------------------------------------------------------

_RT_RANGE = (100, 139)  # 40 frames
_IM_RANGE = (0, 39)  # 40 im bins


def _group_blob_image(blobs):
    n_rt = _RT_RANGE[1] - _RT_RANGE[0] + 1
    n_im = _IM_RANGE[1] - _IM_RANGE[0] + 1
    yy, xx = np.mgrid[0:n_im, 0:n_rt]
    img = np.zeros((n_im, n_rt))
    for rt_c, im_c, amp, sigma in blobs:
        img += amp * np.exp(-(((yy - im_c) ** 2 + (xx - rt_c) ** 2) / (2 * sigma**2)))
    return img


def _write_combined_activation_parquet(act_dir, images_by_mz_rank):
    os.makedirs(act_dir, exist_ok=True)
    rows = []
    for mz_rank, img in images_by_mz_rank.items():
        im_idx, rt_idx = np.nonzero(img > 0.5)
        for i, r in zip(im_idx, rt_idx):
            rows.append(
                {
                    "frame_idx": _RT_RANGE[0] + r,
                    "im_idx": _IM_RANGE[0] + i,
                    "mz_rank": mz_rank,
                    "activation": float(img[i, r]),
                }
            )
    df = pd.DataFrame(rows)
    table = pa.Table.from_pandas(
        df[["frame_idx", "im_idx", "mz_rank", "activation"]].astype(
            {
                "frame_idx": "uint16",
                "im_idx": "uint16",
                "mz_rank": "uint32",
                "activation": "float32",
            }
        ),
        preserve_index=False,
    )
    pq.write_table(table, os.path.join(act_dir, "activation_sorted_by_mz.parquet"))


def _build_group_dict_ref(raw_files, anchors):
    """anchors: dict[mz_rank] -> dict[run] -> (rt_center, im_center) absolute
    frame/im index, mirroring each candidate's own individually-predicted
    apex (independent of the shared merged activation image)."""
    rows = []
    for mz_rank, group_id in [(1, 1001), (2, 1001), (3, -1)]:
        row = {"mz_rank": mz_rank, "confounder_group_id": group_id}
        for i, rf in enumerate(raw_files):
            row[rf] = "Reference" if i == 0 else "Match"
            row[f"MS1_frame_idx_left_ref_{rf}"] = _RT_RANGE[0]
            row[f"MS1_frame_idx_right_ref_{rf}"] = _RT_RANGE[1]
            row[f"mobility_values_index_left_ref_{rf}"] = _IM_RANGE[0]
            row[f"mobility_values_index_right_ref_{rf}"] = _IM_RANGE[1]
            rt_c, im_c = anchors[mz_rank][rf]
            row[f"{rf}_MS1_frame_idx_exp"] = rt_c
            row[f"{rf}_mobility_values_index_exp"] = im_c
        rows.append(row)
    return pd.DataFrame(rows)


def _run_group_scenario(tmp_path, group_blobs, a_anchor_offset, b_anchor_offset):
    """Two runs, three candidates: mz_rank 1 (A) and 2 (B) are a confounder
    group sharing the IDENTICAL activation image `group_blobs` in every run
    (as coSWA's SWA-level merge would produce -- see
    helper.expand_group_ids_to_members); mz_rank 3 (C) is solo with its
    own separate, well-clear activation. A and B's own individually
    predicted anchors are placed at a_anchor_offset/b_anchor_offset within
    that shared image."""
    raw_files = ["run1", "run2"]
    group_img = _group_blob_image(group_blobs)
    solo_img = _group_blob_image([(25, 25, 10.0, 2.0)])
    for rf in raw_files:
        # On-disk format post-fix: the group is stored as ONE row-set keyed
        # by its confounder_group_id (1001, matching _build_group_dict_ref's
        # mz_rank 1 and 2 assignment) -- never duplicated to each member's
        # own mz_rank. match_features_batch lazily re-expands it in memory
        # for this batch via load_peptide_batch_df_from_partquet.
        _write_combined_activation_parquet(
            os.path.join(tmp_path, rf, "activation"),
            {1001: group_img, 3: solo_img},
        )

    def _abs(offset):
        return (_RT_RANGE[0] + offset[0], _IM_RANGE[0] + offset[1])

    anchors = {
        1: {rf: _abs(a_anchor_offset) for rf in raw_files},
        2: {rf: _abs(b_anchor_offset) for rf in raw_files},
        3: {rf: _abs((25, 25)) for rf in raw_files},
    }
    dict_ref = _build_group_dict_ref(raw_files, anchors)

    return match_features_batch(
        dict_ref=dict_ref,
        raw_file_list=raw_files,
        result_dir=str(tmp_path),
        batch=[1, 2, 3],
        processing_kwargs={"apply_seg": True},
        match_decoy=False,
    )


class TestSelectGroupReferenceRun:
    def test_picks_run_with_most_anchors(self):
        # run1: both members anchored (m1 Reference, m2 Quant_Only) -> count 2.
        # run2: only m1 anchored (Quant_Only) -> count 1.
        member_roles = {
            1: ("run1", ["run2"], ["run2"]),
            2: ("run3", ["run1"], ["run1", "run3"]),
        }
        assert _select_group_reference_run(member_roles) == "run1"

    def test_tie_broken_by_reference_role_count(self):
        # run1 and run2 both have 2 anchored members (tied), but run2 has 2
        # members with the Reference role specifically vs run1's 1.
        member_roles = {
            1: ("run1", ["run2"], ["run2"]),
            2: ("run2", ["run1"], ["run1"]),
            3: ("run2", [], []),
        }
        assert _select_group_reference_run(member_roles) == "run2"

    def test_final_tie_broken_randomly_among_remaining(self):
        # run1 and run2 are identical in both anchor count and Reference
        # count -- the result must be one of the two, deterministically
        # reproducible only up to that random choice.
        member_roles = {
            1: ("run1", [], []),
            2: ("run2", [], []),
        }
        result = _select_group_reference_run(member_roles)
        assert result in ("run1", "run2")


class TestCropConsensusFeatureBundleToWindow:
    """Unit coverage for the decoy-scoring crop helper: a coSWA group's
    shared (group-window-scale) bundle cropped down to one member's own
    (smaller) individual window -- the mechanism match_features_batch uses
    so decoy feature scale stays comparable to a solo candidate's own decoy."""

    def test_crop_shapes_and_preserves_contained_quantification(self):
        img = _two_blob_image()  # 40x40, blobs at (row10,col10) and (row10,col30)
        bundle = build_consensus_feature_bundle(
            images=[img], anchors=[(10, 10)], raw_images=[img], labels=["run1"]
        )
        assert bundle.consensus_pp is not None

        cropped = _crop_consensus_feature_bundle_to_window(
            bundle,
            crop_origin=(0, 0),
            crop_shape=(40, 20),  # cols [0, 20) -- fully contains blob at col 10
            member_ref_anchor_local=(10, 10),
            template_frac=0.3,
            labels=["run1"],
        )
        assert cropped.alignment.target_shape == (40, 20)
        assert cropped.raw_aligned_images[0].shape == (40, 20)
        assert cropped.raw_aligned_denoised_images[0].shape == (40, 20)
        assert cropped.segmentation.watershed_labels.shape == (40, 20)
        assert cropped.consensus_pp is not None
        # The far blob (col 30) was never part of this label's own mask, so
        # cropping it out of frame changes nothing about this label's area.
        assert (
            cropped.consensus_pp["area"].iloc[0] == bundle.consensus_pp["area"].iloc[0]
        )
        assert cropped.consensus_pp["intensity_sum"].iloc[0] == pytest.approx(
            bundle.consensus_pp["intensity_sum"].iloc[0]
        )

    def test_crop_that_clips_the_blob_reduces_area(self):
        img = _two_blob_image()
        bundle = build_consensus_feature_bundle(
            images=[img], anchors=[(10, 10)], raw_images=[img], labels=["run1"]
        )
        cropped = _crop_consensus_feature_bundle_to_window(
            bundle,
            crop_origin=(0, 5),
            crop_shape=(40, 10),  # cols [5, 15) -- clips the blob at col 10
            member_ref_anchor_local=(10, 10),
            template_frac=0.3,
            labels=["run1"],
        )
        assert cropped.consensus_pp is not None
        assert (
            cropped.consensus_pp["area"].iloc[0] < bundle.consensus_pp["area"].iloc[0]
        )


class TestMatchFeaturesBatchConfounderGroups:
    def test_bimodal_signal_separates_group_members_no_collision(self, tmp_path):
        """Two distinguishable sub-peaks within the ONE shared group
        segmentation -> A and B's own anchors snap to DIFFERENT watershed
        labels (no forced collapse needed since neither spans more than one
        label) -> both correctly quantified independently, no
        undistinguishable flag, and neither member's own pixels/intensity
        are claimed by the other (0.0 overlap fraction)."""
        (
            results_target,
            results_decoy,
            pp_reference_list,
            pp_match_target_list,
            pp_match_decoy_list,
            no_quant_log,
            no_match_log,
            snap_log_collection,
        ) = _run_group_scenario(
            tmp_path,
            group_blobs=[(15, 8, 10.0, 2.0), (15, 18, 10.0, 2.0)],
            a_anchor_offset=(15, 8),
            b_anchor_offset=(15, 18),
        )
        pp_ref = pd.concat(pp_reference_list)
        pp_match = pd.concat(pp_match_target_list)
        by_rank_ref = pp_ref.set_index("mz_rank")
        by_rank_match = pp_match.set_index("mz_rank")

        assert by_rank_ref.loc[1, "undistinguishable_group_id"] == -1
        assert by_rank_ref.loc[2, "undistinguishable_group_id"] == -1
        assert by_rank_ref.loc[3, "undistinguishable_group_id"] == -1
        assert by_rank_match.loc[1, "undistinguishable_group_id"] == -1
        assert by_rank_match.loc[2, "undistinguishable_group_id"] == -1
        # correctly told apart: distinct quantification for A vs B
        assert by_rank_ref.loc[1, "area"] != by_rank_ref.loc[2, "area"]
        # neither member's own assigned pixels/intensity are shared
        assert by_rank_ref.loc[1, "undistinguishable_pixel_fraction"] == 0.0
        assert by_rank_ref.loc[2, "undistinguishable_pixel_fraction"] == 0.0
        assert by_rank_ref.loc[1, "undistinguishable_intensity_fraction"] == 0.0
        assert by_rank_ref.loc[2, "undistinguishable_intensity_fraction"] == 0.0

    def test_unimodal_signal_flags_group_members_as_undistinguishable(self, tmp_path):
        """A single peak in the shared group segmentation -> A and B's own
        (identical) anchors both snap to the SAME single watershed label ->
        flagged with a shared undistinguishable_group_id, and BOTH members'
        own assigned pixels/intensity are entirely (fraction 1.0) claimed by
        the other, since they share the one label."""
        (
            results_target,
            results_decoy,
            pp_reference_list,
            pp_match_target_list,
            pp_match_decoy_list,
            no_quant_log,
            no_match_log,
            snap_log_collection,
        ) = _run_group_scenario(
            tmp_path,
            group_blobs=[(15, 12, 10.0, 2.0)],
            a_anchor_offset=(15, 12),
            b_anchor_offset=(15, 12),
        )
        pp_ref = pd.concat(pp_reference_list)
        pp_match = pd.concat(pp_match_target_list)
        by_rank_ref = pp_ref.set_index("mz_rank")
        by_rank_match = pp_match.set_index("mz_rank")

        tag_ref = by_rank_ref.loc[1, "undistinguishable_group_id"]
        assert tag_ref != -1
        assert by_rank_ref.loc[2, "undistinguishable_group_id"] == tag_ref
        assert by_rank_ref.loc[3, "undistinguishable_group_id"] == -1  # solo untouched
        tag_match = by_rank_match.loc[1, "undistinguishable_group_id"]
        assert tag_match != -1
        assert by_rank_match.loc[2, "undistinguishable_group_id"] == tag_match

        # both are still genuinely quantified (not dropped just for overlapping)
        assert by_rank_ref.loc[1, "area"] > 0
        assert by_rank_ref.loc[2, "area"] > 0
        # sharing the ONE label -> entirely overlapping, both directions
        assert by_rank_ref.loc[1, "undistinguishable_pixel_fraction"] == 1.0
        assert by_rank_ref.loc[2, "undistinguishable_pixel_fraction"] == 1.0
        assert by_rank_ref.loc[1, "undistinguishable_intensity_fraction"] == 1.0
        assert by_rank_ref.loc[2, "undistinguishable_intensity_fraction"] == 1.0
        # C (solo) has no group to overlap with
        assert by_rank_ref.loc[3, "undistinguishable_pixel_fraction"] == 0.0

    def test_solo_candidate_unaffected_by_group_presence(self, tmp_path):
        """C's own quantification and -1 tag hold regardless of whether the
        other group members' signal happens to be separable or not."""
        for group_blobs, a_off, b_off in [
            ([(15, 8, 10.0, 2.0), (15, 18, 10.0, 2.0)], (15, 8), (15, 18)),
            ([(15, 12, 10.0, 2.0)], (15, 12), (15, 12)),
        ]:
            (_, _, pp_reference_list, pp_match_target_list, *_rest) = (
                _run_group_scenario(tmp_path, group_blobs, a_off, b_off)
            )
            pp_ref = pd.concat(pp_reference_list).set_index("mz_rank")
            pp_match = pd.concat(pp_match_target_list).set_index("mz_rank")
            assert pp_ref.loc[3, "undistinguishable_group_id"] == -1
            assert pp_match.loc[3, "undistinguishable_group_id"] == -1
            assert pp_ref.loc[3, "area"] > 0

    def test_merge_confounders_disabled_ignores_stale_group_id_column(
        self, tmp_path
    ):
        """Backward compatibility: dict_ref may still carry a confounder_group_id
        column left over from a previous run with coSWA enabled (e.g. a
        reused dict_ref.pkl), but if PREPARE_DICT.MERGE_CONFOUNDERS.ENABLED
        is now False for this run, every candidate must be treated as solo,
        fetching its own individually-stored activation (mirroring
        test_no_confounder_group_id_column_behaves_as_before, just with the
        stale column present) -- undistinguishable_group_id stays -1 for
        everyone and no candidate silently loses its data."""
        raw_files = ["run1", "run2"]
        img_a = _group_blob_image([(15, 12, 10.0, 2.0)])
        img_b = _group_blob_image([(15, 12, 10.0, 2.0)])
        img_c = _group_blob_image([(25, 25, 10.0, 2.0)])
        for rf in raw_files:
            _write_combined_activation_parquet(
                os.path.join(tmp_path, rf, "activation"),
                {1: img_a, 2: img_b, 3: img_c},
            )

        def _abs(offset):
            return (_RT_RANGE[0] + offset[0], _IM_RANGE[0] + offset[1])

        rows = []
        for mz_rank, group_id, off in [
            (1, 1001, (15, 12)),
            (2, 1001, (15, 12)),
            (3, -1, (25, 25)),
        ]:
            row = {"mz_rank": mz_rank, "confounder_group_id": group_id}
            for i, rf in enumerate(raw_files):
                row[rf] = "Reference" if i == 0 else "Match"
                row[f"MS1_frame_idx_left_ref_{rf}"] = _RT_RANGE[0]
                row[f"MS1_frame_idx_right_ref_{rf}"] = _RT_RANGE[1]
                row[f"mobility_values_index_left_ref_{rf}"] = _IM_RANGE[0]
                row[f"mobility_values_index_right_ref_{rf}"] = _IM_RANGE[1]
                rt_c, im_c = _abs(off)
                row[f"{rf}_MS1_frame_idx_exp"] = rt_c
                row[f"{rf}_mobility_values_index_exp"] = im_c
            rows.append(row)
        dict_ref = pd.DataFrame(rows)

        (_, _, pp_reference_list, pp_match_target_list, *_rest) = match_features_batch(
            dict_ref=dict_ref,
            raw_file_list=raw_files,
            result_dir=str(tmp_path),
            batch=[1, 2, 3],
            processing_kwargs={"apply_seg": True},
            match_decoy=False,
            merge_confounders_enabled=False,
        )
        pp_ref = pd.concat(pp_reference_list).set_index("mz_rank")
        pp_match = pd.concat(pp_match_target_list).set_index("mz_rank")

        # all three candidates still got quantified -- none silently dropped
        # for want of an activation lookup keyed by the stale group id.
        assert set(pp_ref.index) == {1, 2, 3}
        assert set(pp_match.index) == {1, 2, 3}
        assert (pp_ref["undistinguishable_group_id"] == -1).all()
        assert (pp_match["undistinguishable_group_id"] == -1).all()

    def test_no_confounder_group_id_column_behaves_as_before(self, tmp_path):
        """Backward compatibility: without confounder_group_id (merging
        disabled at dict-build time), every candidate is processed
        independently as today, undistinguishable_group_id defaults to -1
        for everyone."""
        raw_files = ["run1", "run2"]
        img_a = _group_blob_image([(15, 12, 10.0, 2.0)])
        img_b = _group_blob_image([(15, 12, 10.0, 2.0)])
        img_c = _group_blob_image([(25, 25, 10.0, 2.0)])
        for rf in raw_files:
            _write_combined_activation_parquet(
                os.path.join(tmp_path, rf, "activation"),
                {1: img_a, 2: img_b, 3: img_c},
            )

        def _abs(offset):
            return (_RT_RANGE[0] + offset[0], _IM_RANGE[0] + offset[1])

        rows = []
        for mz_rank, off in [(1, (15, 12)), (2, (15, 12)), (3, (25, 25))]:
            row = {"mz_rank": mz_rank}  # no confounder_group_id column at all
            for i, rf in enumerate(raw_files):
                row[rf] = "Reference" if i == 0 else "Match"
                row[f"MS1_frame_idx_left_ref_{rf}"] = _RT_RANGE[0]
                row[f"MS1_frame_idx_right_ref_{rf}"] = _RT_RANGE[1]
                row[f"mobility_values_index_left_ref_{rf}"] = _IM_RANGE[0]
                row[f"mobility_values_index_right_ref_{rf}"] = _IM_RANGE[1]
                rt_c, im_c = _abs(off)
                row[f"{rf}_MS1_frame_idx_exp"] = rt_c
                row[f"{rf}_mobility_values_index_exp"] = im_c
            rows.append(row)
        dict_ref = pd.DataFrame(rows)

        (_, _, pp_reference_list, pp_match_target_list, *_rest) = match_features_batch(
            dict_ref=dict_ref,
            raw_file_list=raw_files,
            result_dir=str(tmp_path),
            batch=[1, 2, 3],
            processing_kwargs={"apply_seg": True},
            match_decoy=False,
        )
        pp_ref = pd.concat(pp_reference_list)
        assert (pp_ref["undistinguishable_group_id"] == -1).all()
