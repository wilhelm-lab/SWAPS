import numpy as np
import pandas as pd
import pytest

from swaps.postprocessing.match_features import ConsensusAlignmentState
from swaps.prepare_dict.search_engine_output_parser import (
    maxquant_mods_to_fragpipe_modseq,
)
from swaps.peak_detection_2d.dataset.prepare_dataset import (
    _load_fragpipe_manifest_run_labels,
    _match_combined_ion_run_prefix,
    _project_run_boundary_to_bbox,
    _rescale_hint_points,
    _rt_im_to_idx,
    compute_target_shape,
    compute_target_shape_per_experiment,
    invert_pad_or_rescale_to_shape,
    pad_or_crop_to_shape,
    pad_or_rescale_to_shape,
    round_up_to_multiple,
)


def _minimal_alignment_state(target_shape, shifts, use_shift_crop_pad=False):
    """Same minimal-fields helper as tests/test_match_features.py's own --
    only the fields _project_anchor_into_aligned_space (and therefore
    _project_run_boundary_to_bbox) reads are populated."""
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
        use_shift_crop_pad=use_shift_crop_pad,
    )


class TestMaxquantModsToFragpipeModseq:
    def test_unmodified(self):
        assert maxquant_mods_to_fragpipe_modseq("PEPTIDE", "-") == "PEPTIDE"

    def test_single_residue_mod(self):
        result = maxquant_mods_to_fragpipe_modseq(
            "AAAEVGAPFIEIHTGCYADAK", "16C(57.0215)"
        )
        assert result == "AAAEVGAPFIEIHTGC[57.0215]YADAK"

    def test_nterm_mod(self):
        result = maxquant_mods_to_fragpipe_modseq(
            "AAAAAAAAAAGAAGGR", "N-term(42.0106)"
        )
        assert result == "n[42.0106]AAAAAAAAAAGAAGGR"

    def test_unparseable_mod_returns_none(self):
        assert maxquant_mods_to_fragpipe_modseq("PEPTIDE", "garbage") is None

    def test_nan_modifications_returns_sequence(self):
        assert maxquant_mods_to_fragpipe_modseq("PEPTIDE", np.nan) == "PEPTIDE"


class TestMatchCombinedIonRunPrefix:
    def test_exact_match(self):
        mapping = _match_combined_ion_run_prefix(
            ["Run1_1", "Run2_1"], ["Run1", "Run2"]
        )
        assert mapping == {"Run1_1": "Run1", "Run2_1": "Run2"}

    def test_hyphen_normalized_in_column_but_not_raw_dir(self):
        # FragPipe sanitizes "-" to "_" in its own column headers.
        mapping = _match_combined_ion_run_prefix(
            ["Slot2_4_1_12685_1"], ["Slot2-4_1_12685"]
        )
        assert mapping == {"Slot2_4_1_12685_1": "Slot2-4_1_12685"}

    def test_ambiguous_prefix_is_skipped(self):
        # Both "Run1" and "Run1_extra" are valid prefixes of "Run1_extra_1" --
        # genuinely ambiguous, so the safe behavior is to skip rather than guess.
        mapping = _match_combined_ion_run_prefix(
            ["Run1_extra_1"], ["Run1", "Run1_extra"]
        )
        assert mapping == {}

    def test_no_match_is_skipped(self):
        mapping = _match_combined_ion_run_prefix(["Unrelated_1"], ["Run1"])
        assert mapping == {}

    def test_manifest_label_bearing_no_prefix_relationship_to_raw_dir(self):
        # Custom FragPipe experiment/replicate naming: combined_ion.tsv's
        # column ("HYE124_A_1") shares no string prefix at all with the raw
        # .d file's own (much longer) name -- only the manifest resolves it.
        mapping = _match_combined_ion_run_prefix(
            ["HYE124_A_1"],
            ["HYE124_A_200ng_5min_ddaPASEF_IntThres1600_Slot2-5_1_12777"],
            manifest_run_labels={
                "HYE124_A_1": "HYE124_A_200ng_5min_ddaPASEF_IntThres1600_Slot2-5_1_12777"
            },
        )
        assert mapping == {
            "HYE124_A_1": "HYE124_A_200ng_5min_ddaPASEF_IntThres1600_Slot2-5_1_12777"
        }

    def test_manifest_pointing_to_unknown_raw_dir_falls_back(self):
        # Manifest entry present but its raw dir isn't in raw_dirs (e.g.
        # stale manifest) -- falls back to the prefix heuristic rather than
        # trusting a dangling reference.
        mapping = _match_combined_ion_run_prefix(
            ["Run1_1"],
            ["Run1"],
            manifest_run_labels={"Run1_1": "SomeOtherRun"},
        )
        assert mapping == {"Run1_1": "Run1"}

    def test_manifest_takes_priority_over_prefix_heuristic(self):
        # col_prefix "Run1_1" would ALSO match "Run1" by the prefix
        # heuristic, but an explicit (different) manifest mapping wins.
        mapping = _match_combined_ion_run_prefix(
            ["Run1_1"],
            ["Run1", "Run1_variant"],
            manifest_run_labels={"Run1_1": "Run1_variant"},
        )
        assert mapping == {"Run1_1": "Run1_variant"}


class TestLoadFragpipeManifestRunLabels:
    def test_missing_manifest_returns_empty(self, tmp_path):
        assert _load_fragpipe_manifest_run_labels(str(tmp_path)) == {}

    def test_parses_experiment_replicate_label_to_raw_dir(self, tmp_path):
        manifest = tmp_path / "fragpipe-files.fp-manifest"
        manifest.write_text(
            "/data/HYE/ddaPASEF/HYE124_A_200ng_5min_ddaPASEF_Slot2-5_1_12777.d\tHYE124_A\t1\tDDA+\n"
            "/data/HYE/ddaPASEF/HYE124_B_200ng_5min_ddaPASEF_Slot2-6_1_12778.d\tHYE124_B\t1\tDDA+\n"
        )
        result = _load_fragpipe_manifest_run_labels(str(tmp_path))
        assert result == {
            "HYE124_A_1": "HYE124_A_200ng_5min_ddaPASEF_Slot2-5_1_12777",
            "HYE124_B_1": "HYE124_B_200ng_5min_ddaPASEF_Slot2-6_1_12778",
        }


class TestRtImToIdx:
    def test_basic_lookup(self):
        ms1scans = pd.DataFrame({"Time_minute": [0.0, 1.0, 2.0, 3.0, 4.0]})
        mobility_values_df = pd.DataFrame(
            {"mobility_values": [0.8, 0.9, 1.0, 1.1, 1.2]}
        )
        frame_left, frame_right, mob_left, mob_right = _rt_im_to_idx(
            rt_start_min=np.array([1.0]),
            rt_end_min=np.array([3.0]),
            im_start=np.array([0.9]),
            im_end=np.array([1.1]),
            ms1scans=ms1scans,
            mobility_values_df=mobility_values_df,
        )
        assert frame_left[0] == 0
        assert frame_right[0] == 4
        assert mob_left[0] == 0
        assert mob_right[0] == 4

    def test_clips_at_zero(self):
        ms1scans = pd.DataFrame({"Time_minute": [0.0, 1.0, 2.0]})
        mobility_values_df = pd.DataFrame({"mobility_values": [0.8, 0.9, 1.0]})
        frame_left, _, mob_left, _ = _rt_im_to_idx(
            rt_start_min=np.array([-5.0]),
            rt_end_min=np.array([0.5]),
            im_start=np.array([-5.0]),
            im_end=np.array([0.85]),
            ms1scans=ms1scans,
            mobility_values_df=mobility_values_df,
        )
        assert frame_left[0] == 0
        assert mob_left[0] == 0


class TestProjectRunBoundaryToBbox:
    def test_projects_and_clips_to_target_shape(self):
        state = _minimal_alignment_state(target_shape=(20, 20), shifts=[(0, 0)])
        bbox = _project_run_boundary_to_bbox(
            frame_idx_left=5,
            frame_idx_right=10,
            mobility_idx_left=3,
            mobility_idx_right=8,
            window_origin=(0, 0),
            raw_crop_shape=(20, 20),
            alignment_state=state,
            run_position_i=0,
        )
        assert bbox == (5, 3, 10, 8)

    def test_subtracts_window_origin(self):
        state = _minimal_alignment_state(target_shape=(20, 20), shifts=[(0, 0)])
        bbox = _project_run_boundary_to_bbox(
            frame_idx_left=105,
            frame_idx_right=110,
            mobility_idx_left=53,
            mobility_idx_right=58,
            window_origin=(100, 50),
            raw_crop_shape=(20, 20),
            alignment_state=state,
            run_position_i=0,
        )
        assert bbox == (5, 3, 10, 8)

    def test_applies_run_shift(self):
        state = _minimal_alignment_state(target_shape=(20, 20), shifts=[(2, -1)])
        bbox = _project_run_boundary_to_bbox(
            frame_idx_left=5,
            frame_idx_right=10,
            mobility_idx_left=3,
            mobility_idx_right=8,
            window_origin=(0, 0),
            raw_crop_shape=(20, 20),
            alignment_state=state,
            run_position_i=0,
        )
        assert bbox == (7, 2, 12, 7)

    def test_out_of_bounds_returns_none(self):
        state = _minimal_alignment_state(target_shape=(20, 20), shifts=[(0, 0)])
        bbox = _project_run_boundary_to_bbox(
            frame_idx_left=25,
            frame_idx_right=30,
            mobility_idx_left=3,
            mobility_idx_right=8,
            window_origin=(0, 0),
            raw_crop_shape=(20, 20),
            alignment_state=state,
            run_position_i=0,
        )
        assert bbox is None


class TestPadOrCropToShape:
    def test_exact_shape_is_noop(self):
        arr = np.arange(16).reshape(4, 4).astype(float)
        result, was_cropped = pad_or_crop_to_shape(arr, (4, 4))
        np.testing.assert_array_equal(result, arr)
        assert not was_cropped

    def test_pads_smaller_symmetrically(self):
        arr = np.ones((2, 2))
        result, was_cropped = pad_or_crop_to_shape(arr, (4, 4))
        assert result.shape == (4, 4)
        assert not was_cropped
        assert result.sum() == 4  # original values preserved, rest zero
        np.testing.assert_array_equal(result[1:3, 1:3], arr)

    def test_crops_larger_symmetrically(self):
        arr = np.arange(36).reshape(6, 6).astype(float)
        result, was_cropped = pad_or_crop_to_shape(arr, (4, 4))
        assert result.shape == (4, 4)
        assert was_cropped
        np.testing.assert_array_equal(result, arr[1:5, 1:5])

    def test_mixed_pad_one_axis_crop_other(self):
        arr = np.ones((2, 6))
        result, was_cropped = pad_or_crop_to_shape(arr, (4, 4))
        assert result.shape == (4, 4)
        assert was_cropped  # cropped on the width axis
        # height 2->4 pads (preserves all values); width 6->4 crops the
        # outer column on each side (2 ones lost per row, 4 total).
        assert result.sum() == 8


class TestPadOrRescaleToShape:
    def test_fits_within_target_is_pure_pad_no_interpolation(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        result, was_rescaled = pad_or_rescale_to_shape(arr, (4, 4))
        assert result.shape == (4, 4)
        assert not was_rescaled
        np.testing.assert_array_equal(result[1:3, 1:3], arr)

    def test_oversized_axis_is_rescaled_not_cropped(self):
        # Full-intensity block spanning the whole oversized source; if this
        # were pad_or_crop_to_shape, cropping would discard part of it.
        arr = np.ones((8, 3))
        result, was_rescaled = pad_or_rescale_to_shape(arr, (4, 4))
        assert result.shape == (4, 4)
        assert was_rescaled
        # Entire original extent survives (rescaled down), unlike cropping
        # which would drop half the rows.
        assert result.sum() > 0
        assert np.count_nonzero(result[:, 1]) == 4  # full rescaled height column

    def test_only_oversized_axis_is_rescaled_fitting_axis_stays_exact(self):
        # H=8 exceeds target 4 (must rescale); W=3 already fits target 4
        # (must NOT be stretched to fill it -- only padded).
        arr = np.zeros((8, 3))
        arr[:, 1] = 1.0  # a single full-height column at W-index 1
        result, was_rescaled = pad_or_rescale_to_shape(arr, (4, 4))
        assert was_rescaled
        # W axis: still a single populated column (padded, not stretched
        # into a wider band of nonzero columns).
        nonzero_cols = np.any(result > 0.01, axis=0)
        assert nonzero_cols.sum() == 1

    def test_no_resize_needed_is_identity_when_shape_matches(self):
        arr = np.arange(16).reshape(4, 4).astype(float)
        result, was_rescaled = pad_or_rescale_to_shape(arr, (4, 4))
        np.testing.assert_array_equal(result, arr)
        assert not was_rescaled

    def test_binary_mask_area_roughly_preserved_under_downscale(self):
        mask = np.zeros((20, 20))
        mask[5:15, 5:15] = 1.0  # 100 px, half the image linearly in each dim
        result, was_rescaled = pad_or_rescale_to_shape(mask, (10, 10))
        assert was_rescaled
        thresholded = (result > 0.5).astype(float)
        # Downscaled 20x20->10x10, so a 10x10 block should map to ~5x5=25 px
        # (interpolation/threshold introduces some slack, not exactness).
        assert 15 <= thresholded.sum() <= 36


class TestInvertPadOrRescaleToShape:
    """Inverse of pad_or_rescale_to_shape -- used at CNN inference time
    (match_features._segment_consensus_with_cnn) to map a predicted mask
    back from the model's fixed training shape onto a real consensus
    image's own native per-peptide shape."""

    def test_pure_pad_case_round_trips_exactly(self):
        # Both axes fit within target_shape -> pad_or_rescale_to_shape only
        # pads, no interpolation; inverting should recover the original
        # array exactly (crop-only, no resize).
        arr = np.arange(6).reshape(2, 3).astype(float)
        padded, was_rescaled = pad_or_rescale_to_shape(arr, (6, 7))
        assert not was_rescaled
        recovered = invert_pad_or_rescale_to_shape(padded, (2, 3), (6, 7))
        np.testing.assert_array_equal(recovered, arr)

    def test_rescaled_axis_round_trips_to_original_shape(self):
        arr = np.ones((8, 3))
        rescaled, was_rescaled = pad_or_rescale_to_shape(arr, (4, 4))
        assert was_rescaled
        recovered = invert_pad_or_rescale_to_shape(rescaled, (8, 3), (4, 4))
        assert recovered.shape == (8, 3)

    def test_binary_mask_survives_resize_and_inverse_resize_roughly(self):
        # A full-extent foreground block, downscaled then inverted back up,
        # should still cover most of its original footprint after
        # rethresholding (some interpolation slack is expected, exactness
        # is not -- this is a smoke test for the round trip, not a
        # pixel-exact one).
        mask = np.zeros((20, 6))
        mask[:, 2:4] = 1.0
        model_shape = (10, 10)
        rescaled, was_rescaled = pad_or_rescale_to_shape(mask, model_shape)
        assert was_rescaled
        recovered = invert_pad_or_rescale_to_shape(rescaled, (20, 6), model_shape)
        assert recovered.shape == (20, 6)
        recovered_binary = (recovered > 0.5).astype(float)
        assert recovered_binary.sum() > 0

    def test_mixed_pad_one_axis_rescale_other_round_trips_shape(self):
        arr = np.arange(8 * 3).reshape(8, 3).astype(float)
        rescaled, was_rescaled = pad_or_rescale_to_shape(arr, (4, 10))
        assert was_rescaled
        recovered = invert_pad_or_rescale_to_shape(rescaled, (8, 3), (4, 10))
        assert recovered.shape == (8, 3)


class TestRescaleHintPoints:
    def test_isolated_point_never_dropped_under_aggressive_downscale(self):
        # A naive nearest-neighbor raster resize can skip an isolated
        # 1-pixel point entirely on a big downscale -- this must not.
        hint = np.zeros((100, 100))
        hint[7, 93] = 1.0
        result = _rescale_hint_points(hint, (10, 10))
        assert result.sum() == 1

    def test_no_rescale_needed_point_lands_at_same_relative_position(self):
        hint = np.zeros((10, 10))
        hint[3, 4] = 1.0
        result = _rescale_hint_points(hint, (10, 10))
        assert result[3, 4] == 1.0
        assert result.sum() == 1

    def test_multiple_points_all_preserved(self):
        hint = np.zeros((50, 50))
        for r, c in [(1, 1), (10, 40), (49, 0), (25, 25)]:
            hint[r, c] = 1.0
        result = _rescale_hint_points(hint, (12, 12))
        assert result.sum() >= 1  # at least one pixel positive
        # every distinct source point maps somewhere inside bounds -- no
        # exception, no point silently vanishing into an empty result
        assert result.sum() <= 4

    def test_empty_hint_stays_empty(self):
        hint = np.zeros((20, 20))
        result = _rescale_hint_points(hint, (10, 10))
        assert result.sum() == 0
        assert result.shape == (10, 10)


class TestComputeTargetShape:
    def test_percentile_rounded_up_to_multiple(self):
        shapes = [(10, 100)] * 98 + [(100, 700)] * 2
        target = compute_target_shape(shapes, percentile=99.0, multiple=16)
        # p99 of H is ~100 (near the top of the distribution), rounds up to 112
        assert target[0] % 16 == 0
        assert target[1] % 16 == 0
        assert target[0] >= 100
        assert target[1] >= 700

    def test_round_up_to_multiple(self):
        assert round_up_to_multiple(0, 16) == 0
        assert round_up_to_multiple(1, 16) == 16
        assert round_up_to_multiple(16, 16) == 16
        assert round_up_to_multiple(17, 16) == 32


class TestComputeTargetShapePerExperiment:
    def test_takes_max_across_experiments_not_pooled(self):
        # Experiment A: narrow RT, wide IM (uniform, so percentile == the
        # exact value, no interpolation surprises). Experiment B: wide RT,
        # narrow IM. Pooling both into one distribution would understate each
        # experiment's own worst axis.
        shapes_by_experiment = {
            "A": [(20, 500)] * 20,
            "B": [(200, 50)] * 20,
        }
        target = compute_target_shape_per_experiment(
            shapes_by_experiment, percentile=99.0, multiple=16
        )
        # Must cover experiment B's RT extent (200) AND experiment A's IM
        # extent (500), even though neither single experiment has both.
        assert target[0] >= 200
        assert target[1] >= 500
        assert target[0] % 16 == 0
        assert target[1] % 16 == 0

    def test_unbalanced_group_sizes_dont_dilute_the_minority_tail(self):
        # Experiment B is under 1% of the pooled sample count -- pooling
        # dilutes its RT extent below the p99 threshold entirely; per-
        # experiment percentiles can't lose a minority experiment's tail
        # that way since each experiment's own percentile is computed alone.
        shapes_by_experiment = {
            "A": [(20, 100)] * 19800,
            "B": [(200, 50)] * 100,
        }
        pooled_target = compute_target_shape(
            shapes_by_experiment["A"] + shapes_by_experiment["B"],
            percentile=99.0,
            multiple=16,
        )
        per_exp_target = compute_target_shape_per_experiment(
            shapes_by_experiment, percentile=99.0, multiple=16
        )
        assert pooled_target[0] < 200  # B's RT tail is diluted away
        assert per_exp_target[0] >= 200  # per-experiment catches it
