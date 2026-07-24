import numpy as np
import pandas as pd
import pytest

from swaps.peak_detection_2d.dataset.gt_correction import FEATURE_COLUMNS, ReviewStore
from swaps.peak_detection_2d.dataset.label_correction_model import (
    _region_targets_from_mask,
    build_region_training_table,
    correct_ground_truth_datapoint,
    train_region_keep_classifier,
)
from swaps.peak_detection_2d.dataset.prepare_dataset import GroundTruthDatapoint


def _populated_store(tmp_path):
    """A ReviewStore with 2 reviewed samples, each 2 watershed regions, one
    kept (fully covered by corrected_mask) and one discarded (uncovered)."""
    store = ReviewStore(str(tmp_path))
    for mz_rank in (1, 2):
        labels = np.array([[1, 1, 2, 2]] * 4)  # region 1 cols[0:2], region 2 cols[2:4]
        crop_image = np.full((4, 4), 0.5, dtype=np.float32)
        crop_image[:, 0:2] = 2.0
        hint_crop = np.zeros((4, 4), dtype=np.float32)
        hint_crop[1, 1] = 1.0  # inside region 1
        corrected_mask = np.zeros((10, 10), dtype=np.float32)
        corrected_mask[0:4, 0:2] = 1.0  # covers only region 1 -> region 1 kept, region 2 not
        store.record_reviewed(
            "expA", mz_rank, "labels",
            crop_image=crop_image, watershed_labels=labels, hint_crop=hint_crop,
            bbox=(0, 0, 4, 4), full_shape=(10, 10), corrected_mask=corrected_mask,
        )
    return store


class TestRegionTargetsFromMask:
    def test_majority_overlap_is_kept(self):
        labels = np.array([[1, 1, 2, 2]])
        corrected_mask_crop = np.array([[1, 1, 0, 0]])
        targets = _region_targets_from_mask(labels, corrected_mask_crop)
        assert targets == {1: 1, 2: 0}

    def test_partial_overlap_below_threshold_is_dropped(self):
        labels = np.array([[1, 1, 1, 1]])
        corrected_mask_crop = np.array([[1, 0, 0, 0]])  # 25% overlap
        assert _region_targets_from_mask(labels, corrected_mask_crop) == {1: 0}


class TestBuildRegionTrainingTable:
    def test_two_regions_per_sample_with_expected_targets(self, tmp_path):
        store = _populated_store(tmp_path)
        table = build_region_training_table(store)
        assert len(table) == 4  # 2 samples x 2 regions
        assert set(table.columns) >= set(FEATURE_COLUMNS) | {"keep", "source_experiment", "mz_rank"}
        assert set(table["mz_rank"]) == {1, 2}
        # region 1 (touches the hint) always kept, region 2 (doesn't) never
        assert (table.loc[table["touches_hint"] == 1.0, "keep"] == 1).all()
        assert (table.loc[table["touches_hint"] == 0.0, "keep"] == 0).all()

    def test_empty_store_gives_empty_table(self, tmp_path):
        store = ReviewStore(str(tmp_path))
        table = build_region_training_table(store)
        assert table.empty


class TestTrainRegionKeepClassifier:
    def test_fits_and_reports_expected_keys(self):
        rng = np.random.default_rng(0)
        n_images, n_regions = 8, 3
        rows = []
        for img_i in range(n_images):
            for region_i in range(n_regions):
                touches_hint = float(region_i == 0)
                rows.append(
                    {
                        **{col: rng.random() for col in FEATURE_COLUMNS},
                        "touches_hint": touches_hint,
                        "keep": int(touches_hint == 1.0),
                        "source_experiment": "expA",
                        "mz_rank": img_i,
                    }
                )
        table = pd.DataFrame(rows)
        model, report = train_region_keep_classifier(table, n_splits=4, random_seed=0)
        assert hasattr(model, "predict_proba")
        assert report["n_regions"] == len(table)
        assert report["n_images"] == n_images
        assert report["cv_accuracy"] is not None
        assert report["baseline_touches_hint_accuracy"] == pytest.approx(1.0)

    def test_raises_on_all_nan_targets(self):
        table = pd.DataFrame({col: [0.1, 0.2] for col in FEATURE_COLUMNS})
        table["keep"] = [np.nan, np.nan]
        table["source_experiment"] = ["expA", "expA"]
        table["mz_rank"] = [1, 2]
        with pytest.raises(ValueError):
            train_region_keep_classifier(table)


class TestCorrectGroundTruthDatapoint:
    def _make_datapoint(self):
        h, w = 40, 40
        full_image = np.full((h, w), 0.01, dtype=np.float32)
        yy, xx = np.mgrid[0:h, 0:w]
        center1, center2 = (20, 10), (20, 30)
        for center in (center1, center2):
            full_image += 5.0 * np.exp(
                -(((yy - center[0]) ** 2 + (xx - center[1]) ** 2) / (2 * 3.0**2))
            )
        mask = np.zeros((h, w), dtype=np.float32)
        mask[5:35, 0:40] = 1.0  # old (over-sized) bbox mask spanning both blobs
        hint_channel = np.zeros((h, w), dtype=np.float32)
        hint_channel[center1] = 1.0
        return GroundTruthDatapoint(
            mz_rank=1,
            image=full_image.astype(np.float32),
            hint_channel=hint_channel,
            mask=mask,
            contributing_runs=["run1"],
            reference_raw_file="run1",
        )

    def test_heuristic_path_shrinks_mask_to_bbox_interior(self):
        dp = self._make_datapoint()
        kwargs = {
            "int_threshold": 1.0, "h_rel": 0.1, "norm_percentile": 95,
            "compactness": 0.001, "normalize_before_hmaxima": True,
        }
        corrected_mask, confidence = correct_ground_truth_datapoint(dp, kwargs, model=None)
        assert corrected_mask.shape == dp.mask.shape
        assert confidence is None
        assert corrected_mask.sum() > 0
        # never extends outside the original bbox
        assert np.all(corrected_mask[dp.mask == 0] == 0)
        # only touches the hinted blob's region, not the whole old bbox
        assert corrected_mask.sum() < dp.mask.sum()
