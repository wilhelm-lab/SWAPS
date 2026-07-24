import numpy as np
import pandas as pd
import pytest

from swaps.peak_detection_2d.dataset.gt_correction import (
    FEATURE_COLUMNS,
    ReviewStore,
    compute_watershed_crop,
    default_kept_label_ids,
    extract_bbox_from_mask,
    mask_from_label_ids,
    mask_from_polygon_vertices,
    regionprops_features,
    select_diverse_subset,
)


def _two_blob_crop(h=40, w=40):
    """Synthetic crop with two well-separated Gaussian blobs, and the
    (peak_consensus_kwargs, hint_crop marking blob 1's apex) needed to
    reliably watershed-split them into >= 2 labels."""
    yy, xx = np.mgrid[0:h, 0:w]
    center1, center2 = (h // 2, w // 4), (h // 2, 3 * w // 4)
    blob1 = 5.0 * np.exp(-(((yy - center1[0]) ** 2 + (xx - center1[1]) ** 2) / (2 * 3.0**2)))
    blob2 = 5.0 * np.exp(-(((yy - center2[0]) ** 2 + (xx - center2[1]) ** 2) / (2 * 3.0**2)))
    crop = (blob1 + blob2 + 0.01).astype(np.float32)
    hint_crop = np.zeros((h, w), dtype=np.float32)
    hint_crop[center1] = 1.0
    kwargs = {
        "int_threshold": 1.0,
        "h_rel": 0.1,
        "norm_percentile": 95,
        "compactness": 0.001,
        "normalize_before_hmaxima": True,
    }
    return crop, hint_crop, kwargs, center1, center2


class TestExtractBboxFromMask:
    def test_recovers_the_filled_rectangle(self):
        mask = np.zeros((10, 10))
        mask[2:5, 3:7] = 1
        assert extract_bbox_from_mask(mask) == (2, 3, 5, 7)

    def test_raises_on_empty_mask(self):
        with pytest.raises(ValueError):
            extract_bbox_from_mask(np.zeros((5, 5)))


class TestComputeWatershedCrop:
    def test_splits_two_separated_blobs(self):
        crop, _hint, kwargs, *_ = _two_blob_crop()
        cropped, labels = compute_watershed_crop(crop, (0, 0, *crop.shape), kwargs)
        assert cropped.shape == labels.shape == crop.shape
        assert labels.max() >= 2

    def test_uses_bbox_offset_not_padding(self):
        crop, _hint, kwargs, *_ = _two_blob_crop()
        full = np.zeros((80, 80), dtype=np.float32)
        full[10:50, 5:45] = crop
        cropped, labels = compute_watershed_crop(full, (10, 5, 50, 45), kwargs)
        np.testing.assert_array_equal(cropped, crop)
        assert labels.shape == crop.shape


class TestMaskFromLabelIds:
    def test_embeds_selected_labels_at_bbox_offset(self):
        labels = np.array([[1, 1, 2], [1, 1, 2], [0, 0, 2]])
        mask = mask_from_label_ids(labels, {1}, bbox=(2, 3, 5, 6), full_shape=(6, 8))
        expected = np.zeros((6, 8), dtype=np.float32)
        expected[2:4, 3:5] = 1.0
        np.testing.assert_array_equal(mask, expected)

    def test_empty_selection_gives_all_zero_mask(self):
        labels = np.array([[1, 2], [1, 2]])
        mask = mask_from_label_ids(labels, set(), bbox=(0, 0, 2, 2), full_shape=(4, 4))
        assert mask.sum() == 0


class TestMaskFromPolygonVertices:
    def test_rectangle_polygon_matches_expected_footprint(self):
        # bbox-local rectangle covering rows[1,4), cols[1,4) out of a 5x5 crop
        vertices = [(1, 1), (4, 1), (4, 4), (1, 4)]
        mask = mask_from_polygon_vertices(vertices, bbox=(10, 10, 15, 15), full_shape=(20, 20))
        assert mask[12, 12] == 1.0  # clearly inside
        assert mask[10, 10] == 0.0  # clearly outside (corner of bbox, outside polygon)
        # exact area depends on boundary-pixel inclusion semantics; just check
        # it's roughly the 3x3-4x4 rectangle, not e.g. leaking into the full crop
        assert 9 <= mask.sum() <= 16

    def test_rejects_degenerate_polygon(self):
        with pytest.raises(ValueError):
            mask_from_polygon_vertices([(0, 0), (1, 1)], bbox=(0, 0, 5, 5), full_shape=(5, 5))


class TestDefaultKeptLabelIds:
    def test_returns_labels_touching_any_hint_pixel(self):
        labels = np.array([[1, 1, 2], [1, 1, 2]])
        hint_crop = np.zeros((2, 3))
        hint_crop[0, 2] = 1  # sits on label 2
        assert default_kept_label_ids(labels, hint_crop) == {2}

    def test_ignores_hints_on_background(self):
        labels = np.array([[0, 1], [0, 1]])
        hint_crop = np.zeros((2, 2))
        hint_crop[0, 0] = 1  # background pixel
        assert default_kept_label_ids(labels, hint_crop) == set()


class TestRegionpropsFeatures:
    def test_feature_columns_and_touches_hint(self):
        labels = np.zeros((10, 10), dtype=int)
        labels[1:4, 1:4] = 1  # 3x3 region, area 9, RT (row) range [1, 4)
        labels[6:8, 6:9] = 2  # 2x3 region, area 6, RT range [6, 8) -- disjoint from region 1
        labels[2:5, 6:9] = 3  # 3x3 region, RT range [2, 5) -- half-overlaps region 1's RT range
        crop_image = np.full((10, 10), 0.1, dtype=np.float32)
        crop_image[1:4, 1:4] = 2.0
        crop_image[6:8, 6:9] = 1.0
        crop_image[2:5, 6:9] = 1.5
        hint_crop = np.zeros((10, 10))
        hint_crop[2, 2] = 1.0  # inside region 1

        feats = regionprops_features(crop_image, labels, hint_crop)
        assert list(feats.columns) == FEATURE_COLUMNS
        assert set(feats.index) == {1, 2, 3}
        assert feats.loc[1, "touches_hint"] == 1.0
        assert feats.loc[2, "touches_hint"] == 0.0
        assert feats.loc[3, "touches_hint"] == 0.0
        assert feats.loc[1, "area_frac"] == pytest.approx(9 / 100)
        assert feats.loc[2, "area_frac"] == pytest.approx(6 / 100)
        # RT-axis IoU against the touches_hint region (region 1): itself is
        # 1.0, the disjoint-RT region is 0.0, the half-overlapping one is
        # in between (IoU([1,4), [2,5)) = 2 / 4 = 0.5).
        assert feats.loc[1, "hint_rt_overlap_frac"] == pytest.approx(1.0)
        assert feats.loc[2, "hint_rt_overlap_frac"] == pytest.approx(0.0)
        assert feats.loc[3, "hint_rt_overlap_frac"] == pytest.approx(0.5)
        assert (feats["n_regions_in_crop"] == 3).all()

    def test_no_hint_anchor_gives_zero_overlap(self):
        labels = np.array([[1, 1, 2, 2]])
        feats = regionprops_features(np.ones((1, 4)), labels, np.zeros((1, 4)))
        assert (feats["hint_rt_overlap_frac"] == 0.0).all()

    def test_empty_labels_gives_empty_frame(self):
        labels = np.zeros((5, 5), dtype=int)
        feats = regionprops_features(np.zeros((5, 5)), labels, np.zeros((5, 5)))
        assert feats.empty
        assert list(feats.columns) == FEATURE_COLUMNS


class TestSelectDiverseSubset:
    def test_returns_whole_pool_when_pool_not_larger_than_n_samples(self):
        pool = pd.DataFrame(
            {
                "bbox_area": [1, 2, 3],
                "aspect_ratio": [1, 1, 1],
                "max_intensity": [1, 1, 1],
                "n_hint_anchors": [1, 1, 1],
                "n_watershed_regions": [1, 1, 1],
            }
        )
        out = select_diverse_subset(pool, n_samples=5)
        assert len(out) == 3

    def test_spreads_selection_across_well_separated_clusters(self):
        rng = np.random.default_rng(0)
        groups = {"A": -10.0, "B": 0.0, "C": 10.0}
        rows = []
        for name, center in groups.items():
            for _ in range(20):
                rows.append(
                    {
                        "bbox_area": center + rng.normal(0, 0.1),
                        "aspect_ratio": center + rng.normal(0, 0.1),
                        "max_intensity": center + rng.normal(0, 0.1),
                        "n_hint_anchors": center + rng.normal(0, 0.1),
                        "n_watershed_regions": center + rng.normal(0, 0.1),
                        "true_group": name,
                    }
                )
        pool = pd.DataFrame(rows)
        selected = select_diverse_subset(pool, n_samples=9, n_clusters=3, random_seed=0)
        assert len(selected) == 9
        assert set(selected["true_group"]) == {"A", "B", "C"}


class TestReviewStore:
    def test_record_and_reload_roundtrip(self, tmp_path):
        store = ReviewStore(str(tmp_path), reviewer="tester")
        crop_image = np.random.default_rng(0).random((5, 6)).astype(np.float32)
        labels = np.ones((5, 6), dtype=int)
        hint_crop = np.zeros((5, 6), dtype=np.float32)
        corrected_mask = np.zeros((20, 20), dtype=np.float32)
        corrected_mask[2:7, 3:9] = 1.0

        assert not store.is_done("expA", 123)
        store.record_reviewed(
            "expA", 123, "labels",
            crop_image=crop_image, watershed_labels=labels, hint_crop=hint_crop,
            bbox=(2, 3, 7, 9), full_shape=(20, 20), corrected_mask=corrected_mask,
        )
        assert store.is_done("expA", 123)

        manifest = store.load_manifest()
        row = manifest[(manifest.source_experiment == "expA") & (manifest.mz_rank == 123)].iloc[0]
        assert row["status"] == "reviewed"
        assert row["method"] == "labels"
        assert row["reviewer"] == "tester"

        sample = store.load_sample_npz("expA", 123)
        np.testing.assert_array_equal(sample["crop_image"], crop_image)
        np.testing.assert_array_equal(sample["corrected_mask"], corrected_mask)
        np.testing.assert_array_equal(sample["bbox"], np.array([2, 3, 7, 9]))

        store.record_discarded("expA", 456)
        manifest = store.load_manifest()
        assert len(manifest) == 2
        discarded_row = manifest[manifest.mz_rank == 456].iloc[0]
        assert discarded_row["status"] == "discarded"
        assert pd.isna(discarded_row["method"])

    def test_re_recording_replaces_prior_row(self, tmp_path):
        store = ReviewStore(str(tmp_path))
        store.record_discarded("expA", 1)
        store.record_reviewed(
            "expA", 1, "polygon",
            crop_image=np.zeros((3, 3), dtype=np.float32),
            watershed_labels=np.zeros((3, 3), dtype=int),
            hint_crop=np.zeros((3, 3), dtype=np.float32),
            bbox=(0, 0, 3, 3), full_shape=(3, 3),
            corrected_mask=np.zeros((3, 3), dtype=np.float32),
        )
        manifest = store.load_manifest()
        assert len(manifest) == 1
        assert manifest.iloc[0]["status"] == "reviewed"
