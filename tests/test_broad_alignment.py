import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from swaps.postprocessing.broad_alignment import (
    ShiftLookup,
    build_shift_lookup,
    build_shift_table,
    calibrate_broad_alignment,
    load_shift_table,
    run_pairwise_calibration_alignment,
    select_calibration_peptides,
)

_RT_RANGE = (100, 139)  # 40 frames
_IM_RANGE = (0, 39)  # 40 im bins


def _blob_image(center, amp=10.0, sigma=3.0):
    n_rt = _RT_RANGE[1] - _RT_RANGE[0] + 1
    n_im = _IM_RANGE[1] - _IM_RANGE[0] + 1
    yy, xx = np.mgrid[0:n_im, 0:n_rt]
    im_c, rt_c = center
    return amp * np.exp(-(((yy - im_c) ** 2 + (xx - rt_c) ** 2) / (2 * sigma**2)))


def _write_activation_parquet(act_dir, images_by_mz_rank):
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


def _build_dict_ref(raw_files, rt_centers, blob_centers):
    """blob_centers: dict[mz_rank] -> dict[run] -> (im_c, rt_c) absolute frame/im index."""
    rows = []
    for mz_rank, rt_center in rt_centers.items():
        row = {
            "mz_rank": mz_rank,
            "RT_search_center": rt_center,
            "IM_search_center": 1.0,
            "reference_score": 100.0 - mz_rank,
            "n_identifications": 5,
            "score_std": 1.0,
        }
        for rf in raw_files:
            row[f"MS1_frame_idx_left_ref_{rf}"] = _RT_RANGE[0]
            row[f"MS1_frame_idx_right_ref_{rf}"] = _RT_RANGE[1]
            row[f"mobility_values_index_left_ref_{rf}"] = _IM_RANGE[0]
            row[f"mobility_values_index_right_ref_{rf}"] = _IM_RANGE[1]
            im_c, rt_c = blob_centers[mz_rank][rf]
            row[f"{rf}_MS1_frame_idx_exp"] = rt_c
            row[f"{rf}_mobility_values_index_exp"] = im_c
        rows.append(row)
    return pd.DataFrame(rows)


class TestSelectCalibrationPeptides:
    def test_requires_mz_rank_and_rt_column(self):
        df = pd.DataFrame({"RT_search_center": [1.0, 2.0]})
        with pytest.raises(KeyError):
            select_calibration_peptides(df, n_peptides=1)

    def test_falls_back_to_unranked_sampling_when_score_columns_absent(self):
        """dict_ref predating reference_score/n_identifications/score_std
        (added after some datasets' Stage 1 already ran) must not hard-fail
        calibration -- these columns are a ranking preference, not a
        requirement, since calibration only needs the peptides' own
        activation images (always present after Stage 2)."""
        df = pd.DataFrame({"mz_rank": [1, 2], "RT_search_center": [1.0, 2.0]})
        selected = select_calibration_peptides(df, n_peptides=2)
        assert set(selected) <= {1, 2}
        assert len(selected) > 0

    def test_prefers_higher_reference_score_within_rt_bin(self):
        df = pd.DataFrame(
            {
                "mz_rank": [1, 2, 3, 4],
                "RT_search_center": [1.0, 1.0, 1.0, 1.0],
                "reference_score": [10.0, 90.0, 50.0, 30.0],
                "n_identifications": [3, 3, 3, 3],
                "score_std": [1.0, 1.0, 1.0, 1.0],
            }
        )
        selected = select_calibration_peptides(df, n_peptides=2, n_rt_bins=1)
        assert set(selected.tolist()) == {2, 3}

    def test_covers_multiple_rt_bins(self):
        df = pd.DataFrame(
            {
                "mz_rank": [1, 2, 3, 4],
                "RT_search_center": [1.0, 2.0, 3.0, 4.0],
                "reference_score": [50.0, 50.0, 50.0, 50.0],
                "n_identifications": [3, 3, 3, 3],
                "score_std": [1.0, 1.0, 1.0, 1.0],
            }
        )
        selected = select_calibration_peptides(df, n_peptides=4, n_rt_bins=4)
        assert set(selected.tolist()) == {1, 2, 3, 4}


class TestPairwiseCalibrationAlignment:
    def test_recovers_known_shift_between_two_raw_files(self, tmp_path):
        raw_files = ["R1", "R2"]
        # _blob_image's center is (im_c, rt_c) in ITS OWN local array axes,
        # but parquet_df_to_dense_frame (used inside get_pept_act_from_parquet)
        # reconstructs images as (rt, im) -- transposed relative to that local
        # array. R2's blob is offset by (im: +3, rt: -2) relative to R1's in
        # local-array terms, which is (rt: -2, im: +3) in the reconstructed
        # image; the recovered shift is the reference's position minus the
        # match's, in (rt, im) order: (15-13, 15-18) = (2, -3). Verified
        # empirically against the real reconstructed images, not assumed.
        blob_centers = {
            1: {"R1": (15, 15), "R2": (18, 13)},
        }
        for rf in raw_files:
            act_dir = tmp_path / rf / "activation"
            img = _blob_image(blob_centers[1][rf])
            _write_activation_parquet(str(act_dir), {1: img})

        dict_ref = _build_dict_ref(raw_files, {1: 3.0}, blob_centers)
        samples = run_pairwise_calibration_alignment(
            np.array([1]), raw_files, str(tmp_path), dict_ref, max_workers=1
        )
        assert len(samples) == 2  # (R1->R2) and (R2->R1)
        r1_to_r2 = samples[
            (samples.reference_run == "R1") & (samples.matched_run == "R2")
        ].iloc[0]
        assert r1_to_r2["shift_rt"] == 2
        assert r1_to_r2["shift_im"] == -3
        assert r1_to_r2["template_matching_score"] > 0.9


class TestBuildShiftTable:
    def _samples(self, rows):
        return pd.DataFrame(rows)

    def test_joint_pair_mode_not_independent_axes(self):
        # (shift_rt, shift_im) pairs: (1, 5) x3, (2, 6) x2 -- independent
        # per-axis modes would synthesize (1, 5) anyway here, so also check
        # a case where independent modes would diverge from the joint mode.
        rows = [
            {"reference_run": "A", "matched_run": "B", "rt_position": 1.0,
             "shift_rt": 1, "shift_im": 5, "template_matching_score": 0.9},
            {"reference_run": "A", "matched_run": "B", "rt_position": 1.0,
             "shift_rt": 1, "shift_im": 5, "template_matching_score": 0.9},
            {"reference_run": "A", "matched_run": "B", "rt_position": 1.0,
             "shift_rt": 2, "shift_im": 5, "template_matching_score": 0.9},
            {"reference_run": "A", "matched_run": "B", "rt_position": 1.0,
             "shift_rt": 1, "shift_im": 6, "template_matching_score": 0.9},
        ]
        table = build_shift_table(
            self._samples(rows), confidence_frac=1.0, bin_width_minutes=10.0,
            min_samples_per_bin=1,
        )
        row = table.iloc[0]
        assert (row["shift_rt"], row["shift_im"]) == (1, 5)

    def test_confidence_frac_keeps_only_top_scores(self):
        rows = [
            {"reference_run": "A", "matched_run": "B", "rt_position": 1.0,
             "shift_rt": 1, "shift_im": 1, "template_matching_score": 0.9},
            {"reference_run": "A", "matched_run": "B", "rt_position": 1.0,
             "shift_rt": 99, "shift_im": 99, "template_matching_score": 0.1},
        ]
        table = build_shift_table(
            self._samples(rows), confidence_frac=0.5, bin_width_minutes=10.0,
            min_samples_per_bin=1,
        )
        row = table.iloc[0]
        assert (row["shift_rt"], row["shift_im"]) == (1, 1)

    def test_nan_score_rows_excluded(self):
        rows = [
            {"reference_run": "A", "matched_run": "B", "rt_position": 1.0,
             "shift_rt": 1, "shift_im": 1, "template_matching_score": 0.9},
            {"reference_run": "A", "matched_run": "B", "rt_position": 1.0,
             "shift_rt": 99, "shift_im": 99, "template_matching_score": np.nan},
        ]
        table = build_shift_table(
            self._samples(rows), confidence_frac=1.0, bin_width_minutes=10.0,
            min_samples_per_bin=1,
        )
        row = table.iloc[0]
        assert (row["shift_rt"], row["shift_im"]) == (1, 1)

    def test_sparse_bin_borrows_neighbor(self):
        # bin 0 gets 20 confident samples of (1,1); bin 1 gets a single (9,9)
        # sample -- too few to trust on its own -> should borrow bin 0's mode.
        rows = [
            {"reference_run": "A", "matched_run": "B", "rt_position": 0.1,
             "shift_rt": 1, "shift_im": 1, "template_matching_score": 0.9}
            for _ in range(20)
        ] + [
            {"reference_run": "A", "matched_run": "B", "rt_position": 1.1,
             "shift_rt": 9, "shift_im": 9, "template_matching_score": 0.9}
        ]
        table = build_shift_table(
            self._samples(rows), confidence_frac=1.0, bin_width_minutes=1.0,
            min_samples_per_bin=10, max_neighbor_search_bins=5,
        )
        sparse_bin = table[table["n_samples"] < 10]
        assert (sparse_bin["shift_rt"] == 1).all()
        assert (sparse_bin["shift_im"] == 1).all()
        assert sparse_bin["is_fallback"].all()

    def test_every_bin_in_range_is_filled(self):
        rows = [
            {"reference_run": "A", "matched_run": "B", "rt_position": rt,
             "shift_rt": 1, "shift_im": 1, "template_matching_score": 0.9}
            for rt in (0.1, 3.9)  # far apart -> several empty bins between
        ]
        table = build_shift_table(
            self._samples(rows), confidence_frac=1.0, bin_width_minutes=1.0,
            min_samples_per_bin=1,
        )
        assert table["rt_bin_index"].tolist() == list(range(len(table)))
        assert table["shift_rt"].notna().all()


class TestShiftLookup:
    def test_lookup_resolves_correct_bin(self):
        table = pd.DataFrame(
            [
                {"reference_run": "A", "matched_run": "B", "rt_bin_index": 0,
                 "rt_bin_start": 0.0, "rt_bin_end": 1.0, "shift_rt": 1,
                 "shift_im": 2, "n_samples": 10, "is_fallback": False},
                {"reference_run": "A", "matched_run": "B", "rt_bin_index": 1,
                 "rt_bin_start": 1.0, "rt_bin_end": 2.0, "shift_rt": 3,
                 "shift_im": 4, "n_samples": 10, "is_fallback": False},
            ]
        )
        lookup = build_shift_lookup(table)
        assert lookup.lookup("A", "B", 0.5) == (1, 2)
        assert lookup.lookup("A", "B", 1.5) == (3, 4)

    def test_lookup_clamps_out_of_range(self):
        table = pd.DataFrame(
            [
                {"reference_run": "A", "matched_run": "B", "rt_bin_index": 0,
                 "rt_bin_start": 1.0, "rt_bin_end": 2.0, "shift_rt": 1,
                 "shift_im": 2, "n_samples": 10, "is_fallback": False},
            ]
        )
        lookup = build_shift_lookup(table)
        assert lookup.lookup("A", "B", -5.0) == (1, 2)
        assert lookup.lookup("A", "B", 50.0) == (1, 2)

    def test_lookup_returns_none_for_unknown_pair(self):
        table = pd.DataFrame(
            [
                {"reference_run": "A", "matched_run": "B", "rt_bin_index": 0,
                 "rt_bin_start": 0.0, "rt_bin_end": 1.0, "shift_rt": 1,
                 "shift_im": 2, "n_samples": 10, "is_fallback": False},
            ]
        )
        lookup = build_shift_lookup(table)
        assert lookup.lookup("A", "C", 0.5) is None


class TestCalibrateBroadAlignment:
    def test_round_trip_writes_and_reloads_table(self, tmp_path):
        raw_files = ["R1", "R2"]
        blob_centers = {
            1: {"R1": (15, 15), "R2": (16, 14)},
            2: {"R1": (20, 20), "R2": (21, 19)},
        }
        for rf in raw_files:
            act_dir = tmp_path / rf / "activation"
            images = {mz: _blob_image(blob_centers[mz][rf]) for mz in blob_centers}
            _write_activation_parquet(str(act_dir), images)

        dict_ref = _build_dict_ref(raw_files, {1: 3.0, 2: 4.0}, blob_centers)
        dict_ref_path = tmp_path / "dict_ref.pkl"
        dict_ref.to_pickle(dict_ref_path)

        table = calibrate_broad_alignment(
            str(tmp_path),
            raw_files,
            str(dict_ref_path),
            n_peptides=2,
            confidence_frac=1.0,
            min_samples_per_bin=1,
            max_workers=1,
        )
        output_path = tmp_path / "broad_alignment_shift_table.parquet"
        assert output_path.exists()
        reloaded = load_shift_table(str(output_path))
        pd.testing.assert_frame_equal(table.reset_index(drop=True), reloaded)
