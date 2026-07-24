"""Human-in-the-loop ground-truth mask correction for CNN consensus segmentation.

The bulk-built ground truth (prepare_dataset.py) uses FragPipe combined_ion's
RT/IM window as a straight bounding box -- coarser than the true peak
footprint, so masks systematically over-segment (include background the true
peak doesn't touch). This module turns manually correcting that into a fast
selection task instead of pixel painting: watershed-segment the *existing*
bbox crop (no padding -- the true peak is assumed to fall entirely inside
it; if it doesn't, discard the sample instead of trying to correct it) with
the exact production peak_consensus_kwargs, and let a reviewer pick which
resulting label(s) belong to the true peak. A free-form polygon override
covers the minority of cases where no label combination matches the true
(possibly irregular) footprint.

Corrections recorded here (via ReviewStore) also double as training data for
a "region keep/discard" classifier (label_correction_model.py), so a few
hundred manual reviews can bootstrap corrected masks across the rest of an
experiment instead of requiring every sample to be hand-reviewed.
"""

import logging
import os
import time

import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from skimage.measure import regionprops_table
from sklearn.cluster import KMeans

from postprocessing.image_processing import detect_2d_peak_with_watershed

from .prepare_dataset import (
    GroundTruthDatapoint,
    build_experiment_ground_truth,
    load_experiment_context,
)

Logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "area_frac",
    "touches_hint",
    "hint_dist_frac",
    "hint_rt_overlap_frac",
    "mean_intensity_frac",
    "max_intensity_frac",
    "eccentricity",
    "solidity",
    "extent",
    "touches_border",
    "n_regions_in_crop",
]

_POOL_FEATURE_COLUMNS = [
    "bbox_area",
    "aspect_ratio",
    "max_intensity",
    "n_hint_anchors",
    "n_watershed_regions",
]

_MANIFEST_COLUMNS = [
    "source_experiment",
    "mz_rank",
    "status",
    "method",
    "reviewer",
    "timestamp",
]


def extract_bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    """(row0, col0, row1, col1) tight bounding box of a binary mask's
    nonzero extent -- prepare_dataset's ground-truth mask is already exactly
    this rectangle (see _project_run_boundary_to_bbox), so this just
    recovers the 4 corners it was filled from."""
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        raise ValueError("mask has no nonzero pixels; no bbox to extract.")
    return int(rows.min()), int(cols.min()), int(rows.max()) + 1, int(cols.max()) + 1


def compute_watershed_crop(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    peak_consensus_kwargs: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop `image` to `bbox` (no padding) and watershed-segment the crop
    with the same peak_consensus_kwargs match_features.py's own
    segment_consensus_from_aligned uses in production, so labels line up
    with what the pipeline would produce at inference time.

    Returns (crop_image, watershed_labels) -- both bbox-local, same shape.
    watershed_labels is skimage-style: 0 = background, each basin a distinct
    positive int (the 4th tuple element of detect_2d_peak_with_watershed,
    its full multi-marker segmentation -- see image_processing.py).
    """
    row0, col0, row1, col1 = bbox
    crop = image[row0:row1, col0:col1]
    kwargs = dict(peak_consensus_kwargs or {})
    _, _, _, labels, _ = detect_2d_peak_with_watershed(
        crop,
        int_threshold=kwargs.get("int_threshold", 0.5),
        h_rel=kwargs.get("h_rel", 0.15),
        norm_percentile=kwargs.get("norm_percentile", 95),
        compactness=kwargs.get("compactness", 0.001),
        normalize_before_hmaxima=kwargs.get("normalize_before_hmaxima", True),
    )
    return crop, labels


def mask_from_label_ids(
    labels: np.ndarray,
    label_ids: set[int] | list[int],
    bbox: tuple[int, int, int, int],
    full_shape: tuple[int, int],
) -> np.ndarray:
    """Full-consensus-image-shape binary mask: union of the given
    bbox-local watershed label ids, embedded back at `bbox`'s offset."""
    row0, col0, row1, col1 = bbox
    mask = np.zeros(full_shape, dtype=np.float32)
    if label_ids:
        mask[row0:row1, col0:col1] = np.isin(labels, list(label_ids)).astype(np.float32)
    return mask


def mask_from_polygon_vertices(
    vertices_xy: list[tuple[float, float]],
    bbox: tuple[int, int, int, int],
    full_shape: tuple[int, int],
) -> np.ndarray:
    """Rasterize a freehand/polygon override -- the escape hatch for
    corrections no watershed label combination can represent (irregular
    boundary) -- into a full-consensus-image-shape binary mask.

    `vertices_xy` are bbox-local (x, y) = (col, row) coordinates, matching
    both matplotlib's PolygonSelector/LassoSelector output and imshow's own
    (origin="lower") pixel-coordinate convention.
    """
    row0, col0, row1, col1 = bbox
    h, w = row1 - row0, col1 - col0
    if len(vertices_xy) < 3:
        raise ValueError("polygon needs at least 3 vertices.")
    yy, xx = np.mgrid[0:h, 0:w]
    points = np.column_stack((xx.ravel(), yy.ravel()))
    inside = MplPath(vertices_xy).contains_points(points).reshape(h, w)
    mask = np.zeros(full_shape, dtype=np.float32)
    mask[row0:row1, col0:col1] = inside.astype(np.float32)
    return mask


def default_kept_label_ids(labels: np.ndarray, hint_crop: np.ndarray) -> set[int]:
    """Baseline heuristic starting selection: every watershed label
    containing at least one known-anchor (hint channel) pixel -- a
    zero-review-cost default a reviewer only has to correct when it's
    wrong, and also label_correction_model's baseline to beat."""
    rows, cols = np.nonzero(hint_crop)
    return {int(labels[r, c]) for r, c in zip(rows, cols) if labels[r, c] > 0}


def regionprops_features(
    crop_image: np.ndarray, labels: np.ndarray, hint_crop: np.ndarray
) -> pd.DataFrame:
    """Per-watershed-label feature table (index = label id), used both to
    seed the reviewer's default selection's plausibility and as
    label_correction_model's classifier input. Every feature is normalized
    by this crop's own size/intensity/diagonal, so they transfer across the
    wide range of peptide/window sizes in the dataset instead of being tied
    to one image's absolute scale.

    `hint_rt_overlap_frac` in particular targets watershed *over*-segmenting
    one true peak along ion mobility (the array's column axis) into several
    regions that share the same retention-time (row axis) window: a true
    peak's RT elution window is a property of the whole peak, so an
    IM-adjacent fragment of the same real peak should have an RT range that
    closely overlaps the hint-touching region's RT range even when it
    doesn't touch the hint pixel itself, while an unrelated neighboring
    peak usually won't. It's the IoU (not plain containment) between a
    region's own [row_min, row_max) and the union of every touches_hint
    region's own RT range, in this crop's own row-index units, so it stays
    comparable across crops of very different height."""
    if labels.max() == 0:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    props = regionprops_table(
        labels,
        intensity_image=crop_image,
        properties=(
            "label",
            "area",
            "centroid",
            "mean_intensity",
            "max_intensity",
            "eccentricity",
            "solidity",
            "extent",
            "bbox",
        ),
    )
    df = pd.DataFrame(props).set_index("label")

    h, w = labels.shape
    crop_area = h * w
    crop_max_intensity = float(crop_image.max()) or 1.0
    diag = float(np.hypot(h, w)) or 1.0

    hint_rc = np.argwhere(hint_crop > 0)
    touching = default_kept_label_ids(labels, hint_crop)

    if hint_rc.size:
        centroids = df[["centroid-0", "centroid-1"]].to_numpy()
        dists = np.min(
            np.hypot(
                centroids[:, 0:1] - hint_rc[:, 0][None, :],
                centroids[:, 1:2] - hint_rc[:, 1][None, :],
            ),
            axis=1,
        )
    else:
        dists = np.full(len(df), diag)

    row_min = df["bbox-0"].to_numpy(dtype=float)
    row_max = df["bbox-2"].to_numpy(dtype=float)
    if touching:
        touching_mask = df.index.isin(touching)
        hint_row_min = row_min[touching_mask].min()
        hint_row_max = row_max[touching_mask].max()
        inter = np.clip(
            np.minimum(row_max, hint_row_max) - np.maximum(row_min, hint_row_min), 0, None
        )
        union = np.maximum(row_max, hint_row_max) - np.minimum(row_min, hint_row_min)
        rt_overlap_frac = inter / np.maximum(union, 1e-8)
    else:
        rt_overlap_frac = np.zeros(len(df))

    touches_border = (
        (df["bbox-0"] <= 0) | (df["bbox-1"] <= 0) | (df["bbox-2"] >= h) | (df["bbox-3"] >= w)
    ).astype(float)

    return pd.DataFrame(
        {
            "area_frac": df["area"] / crop_area,
            "touches_hint": df.index.isin(touching).astype(float),
            "hint_dist_frac": dists / diag,
            "hint_rt_overlap_frac": rt_overlap_frac,
            "mean_intensity_frac": df["mean_intensity"] / crop_max_intensity,
            "max_intensity_frac": df["max_intensity"] / crop_max_intensity,
            "eccentricity": df["eccentricity"],
            "solidity": df["solidity"],
            "extent": df["extent"],
            "touches_border": touches_border,
            "n_regions_in_crop": float(len(df)),
        },
        index=df.index,
    )


def candidate_pool_features(dp: GroundTruthDatapoint, peak_consensus_kwargs: dict) -> dict:
    """Cheap per-candidate scalar features used only to spread the manual
    review sample across the dataset's diversity -- not the classifier's own
    (region-level) feature set, see regionprops_features."""
    bbox = extract_bbox_from_mask(dp.mask)
    row0, col0, row1, col1 = bbox
    h, w = row1 - row0, col1 - col0
    crop, labels = compute_watershed_crop(dp.image, bbox, peak_consensus_kwargs)
    return {
        "bbox_area": h * w,
        "aspect_ratio": h / max(w, 1),
        "max_intensity": float(crop.max()) if crop.size else 0.0,
        "n_hint_anchors": int(dp.hint_channel.sum()),
        "n_watershed_regions": int(labels.max()),
    }


def select_diverse_subset(
    pool: pd.DataFrame,
    n_samples: int,
    feature_cols: list[str] = _POOL_FEATURE_COLUMNS,
    n_clusters: int = 20,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Pure clustering + round-robin selection over an already-featurized
    candidate pool -- split out from sample_diverse_review_batch (which
    builds `pool` via real per-experiment I/O) so the selection logic itself
    is unit-testable without a real SWAPS experiment directory.

    Standardizes `feature_cols`, k-means clusters into `n_clusters` groups,
    then round-robins across clusters (shuffled within each) until
    `n_samples` are picked -- spreads the review batch across the pool's
    feature diversity instead of reproducing its natural skew (as plain
    random sampling would), so a classifier trained on it later isn't blind
    to whatever regime is under-represented in the raw pool.
    """
    if len(pool) <= n_samples:
        return pool.reset_index(drop=True)

    rng = np.random.default_rng(random_seed)
    X = pool[feature_cols].to_numpy(dtype=float)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    k = min(n_clusters, len(pool))
    cluster_ids = KMeans(n_clusters=k, random_state=random_seed, n_init=10).fit_predict(X)
    pool = pool.assign(cluster=cluster_ids)

    per_cluster = {c: rng.permutation(sub.index.to_numpy()) for c, sub in pool.groupby("cluster")}
    order = list(per_cluster.keys())
    rng.shuffle(order)
    cursors = {c: 0 for c in order}
    picked = []
    while len(picked) < n_samples:
        progressed = False
        for c in order:
            idxs = per_cluster[c]
            cur = cursors[c]
            if cur < len(idxs):
                picked.append(idxs[cur])
                cursors[c] = cur + 1
                progressed = True
                if len(picked) >= n_samples:
                    break
        if not progressed:
            break

    Logger.info(
        "Selected %d/%d candidates across %d clusters (pool=%d).",
        len(picked), n_samples, k, len(pool),
    )
    return pool.loc[picked].reset_index(drop=True)


def sample_diverse_review_batch(
    swaps_dirs: list[str],
    n_samples: int,
    pool_size_per_experiment: int = 500,
    n_clusters: int = 20,
    batch_size: int = 200,
    random_seed: int = 42,
    include_decoys: bool = False,
) -> pd.DataFrame:
    """Pick `n_samples` mz_ranks to hand-review across `swaps_dirs`, spread
    over the feature diversity of a randomly-drawn pool (see
    select_diverse_subset) rather than picked randomly/sequentially.

    Returns a plan DataFrame [source_experiment, mz_rank] -- feed straight
    to review_widget.GroundTruthReviewSession.
    """
    rows = []
    for swaps_dir in swaps_dirs:
        source_experiment = os.path.basename(swaps_dir.rstrip("/"))
        ctx = load_experiment_context(swaps_dir)
        peak_consensus_kwargs = dict(ctx["processing_kwargs"].get("peak_consensus_kwargs", {}))
        dps = build_experiment_ground_truth(
            swaps_dir,
            pool_size_per_experiment,
            batch_size=batch_size,
            random_seed=random_seed,
            include_decoys=include_decoys,
        )
        for dp in dps:
            feats = candidate_pool_features(dp, peak_consensus_kwargs)
            feats["source_experiment"] = source_experiment
            feats["mz_rank"] = dp.mz_rank
            rows.append(feats)

    pool = pd.DataFrame(rows)
    if pool.empty:
        raise ValueError("No candidates found across the given experiments.")

    selected = select_diverse_subset(
        pool, n_samples, n_clusters=n_clusters, random_seed=random_seed
    )
    return selected[["source_experiment", "mz_rank"]].reset_index(drop=True)


class ReviewStore:
    """Filesystem-backed, resumable storage for one review session's output:
    a manifest parquet (one row per reviewed-or-discarded mz_rank) plus one
    .npz per accepted correction (crop_image, watershed_labels, hint_crop,
    bbox, full_shape, corrected_mask) -- everything label_correction_model.py
    needs to re-derive per-region training targets without recomputing the
    consensus image.

    One .npz per sample (not one shared HDF5 dataset) on purpose: a review
    session is interactive and resumable across kernel restarts / remote
    disconnects, so each accept must durably commit on its own rather than
    batching writes like prepare_dataset's bulk HDF5 writer does.
    """

    def __init__(self, output_dir: str, reviewer: str = "unknown"):
        self.output_dir = output_dir
        self.masks_dir = os.path.join(output_dir, "masks")
        self.manifest_path = os.path.join(output_dir, "manifest.parquet")
        self.reviewer = reviewer
        os.makedirs(self.masks_dir, exist_ok=True)

    def load_manifest(self) -> pd.DataFrame:
        if os.path.exists(self.manifest_path):
            return pd.read_parquet(self.manifest_path)
        return pd.DataFrame(columns=_MANIFEST_COLUMNS)

    def is_done(self, source_experiment: str, mz_rank: int) -> bool:
        manifest = self.load_manifest()
        return bool(
            (
                (manifest["source_experiment"] == source_experiment)
                & (manifest["mz_rank"] == mz_rank)
            ).any()
        )

    def _npz_path(self, source_experiment: str, mz_rank: int) -> str:
        return os.path.join(self.masks_dir, f"{source_experiment}__{mz_rank}.npz")

    def record_reviewed(
        self,
        source_experiment: str,
        mz_rank: int,
        method: str,
        crop_image: np.ndarray,
        watershed_labels: np.ndarray,
        hint_crop: np.ndarray,
        bbox: tuple[int, int, int, int],
        full_shape: tuple[int, int],
        corrected_mask: np.ndarray,
    ) -> None:
        np.savez_compressed(
            self._npz_path(source_experiment, mz_rank),
            crop_image=crop_image,
            watershed_labels=watershed_labels,
            hint_crop=hint_crop,
            bbox=np.array(bbox),
            full_shape=np.array(full_shape),
            corrected_mask=corrected_mask,
        )
        self._append_manifest_row(source_experiment, mz_rank, "reviewed", method)

    def record_discarded(self, source_experiment: str, mz_rank: int) -> None:
        self._append_manifest_row(source_experiment, mz_rank, "discarded", None)

    def _append_manifest_row(
        self, source_experiment: str, mz_rank: int, status: str, method: str | None
    ) -> None:
        manifest = self.load_manifest()
        manifest = manifest[
            ~(
                (manifest["source_experiment"] == source_experiment)
                & (manifest["mz_rank"] == mz_rank)
            )
        ]
        new_row = pd.DataFrame(
            [
                {
                    "source_experiment": source_experiment,
                    "mz_rank": mz_rank,
                    "status": status,
                    "method": method,
                    "reviewer": self.reviewer,
                    "timestamp": time.time(),
                }
            ]
        )
        manifest = new_row if manifest.empty else pd.concat([manifest, new_row], ignore_index=True)
        manifest.to_parquet(self.manifest_path, index=False)

    def load_sample_npz(self, source_experiment: str, mz_rank: int) -> dict:
        with np.load(self._npz_path(source_experiment, mz_rank)) as f:
            return {k: f[k] for k in f.files}
