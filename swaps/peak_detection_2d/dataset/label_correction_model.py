"""Bootstraps ground-truth mask corrections across a full experiment from a
few hundred manual reviews (see gt_correction.py / review_widget.py).

Each manual review (gt_correction.ReviewStore) implicitly labels every
watershed region in that sample's bbox crop as "belongs to the true peak" or
not -- not just the image as a whole. That per-region signal is what's
learned here: a classifier over regionprops_features' cheap per-region
features, trained on the reviewed samples' regions and applied to every
other eligible mz_rank's own watershed-in-bbox regions. This turns "hand
correct thousands of masks" into "hand correct a few hundred, model the
rest" -- see build_region_training_table / train_region_keep_classifier /
apply_corrections_to_experiment.
"""

import logging
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

from .gt_correction import (
    FEATURE_COLUMNS,
    ReviewStore,
    candidate_pool_features,
    compute_watershed_crop,
    default_kept_label_ids,
    extract_bbox_from_mask,
    mask_from_label_ids,
    regionprops_features,
    select_diverse_subset,
)
from .prepare_dataset import (
    GroundTruthDatapoint,
    _eligible_mz_ranks,
    load_experiment_context,
    prepare_ground_truth_batch,
)

Logger = logging.getLogger(__name__)


def _region_targets_from_mask(
    labels: np.ndarray, corrected_mask_crop: np.ndarray, overlap_thres: float = 0.5
) -> dict[int, int]:
    """Per-label binary target: 1 if more than `overlap_thres` of that
    watershed region's area is covered by the reviewer's final corrected
    mask. Works identically whether that mask came from label-selection or
    a polygon override, so both review methods contribute training data."""
    targets = {}
    for lbl in np.unique(labels):
        if lbl == 0:
            continue
        region = labels == lbl
        targets[int(lbl)] = int(float(corrected_mask_crop[region].mean()) > overlap_thres)
    return targets


def build_region_training_table(review_store: ReviewStore) -> pd.DataFrame:
    """One row per watershed region across every 'reviewed' manifest entry:
    regionprops_features' feature columns + a `keep` target (see
    _region_targets_from_mask) + (source_experiment, mz_rank) grouping keys
    -- regions from the same image are correlated, so any train/val split
    must keep them together (see train_region_keep_classifier)."""
    manifest = review_store.load_manifest()
    reviewed = manifest[manifest["status"] == "reviewed"]
    tables = []
    for _, row in reviewed.iterrows():
        source_experiment, mz_rank = row["source_experiment"], int(row["mz_rank"])
        sample = review_store.load_sample_npz(source_experiment, mz_rank)
        labels = sample["watershed_labels"]
        crop_image = sample["crop_image"]
        hint_crop = sample["hint_crop"]
        row0, col0, row1, col1 = (int(x) for x in sample["bbox"])
        corrected_mask_crop = sample["corrected_mask"][row0:row1, col0:col1]

        feats = regionprops_features(crop_image, labels, hint_crop)
        if feats.empty:
            continue
        targets = _region_targets_from_mask(labels, corrected_mask_crop)
        feats = feats.copy()
        feats["keep"] = feats.index.map(targets)
        feats["source_experiment"] = source_experiment
        feats["mz_rank"] = mz_rank
        tables.append(feats)

    if not tables:
        return pd.DataFrame(columns=FEATURE_COLUMNS + ["keep", "source_experiment", "mz_rank"])
    return pd.concat(tables, ignore_index=False)


def _binary_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None
) -> dict[str, float | None]:
    """Metrics that stay informative when "keep" is the rare class (most
    watershed regions in a crop are background, not the true peak) --
    plain accuracy is dominated by the easy majority-class ("discard")
    predictions and looks deceptively high for *any* classifier (including
    the touches_hint baseline) that just gets the common case right.

    - balanced_accuracy: mean of per-class recall, so the majority class
      can't inflate the score on its own.
    - precision/recall/f1 for the positive ("keep") class specifically:
      precision = of predicted-keep regions, how many are truly the peak;
      recall = of truly-peak regions, how many did we find. These map
      directly onto downstream mask quality (low precision -> mask still
      too big; low recall -> mask missing peak area).
    - average_precision (area under the precision-recall curve): only
      computed when `y_proba` is given (a hard 0/1 baseline decision has no
      ranking to score). More sensitive than ROC-AUC to minority-class
      performance under heavy imbalance, since ROC-AUC's false-positive
      rate is diluted by a large true-negative pool.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_keep": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_keep": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_keep": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    has_both_classes = len(set(y_true)) > 1
    metrics["auc"] = (
        float(roc_auc_score(y_true, y_proba)) if y_proba is not None and has_both_classes else None
    )
    metrics["average_precision"] = (
        float(average_precision_score(y_true, y_proba))
        if y_proba is not None and has_both_classes
        else None
    )
    return metrics


def _mean_metrics(fold_metrics: list[dict[str, float | None]]) -> dict[str, float | None]:
    keys = fold_metrics[0].keys()
    return {
        k: (float(np.mean(vals)) if (vals := [m[k] for m in fold_metrics if m[k] is not None]) else None)
        for k in keys
    }


def train_region_keep_classifier(
    training_table: pd.DataFrame,
    n_splits: int = 5,
    random_seed: int = 42,
) -> tuple[GradientBoostingClassifier, dict]:
    """Trains a GradientBoostingClassifier on every reviewed region.

    Reports GroupKFold (grouped by (source_experiment, mz_rank), so no
    image's regions leak across train/val) held-out metrics for the
    classifier *and* for the zero-cost "touches_hint" baseline -- see
    _binary_classification_metrics for why plain accuracy alone is a poor
    comparison under the class imbalance every crop has (few "keep"
    regions, many "discard" ones): compare `cv_*` against `baseline_*` on
    balanced_accuracy/f1_keep/average_precision, not accuracy, to tell
    whether the model earns its keep over the free heuristic.
    """
    df = training_table.dropna(subset=["keep"])
    if df.empty:
        raise ValueError("training_table has no labeled regions to train on.")
    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df["keep"].to_numpy(dtype=int)
    groups = (df["source_experiment"].astype(str) + "_" + df["mz_rank"].astype(str)).to_numpy()

    n_groups = len(set(groups))
    splits = min(n_splits, n_groups)
    fold_model_metrics, fold_baseline_metrics = [], []
    touches_hint_col = FEATURE_COLUMNS.index("touches_hint")
    if splits >= 2:
        gkf = GroupKFold(n_splits=splits)
        for train_idx, val_idx in gkf.split(X, y, groups):
            clf = GradientBoostingClassifier(random_state=random_seed)
            clf.fit(X[train_idx], y[train_idx])
            pred = clf.predict(X[val_idx])
            proba = clf.predict_proba(X[val_idx])[:, 1]
            fold_model_metrics.append(_binary_classification_metrics(y[val_idx], pred, proba))

            baseline_pred = (X[val_idx][:, touches_hint_col] > 0.5).astype(int)
            fold_baseline_metrics.append(_binary_classification_metrics(y[val_idx], baseline_pred))
    else:
        Logger.warning(
            "Only %d distinct reviewed image(s); skipping cross-validation "
            "(need >= 2 for GroupKFold). Fitting on all data with no held-out report.",
            n_groups,
        )

    model = GradientBoostingClassifier(random_state=random_seed)
    model.fit(X, y)
    report = {
        "n_regions": len(df),
        "n_images": n_groups,
        "keep_rate": float(y.mean()),
    }
    if fold_model_metrics:
        report.update({f"cv_{k}": v for k, v in _mean_metrics(fold_model_metrics).items()})
        report.update(
            {
                f"baseline_touches_hint_{k}": v
                for k, v in _mean_metrics(fold_baseline_metrics).items()
            }
        )
    else:
        report.update({f"cv_{k}": None for k in ("accuracy", "balanced_accuracy", "precision_keep", "recall_keep", "f1_keep", "auc", "average_precision")})
        report.update({f"baseline_touches_hint_{k}": None for k in ("accuracy", "balanced_accuracy", "precision_keep", "recall_keep", "f1_keep")})
    Logger.info("Region-keep classifier: %s", report)
    return model, report


def correct_ground_truth_datapoint(
    dp: GroundTruthDatapoint,
    peak_consensus_kwargs: dict,
    model: GradientBoostingClassifier | None = None,
    keep_threshold: float = 0.5,
) -> tuple[np.ndarray, float | None]:
    """Corrected mask for one GroundTruthDatapoint's existing (bbox) mask:
    watershed the bbox crop, then select regions either via `model` (if
    given -- predict_proba > keep_threshold) or the zero-cost
    default_kept_label_ids heuristic (touches a known anchor).

    Returns (corrected_mask, mean_confidence). mean_confidence is
    mean(|predict_proba - 0.5| * 2) across this crop's regions when a model
    is used (1.0 = certain, 0.0 = coin flip); None when falling back to the
    heuristic, so callers can flag low-confidence model predictions for a
    follow-up manual look instead of trusting every prediction equally.
    """
    bbox = extract_bbox_from_mask(dp.mask)
    crop, labels = compute_watershed_crop(dp.image, bbox, peak_consensus_kwargs)
    row0, col0, row1, col1 = bbox
    hint_crop = dp.hint_channel[row0:row1, col0:col1]

    if labels.max() == 0:
        return dp.mask, None

    if model is None:
        kept = default_kept_label_ids(labels, hint_crop)
        confidence = None
    else:
        feats = regionprops_features(crop, labels, hint_crop)
        proba = model.predict_proba(feats[FEATURE_COLUMNS].to_numpy(dtype=float))[:, 1]
        kept = {int(lbl) for lbl, p in zip(feats.index, proba) if p > keep_threshold}
        confidence = float(np.mean(np.abs(proba - 0.5) * 2)) if len(proba) else None

    corrected_mask = mask_from_label_ids(labels, kept, bbox, dp.mask.shape)
    return corrected_mask, confidence


def apply_corrections_to_experiment(
    swaps_dir: str,
    model: GradientBoostingClassifier | None,
    review_store: ReviewStore | None = None,
    n_samples: int | None = None,
    batch_size: int = 500,
    random_seed: int = 42,
    include_decoys: bool = False,
) -> pd.DataFrame:
    """Corrected-mask table for one experiment's eligible mz_ranks.

    Human corrections from `review_store` are used verbatim where available
    (samples marked "discarded" there are dropped entirely -- the true
    region fell outside the existing bbox); everything else gets `model`'s
    prediction (or the touches_hint heuristic, if model is None) via
    correct_ground_truth_datapoint.

    Returns [mz_rank, corrected_mask, source, confidence] -- `source` is
    "human"/"model"/"heuristic". Feed into prepare_dataset's own
    GroundTruthDatapoint.mask before the pad/rescale + HDF5-write step to
    replace the raw bbox ground truth.
    """
    source_experiment = os.path.basename(swaps_dir.rstrip("/"))
    ctx = load_experiment_context(swaps_dir)
    peak_consensus_kwargs = dict(ctx["processing_kwargs"].get("peak_consensus_kwargs", {}))
    eligible = _eligible_mz_ranks(ctx["dict_ref"], ctx["boundary_table"], include_decoys)

    reviewed_status: dict[int, str] = {}
    if review_store is not None:
        manifest = review_store.load_manifest()
        sub = manifest[manifest["source_experiment"] == source_experiment]
        reviewed_status = dict(zip(sub["mz_rank"].astype(int), sub["status"]))

    discarded = {mz for mz, status in reviewed_status.items() if status == "discarded"}
    eligible = np.array([mz for mz in eligible if mz not in discarded])
    if n_samples is not None:
        rng = np.random.default_rng(random_seed)
        eligible = rng.permutation(eligible)[:n_samples]

    rows = []
    for start in range(0, len(eligible), batch_size):
        chunk = eligible[start : start + batch_size]
        batch_results = prepare_ground_truth_batch(
            ctx["dict_ref"],
            ctx["raw_file_list"],
            swaps_dir,
            chunk,
            ctx["boundary_table"],
            ctx["processing_kwargs"],
        )
        for mz_rank, dp in batch_results.items():
            if reviewed_status.get(mz_rank) == "reviewed":
                sample = review_store.load_sample_npz(source_experiment, mz_rank)
                rows.append(
                    {
                        "mz_rank": mz_rank,
                        "corrected_mask": sample["corrected_mask"],
                        "source": "human",
                        "confidence": 1.0,
                    }
                )
                continue
            corrected_mask, confidence = correct_ground_truth_datapoint(
                dp, peak_consensus_kwargs, model=model
            )
            rows.append(
                {
                    "mz_rank": mz_rank,
                    "corrected_mask": corrected_mask,
                    "source": "model" if model is not None else "heuristic",
                    "confidence": confidence,
                }
            )
        Logger.info(
            "%s: corrected %d/%d eligible mz_ranks",
            source_experiment,
            min(start + batch_size, len(eligible)),
            len(eligible),
        )
    return pd.DataFrame(rows)


def sample_active_learning_review_batch(
    swaps_dirs: list[str],
    model: GradientBoostingClassifier,
    n_samples: int,
    review_store: ReviewStore,
    pool_size_per_experiment: int = 500,
    uncertainty_pool_factor: int = 3,
    n_clusters: int = 20,
    batch_size: int = 200,
    random_seed: int = 42,
    include_decoys: bool = False,
) -> pd.DataFrame:
    """Active-learning follow-up to gt_correction.sample_diverse_review_batch:
    picks the *next* review batch from wherever `model` is least confident,
    instead of spreading evenly across the dataset's diversity -- once you
    already have a trained region-keep classifier, the most useful thing a
    reviewer's limited time can do is resolve exactly the cases the model
    itself can't already handle (e.g. a low recall on the "keep" class means
    the model is confidently missing true-peak regions somewhere -- this
    surfaces the crops where it *isn't* confident, which is where those
    misses are concentrated).

    For every candidate mz_rank not already in `review_store` (both
    "reviewed" and "discarded" ones are excluded -- no point re-showing a
    decision you already made), computes the model's predicted correction
    and its mean per-region confidence (correct_ground_truth_datapoint's
    second return value; lower means the model's region decisions in that
    crop were closer to a coin flip). Keeps the
    `uncertainty_pool_factor * n_samples` least-confident candidates, then
    applies gt_correction.select_diverse_subset to *that* uncertain subset
    (same cheap pool features as sample_diverse_review_batch) so the batch
    isn't just N near-duplicate hard cases.

    Returns a plan DataFrame [source_experiment, mz_rank] -- feed straight
    into review_widget.GroundTruthReviewSession, same as
    sample_diverse_review_batch's output.
    """
    rng = np.random.default_rng(random_seed)
    rows = []
    for swaps_dir in swaps_dirs:
        source_experiment = os.path.basename(swaps_dir.rstrip("/"))
        ctx = load_experiment_context(swaps_dir)
        peak_consensus_kwargs = dict(ctx["processing_kwargs"].get("peak_consensus_kwargs", {}))
        eligible = _eligible_mz_ranks(ctx["dict_ref"], ctx["boundary_table"], include_decoys)

        manifest = review_store.load_manifest()
        done = set(
            manifest.loc[manifest["source_experiment"] == source_experiment, "mz_rank"].astype(int)
        )
        remaining = np.array([mz for mz in eligible if mz not in done])
        if len(remaining) == 0:
            Logger.info("%s: nothing left unreviewed to score.", source_experiment)
            continue
        pool_mz_ranks = rng.permutation(remaining)[:pool_size_per_experiment]

        for start in range(0, len(pool_mz_ranks), batch_size):
            chunk = pool_mz_ranks[start : start + batch_size]
            batch_results = prepare_ground_truth_batch(
                ctx["dict_ref"],
                ctx["raw_file_list"],
                swaps_dir,
                chunk,
                ctx["boundary_table"],
                ctx["processing_kwargs"],
            )
            for mz_rank, dp in batch_results.items():
                _, confidence = correct_ground_truth_datapoint(
                    dp, peak_consensus_kwargs, model=model
                )
                if confidence is None:
                    continue  # no watershed regions in this crop -- nothing to review
                feats = candidate_pool_features(dp, peak_consensus_kwargs)
                feats["confidence"] = confidence
                feats["source_experiment"] = source_experiment
                feats["mz_rank"] = mz_rank
                rows.append(feats)
        Logger.info(
            "%s: scored %d/%d unreviewed candidates for model uncertainty",
            source_experiment,
            len(pool_mz_ranks),
            len(remaining),
        )

    pool = pd.DataFrame(rows)
    if pool.empty:
        raise ValueError("No unreviewed candidates found across the given experiments.")

    n_uncertain = min(len(pool), max(n_samples, uncertainty_pool_factor * n_samples))
    uncertain_pool = pool.nsmallest(n_uncertain, "confidence").reset_index(drop=True)
    Logger.info(
        "Uncertain-pool confidence range: %.3f-%.3f (median %.3f) vs. overall pool median %.3f",
        uncertain_pool["confidence"].min(),
        uncertain_pool["confidence"].max(),
        uncertain_pool["confidence"].median(),
        pool["confidence"].median(),
    )

    selected = select_diverse_subset(
        uncertain_pool, n_samples, n_clusters=n_clusters, random_seed=random_seed
    )
    return selected[["source_experiment", "mz_rank"]].reset_index(drop=True)
