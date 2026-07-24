"""CLI entry point: build a diverse-sampled review plan for
peak_detection_2d.dataset.gt_correction's manual ground-truth review
workflow (see gt_correction.py's module docstring).

Usage:
    python swaps/run_sample_gt_review_batch.py --output_dir=/path/to/gt_review \
        --n_samples=300 [--experiment_dirs dir1 dir2 dir3]

Writes <output_dir>/review_plan.parquet ([source_experiment, mz_rank]) plus
review_plan_experiments.yaml (source_experiment -> swaps_dir), so
review_widget.GroundTruthReviewSession / label_correction_model don't need
that mapping re-typed.
"""

import argparse
import logging
import os

import yaml

from peak_detection_2d.dataset.gt_correction import sample_diverse_review_batch

DEFAULT_EXPERIMENT_DIRS = [
    "/cmnfs/proj/ORIGINS/data/SWAPS_FFM_timsTOF_benchmark/coSWA/HT_microflow_30min_5ug_HYE_mergeConfOFF_20260710_103949_916715",
    "/cmnfs/proj/ORIGINS/data/SWAPS_FFM_timsTOF_benchmark/HT_nanoflow_5min_200ng_HeLa_mergeConfOFF_20260714_084332_487834",
    "/cmnfs/proj/ORIGINS/data/SWAPS_FFM_timsTOF_benchmark/Ultra_nanoflow_5min_125pg_K562_mergeConfOFF_20260714_084332_487963",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment_dirs",
        nargs="+",
        default=DEFAULT_EXPERIMENT_DIRS,
        help="SWAPS RESULT_PATH dirs to pool review candidates from.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_samples", type=int, default=300)
    parser.add_argument("--pool_size_per_experiment", type=int, default=500)
    parser.add_argument("--n_clusters", type=int, default=20)
    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    plan = sample_diverse_review_batch(
        args.experiment_dirs,
        args.n_samples,
        pool_size_per_experiment=args.pool_size_per_experiment,
        n_clusters=args.n_clusters,
        random_seed=args.random_seed,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    plan_path = os.path.join(args.output_dir, "review_plan.parquet")
    plan.to_parquet(plan_path, index=False)

    experiments_map = {os.path.basename(d.rstrip("/")): d for d in args.experiment_dirs}
    with open(os.path.join(args.output_dir, "review_plan_experiments.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(experiments_map, f)

    logging.info("Wrote %d-sample review plan to %s", len(plan), plan_path)


if __name__ == "__main__":
    main()
