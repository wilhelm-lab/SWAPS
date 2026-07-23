"""CLI entry point: bulk-build the CNN consensus-segmentation ground-truth
training dataset (see peak_detection_2d/dataset/prepare_dataset.py).

Usage:
    python swaps/run_build_cnn_training_dataset.py --output_dir=/path/to/output \
        [--n_per_experiment=5000] [--target_shape_percentile=90.0] \
        [--experiment_dirs dir1 dir2 dir3]

Run from the repo root (or anywhere -- this script's own directory, swaps/,
needs to end up on sys.path for the bare `peak_detection_2d...` import below;
running it as `python swaps/run_build_cnn_training_dataset.py` does that
automatically, same as swaps/sbs_runner_ims.py).
"""

import argparse
import logging

from peak_detection_2d.dataset.prepare_dataset import build_training_dataset

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
        help="SWAPS RESULT_PATH dirs to pool ground truth from.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_per_experiment", type=int, default=5000)
    parser.add_argument("--target_shape_percentile", type=float, default=90.0)
    parser.add_argument("--shape_sample_size", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--include_decoys", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logging.info("Experiment dirs: %s", args.experiment_dirs)
    logging.info(
        "n_per_experiment=%d, target_shape_percentile=%.1f, batch_size=%d",
        args.n_per_experiment,
        args.target_shape_percentile,
        args.batch_size,
    )

    manifest = build_training_dataset(
        experiment_dirs=args.experiment_dirs,
        output_dir=args.output_dir,
        n_per_experiment=args.n_per_experiment,
        target_shape_percentile=args.target_shape_percentile,
        shape_sample_size=args.shape_sample_size,
        batch_size=args.batch_size,
        random_seed=args.random_seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        include_decoys=args.include_decoys,
    )
    logging.info("Done. Manifest: %s", manifest)


if __name__ == "__main__":
    main()
