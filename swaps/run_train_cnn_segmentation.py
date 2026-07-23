"""CLI entry point: train the CNN consensus-segmentation UNET
(see peak_detection_2d/train.py, peak_detection_2d/config/singleton_peak_detection.py).

Usage:
    python swaps/run_train_cnn_segmentation.py --output_dir=/path/to/run \
        [--config=/path/to/override.yaml] \
        [-- MODEL.SOLVER.TOTAL_EPOCHS 50 DATASET.TRAIN_BATCH_SIZE 64 ...]

Run as `python swaps/run_train_cnn_segmentation.py` (same sys.path
convention as swaps/run_build_cnn_training_dataset.py / swaps/sbs_runner_ims.py).
"""

import argparse
import logging
import os
import sys

from peak_detection_2d.config.singleton_peak_detection import peak_detection_cfg
from peak_detection_2d.train import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Run directory for checkpoints, tensorboard logs, and a config.yaml snapshot.",
    )
    parser.add_argument(
        "--config", default=None, help="YAML file to merge on top of the default config."
    )
    args, opts = parser.parse_known_args()
    if opts and opts[0] == "--":
        opts = opts[1:]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    cfg = peak_detection_cfg.clone()
    if args.config:
        cfg.merge_from_file(args.config)
    if opts:
        cfg.merge_from_list(opts)
    cfg.freeze()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.yaml"), "w", encoding="utf-8") as f:
        f.write(cfg.dump())
    logging.info("Config snapshot written to %s", os.path.join(args.output_dir, "config.yaml"))

    best_ckpt = train(cfg, args.output_dir)
    logging.info("Done. Best checkpoint: %s", best_ckpt)


if __name__ == "__main__":
    main()
