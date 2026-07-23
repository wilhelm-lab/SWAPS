"""CLI entry point: evaluate a trained CNN consensus-segmentation checkpoint
on a Phase B dataset split (see peak_detection_2d/evaluate.py).

Usage:
    python swaps/run_evaluate_cnn_segmentation.py --checkpoint=/path/to/epochXXXX.pt \
        [--split=test] [--threshold=0.5] [--config=/path/to/override.yaml]

Run as `python swaps/run_evaluate_cnn_segmentation.py` (same sys.path
convention as swaps/run_train_cnn_segmentation.py / swaps/sbs_runner_ims.py).
"""

import argparse
import logging

import numpy as np

from peak_detection_2d.config.singleton_peak_detection import peak_detection_cfg
from peak_detection_2d.evaluate import evaluate_weighted_iou


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to a trained checkpoint (.pt).")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Sigmoid threshold for binarizing predictions."
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

    scores = evaluate_weighted_iou(
        cfg, args.checkpoint, split=args.split, threshold=args.threshold
    )
    logging.info(
        "%s weighted IoU (n=%d): mean=%.4f median=%.4f std=%.4f",
        args.split,
        len(scores),
        np.mean(scores),
        np.median(scores),
        np.std(scores),
    )
    for q in (0.05, 0.25, 0.5, 0.75, 0.95):
        logging.info("  p%d: %.4f", int(q * 100), np.quantile(scores, q))
    logging.info("  fraction > 0.8: %.4f", float((scores > 0.8).mean()))
    logging.info("  fraction < 0.2: %.4f", float((scores < 0.2).mean()))


if __name__ == "__main__":
    main()
