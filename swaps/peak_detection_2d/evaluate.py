"""Evaluate a trained checkpoint's segmentation quality on a Phase B split.

Imported as `peak_detection_2d.evaluate` -- run via the top-level
`swaps/run_evaluate_cnn_segmentation.py` CLI script, not directly.
"""

import logging

import numpy as np
import torch
from torch.utils.data import DataLoader
from yacs.config import CfgNode

from .dataset.torch_dataset import PeakSegmentationDataset
from .loss.combo_loss import per_image_weighted_iou_metric
from .model.build_model import build_model

Logger = logging.getLogger(__name__)


def evaluate_weighted_iou(
    cfg: CfgNode,
    checkpoint_path: str,
    split: str = "test",
    threshold: float = 0.5,
) -> np.ndarray:
    """Per-sample intensity-weighted IoU (per_image_weighted_iou_metric,
    ported from multi_output_PS_model's custom_loss.py) for `checkpoint_path`
    on `split` of cfg.DATASET.DATA_DIR. Returns one value per sample.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Logger.info("Using device: %s", device)

    add_log_channel = cfg.MODEL.PARAMS.IN_CHANNELS == 3
    dataset = PeakSegmentationDataset.from_split(
        cfg.DATASET.DATA_DIR, split, add_log_channel=add_log_channel
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.DATASET.TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.DATASET.NUM_WORKERS,
    )
    Logger.info("%s dataset: %d samples", split, len(dataset))

    model = build_model(cfg.MODEL).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    scores = np.empty(0, dtype=np.float32)
    with torch.no_grad():
        for image_batch, mask_batch in loader:
            image_batch = image_batch.to(device).float()
            mask_batch = mask_batch.to(device).float()
            out = model(image_batch)
            batch_scores = per_image_weighted_iou_metric(
                out,
                mask_batch,
                image_batch,
                threshold=threshold,
                device=device,
                channel=0,
            )
            scores = np.append(scores, batch_scores.cpu().numpy())
    return scores
