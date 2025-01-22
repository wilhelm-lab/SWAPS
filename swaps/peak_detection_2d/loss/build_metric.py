from multiprocessing import reduction
from yacs.config import CfgNode
from .combo_loss import ComboLoss, WeightedDiceLoss
from .custom_loss import (
    per_image_weighted_dice_metric,
    per_image_weighted_iou_metric,
)
import torch
from torcheval.metrics.functional import binary_auprc, binary_auroc


def build_metric(metric_cfg: CfgNode):
    if metric_cfg.METRIC == "WDICE":
        metric = per_image_weighted_dice_metric
    if metric_cfg.METRIC == "WIOU":
        metric = per_image_weighted_iou_metric
    if metric_cfg.METRIC == "MAE":
        metric = torch.nn.L1Loss(reduction="none")
    if metric_cfg.METRIC == "AUPRC":
        metric = binary_auprc
    if metric_cfg.METRIC == "ROCAUC":
        metric = binary_auroc
    return metric
