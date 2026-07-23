"""Loss factory, ported from `multi_output_PS_model`'s `loss/build_criterion.py`.

Judgment call: `custom_loss.py` is NOT ported. Its only symbol this factory
used (`FocalLoss1D`, for `loss_cfg.NAME == "FocalLoss"`) is a 1D box/cls loss
unrelated to the 2D segmentation `ComboLoss` path, is not needed by the
`mask_segmentation` default config, and is actually broken in the source
branch (`FocalLoss1D.__init__` calls `super(FocalLoss, ...)`, an undefined
name -- `NameError` on instantiation). `custom_loss.py`'s other contents
(`metric_iou_batch`, `WeightedBoundingBoxIoULoss`, etc.) are box-regression
metrics not called anywhere in the ported `seg_model.py`. That `FocalLoss`
branch is dropped here; everything else ported as-is.
"""

from yacs.config import CfgNode
from torch import nn, tensor

from .combo_loss import ComboLoss, WeightedDiceLoss


def build_criterion(loss_cfg: CfgNode, device=None):
    if loss_cfg.NAME == "ComboLoss":
        weights = dict(zip(loss_cfg.LOSSTYPES, loss_cfg.WEIGHTS))
        return ComboLoss(
            weights=weights,
            channel_weights=loss_cfg.CHANNEL_WEIGHTS,
            per_image=loss_cfg.PER_IMAGE,
        )
    if loss_cfg.NAME == "WeightedDiceLoss":
        return WeightedDiceLoss(per_image=loss_cfg.PER_IMAGE, manual_sigmoid=True)
    if loss_cfg.NAME == "L1Loss":
        return nn.L1Loss()
    if loss_cfg.NAME == "MSELoss":
        return nn.MSELoss()
    if loss_cfg.NAME == "BCELoss":
        return nn.BCELoss()
    if loss_cfg.NAME == "BCEWithLogitsLoss":
        pos_weight = loss_cfg.POS_WEIGHT
        if pos_weight is not None:
            pos_weight = tensor(pos_weight).to(device)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    raise ValueError(f"Unsupported loss_cfg.NAME: {loss_cfg.NAME!r}")
