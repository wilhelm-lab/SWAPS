from yacs.config import CfgNode
from .combo_loss import ComboLoss, WeightedDiceLoss
from torch import nn


def build_criterion(loss_cfg: CfgNode):
    if loss_cfg.NAME == "ComboLoss":
        weights = {k: v for k, v in zip(loss_cfg.LOSSTYPES, loss_cfg.WEIGHTS)}
        criterion = ComboLoss(
            **{
                "weights": weights,
                "channel_weights": loss_cfg.CHANNEL_WEIGHTS,
                "per_image": loss_cfg.PER_IMAGE,
            },
        )
    if loss_cfg.NAME == "WeightedDiceLoss":
        criterion = WeightedDiceLoss(per_image=loss_cfg.PER_IMAGE, manual_sigmoid=True)
    if loss_cfg.NAME == "L1Loss":
        criterion = nn.L1Loss()
    if loss_cfg.NAME == "BCELoss":
        criterion = nn.BCELoss()
    if loss_cfg.NAME == "BCEWithLogitsLoss":
        criterion = nn.BCEWithLogitsLoss()
    return criterion
