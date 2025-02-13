from yacs.config import CfgNode
from .combo_loss import ComboLoss, WeightedDiceLoss
from .custom_loss import FocalLoss1D
from torch import nn, tensor


def build_criterion(loss_cfg: CfgNode, device=None):
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
        pos_weight = loss_cfg.POS_WEIGHT
        if pos_weight is not None:
            pos_weight = tensor(pos_weight).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if loss_cfg.NAME == "FocalLoss":
        criterion = FocalLoss1D(alpha=loss_cfg.ALPHA, gamma=loss_cfg.GAMMA)
    return criterion
