"""Optimizer/scheduler/early-stopper factories, ported as-is from
`multi_output_PS_model`'s `solver/build_optimizer.py` -- no seg/cls
entanglement here. `EarlyStopping` now lives in `solver/early_stopping.py`
(see that module's docstring) rather than the un-ported `..utils`.
"""

import torch
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR, OneCycleLR
from yacs.config import CfgNode

from .early_stopping import EarlyStopping


def build_optimizer(model: torch.nn.Module, opti_cfg: CfgNode) -> Optimizer:
    parameters = model.parameters()
    opti_type = opti_cfg.NAME
    lr = opti_cfg.BASE_LR
    weight_decay = opti_cfg.WEIGHT_DECAY
    if opti_type == "adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    if opti_type == "sgd":
        sgd_cfg = opti_cfg.SGD
        return torch.optim.SGD(
            parameters,
            lr=lr,
            momentum=sgd_cfg.MOMENTUM,
            nesterov=sgd_cfg.NESTEROV,
            weight_decay=weight_decay,
        )
    raise ValueError("invalid optimizer, available choices adam/sgd")


def build_scheduler(
    optimizer: Optimizer,
    scheduler_cfg: CfgNode,
    steps_per_epoch: int = 0,
    epochs: int = 0,
):
    scheduler_type = scheduler_cfg.NAME
    if scheduler_type == "unchange":
        return None
    if scheduler_type == "multi_steps":
        return MultiStepLR(
            optimizer,
            scheduler_cfg.MULTI_STEPS_LR_MILESTONES,
            gamma=scheduler_cfg.LR_REDUCE_GAMMA,
            last_epoch=-1,
        )
    if scheduler_type == "reduce_on_plateau":
        return ReduceLROnPlateau(
            optimizer,
            patience=scheduler_cfg.PATIENCE,
            factor=scheduler_cfg.LR_REDUCE_GAMMA,
        )
    if scheduler_type == "one_cycle":
        return OneCycleLR(
            optimizer,
            max_lr=scheduler_cfg.MAX_LR,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            pct_start=scheduler_cfg.PCT_START,
            anneal_strategy=scheduler_cfg.ANNEAL_STRATEGY,
            div_factor=scheduler_cfg.DIV_FACTOR,
            cycle_momentum=True,
        )
    raise ValueError(
        "scheduler name invalid, choices are "
        "unchange/multi_steps/reduce_on_plateau/one_cycle"
    )


def build_early_stopper(early_stopper_cfg: CfgNode) -> EarlyStopping:
    return EarlyStopping(
        patience=early_stopper_cfg.PATIENCE, mode=early_stopper_cfg.MODE
    )
