import torch
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR, OneCycleLR
from yacs.config import CfgNode
from typing import Union
from ..utils import EarlyStopping


def build_optimizer(model: torch.nn.Module, opti_cfg: CfgNode) -> Optimizer:
    """
    simple optimizer builder
    :param model: already gpu pushed model
    :param opti_cfg:  config node
    :return: the optimizer
    """
    parameters = model.parameters()
    opti_type = opti_cfg.NAME
    lr = opti_cfg.BASE_LR
    weight_decay = opti_cfg.WEIGHT_DECAY
    if opti_type == "adam":
        optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif opti_type == "sgd":
        sgd_cfg = opti_cfg.SGD
        momentum = sgd_cfg.MOMENTUM
        nesterov = sgd_cfg.NESTEROV
        optimizer = torch.optim.SGD(
            parameters,
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
        )
    else:
        raise Exception("invalid optimizer, available choices adam/sgd")
    return optimizer


def build_scheduler(
    optimizer: Optimizer,
    scheduler_cfg: CfgNode,
    steps_per_epoch: int = 0,
    epochs: int = 0,
):
    """

    :param optimizer:
    :param optimizer: Optimizer
    :param scheduler_cfg:
    "param solver_cfg: CfgNode
    :return:
    """
    scheduler_type = scheduler_cfg.NAME
    if scheduler_type == "unchange":
        return None
    elif scheduler_type == "multi_steps":
        gamma = scheduler_cfg.LR_REDUCE_GAMMA
        milestones = scheduler_cfg.MULTI_STEPS_LR_MILESTONES
        scheduler = MultiStepLR(optimizer, milestones, gamma=gamma, last_epoch=-1)
        return scheduler
    elif scheduler_type == "reduce_on_plateau":
        gamma = scheduler_cfg.LR_REDUCE_GAMMA
        scheduler = ReduceLROnPlateau(
            optimizer, patience=scheduler_cfg.PATIENCE, factor=gamma
        )
        return scheduler
    elif scheduler_type == "one_cycle":
        scheduler = OneCycleLR(
            optimizer,
            max_lr=scheduler_cfg.MAX_LR,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            pct_start=scheduler_cfg.PCT_START,
            anneal_strategy=scheduler_cfg.ANNEAL_STRATEGY,
            div_factor=scheduler_cfg.DIV_FACTOR,
            cycle_momentum=True,
        )
    else:
        raise Exception(
            "scheduler name invalid, choices are"
            " unchange/multi_steps/reduce_on_plateau/one_cycle"
        )


def build_early_stopper(early_stopper_cfg: CfgNode):
    """
    simple early stopper builder
    :param early_stopper_cfg:  config node
    :return: the early stopper
    """
    return EarlyStopping(
        patience=early_stopper_cfg.PATIENCE, mode=early_stopper_cfg.MODE
    )
