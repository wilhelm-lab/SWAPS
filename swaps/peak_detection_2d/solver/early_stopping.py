"""`EarlyStopping`, extracted from `multi_output_PS_model`'s `utils.py`
(a 32KB catch-all module of plotting/inference helpers, not ported) into its
own module since `solver/build_optimizer.py`'s `build_early_stopper` needs
it and nothing else from that file is in scope for this phase.
"""

import logging
import os
from pathlib import Path

import numpy as np
import torch

Logger = logging.getLogger(__name__)


class EarlyStopping:
    def __init__(self, patience=7, mode="max", delta=0.0001):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.val_score = np.inf if self.mode == "min" else -np.inf

    def __call__(
        self, epoch_score, epoch_num, loss, optimizer, model, model_path, scheduler=None
    ):
        score = -1.0 * epoch_score if self.mode == "min" else np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(
                epoch_score, epoch_num, loss, optimizer, model, model_path, scheduler
            )
        elif score < self.best_score + self.delta:
            self.counter += 1
            Logger.info(
                "EarlyStopping counter: %d out of %d", self.counter, self.patience
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(
                epoch_score, epoch_num, loss, optimizer, model, model_path, scheduler
            )
            self.counter = 0

    def save_checkpoint(
        self, epoch_score, epoch_num, loss, optimizer, model, model_path, scheduler=None
    ):
        model_path = Path(model_path)
        os.makedirs(model_path.parent, exist_ok=True)
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            Logger.info(
                "Validation score improved (%s --> %s). Model saved at %s!",
                self.val_score,
                epoch_score,
                model_path,
            )
            save_state = {
                "epoch": epoch_num,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
            }
            if scheduler is not None:
                save_state["scheduler_state_dict"] = scheduler.state_dict()
            torch.save(save_state, model_path)
        self.val_score = epoch_score
