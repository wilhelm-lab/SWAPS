"""Training loop for the CNN consensus-segmentation UNET.

Imported as `peak_detection_2d.train` (relative imports below, matching
`multi_output_PS_model`'s own `train.py`) -- run via the top-level
`swaps/run_train_cnn_segmentation.py` CLI script, not directly.
"""

import logging
import os

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from yacs.config import CfgNode

from .dataset.torch_dataset import PeakSegmentationDataset, load_target_shape
from .loss.build_criterion import build_criterion
from .model.build_model import build_model
from .model.seg_model import evaluate, train_one_epoch
from .solver.build_optimizer import build_early_stopper, build_optimizer, build_scheduler

Logger = logging.getLogger(__name__)


def train(cfg: CfgNode, run_dir: str) -> str:
    """Train the UNET per `cfg`, checkpointing/logging under `run_dir`.

    Returns the path of the best (lowest validation loss) checkpoint.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Logger.info("Using device: %s", device)

    target_shape = load_target_shape(cfg.DATASET.DATA_DIR)
    if tuple(target_shape) != tuple(cfg.DATASET.TARGET_SHAPE):
        Logger.warning(
            "manifest.yaml target_shape %s differs from cfg.DATASET.TARGET_SHAPE %s; "
            "manifest.yaml is authoritative, proceeding with it.",
            target_shape,
            tuple(cfg.DATASET.TARGET_SHAPE),
        )

    add_log_channel = cfg.MODEL.PARAMS.IN_CHANNELS == 3
    train_dataset = PeakSegmentationDataset.from_split(
        cfg.DATASET.DATA_DIR, "train", add_log_channel=add_log_channel
    )
    val_dataset = PeakSegmentationDataset.from_split(
        cfg.DATASET.DATA_DIR, "val", add_log_channel=add_log_channel
    )
    Logger.info(
        "Train dataset: %d samples, val dataset: %d samples",
        len(train_dataset),
        len(val_dataset),
    )

    pin_memory = device == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.DATASET.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.DATASET.NUM_WORKERS,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.DATASET.VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.DATASET.NUM_WORKERS,
        pin_memory=pin_memory,
    )

    model = build_model(cfg.MODEL).to(device)
    optimizer = build_optimizer(model, cfg.MODEL.SOLVER.OPTIMIZER)
    total_epochs = cfg.MODEL.SOLVER.TOTAL_EPOCHS
    scheduler = build_scheduler(
        optimizer,
        cfg.MODEL.SOLVER.SCHEDULER,
        steps_per_epoch=len(train_loader),
        epochs=total_epochs,
    )
    criterion = build_criterion(cfg.MODEL.SOLVER.LOSS, device=device)
    early_stopper = build_early_stopper(cfg.MODEL.SOLVER.EARLY_STOPPING)

    start_epoch = 0
    if cfg.RESUME_PATH:
        checkpoint = torch.load(cfg.RESUME_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        Logger.info("Resumed from %s at epoch %d", cfg.RESUME_PATH, start_epoch)

    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(run_dir, "tensorboard"))
    best_model_path = ""

    for epoch in range(start_epoch, total_epochs):
        Logger.info("Epoch %d/%d", epoch, total_epochs)
        train_loss = train_one_epoch(
            train_loader, model, optimizer, criterion, device=device, scheduler=scheduler
        )
        val_loss = evaluate(val_loader, model, metric=criterion, device=device)
        Logger.info(
            "Epoch %d: train_loss=%.4f val_loss=%.4f", epoch, train_loss, val_loss
        )

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        if scheduler is not None and scheduler.__class__.__name__ != "OneCycleLR":
            if scheduler.__class__.__name__ == "ReduceLROnPlateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

        model_path = os.path.join(ckpt_dir, f"epoch{epoch:04d}_valloss{val_loss:.4f}.pt")
        early_stopper(
            epoch_score=val_loss,
            epoch_num=epoch,
            loss=val_loss,
            optimizer=optimizer,
            model=model,
            model_path=model_path,
            scheduler=scheduler,
        )
        if early_stopper.counter == 0:
            best_model_path = model_path
        if early_stopper.early_stop:
            Logger.info("Early stopping triggered at epoch %d", epoch)
            break

    writer.close()
    Logger.info("Training finished. Best checkpoint: %s", best_model_path)
    return best_model_path
