"""UNET segmentation model + minimal train/eval loop helpers.

Ported from the `multi_output_PS_model` branch's `model/seg_model.py` (742
lines), trimmed to the `mask_segmentation` (`model_type="seg"`) path only --
the combined seg+cls two-head model (`model_type="cls"`, `add_ps_channel`,
`seg_model=` chaining), `binary_auroc`/classification metrics, and the
inference/calibration helpers (`inference_and_sum_intensity`,
`inference_flatten_output`, `label_and_sum_intensity`, `naive_sum_intensity`)
are dropped -- out of scope for this phase (see Phase A report).

`train_one_epoch`/`evaluate` are also adapted to the new (image, mask)
2-tuple batch shape (`PeakSegmentationDataset`) instead of the old
(image, hint, label_dict) 3-tuple -- Phase B's hint channel is already
stacked into the image tensor dataset-side, so there is no separate
`label_batch["mask"]` dict to unpack anymore.
"""

import logging

import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms.v2.functional import center_crop
from tqdm import tqdm

Logger = logging.getLogger(__name__)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_one_epoch(
    train_loader,
    model,
    optimizer,
    loss_fn,
    device="cuda",
    scheduler=None,
):
    """One training epoch. `train_loader` yields (image_batch, mask_batch)
    pairs; `loss_fn` is assumed to accept (outputs, targets, images) like
    ComboLoss."""
    epoch_losses = AverageMeter()
    model = model.to(device)
    model.train()
    tk0 = tqdm(train_loader, total=len(train_loader))
    for image_batch, mask_batch in tk0:
        image_batch = image_batch.to(device).float()
        mask_batch = mask_batch.to(device).float()
        out = model(image_batch)
        b_loss = loss_fn(out, mask_batch, image_batch)
        optimizer.zero_grad()
        b_loss.backward()
        optimizer.step()
        epoch_losses.update(b_loss.mean().item(), train_loader.batch_size)
        tk0.set_postfix(
            total_loss=epoch_losses.avg,
            learning_rate=optimizer.param_groups[0]["lr"],
        )
        # OneCycleLR steps per-batch, not per-epoch like other schedulers
        if scheduler is not None and scheduler.__class__.__name__ == "OneCycleLR":
            scheduler.step()
    return epoch_losses.avg


def evaluate(
    valid_loader,
    model,
    metric,
    device="cuda",
    save_all_loss: bool = False,
    **kwargs,
):
    """`metric` is assumed to accept (outputs, targets, images) like
    ComboLoss (or any of its component losses)."""
    epoch_loss = AverageMeter()
    model = model.to(device)
    model.eval()
    tk0 = tqdm(valid_loader, total=len(valid_loader))
    if save_all_loss:
        losses = np.empty((0))
    with torch.no_grad():
        for image_batch, mask_batch in tk0:
            image_batch = image_batch.to(device).float()
            mask_batch = mask_batch.to(device).float()
            out = model(image_batch)
            b_loss = metric(out, mask_batch, image_batch, **kwargs)
            epoch_loss.update(b_loss.mean().item(), valid_loader.batch_size)
            tk0.set_postfix({"loss": epoch_loss.avg})
            if save_all_loss:
                losses = np.append(losses, b_loss.cpu().numpy())
    if save_all_loss:
        return epoch_loss.avg, losses
    return epoch_loss.avg


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super().__init__()
        self.seq_block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.seq_block(x)


class CNNBlocks(nn.Module):
    """Parameters:
    n_conv (int): creates a block of n_conv convolutions
    in_channels (int): number of in_channels of the first block's convolution
    out_channels (int): number of out_channels of the first block's convolution
    """

    def __init__(self, n_conv, in_channels, out_channels, padding):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_conv):
            self.layers.append(CNNBlock(in_channels, out_channels, padding=padding))
            in_channels = out_channels

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Encoder(nn.Module):
    """Parameters:
    in_channels (int): number of in_channels of the first CNNBlocks
    out_channels (int): number of out_channels of the first CNNBlocks
    padding (int): padding applied in each convolution
    downhill (int): number times a CNNBlocks + MaxPool2D it's applied.
    """

    def __init__(self, in_channels, out_channels, padding, downhill=4):
        super().__init__()
        self.enc_layers = nn.ModuleList()
        for _ in range(downhill):
            self.enc_layers += [
                CNNBlocks(
                    n_conv=2,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    padding=padding,
                ),
                nn.MaxPool2d(2, 2),
            ]
            in_channels = out_channels
            out_channels *= 2
        # doubling the depth of the last CNN block
        self.enc_layers.append(
            CNNBlocks(
                n_conv=2,
                in_channels=in_channels,
                out_channels=out_channels,
                padding=padding,
            )
        )

    def forward(self, x):
        route_connection = []
        for layer in self.enc_layers:
            if isinstance(layer, CNNBlocks):
                x = layer(x)
                route_connection.append(x)
            else:
                x = layer(x)
        return x, route_connection


class Decoder(nn.Module):
    """Parameters:
    in_channels (int): number of in_channels of the first ConvTranspose2d
    out_channels (int): number of out_channels of the first ConvTranspose2d
    padding (int): padding applied in each convolution
    uphill (int): number times a ConvTranspose2d + CNNBlocks it's applied.
    """

    def __init__(self, in_channels, out_channels, exit_channels, padding, uphill=4):
        super().__init__()
        self.exit_channels = exit_channels
        self.layers = nn.ModuleList()
        for _ in range(uphill):
            self.layers += [
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                CNNBlocks(
                    n_conv=2,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    padding=padding,
                ),
            ]
            in_channels //= 2
            out_channels //= 2
        # cannot be a CNNBlock (has ReLU baked in); no Sigmoid here either --
        # loss functions expect raw logits (e.g. BCEWithLogitsLoss-style).
        # ADAPTED from upstream: the source branch reused `padding` (=1) here
        # too, but with kernel_size=1 that pads without anything for the
        # kernel to "consume", growing H/W by 2*padding instead of preserving
        # them -- always padding=0 for this 1x1 projection, regardless of the
        # padding used by the preceding 3x3 CNNBlocks.
        self.layers.append(
            nn.Conv2d(in_channels, exit_channels, kernel_size=1, padding=0),
        )

    def forward(self, x, routes_connection):
        # pop the last element of the list since it's not used for concatenation
        routes_connection.pop(-1)
        for layer in self.layers:
            if isinstance(layer, CNNBlocks):
                # center-crop the skip-connection route to match x's spatial size.
                # ADAPTED from upstream: the source branch passed only
                # x.shape[2] (height), which torchvision's center_crop treats
                # as a *square* crop size -- silently correct there only
                # because that branch's images were always square
                # (PADDING_SHAPE=(258, 258)); our target shape (112, 528) is
                # not square, so both spatial dims must be passed explicitly.
                routes_connection[-1] = center_crop(
                    routes_connection[-1], [x.shape[2], x.shape[3]]
                )
                x = torch.cat([x, routes_connection.pop(-1)], dim=1)
                x = layer(x)
            else:
                x = layer(x)
        return x


class UNET(nn.Module):
    """Standard U-Net: Encoder (repeated CNNBlocks + MaxPool2d downsampling)
    -> Decoder (ConvTranspose2d + skip-connection upsampling) -> single-
    channel logit mask. Segmentation-only: the upstream branch's `seg_head`/
    `cls_head` toggle and classification head are dropped (see build_model).
    """

    def __init__(self, in_channels, first_out_channels, exit_channels, downhill, padding=0):
        super().__init__()
        self.encoder = Encoder(
            in_channels, first_out_channels, padding=padding, downhill=downhill
        )
        self.decoder = Decoder(
            first_out_channels * (2**downhill),
            first_out_channels * (2 ** (downhill - 1)),
            exit_channels,
            padding=padding,
            uphill=downhill,
        )

    def forward(self, x):
        enc_out, routes = self.encoder(x)
        return self.decoder(enc_out, routes)
