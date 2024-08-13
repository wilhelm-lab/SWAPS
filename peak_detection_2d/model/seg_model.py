from ast import Call
from typing import List, Callable
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision.transforms.v2.functional import center_crop
from tqdm import tqdm

from peak_detection_2d.loss.custom_loss import metric_iou_batch

Logger = logging.getLogger(__name__)


class AverageMeter:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

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
    accumulation_steps=1,
    device="cuda",
    scheduler=None,
    use_image_as_input: bool = False,
):
    epoch_losses = AverageMeter()
    model = model.to(device)
    model.train()
    if accumulation_steps > 1:
        optimizer.zero_grad()
    tk0 = tqdm(train_loader, total=len(train_loader))
    for image_batch, hint_batch, label_batch in tk0:
        out = model(image_batch.float())
        if use_image_as_input:
            b_loss = loss_fn(out, label_batch["mask"].to(device), image_batch)
        else:
            b_loss = loss_fn(out, label_batch["mask"].to(device))
        with torch.set_grad_enabled(True):
            b_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        epoch_losses.update(b_loss.mean().item(), train_loader.batch_size)
        tk0.set_postfix(
            loss=epoch_losses.avg, learning_rate=optimizer.param_groups[0]["lr"]
        )
        # Update Scheduler at this point only if scheduler_type is 'OneCycleLR'
        if scheduler is not None and scheduler.__class__.__name__ == "OneCycleLR":
            scheduler.step()
    return epoch_losses.avg


def evaluate(
    valid_loader,
    model,
    metric,
    device="cuda",
    save_all_loss: bool = False,
    use_image_for_metric: bool = False,
    **kwargs,
):
    epoch_losses = AverageMeter()
    model = model.to(device)
    model.eval()
    tk0 = tqdm(valid_loader, total=len(valid_loader))
    if save_all_loss:
        losses = np.empty((0))
        pept_mz_rank = np.empty((0))
    with torch.no_grad():
        for image_batch, hint_batch, label_batch in tk0:
            out = model(image_batch.float())
            if use_image_for_metric:
                b_loss = metric(
                    out.to(device),
                    label_batch["mask"].to(device),
                    image_batch.to(device),
                    **kwargs,
                )
            else:
                b_loss = metric(
                    out.to(device), label_batch["mask"].to(device), **kwargs
                )
            epoch_losses.update(b_loss.mean().item(), valid_loader.batch_size)
            tk0.set_postfix(loss=epoch_losses.avg)
            if save_all_loss:
                losses = np.append(losses, b_loss.cpu().numpy())
                pept_mz_rank = np.append(pept_mz_rank, label_batch["pept_mz_rank"])
            # Logger.info("Loss: %s", b_loss.cpu().numpy())
            # Logger.info("all_losses: %s", all_losses)
    if save_all_loss:
        all_losses = dict(losses=losses, ranks=pept_mz_rank)
        return epoch_losses.avg, all_losses
    else:
        return epoch_losses.avg


def inference_flatten_output(
    data_loader, model, device, get_labels: bool = False, threshold: float = 0.5
):
    model = model.to(device)
    model.eval()
    out_score_list = []
    out_final_list = []
    label_list = []
    out_list = []
    with torch.no_grad():
        for image_batch, hint_batch, label_batch in data_loader:
            out = model(image_batch.float())
            out_list.extend(out.view(-1).cpu().numpy())
            out_prob_1 = torch.sigmoid(out)  # confidence of label = 1
            # out_prob_0 = 1 - out_prob_1  # confidence of label = 0
            # out_confidence = torch.where(out_prob_1 > threshold, out_prob_1, out_prob_0)
            out_score_list.extend(out_prob_1.view(-1).cpu().numpy())

            out_final = torch.where(
                out_prob_1 > threshold,
                torch.ones_like(out_prob_1),
                torch.zeros_like(out_prob_1),
            )
            out_final_list.extend(out_final.view(-1).cpu().numpy())
            if get_labels:
                label_list.extend(label_batch["mask"].view(-1).cpu().numpy())
    return out_score_list, out_final_list, label_list, out_list


def inference_and_sum_intensity(
    data_loader,
    model,
    device="cuda",
    threshold: float = 0.5,
    int_channel: int = 0,
    calc_score: bool = False,
    calib=None,
    per_image_metric: List[Callable] = None,
    use_image_for_metric: List[bool] = None,
    resize: bool = False,
    **kwargs,
):

    tk0 = tqdm(data_loader, total=len(data_loader))

    model = model.to(device)
    model.eval()

    sum_intensity = np.empty((0))
    pept_mz_rank = np.empty((0))
    out_score = np.empty((0))

    if per_image_metric is not None:
        assert len(per_image_metric) == len(use_image_for_metric)
        epoch_losses = {}
        losses = {}
        for metric in per_image_metric:
            epoch_losses[metric.__name__] = AverageMeter()
            losses[metric.__name__] = np.empty((0))
    with torch.no_grad():
        for image_batch, hint_batch, label_batch in data_loader:
            batch_size, n_channels = label_batch["mask"].size(0), label_batch[
                "mask"
            ].size(1)
            pept_mz_rank = np.append(pept_mz_rank, label_batch["pept_mz_rank"])
            out = model(image_batch.float())
            Logger.debug(
                "out non zero value distribution: %s, %s",
                out.nonzero().min().item(),
                out.nonzero().max().item(),
            )
            if per_image_metric is not None:
                for metric, use_image in zip(per_image_metric, use_image_for_metric):
                    if use_image:
                        b_loss = metric(
                            out.to(device),
                            label_batch["mask"].to(device),
                            image_batch.to(device),
                            **kwargs,
                        )
                    else:
                        b_loss = metric(
                            out.to(device), label_batch["mask"].to(device), **kwargs
                        )
                    Logger.debug("b_loss: %s", b_loss)

                    epoch_losses[metric.__name__].update(
                        b_loss.mean().item(), data_loader.batch_size
                    )
                    tk0.set_postfix(loss=epoch_losses[metric.__name__].avg)

                    losses[metric.__name__] = np.append(
                        losses[metric.__name__], b_loss.cpu().numpy()
                    )
            if calib is not None:
                out_final = calib.predict(out.contiguous().view(-1).float().cpu())
                out_final = torch.tensor(out_final).view(out.shape).to(device)
            else:
                out_final = torch.sigmoid(out)
            if threshold is not None:
                if calc_score:
                    # out_score = torch.where(out_final > threshold, out_final, torch.zeros_like(out_final))
                    out_score_view = (
                        out_final.contiguous().view(batch_size, -1).float().to(device)
                    )
                    valid_pixel = out_score_view > threshold
                    masked_tensor = torch.where(
                        valid_pixel, out_score_view, torch.tensor(float("nan"))
                    )
                    Logger.debug("valid_pixel shape %s", valid_pixel.shape)
                    out_score = np.append(
                        out_score,
                        torch.nanmean(masked_tensor, dim=(1)).cpu().numpy(),
                    )
                    Logger.debug("out_score shape %s", out_score.shape)

                out_final = torch.where(
                    out_final > threshold,
                    torch.ones_like(out_final),
                    torch.zeros_like(out_final),
                )
            out_final = torch.where(
                out_final > threshold,
                torch.ones_like(out_final),
                torch.zeros_like(out_final),
            )
            out_final_reshaped = (
                out_final.contiguous().view(batch_size, -1).float().to(device)
            )

            intensity_reshaped = (
                image_batch[:, int_channel, :, :]
                .contiguous()
                .view(batch_size, -1)
                .float()
                .to(device)
            )
            Logger.debug("pept_mz_rank shape %s", pept_mz_rank.shape)
            try:
                sum_intensity = np.append(
                    sum_intensity,
                    torch.sum(intensity_reshaped * out_final_reshaped, dim=(1))
                    .cpu()
                    .numpy(),
                )
                Logger.debug("sum_intensity shape %s", sum_intensity.shape)
            except RuntimeError:
                Logger.debug(
                    "intensity shape %s, out_final shape %s",
                    image_batch[:, int_channel, :, :].shape,
                    out_final.shape,
                )
    if out_score.size == 0:
        out_score = np.zeros_like(sum_intensity)
    result = dict(
        sum_intensity=sum_intensity, mz_rank=pept_mz_rank, out_score=out_score
    )
    if per_image_metric is not None:
        for metric in per_image_metric:
            result[metric.__name__] = losses[metric.__name__]
    return pd.DataFrame(result)


def label_and_sum_intensity(data_loader, channel: int = 0, device="cuda"):
    sum_intensity = np.empty((0))
    pept_mz_rank = np.empty((0))
    with torch.no_grad():
        for image_batch, hint_batch, label_batch in data_loader:
            batch_size, n_channels = label_batch["mask"].size(0), label_batch[
                "mask"
            ].size(1)
            pept_mz_rank = np.append(pept_mz_rank, label_batch["pept_mz_rank"])
            label = (
                label_batch["mask"].contiguous().view(batch_size, -1).float().to(device)
            )
            intensity = (
                image_batch[:, channel, :, :]
                .contiguous()
                .view(batch_size, -1)
                .float()
                .to(device)
            )
            # Logger.debug("label shape %s", label.shape)
            # Logger.debug("intensity shape %s", intensity.shape)
            Logger.debug("pept_mz_rank shape %s", pept_mz_rank.shape)
            # sum_intensity = np.append(
            #     sum_intensity,
            #     np.sum(intensity * label, axis=(1, 2, 3)),
            # )
            sum_intensity = np.append(
                sum_intensity, torch.sum(intensity * label, dim=(1)).cpu().numpy()
            )
            # sum_intensity_all = np.append(sum_intensity)
            Logger.debug("sum_intensity shape %s", sum_intensity.shape)
    result = dict(sum_intensity=sum_intensity, pept_mz_rank=pept_mz_rank)
    return pd.DataFrame(result)


def naive_sum_intensity(data_loader, channel: int = 0, device="cuda"):
    sum_intensity = np.empty((0))
    pept_mz_rank = np.empty((0))

    for image_batch, hint_batch, label_batch in data_loader:
        batch_size = label_batch["mask"].size(0)
        pept_mz_rank = np.append(pept_mz_rank, label_batch["pept_mz_rank"])
        intensity = (
            image_batch[:, channel, :, :]
            .contiguous()
            .view(batch_size, -1)
            .float()
            .to(device)
        )
        sum_intensity = np.append(
            sum_intensity, torch.sum(intensity, dim=(1)).cpu().numpy()
        )
        Logger.debug("sum_intensity shape %s", sum_intensity.shape)
    result = dict(sum_intensity=sum_intensity, pept_mz_rank=pept_mz_rank)
    return pd.DataFrame(result)


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super(CNNBlock, self).__init__()

        self.seq_block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.seq_block(x)
        return x


class CNNBlocks(nn.Module):
    """
    Parameters:
    n_conv (int): creates a block of n_conv convolutions
    in_channels (int): number of in_channels of the first block's convolution
    out_channels (int): number of out_channels of the first block's convolution
    expand (bool) : if True after the first convolution of a blocl the number of channels doubles
    """

    def __init__(self, n_conv, in_channels, out_channels, padding):
        super(CNNBlocks, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(n_conv):
            self.layers.append(CNNBlock(in_channels, out_channels, padding=padding))
            # after each convolution we set (next) in_channel to (previous) out_channels
            in_channels = out_channels

    def forward(self, x):
        for layer in self.layers:
            # Logger.debug("Input x shape: %s", x.shape)
            x = layer(x)
            # Logger.debug("Output shape: %s", x.shape)
            return x


class Encoder(nn.Module):
    """
    Parameters:
    in_channels (int): number of in_channels of the first CNNBlocks
    out_channels (int): number of out_channels of the first CNNBlocks
    padding (int): padding applied in each convolution
    downhill (int): number times a CNNBlocks + MaxPool2D it's applied.
    """

    def __init__(self, in_channels, out_channels, padding, downhill=4):
        super(Encoder, self).__init__()
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
        # doubling the dept of the last CNN block
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
            # Logger.debug("Layer: %s", layer)
            # Logger.debug("Input x shape: %s", x.shape)
            if isinstance(layer, CNNBlocks):
                x = layer(x)
                route_connection.append(x)
            else:
                x = layer(x)
            # Logger.debug("Output shape: %s", x.shape)
        return x, route_connection


class Decoder(nn.Module):
    """
    Parameters:
    in_channels (int): number of in_channels of the first ConvTranspose2d
    out_channels (int): number of out_channels of the first ConvTranspose2d
    padding (int): padding applied in each convolution
    uphill (int): number times a ConvTranspose2d + CNNBlocks it's applied.
    """

    def __init__(self, in_channels, out_channels, exit_channels, padding, uphill=4):
        super(Decoder, self).__init__()
        self.exit_channels = exit_channels
        self.layers = nn.ModuleList()

        for i in range(uphill):
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

        # cannot be a CNNBlock because it has ReLU incorpored
        # cannot append nn.Sigmoid here because you should be later using
        # BCELoss () which will trigger the amp error "are unsafe to autocast".
        self.layers.append(
            nn.Conv2d(in_channels, exit_channels, kernel_size=1, padding=padding),
        )

    def forward(self, x, routes_connection):
        # pop the last element of the list since
        # it's not used for concatenation
        routes_connection.pop(-1)
        for layer in self.layers:
            if isinstance(layer, CNNBlocks):
                # center_cropping the route tensor to make width and height match
                routes_connection[-1] = center_crop(routes_connection[-1], x.shape[2])
                # Logger.debug("Route shape: %s", routes_connection[-1].shape)
                # concatenating tensors channel-wise
                x = torch.cat([x, routes_connection.pop(-1)], dim=1)
                # Logger.debug("Concatenated x shape: %s", x.shape)
                x = layer(x)
            else:
                x = layer(x)
        return x


class UNET(nn.Module):
    def __init__(
        self, in_channels, first_out_channels, exit_channels, downhill, padding=0
    ):
        super(UNET, self).__init__()
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
        # Logger.debug("Routes: %s", routes.shape)
        out = self.decoder(enc_out, routes)
        # out = torch.sigmoid(out)
        return out
