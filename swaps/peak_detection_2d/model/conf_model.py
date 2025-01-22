import logging
import os
from re import T
from typing import Literal
import matplotlib.pyplot as plt
import numpy as np
from sympy import plot
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from .seg_model import AverageMeter, Encoder
import pandas as pd
from ..loss.custom_loss import per_image_weighted_iou_metric

Logger = logging.getLogger(__name__)


def _calc_linear_line_size(image_size: int, downhill: int):
    for i in range(downhill):
        image_size = (image_size - 2) // 2
    return image_size - 2


class CNNEncoderRegressor(nn.Module):
    def __init__(
        self,
        image_size=258,
        in_channels=1,
        first_out_channels=16,
        downhill=4,
        padding=0,
        dropout_rate=0.25,
        sigmoid_output=False,
    ):
        super(CNNEncoderRegressor, self).__init__()
        self.encoder = Encoder(
            in_channels, first_out_channels, padding=padding, downhill=downhill
        )
        # self.linear_line_size = 12 * 12 * first_out_channels * 2**downhill
        self.linear_line_size = (
            _calc_linear_line_size(image_size, downhill) ** 2
            * first_out_channels
            * 2**downhill
        )
        self.fc1 = nn.Linear(in_features=self.linear_line_size, out_features=128)
        self.dropout = nn.Dropout(p=dropout_rate)  # Add dropout
        self.fc2 = nn.Linear(in_features=128, out_features=1)
        if sigmoid_output:
            self.fc2 = nn.Sequential(self.fc2, nn.Sigmoid())

    def forward(self, x):
        enc_out, routes = self.encoder(x)
        x = enc_out.view(-1, self.linear_line_size)
        # out = torch.sigmoid(out)
        x = self.fc1(x)
        # print(f'fc1 {x.size()}')
        x = nn.functional.relu(x)
        # print(f'relu2 {x.size()}')
        x = self.dropout(x)
        x = self.fc2(x)
        # x = torch.sigmoid(x)
        return x


class CNNRegression(nn.Module):
    """
    This will be the very basic CNN model we will use for the regression task.
    """

    def __init__(self, image_size: tuple[int, int, int] = (1, 258, 258)):
        super(CNNRegression, self).__init__()
        self.image_size = image_size
        self.conv1 = nn.Conv2d(
            in_channels=self.image_size[0],
            out_channels=4,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=4, out_channels=16, kernel_size=3, stride=1, padding=1
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear_line_size = int(16 * (image_size[1] // 4) * (image_size[2] // 4))
        self.fc1 = nn.Linear(in_features=self.linear_line_size, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=1)

    def forward(self, x):
        """
        Passes the data through the network.
        There are commented out print statements that can be used to
        check the size of the tensor at each layer. These are very useful when
        the image size changes and you want to check that the network layers are
        still the correct shape.
        """
        # x = torch.sigmoid(x)
        x = self.conv1(x)
        # print('Size of tensor after each layer')
        # print(f'conv1 {x.size()}')
        x = nn.functional.relu(x)
        # print(f'relu1 {x.size()}')
        x = self.pool1(x)
        # print(f'pool1 {x.size()}')
        x = self.conv2(x)
        # print(f'conv2 {x.size()}')
        x = nn.functional.relu(x)
        # print(f'relu2 {x.size()}')
        x = self.pool2(x)
        # print(f'pool2 {x.size()}')
        x = x.view(-1, self.linear_line_size)
        # print(f'view1 {x.size()}')
        x = self.fc1(x)
        # print(f'fc1 {x.size()}')
        x = nn.functional.relu(x)
        # print(f'relu2 {x.size()}')
        x = self.fc2(x)
        # print(f'fc2 {x.size()}')
        return x


class ConfidenceModel(nn.Module):
    def __init__(self, in_channels=1):
        super(ConfidenceModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        # Confidence head
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # Global average pooling
        self.confidence_fc = nn.Linear(
            64, 1
        )  # Fully connected layer for confidence score

    def forward(self, x):
        x = torch.sigmoid(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        confidence_features = self.global_pool(x)

        confidence_features = confidence_features.view(
            confidence_features.size(0), -1
        )  # Flatten
        confidence_output = self.confidence_fc(confidence_features)
        # Apply sigmoid

        return confidence_output


def train_one_epoch(
    train_loader,
    model,
    optimizer,
    loss_fn,
    accumulation_steps=1,
    device="cuda",
    scheduler=None,
):
    epoch_losses = AverageMeter()
    model = model.to(device)
    model.train()
    if accumulation_steps > 1:
        optimizer.zero_grad()
    tk0 = tqdm(train_loader, total=len(train_loader))
    for seg_output_batch, conf_score_batch in tk0:
        out = model(seg_output_batch.float().to(device))
        b_loss = loss_fn(out.view(-1).to(device), conf_score_batch.to(device))
        with torch.set_grad_enabled(True):
            b_loss.backward()
            # Gradient check
            # for param in model.parameters():
            #     if param.grad is not None:
            #         print(f"Gradient for {param.shape}: {param.grad.abs().mean()}")

            optimizer.step()
            optimizer.zero_grad()
        epoch_losses.update(b_loss.mean().item(), train_loader.batch_size)
        tk0.set_postfix(
            loss=epoch_losses.avg,
            learning_rate=optimizer.param_groups[0]["lr"],
            # gradient=model.parameters().grad.abs().mean(),
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
    save_output_prediction: bool = False,
    save_target: bool = False,
    **kwargs,
):
    epoch_losses = AverageMeter()
    model = model.to(device)
    model.eval()
    tk0 = tqdm(valid_loader, total=len(valid_loader))
    output_pred = np.empty((0))
    target = np.empty((0))
    losses = np.empty((0))

    with torch.no_grad():
        for seg_output_batch, conf_score_batch in tk0:
            out = model(seg_output_batch.float().to(device))
            # Logger.debug("out range %s, %s", out.min().item(), out.max().item())
            # Logger.debug(
            #     "conf_score_batch range %s, %s",
            #     conf_score_batch.min().item(),
            #     conf_score_batch.max().item(),
            # )
            b_loss = metric(
                out.view(-1).to(device), conf_score_batch.to(device), **kwargs
            )
            epoch_losses.update(b_loss.mean().item(), valid_loader.batch_size)
            tk0.set_postfix(loss=epoch_losses.avg)
            if save_all_loss:
                losses = np.append(losses, b_loss.cpu().numpy())
            if save_output_prediction:
                output_pred = np.append(output_pred, out.cpu().numpy())
            if save_target:
                target = np.append(target, conf_score_batch.cpu().numpy())
            # Logger.info("Loss: %s", b_loss.cpu().numpy())
            # Logger.info("all_losses: %s", all_losses)
    if any([save_all_loss, save_output_prediction, save_target]):
        other_outputs = dict(losses=losses, output_pred=output_pred, target=target)
        return epoch_losses.avg, other_outputs
    else:
        return epoch_losses.avg


def _average_pixel_prob(
    out_pred_score,
    batch_size,
    device,
    threshold: float = 0.5,
    convert_to_confidence: bool = False,
):
    if isinstance(out_pred_score, np.ndarray):
        out_pred_score = torch.tensor(out_pred_score).to(device)
    out_score_view = out_pred_score.contiguous().view(batch_size, -1).float().to(device)
    valid_pixel = out_score_view > threshold

    if convert_to_confidence:
        out_score_view = (
            abs(0.5 - out_score_view) * 2
        )  # convert probability of class 1 to binary class confidence
    masked_tensor = torch.where(valid_pixel, out_score_view, torch.tensor(float("nan")))
    Logger.debug("valid_pixel shape %s", valid_pixel.shape)
    # out_score = np.append(
    #     out_score,
    #     torch.nanmean(masked_tensor, dim=(1)).cpu().numpy(),
    # )
    # Logger.debug("out_score shape %s", out_score.shape)
    return torch.nanmean(masked_tensor, dim=(1)).cpu()


def inference_and_sum_intensity(
    data_loader,
    model,
    device="cuda",
    threshold: float = 0.5,
    channel: int = 0,
    calc_score: Literal["conf_model", "sigmoid", "iso_reg"] = "sigmoid",
    calib_model=None,
    conf_model=None,
    plot_calib_score_distribution: bool = False,
    result_dir: str = None,
    exp: bool = False,
):
    model = model.to(device)
    model.eval()
    sum_intensity = np.empty((0))
    pept_mz_rank = np.empty((0))
    out_score = np.empty((0))
    wiou_array = np.empty((0))
    label_sum_intensity = np.empty((0))
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

            match calc_score:
                case "conf_model":
                    assert conf_model is not None, "Confidence model is not provided"
                    conf_model.eval()
                    out_score = np.append(
                        out_score, conf_model(out.float().to(device)).cpu().numpy()
                    )
                    out_pred_score = torch.sigmoid(out)
                    Logger.debug(
                        "out_score range %s, %s",
                        out_score.min().item(),
                        out_score.max().item(),
                    )
                case "sigmoid":
                    out_pred_score = torch.sigmoid(out)
                    image_out_score = _average_pixel_prob(
                        out_pred_score,
                        batch_size,
                        device,
                        threshold=threshold,
                        convert_to_confidence=True,
                    )
                    out_score = np.append(out_score, image_out_score)
                case "iso_reg":
                    out_pred_score = calib_model.predict(
                        out.contiguous().view(-1).float().cpu()
                    )
                    if plot_calib_score_distribution:
                        plt.hist(
                            out_pred_score, bins=100, log=True
                        )  # TODO: for one image
                        plt.title("Isotonic regression output distribution")
                        plt.savefig(
                            os.path.join(result_dir, "isoreg_output_distribution.png"),
                            dpi=100,
                        )
                        plt.close()
                    image_out_score = _average_pixel_prob(
                        out_pred_score,
                        batch_size,
                        device,
                        threshold=threshold,
                        convert_to_confidence=True,
                    )
                    out_pred_score = torch.tensor(out_pred_score).to(device)
                    out_score = np.append(out_score, image_out_score)

            out_final = torch.where(
                out_pred_score > threshold,
                torch.ones_like(out_pred_score),
                torch.zeros_like(out_pred_score),
            )

            out_final = out_final.contiguous().view(batch_size, -1).float().to(device)
            if exp: # TODO: remove this and always use ori_image_raw
                Logger.warning(
                    "Exponential transformation is applied on transformed image"
                )
                image_batch = torch.exp(image_batch) - 1
            intensity = (
                # image_batch[:, channel, :, :]
                label_batch["ori_image_raw"]  # TODO: change to ori_image_raw
                .contiguous()
                .view(batch_size, -1)
                .float()
                .to(device)
            )
            wiou = per_image_weighted_iou_metric(
                out.to(device),
                label_batch["mask"].to(device),
                # image_batch.to(device),
                label_batch["ori_image_raw"].to(device),
                device=device,
                channel=None,  # TODO: changed to ori_image_raw, channel not considered anymore
            )
            wiou_array = np.append(wiou_array, wiou.cpu().numpy())
            Logger.debug("pept_mz_rank shape %s", pept_mz_rank.shape)
            sum_intensity = np.append(
                sum_intensity, torch.sum(intensity * out_final, dim=(1)).cpu().numpy()
            )
            Logger.debug("sum_intensity shape %s", sum_intensity.shape)
            label_sum_intensity = np.append(
                label_sum_intensity,
                torch.sum(
                    intensity * label_batch["mask"].view(batch_size, -1).to(device),
                    dim=(1),
                )
                .cpu()
                .numpy(),
            )
    result = dict(
        sum_intensity=sum_intensity,
        pept_mz_rank=pept_mz_rank,
        out_score=out_score,  # confidence score
        wiou=wiou_array,
        label_sum_intensity=label_sum_intensity,
    )
    return pd.DataFrame(result)
