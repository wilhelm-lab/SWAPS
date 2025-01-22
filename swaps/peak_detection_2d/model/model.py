import logging

import torch
import torch.nn as nn
from ..loss.custom_loss import batch_loss, batch_wiou_loss

Logger = logging.getLogger(__name__)


def train_val_step(dataloader, model, loss_func, optimizer=None):
    if optimizer is not None:
        model.train()
    else:
        model.eval()

    running_loss = 0
    running_iou = 0

    for image_batch, hint_batch, label_batch in dataloader:
        output_labels = model(image_batch.float(), hint_batch.float())
        loss_value, iou_metric_value = batch_loss(
            loss_func, output_labels, label_batch, optimizer
        )
        running_loss += loss_value
        running_iou += iou_metric_value
    step_val_loss = running_loss / len(dataloader.dataset)
    step_val_iou = running_iou / len(dataloader.dataset)
    return step_val_loss, step_val_iou


def train_val_step_wiou(dataloader, model, loss_func, optimizer=None):
    if optimizer is not None:
        Logger.info("model is in training mode")
        model.train()
    else:
        Logger.info("model is in evaluation mode")
        model.eval()

    running_loss = 0
    running_iou = 0
    n_batch = 0
    for image_batch, hint_batch, label_batch in dataloader:
        output_labels = model(image_batch.float(), hint_batch.float())
        loss_value, iou_metric_value = batch_wiou_loss(
            loss_func,
            output_labels,
            label_batch,
            target_heatmap=image_batch.float(),
            optimizer=optimizer,
        )
        running_loss += loss_value
        running_iou += iou_metric_value
        Logger.info("running iou: %s", running_iou)
        n_batch += 1
    step_val_loss = running_loss / n_batch
    step_val_iou = running_iou / n_batch
    return step_val_loss, step_val_iou


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.base1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding="same"),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
        )
        self.base2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, x):
        x = self.base1(x) + x
        x = self.base2(x)
        return x


class PeakDetectionNet(nn.Module):
    def __init__(self, in_channels, first_output_channels):
        super().__init__()
        self.conv = nn.Sequential(
            ResBlock(in_channels, first_output_channels),
            nn.MaxPool2d(2),
            ResBlock(first_output_channels, 2 * first_output_channels),
            nn.MaxPool2d(2),
            ResBlock(2 * first_output_channels, 4 * first_output_channels),
            nn.MaxPool2d(2),
            ResBlock(4 * first_output_channels, 8 * first_output_channels),
            nn.MaxPool2d(2),
            nn.Conv2d(
                8 * first_output_channels, 16 * first_output_channels, kernel_size=3
            ),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(256 * first_output_channels, 4),  # set up first FC layer
        )
        self.hint_fc = nn.Linear(2, out_features=4)  # FC layer for hint feature
        self.all_fc = nn.Linear(
            8, out_features=4
        )  # FC layer combines conv feature and hint

    def forward(self, img_input, hint_input):
        c = self.conv(img_input)
        f = self.hint_fc(hint_input)
        # now we can reshape `c` and `f` to 2D and concat them
        combined = torch.cat((c.view(c.size(0), -1), f.view(f.size(0), -1)), dim=1)
        out = self.all_fc(combined)
        # out = torch.sigmoid(out)
        return out
