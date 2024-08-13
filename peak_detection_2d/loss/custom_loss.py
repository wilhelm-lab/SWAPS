import logging

import torch
import torch.nn as nn
from torchvision.ops import complete_box_iou_loss, box_iou

Logger = logging.getLogger(__name__)


class CIoULoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super(CIoULoss, self).__init__()
        self.reduction = reduction

    def forward(
        self, input_bbox: torch.Tensor, target_bbox: torch.Tensor
    ) -> torch.Tensor:
        return -complete_box_iou_loss(input_bbox, target_bbox, reduction=self.reduction)


class WeightedBoundingBoxIoULoss(nn.Module):
    def __init__(
        self,
        reduction: str = "mean",
        add_diou: bool = False,
        add_smooth_l1: bool = False,
    ):
        super(WeightedBoundingBoxIoULoss, self).__init__()
        self.reduction = reduction
        self.add_diou = add_diou
        self.add_smoothL1 = add_smooth_l1

    def forward(self, predicted_boxes, target_boxes, target_heatmap):
        # Compute intersection and union for each bounding box
        intersection = self.compute_intersection(
            predicted_boxes, target_boxes, target_heatmap
        )
        Logger.debug("intersection dimension: %s", intersection.shape)
        union = self.compute_union(predicted_boxes, target_boxes, target_heatmap)
        Logger.debug("union dimension: %s", union.shape)

        # Compute weights based on target heatmap values
        weights = target_heatmap
        Logger.debug("weights dimension: %s", weights.shape)
        # Compute weighted IoU for each bounding box
        weighted_iou = torch.sum(torch.mul(intersection, weights), dim=(1, 2, 3)) / (
            torch.sum(torch.mul(union, weights), dim=(1, 2, 3)) + (1e-7)
        )
        Logger.debug(
            "intersection value %s",
            torch.sum(torch.mul(intersection, weights), dim=(1, 2, 3)),
        )
        Logger.debug(
            "union value %s",
            torch.sum(torch.mul(union, weights), dim=(1, 2, 3)) + (1e-7),
        )

        # Compute mean loss over all bounding boxes
        weighted_iou = torch.where(
            weighted_iou > 1, torch.zeros_like(weighted_iou), weighted_iou
        )
        weighted_iou = torch.mean(weighted_iou)
        # Logger.info("weighted iou mean: %s", weighted_iou)

        if self.add_diou:
            diou_loss = self._diou_iou_loss(predicted_boxes, target_boxes)
            Logger.debug(
                "diou loss: %s, diou loss shape: %s", diou_loss, diou_loss.shape
            )
            diou_loss = torch.mean(diou_loss)
            total_loss = 1 - weighted_iou + diou_loss
        elif self.add_smoothL1:
            smooth_l1_loss = nn.SmoothL1Loss(reduction="mean")
            smooth_l1_loss = smooth_l1_loss(predicted_boxes, target_boxes)
            total_loss = 1 - weighted_iou + smooth_l1_loss
        else:
            total_loss = 1 - weighted_iou

        return total_loss

    def compute_intersection(self, boxes1, boxes2, target_heatmap):
        intersection = torch.zeros_like(target_heatmap)
        Logger.debug("intersection dimension: %s", intersection.shape)
        Logger.debug("boxes1: %s, boxes2: %s", boxes1.shape, boxes2.shape)
        # Compute intersection for each pair of bounding boxes
        x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
        y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
        x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
        y2 = torch.min(boxes1[:, 3], boxes2[:, 3])
        Logger.debug("Intersection box: x1: %s, y1: %s, x2: %s, y2: %s", x1, y1, x2, y2)
        for i in range(boxes1.shape[0]):
            intersection[i, :, int(y1[i]) : int(y2[i]), int(x1[i]) : int(x2[i])] = 1

        Logger.debug(
            "Non zero count in interestion %s", torch.count_nonzero(intersection)
        )
        return intersection

    def compute_union(self, boxes1, boxes2, target_heatmap):
        union = torch.zeros_like(target_heatmap)
        # Compute union for each pair of bounding boxes
        x1 = torch.min(boxes1[:, 0], boxes2[:, 0])
        y1 = torch.min(boxes1[:, 1], boxes2[:, 1])
        x2 = torch.max(boxes1[:, 2], boxes2[:, 2])
        y2 = torch.max(boxes1[:, 3], boxes2[:, 3])
        for i in range(boxes1.shape[0]):
            union[i, :, int(y1[i]) : int(y2[i]), int(x1[i]) : int(x2[i])] = 1
            if torch.count_nonzero(union[i]) == 0:
                Logger.warning("Union box is empty")
                Logger.info(
                    "Union box: x1: %s, y1: %s, x2: %s, y2: %s",
                    x1[i],
                    y1[i],
                    x2[i],
                    y2[i],
                )
        Logger.debug("Non zero count in union %s", torch.count_nonzero(union))
        return union

    def _diou_iou_loss(
        self,
        boxes1: torch.Tensor,
        boxes2: torch.Tensor,
        eps: float = 1e-7,
    ):
        # intsct, union = _loss_inter_union(boxes1, boxes2)
        # iou = intsct / (union + eps)
        # smallest enclosing box
        x1, y1, x2, y2 = boxes1.unbind(dim=-1)
        x1g, y1g, x2g, y2g = boxes2.unbind(dim=-1)
        xc1 = torch.min(x1, x1g)
        yc1 = torch.min(y1, y1g)
        xc2 = torch.max(x2, x2g)
        yc2 = torch.max(y2, y2g)
        # The diagonal distance of the smallest enclosing box squared
        diagonal_distance_squared = ((xc2 - xc1) ** 2) + ((yc2 - yc1) ** 2) + eps
        # centers of boxes
        x_p = (x2 + x1) / 2
        y_p = (y2 + y1) / 2
        x_g = (x1g + x2g) / 2
        y_g = (y1g + y2g) / 2

        centers_distance_squared = ((x_p - x_g) ** 2) + ((y_p - y_g) ** 2)
        distance = centers_distance_squared / diagonal_distance_squared
        return distance


def metric_iou_batch(output_bbox, target_bbox):
    return torch.trace(box_iou(output_bbox, target_bbox)).item()


def metric_wiou_batch(output_bbox, target_bbox, target_heatmap):
    """Compute the weighted IoU metric for a batch of bounding boxes."""
    wiou_loss = (
        WeightedBoundingBoxIoULoss(add_diou=False)
        .forward(output_bbox, target_bbox, target_heatmap)
        .item()
    )
    # Logger.info("wiou loss: %s", wiou_loss)
    return 1 - wiou_loss


def batch_loss(loss_func, output, target, optimizer=None, **kwargs):
    loss = loss_func(output, target, **kwargs)
    with torch.no_grad():
        iou_metric = metric_iou_batch(output, target)
    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item(), iou_metric


def batch_wiou_loss(loss_func, output, target, target_heatmap, optimizer=None):
    loss = loss_func(output, target, target_heatmap)
    with torch.no_grad():
        iou_metric = metric_wiou_batch(output, target, target_heatmap)
        # Logger.info("iou metric: %s", iou_metric)
    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item(), iou_metric


# pixel-wise accuracy
def acc_metric(input, target):
    inp = torch.where(
        input > 0.5, torch.tensor(1, device="cuda"), torch.tensor(0, device="cuda")
    )
    acc = (inp.squeeze(1) == target).float().mean()
    return acc


# from https://www.kaggle.com/rishabhiitbhu/unet-with-resnet34-encoder-pytorch
def dice_metric_backup(probability, truth, threshold=0.5, reduction="none"):
    batch_size = len(truth)
    with torch.no_grad():
        probability = probability.view(batch_size, -1)
        truth = truth.view(batch_size, -1)
        assert probability.shape == truth.shape

        p = (probability > threshold).float()
        t = (truth > 0.5).float()

        t_sum = t.sum(-1)
        p_sum = p.sum(-1)
        neg_index = torch.nonzero(t_sum == 0)
        pos_index = torch.nonzero(t_sum >= 1)

        dice_neg = (p_sum == 0).float()
        dice_pos = 2 * (p * t).sum(-1) / ((p + t).sum(-1))

        dice_neg = dice_neg[neg_index]
        dice_pos = dice_pos[pos_index]
        dice = torch.cat([dice_pos, dice_neg])

        num_neg = len(neg_index)
        num_pos = len(pos_index)

    return dice


# from https://www.kaggle.com/rishabhiitbhu/unet-with-resnet34-encoder-pytorch
def dice_metric(pred, target):  # handles batch input as well, directly reduced
    pred = torch.sigmoid(pred)
    smooth = 1.0
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    diceloss = (2.0 * intersection + smooth) / (
        pred_flat.sum() + target_flat.sum() + smooth
    )
    # Logger.info("dice loss shape: %s", diceloss.item())
    return diceloss


# from https://www.kaggle.com/rishabhiitbhu/unet-with-resnet34-encoder-pytorch
def per_image_dice_metric(
    pred, target, threshold: None | float = 0.5, device: str = "cuda"
):  # return dice loss for each image in a batch
    batch_size = len(target)
    pred = torch.sigmoid(pred)
    if threshold is not None:
        pred = torch.where(
            pred > threshold,
            torch.tensor(1, device=device),
            torch.tensor(0, device=device),
        )
    smooth = 1.0
    pred_flat = pred.view(batch_size, -1)
    target_flat = target.view(batch_size, -1)
    intersection = (pred_flat * target_flat).sum(axis=1)
    Logger.debug("intersection shape: %s", intersection.shape)
    diceloss = (2.0 * intersection + smooth) / (
        pred_flat.sum(axis=1) + target_flat.sum(axis=1) + smooth
    )
    Logger.info("dice loss shape: %s", diceloss.shape)
    return diceloss


def weighted_dice_metric(pred, target, image):
    pred = torch.sigmoid(pred)
    smooth = 1.0
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    image_flat = image.view(-1)
    weighted_pred_flat = image_flat * pred_flat
    weighted_target_flat = image_flat * target_flat
    weighted_intersection = (image_flat * pred_flat * target_flat).sum()
    diceloss = (2.0 * weighted_intersection + smooth) / (
        weighted_pred_flat.sum() + weighted_target_flat.sum() + smooth
    )
    # Logger.info("dice loss shape: %s", diceloss.item())
    return diceloss


def per_image_weighted_dice_metric(
    pred,
    target,
    image,
    threshold: None | float = 0.5,
    device: str = "cuda",
    channel: int = None,
    exp: bool = False,
):  # return dice loss for each image in a batch
    """Compute the weighted dice loss for each image in a batch.
    Set threshold to None when using as a loss function"""
    batch_size = len(target)
    pred = torch.sigmoid(pred)
    if channel is not None:
        image = image[:, channel, :, :]
    if threshold is not None:
        pred = torch.where(
            pred > threshold,
            torch.tensor(1, device=device),
            torch.tensor(0, device=device),
        )

    smooth = 1e-6
    pred_flat = pred.view(batch_size, -1)
    target_flat = target.view(batch_size, -1)
    image_flat = image.view(batch_size, -1)
    if exp:  # exponential weighting for image
        image_flat = torch.exp(image_flat) - 1
    weighted_pred_flat = image_flat * pred_flat
    weighted_target_flat = image_flat * target_flat
    weighted_intersection = (pred_flat * target_flat * image_flat).sum(axis=1)
    # Logger.info("weighted intersection: %s", weighted_intersection)
    weighted_union = (
        weighted_pred_flat.sum(axis=1) + weighted_target_flat.sum(axis=1) + smooth
    )
    # Logger.info("weighted union: %s", weighted_union)
    # Logger.debug("intersection shape: %s", intersection.shape)
    weighted_diceloss = (2.0 * weighted_intersection + smooth) / weighted_union
    # Logger.info("dice loss shape: %s", diceloss.shape)
    return weighted_diceloss


def per_image_weighted_iou_metric(
    pred,
    target,
    image,
    threshold: None | float = 0.5,
    device: str = "cuda",
    channel: int = None,
    exp: bool = False,
):
    """Compute the weighted dice loss for each image in a batch.
    Set threshold to None when using as a loss function"""
    batch_size = len(target)
    pred = torch.sigmoid(pred)
    if threshold is not None:
        pred = torch.where(
            pred > threshold,
            torch.tensor(1, device=device),
            torch.tensor(0, device=device),
        )
    if channel is not None:
        image = image[:, channel, :, :]
    smooth = 0.00000001
    pred_flat = pred.view(batch_size, -1)
    target_flat = target.view(batch_size, -1)
    image_flat = image.view(batch_size, -1)
    if exp:
        image_flat = torch.exp(image_flat) - 1
    weighted_pred_flat = image_flat * pred_flat
    weighted_target_flat = image_flat * target_flat
    weighted_intersection = (pred_flat * target_flat * image_flat).sum(axis=1)

    # Logger.debug("intersection shape: %s", intersection.shape)
    weighted_iou = (weighted_intersection + smooth) / (
        weighted_pred_flat.sum(axis=1)
        + weighted_target_flat.sum(axis=1)
        - weighted_intersection
        + smooth
    )
    # Logger.info("dice loss shape: %s", diceloss.shape)
    return weighted_iou


class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError(
                "Target size ({}) must be the same as input size ({})".format(
                    target.size(), input.size()
                )
            )
        max_val = (-input).clamp(min=0)
        loss = (
            input
            - input * target
            + max_val
            + ((-max_val).exp() + (-input - max_val).exp()).log()
        )
        invprobs = nn.LogSigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        return loss.mean()

    # from https://www.kaggle.com/rishabhiitbhu/unet-with-resnet34-encoder-pytorch


class MixedLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma)

    def forward(self, input, target):
        loss = self.alpha * self.focal(input, target) - torch.log(
            dice_metric(input, target)
        )
        return loss.mean()
