import logging
from typing import Literal
import numpy as np

from pathlib import Path
from tqdm import tqdm
import os
import h5py
import seaborn as sns
from matplotlib import patches
from matplotlib.colors import ListedColormap, BoundaryNorm
from utils.plot import save_plot
import torch

import matplotlib.pyplot as plt
from peak_detection_2d.loss.custom_loss import (
    metric_iou_batch,
    metric_wiou_batch,
    per_image_dice_metric,
    per_image_weighted_dice_metric,
    per_image_weighted_iou_metric,
)

Logger = logging.getLogger(__name__)


class EarlyStopping:
    def __init__(self, patience=7, mode="max", delta=0.0001):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(
        self, epoch_score, epoch_num, loss, optimizer, model, model_path, scheduler=None
    ):
        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(
                epoch_score, epoch_num, loss, optimizer, model, model_path, scheduler
            )
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(
                "EarlyStopping counter: {} out of {}".format(
                    self.counter, self.patience
                )
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
        self,
        epoch_score,
        epoch_num,
        loss,
        optimizer,
        model,
        model_path,
        scheduler=None,
    ):
        model_path = Path(model_path)
        parent = model_path.parent
        os.makedirs(parent, exist_ok=True)
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            print(
                "Validation score improved ({} --> {}). Model saved at at {}!".format(
                    self.val_score, epoch_score, model_path
                )
            )
            save_state = {
                "epoch": epoch_num,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
            }
            if scheduler is not None:
                save_state["scheduler_state_dict"] = scheduler.state_dict()
            torch.save(
                save_state,
                model_path,
            )
        self.val_score = epoch_score


def plot_data_points(
    dp_dict,
    log_data: bool = False,
    pred_bbox: list | None = None,
    pred_mask: np.ndarray | None = None,
    zoom_in: bool = False,
    label: Literal["bbox", "mask", "hide"] = "bbox",
):
    # ax = sns.heatmap(dp_dict["data"], cmap="icefire")
    fig, ax = plt.subplots()
    if log_data:
        heatmap = ax.imshow(np.log(dp_dict["data"] + 1), cmap="binary")
    else:
        heatmap = ax.imshow(dp_dict["data"], cmap="binary")
    # Define colors with transparency: -1 is blue, 0 is translucent white, 1 is red
    colors = [
        (0, 0, 1, 1),  # Blue for -1
        (1, 1, 1, 0.5),  # Translucent white for 0
        (1, 0, 0, 1),
    ]  # Red for 1

    # Create a colormap
    cmap = ListedColormap(colors)

    # Define boundaries for discrete mapping
    bounds = [-1.5, -0.5, 0.5, 1.5]  # Create boundaries to cover the discrete values
    norm = BoundaryNorm(bounds, cmap.N)
    # Logger.info("Peptide mz rank: %s", dp_dict["pept_mz_rank"])
    match label:
        case "bbox":
            ax.add_patch(
                patches.Rectangle(
                    xy=(dp_dict["bbox"][0], dp_dict["bbox"][1]),
                    width=dp_dict["bbox"][2] - dp_dict["bbox"][0],
                    height=dp_dict["bbox"][3] - dp_dict["bbox"][1],
                    edgecolor="red",
                    fill=False,
                    linestyle="-",
                    lw=3,
                    label="True",
                )
            )
        case "mask":
            ax.imshow(dp_dict["mask"], cmap="Blues", alpha=0.3, label="true")
            try:
                ax.imshow(
                    dp_dict["hint_channel"],
                    cmap=cmap,
                    norm=norm,
                    alpha=0.5,
                    label="hint",
                )
            except KeyError:
                pass
        case "hide":
            pass
        case _:
            raise ValueError(f"Unknown option {label}")
    if pred_bbox is not None:
        ax.add_patch(
            patches.Rectangle(
                xy=(pred_bbox[0], pred_bbox[1]),
                width=pred_bbox[2] - pred_bbox[0],
                height=pred_bbox[3] - pred_bbox[1],
                edgecolor="red",
                fill=False,
                linestyle="--",
                lw=3,
                label="Pred",
            )
        )
    if pred_mask is not None:
        ax.imshow(pred_mask, cmap="Reds", alpha=0.3, label="pred")

    # ax.plot(dp_dict["hint_idx"][1], dp_dict["hint_idx"][0], "ro", label="Hint")
    ax.set_xlabel("ion mobility (AU)")
    ax.set_ylabel("RT (AU)")
    ax.legend()
    if zoom_in:
        if pred_bbox is None:
            pred_bbox = dp_dict["bbox"]
        x_min = min(dp_dict["bbox"][0], dp_dict["hint_idx"][1], pred_bbox[0]) - 10
        x_max = max(dp_dict["bbox"][2], dp_dict["hint_idx"][1], pred_bbox[2]) + 10
        y_min = min(dp_dict["bbox"][1], dp_dict["hint_idx"][0], pred_bbox[1]) - 10
        y_max = max(dp_dict["bbox"][3], dp_dict["hint_idx"][0], pred_bbox[3]) + 10
        plt.axis([x_min, x_max, y_max, y_min])


def save_data_points_to_hdf5(futures, output_filename):
    """save data points to hdf5 file, each data point is a group"""
    with h5py.File(output_filename, "a") as f:
        for future in tqdm(futures):
            result = future.result()
            # Logger.info("group name: %s", f"pept_mz_rank_{result['pept_mz_rank']}")
            group = f.create_group(f"pept_mz_rank_{result['pept_mz_rank']}")
            for key, value in result.items():
                group.create_dataset(key, data=value)


def load_data_from_hdf5(input_filename):
    """Load data points from hdf5 file"""
    data_points = []
    with h5py.File(input_filename, "r") as f:
        # Iterate over each group in the HDF5 file
        for group_name in f.keys():
            data_point = {}
            group = f[group_name]
            # Iterate over each dataset within the group
            for dataset_name in group.keys():
                # Check if the dataset is scalar
                if (
                    isinstance(group[dataset_name], h5py.Dataset)
                    and group[dataset_name].shape == ()
                ):
                    # If scalar, directly assign the value to the data point dictionary
                    data_point[dataset_name] = group[dataset_name][()]
                else:
                    # If not scalar, retrieve the dataset value and store it in the data point dictionary
                    data_point[dataset_name] = group[dataset_name][:]
            # Append the data point dictionary to the list of data points
            data_points.append(data_point)
    return data_points


def plot_history(history, title: str = "loss", save_dir=None):
    plt.plot(history["train"], label="train")
    plt.plot(history["val"], label="val")
    if title is not None:
        plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(title)
    plt.legend()
    save_plot(save_dir, fig_type_name="PS_model_training_history", fig_spec_name=title)


def plot_per_image_metric_distr(
    loss_array, metric_name, save_dir, show_quantiles=[25, 50, 75]
):
    # Calculate quantiles
    quantiles = np.percentile(loss_array, show_quantiles)

    plt.figure(figsize=(10, 5))
    plt.hist(loss_array, bins=100, alpha=0.75)
    correction = 0.05
    # Plot quantile lines
    for show_quantile, quantile in zip(show_quantiles, quantiles):
        print(f"{show_quantile}%: {quantile:.2f}")
        plt.axvline(quantile, color="r", linestyle="dashed", linewidth=1)
        plt.text(
            quantile,
            plt.ylim()[1] * (0.9 + correction),
            f"{show_quantile}%:{quantile:.2f}",
            color="r",
            ha="center",
        )
        correction *= -1
    plt.title(f"{metric_name} Distribution")
    plt.xlabel(metric_name)
    plt.ylabel("Frequency")
    save_plot(
        save_dir=save_dir,
        fig_type_name="PS_model",
        fig_spec_name=f"test_{metric_name}_distribution",
    )


def plot_sample_predictions(
    dataset,
    save_dir,
    model,
    conf_model=None,
    threshold: float = 0.5,
    n: int = 5,
    sample_indices=None,
    # metric: Literal["iou", "wiou", "dice", "wdice", "mask_wiou"] = "wiou",
    metric_list: list = ["iou", "wiou", "dice", "wdice", "mask_wiou"],
    use_hint: bool = True,
    label: Literal["bbox", "mask"] = "bbox",
    zoom_in: bool = True,
    device="cuda",
    channel: int = 0,
    **kwargs,
):
    if sample_indices is None:
        sample_indices = np.random.choice(len(dataset), n, replace=False)
        Logger.info("Sample indices: %s", sample_indices)
    for i in sample_indices:
        with torch.no_grad():
            image, hint, target_dict = dataset[i]
            target = target_dict["mask"].to(device)
            if use_hint:
                output = model(image.unsqueeze(0).float(), hint.unsqueeze(0).float())
            else:
                output = model(image.unsqueeze(0).float())
                if conf_model is not None:
                    conf_model.eval()
                    conf_score = conf_model(output)
                    Logger.info("Confidence score: %s", conf_score.item())
                # output = (output > threshold).float()
        metric_val_list = []
        for metric in metric_list:
            match metric:
                case "iou":
                    metric_val = metric_iou_batch(output, target.unsqueeze(0))

                case "wiou":
                    metric_val = metric_wiou_batch(
                        output, target.unsqueeze(0), image.unsqueeze(0).float()
                    )
                case "dice":
                    metric_val = per_image_dice_metric(output, target.unsqueeze(0))
                    metric_val = metric_val.item()
                case "wdice":
                    metric_val = per_image_weighted_dice_metric(
                        output, target, image.unsqueeze(0), channel=channel, **kwargs
                    )
                    metric_val = metric_val.item()
                case "mask_wiou":
                    metric_val = per_image_weighted_iou_metric(
                        output, target, image.unsqueeze(0), channel=channel, **kwargs
                    )
                    metric_val = metric_val.item()
            metric_val_list.append(metric_val)

        match label:
            case "bbox":
                to_plot = {
                    "data": image[0].cpu(),
                    "hint_idx": hint.cpu(),
                    "bbox": target.cpu(),
                }
                plot_data_points(
                    to_plot, pred_bbox=output[0].cpu().detach().numpy(), zoom_in=zoom_in
                )
            case "mask":
                pred = torch.sigmoid(output[0][0])
                if threshold is not None:
                    pred = torch.where(
                        pred > threshold,
                        torch.tensor(1, device=device),
                        torch.tensor(0, device=device),
                    )
                to_plot = {
                    "data": image[channel].cpu(),
                    "hint_idx": hint.cpu(),
                    "mask": target[0].cpu(),
                }
                plot_data_points(
                    to_plot,
                    pred_mask=pred.cpu().detach().numpy(),
                    zoom_in=zoom_in,
                    label="mask",
                )
                Logger.info("Masked area %s", target[0].cpu().sum().item())

                Logger.info(
                    "Masked intensity sum %.2f",
                    np.nansum(
                        np.multiply(
                            image[channel].cpu().numpy(), target[0].cpu().numpy()
                        )
                    ),
                )
                Logger.info(
                    "Pred masked intensity sum %.2f",
                    np.nansum(
                        np.multiply(
                            image[channel].cpu().numpy(),
                            pred.cpu().numpy(),
                        )
                    ),
                )
            case _:  # noqa
                raise ValueError(f"Unknown option {target}")
        metrics_str = "\n".join(
            [
                f"{name} value: {val:.2f}"
                for name, val in zip(metric_list, metric_val_list)
            ]
        )
        if conf_model is not None:
            metrics_str += f"\nConf. {conf_score.item():.2f}\n"
        plt.title(metrics_str + f"pept_mzrank: {int(target_dict['pept_mz_rank'])}")
        plt.legend()
        save_plot(
            save_dir=save_dir,
            fig_type_name="PS_model_prediction_sample",
            fig_spec_name=f"sample_{i}",
        )


def plot_confidence_distr(test_df, save_dir):
    sns.kdeplot(test_df, x="out_score", y="wiou")
    corr = test_df["wiou"].corr(test_df["out_score"])
    plt.title("Confidence Score Distribution, " + f"Correlation: {corr:.2f}")
    plt.xlabel("Confidence Score")
    plt.ylabel("Weighted IoU")
    save_plot(
        save_dir=save_dir,
        fig_type_name="conf_model",
        fig_spec_name="test_confidence_score_distribution",
    )
