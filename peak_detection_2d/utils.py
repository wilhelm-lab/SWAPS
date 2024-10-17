import logging
from typing import Literal, List
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import os
import math
import h5py
import seaborn as sns
from matplotlib import patches
from matplotlib.colors import ListedColormap, BoundaryNorm
from peak_detection_2d import dataset
from utils.plot import save_plot
import torch
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from peak_detection_2d.loss.custom_loss import (
    metric_iou_batch,
    metric_wiou_batch,
    per_image_dice_metric,
    per_image_weighted_dice_metric,
    per_image_weighted_iou_metric,
)
from peak_detection_2d.dataset.dataset import MultiHDF5_MaskDataset

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
    if "data" in dp_dict:
        if log_data:
            heatmap = ax.imshow(np.log(dp_dict["data"] + 1), cmap="binary")
        else:
            heatmap = ax.imshow(dp_dict["data"], cmap="binary")

    if "hint_channel" in dp_dict:

        # Logger.info("Peptide mz rank: %s", dp_dict["pept_mz_rank"])
        # Define colors with transparency: -1 is blue, 0 is translucent white, 1 is red
        colors = [
            (0, 0, 1, 1),  # Blue for -1
            (1, 1, 1, 0.5),  # Translucent white for 0
            (1, 0, 0, 1),
        ]  # Red for 1

        # Create a colormap
        cmap = ListedColormap(colors)
        # Define boundaries for discrete mapping
        bounds = [
            -1.5,
            -0.5,
            0.5,
            1.5,
        ]  # Create boundaries to cover the discrete values
        norm = BoundaryNorm(bounds, cmap.N)
        ax.imshow(
            dp_dict["hint_channel"],
            cmap=cmap,
            norm=norm,
            alpha=0.5,
            label="hint",
        )
        Logger.info("hint channel sum: %s", dp_dict["hint_channel"].sum().item())
        Logger.info(
            "hint channel non zero values: %s",
            dp_dict["hint_channel"][dp_dict["hint_channel"] != 0],
        )
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
    ax.set_xlabel("Ion Mobility (AU)")
    ax.set_ylabel("Retention Time (AU)")
    ax.legend()
    if zoom_in:
        if pred_bbox is None:
            pred_bbox = dp_dict["bbox"]
        x_min = min(dp_dict["bbox"][0], dp_dict["hint_idx"][1], pred_bbox[0]) - 10
        x_max = max(dp_dict["bbox"][2], dp_dict["hint_idx"][1], pred_bbox[2]) + 10
        y_min = min(dp_dict["bbox"][1], dp_dict["hint_idx"][0], pred_bbox[1]) - 10
        y_max = max(dp_dict["bbox"][3], dp_dict["hint_idx"][0], pred_bbox[3]) + 10
        plt.axis([x_min, x_max, y_max, y_min])


def plot_data_points_illustration(
    dp_dict,
    log_data: bool = False,
    pred_bbox: list | None = None,
    pred_mask: np.ndarray | None = None,
    zoom_in: bool = False,
    label: Literal["bbox", "mask", "hide"] = "bbox",
):
    # ax = sns.heatmap(dp_dict["data"], cmap="icefire")
    fig, ax = plt.subplots()
    if "data" in dp_dict:
        if log_data:
            heatmap = ax.imshow(np.log(dp_dict["data"] + 1), cmap="binary")
        else:
            heatmap = ax.imshow(dp_dict["data"], cmap="binary")
    if "seg_out" in dp_dict:
        Logger.info("seg_out shape: %s", dp_dict["seg_out"].shape)
        ax.imshow(dp_dict["seg_out"], cmap="Reds", label="pred")
    if "hint_channel" in dp_dict:

        # Logger.info("Peptide mz rank: %s", dp_dict["pept_mz_rank"])
        # Define colors with transparency: -1 is blue, 0 is translucent white, 1 is red
        colors = [
            (0, 0, 1, 1),  # Blue for -1
            (1, 1, 1, 0.5),  # Translucent white for 0
            (1, 0, 0, 1),
        ]  # Red for 1

        # Create a colormap
        cmap = ListedColormap(colors)
        # Define boundaries for discrete mapping
        bounds = [
            -1.5,
            -0.5,
            0.5,
            1.5,
        ]  # Create boundaries to cover the discrete values
        norm = BoundaryNorm(bounds, cmap.N)
        ax.imshow(
            dp_dict["hint_channel"],
            cmap=cmap,
            norm=norm,
            alpha=0.5,
            label="hint",
        )
        Logger.info("hint channel sum: %s", dp_dict["hint_channel"].sum().item())
        Logger.info(
            "hint channel non zero values: %s",
            dp_dict["hint_channel"][dp_dict["hint_channel"] != 0],
        )
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
    # ax.set_xlabel("Ion Mobility (AU)")
    # ax.set_ylabel("Retention Time (AU)")
    # ax.legend()
    # plt.axis('off')
    # Remove the axis labels
    ax.set_xticks([])
    ax.set_yticks([])

    # Customize the frame thickness
    for spine in ax.spines.values():
        spine.set_linewidth(3)  # Increase the frame (spine) thickness

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
    loss_array, metric_name, save_dir, show_quantiles=[25, 50, 75], dataset_name=""
):
    # Calculate quantiles
    quantiles = np.percentile(loss_array, show_quantiles)

    plt.figure(figsize=(10, 6))
    plt.hist(loss_array, bins=10, alpha=0.75)
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
    plt.title(f"{dataset_name} {metric_name} Distribution")
    plt.xlabel(metric_name)
    plt.ylabel("Frequency")
    save_plot(
        save_dir=save_dir,
        fig_type_name="PS_model",
        fig_spec_name=f"test_{metric_name}_distribution_{dataset_name}",
    )


def single_inference(
    datapoint,
    seg_model,
    cls_model,
    add_ps_channel: bool = True,
    device="cuda",
):
    seg_model.to(device)
    seg_model.eval()
    cls_model.to(device)
    cls_model.eval()
    with torch.no_grad():
        image, hint, target_dict = datapoint
        if add_ps_channel:
            image_cls = MultiHDF5_MaskDataset.add_ps_channel_to_batch(
                image.unsqueeze(0).float(), device=device, seg_model=seg_model
            )
            Logger.debug("Image shape: %s", image_cls.shape)
        else:
            image_cls = image.unsqueeze(0).float()
        target = target_dict["mask"].to(device)
        ori_image_raw = target_dict["ori_image_raw"].to(device)

        seg_out = seg_model(image.unsqueeze(0).float())
        cls_out = cls_model(image_cls.float())
        # output = (output > threshold).float()
        return seg_out, cls_out


def plot_sample_predictions(
    dataset,
    save_dir,
    seg_model,
    cls_model,
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
    sigmoid_cls_score: bool = True,
    add_ps_channel: bool = False,
    **kwargs,
):
    seg_model.to(device)
    seg_model.eval()
    cls_model.to(device)
    cls_model.eval()
    if sample_indices is None:
        sample_indices = np.random.choice(len(dataset), n, replace=False)
        Logger.info("Sample indices: %s", sample_indices)
    for i in sample_indices:
        with torch.no_grad():
            image, hint, target_dict = dataset[i]
            if add_ps_channel:
                image_cls = MultiHDF5_MaskDataset.add_ps_channel_to_batch(
                    image.unsqueeze(0).float(), device=device, seg_model=seg_model
                )
                Logger.debug("Image shape: %s", image_cls.shape)
            else:
                image_cls = image.unsqueeze(0).float()
            target = target_dict["mask"].to(device)
            ori_image_raw = target_dict["ori_image_raw"].to(device)
            if use_hint:
                (seg_out) = seg_model(
                    image.unsqueeze(0).float(), hint.unsqueeze(0).float()
                )

                cls_out = cls_model(image_cls.float(), hint.unsqueeze(0).float())
            else:
                seg_out = seg_model(image.unsqueeze(0).float())
                cls_out = cls_model(image_cls.float())
                if conf_model is not None:
                    conf_model.eval()
                    conf_score = conf_model(seg_out)
                    Logger.info("Confidence score: %s", conf_score.item())
                # output = (output > threshold).float()
        if sigmoid_cls_score:
            cls_out = torch.sigmoid(cls_out)
        metric_val_list = []
        for metric in metric_list:
            match metric:
                case "iou":
                    metric_val = metric_iou_batch(seg_out, target.unsqueeze(0))

                case "wiou":  # TODO: deprecated, remove
                    metric_val = metric_wiou_batch(
                        seg_out, target.unsqueeze(0), image.unsqueeze(0).float()
                    )
                case "dice":
                    metric_val = per_image_dice_metric(seg_out, target.unsqueeze(0))
                    metric_val = metric_val.item()
                case "wdice":  # TODO: updated ori image raw input, check if it works
                    metric_val = per_image_weighted_dice_metric(
                        seg_out,
                        target,
                        ori_image_raw,
                        channel=None,
                        **kwargs,
                    )
                    metric_val = metric_val.item()
                case (
                    "mask_wiou"
                ):  # TODO: updated ori image raw input, check if it works
                    metric_val = per_image_weighted_iou_metric(
                        seg_out,
                        target,
                        ori_image_raw,
                        channel=None,
                        **kwargs,
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
                    to_plot,
                    pred_bbox=seg_out[0].cpu().detach().numpy(),
                    zoom_in=zoom_in,
                )
            case "mask":
                pred = torch.sigmoid(seg_out[0][0])
                if threshold is not None:
                    pred = torch.where(
                        pred > threshold,
                        torch.tensor(1, device=device),
                        torch.tensor(0, device=device),
                    )
                Logger.info("Ori_image_raw shape: %s", ori_image_raw.shape)
                to_plot = {
                    "data": ori_image_raw[0].cpu(),
                    "hint_idx": hint.cpu(),
                    "hint_channel": image[2].cpu(),
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
                            ori_image_raw[0].cpu().numpy(), target[0].cpu().numpy()
                        )
                    ),
                )
                Logger.info(
                    "Pred masked intensity sum %.2f",
                    np.nansum(
                        np.multiply(
                            ori_image_raw[0].cpu().numpy(),
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
        plt.title(
            metrics_str
            + f", target score: {cls_out.cpu().item():.2f}"
            + "\n"
            + f"m/z Rank: {int(target_dict['pept_mz_rank'])}"
            + f", IsTarget: {target_dict['target']}"
        )
        plt.legend()
        save_plot(
            save_dir=save_dir,
            fig_type_name="PS_model_prediction",
            fig_spec_name=f"sample_{int(target_dict['pept_mz_rank'])}",
            # bbox_inches="tight",
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


def plot_target_decoy_distr(
    ps_df,
    save_dir=None,
    dataset_name="",
    threshold: tuple | None = None,
    main_plot_type: Literal["kde", "scatter"] = "scatter",
):
    """
    Plot target decoy distribution
    ps_df: pandas dataframe with columns: target_decoy_score, sum_intensity, Decoy
    threshold: tuple with two values for thresholding target decoy score, (target_decoy_score, log_sum_intensity)
    """
    ps_df["log_sum_intensity"] = np.log10(ps_df["sum_intensity"] + 1)
    sns.jointplot(
        ps_df,
        x="target_decoy_score",
        y="log_sum_intensity",
        hue="Decoy",
        kind=main_plot_type,
        s=13,
    )
    # plt.title("Target Decoy Distribution")
    plt.xlabel("Target Decoy Score")
    plt.ylabel("Log10(Sum Intensity)")
    plt.suptitle(dataset_name)
    if threshold is not None:
        plt.axvline(threshold[0], color="r", linestyle="--", linewidth=1)
        plt.axhline(threshold[1], color="r", linestyle="--", linewidth=1)
        td_counts = ps_df.loc[
            (ps_df["target_decoy_score"] > threshold[0])
            & (ps_df["log_sum_intensity"] > threshold[1]),
            "Decoy",
        ].value_counts()
        fdr = td_counts[True] / td_counts[False]
        plt.text(x=0.85, y=7.5, s=f"FDR: {fdr:.2f}")
        plt.text(x=0.85, y=8, s=f"N targets: {td_counts[0]}")
        logging.info("target decoy counts: %s", td_counts)
    save_plot(
        save_dir=save_dir,
        fig_type_name="target_decoy_distribution",
        fig_spec_name=dataset_name,
    )


def calc_fdr_and_thres(
    pred_df,
    score_col="target_decoy_score",
    filter_dict: dict | None = None,
    return_plot: bool = False,
    save_dir=None,
    dataset_name="",
    xlim=None,
    mark_x=[0.01, 0.05, 0.1],
    **kwargs,
):
    """Calculate FDR and threshold for a given score column
    Args:
        pred_df (pd.DataFrame): Dataframe with predictions
        score_col (str): Column to use for scoring
        filter_dict (dict): Dictionary with filters, e.g. {"log_sum_intensity": [0, 100]}
    Returns:
        pd.DataFrame: Dataframe with FDR and threshold
    """
    pred_df_new = pred_df.copy()
    if score_col not in pred_df_new.columns:
        if score_col == "log_sum_intensity":
            pred_df_new["log_sum_intensity"] = np.log10(
                pred_df_new["sum_intensity"] + 1
            )
        else:
            raise ValueError(f"score_col {score_col} not in pred_df")
    if filter_dict is not None:
        pred_df_new = _filter_pred(filter_dict, pred_df_new)

    pred_df_new = pred_df_new.sort_values(score_col, ascending=False)
    pred_df_new["Target"] = pred_df_new["Decoy"] == 0
    pred_df_new["fdr"] = (pred_df_new["Decoy"].cumsum()) / np.maximum(
        pred_df_new["Target"].cumsum(), 1
    )
    pred_df_new["N_identified_target"] = pred_df_new["Target"].cumsum()
    if return_plot:
        sns.scatterplot(
            data=pred_df_new,
            y="N_identified_target",
            x="fdr",
            hue="target_decoy_score",
            edgecolor=None,
            palette="Spectral",
            **kwargs,
        )
        n_target_max = pred_df_new["N_identified_target"].max()
        plt.vlines(
            x=mark_x,
            ymin=[0, 0, 0],
            ymax=[n_target_max, n_target_max, n_target_max],
            color="r",
        )
        # Access the legend and set its title
        plt.ylabel("Number of Identified Targets")
        plt.xlabel("FDR")
        plt.legend(title="Threshold", loc="lower right")
        if filter_dict is None:
            plt.title(dataset_name + "FDR vs Identified Targets, no filter")
        if filter_dict is not None:
            plt.title(
                dataset_name
                + " FDR vs Identified Targets, filter by:"
                + "\n"
                + "<br/>".join(
                    [f"{key}: {value}" for key, value in filter_dict.items()]
                )
            )
        if xlim is not None:
            plt.xlim(xlim)
        save_plot(
            save_dir=save_dir,
            fig_type_name="fdr_id_targets",
            fig_spec_name=dataset_name,
        )
    return pred_df_new


def _filter_pred(filter_dict, pred_df):
    pred_df_new = pred_df.copy()
    for key, value in filter_dict.items():
        if key not in pred_df_new.columns:
            if key == "log_sum_intensity":
                pred_df_new["log_sum_intensity"] = np.log10(
                    pred_df_new["sum_intensity"] + 1
                )
            else:
                raise ValueError(f"key {key} not in pred_df")
        Logger.info("Number of entries before filtering: %s", pred_df_new.shape[0])
        pred_df_new = pred_df_new.loc[
            (pred_df_new[key] >= value[0]) & (pred_df_new[key] <= value[1])
        ]
        Logger.info(
            "Number of entries after filtering by %s with condition %s: %s",
            key,
            value,
            pred_df_new.shape[0],
        )

    return pred_df_new


def plot_roc_auc(
    pred_df: pd.DataFrame | None = None,
    pred_df_list: List[pd.DataFrame] | None = None,
    color_list: List[str] | None = None,
    label_list: List[str] | None = None,
    save_dir=None,
    dataset_name="",
    filter_dict=None,
):
    """Plot ROC AUC curve"""
    assert pred_df is not None or pred_df_list is not None
    if pred_df is not None:
        pred_df_list = [pred_df]
        color_list = ["darkorange"]
        label_list = dataset_name
    plt.figure()
    for pred_df, color, label in zip(pred_df_list, color_list, label_list):
        if "Target" not in pred_df.columns:
            pred_df["Target"] = pred_df["Decoy"] == 0
        if filter_dict is not None:
            pred_df = _filter_pred(filter_dict, pred_df)
        fpr, tpr, threshold = roc_curve(
            pred_df["Target"], pred_df["target_decoy_score"]
        )
        roc_auc = roc_auc_score(pred_df["Target"], pred_df["target_decoy_score"])

        lw = 2
        plt.plot(
            fpr,
            tpr,
            color=color,
            lw=lw,
            label=label + " ROC curve (area = %0.2f)" % roc_auc,
        )

    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Target-Decoy ROC Curve")
    plt.legend(loc="lower right")
    save_plot(save_dir=save_dir, fig_type_name="roc_auc", fig_spec_name=dataset_name)
    return roc_auc


def calc_fdr_given_thres(
    data_df, target_decoy_score_thres: float = 0.0, log_sum_intensity_thres: float = 0.0
):
    if "log_sum_intensity" not in data_df.columns:
        data_df["log_sum_intensity"] = np.log10(data_df["sum_intensity"] + 1)
    td_counts = data_df.loc[
        (data_df["target_decoy_score"] >= target_decoy_score_thres)
        & (data_df["log_sum_intensity"] >= log_sum_intensity_thres),
        "Decoy",
    ].value_counts()
    fdr = td_counts[True] / td_counts[False]
    return td_counts, fdr


def compete_target_decoy_pair(
    pept_act_sum_ps: pd.DataFrame,
    maxquant_result_ref: pd.DataFrame,
    filter_dict: dict = {"log_sum_intensity": [2, 100]},
    td_pair_col: str = "TD pair id",
):
    for col in ["Decoy", td_pair_col]:
        if col not in pept_act_sum_ps.columns:
            pept_act_sum_ps = pd.merge(
                pept_act_sum_ps,
                maxquant_result_ref[["mz_rank", col]],
                on="mz_rank",
                how="inner",
            )
    if filter_dict is not None:
        pept_act_sum_ps_full = _filter_pred(filter_dict, pept_act_sum_ps)
    else:
        pept_act_sum_ps_full = pept_act_sum_ps.copy()
    pept_act_sum_ps_full_tdc = (
        pept_act_sum_ps_full.groupby(td_pair_col)
        .apply(lambda x: x.loc[x["target_decoy_score"].idxmax()])
        .reset_index(drop=True)
    )
    fdr_after_tdc = calc_fdr_given_thres(pept_act_sum_ps_full_tdc)
    Logger.info("FDR after TDC: %s", fdr_after_tdc)
    return pept_act_sum_ps_full, pept_act_sum_ps_full_tdc


def sigmoid(x):
    return 1 / (1 + math.exp(-x))
