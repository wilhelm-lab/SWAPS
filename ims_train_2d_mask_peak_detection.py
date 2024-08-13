import logging
import json
import os
import gc
import torch
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.transforms import Compose
from torch.utils.tensorboard import SummaryWriter
from peak_detection_2d.utils import (
    plot_sample_predictions,
    plot_history,
    EarlyStopping,
    plot_per_image_metric_distr,
)
from peak_detection_2d.dataset.dataset import (
    MultiHDF5_MaskDataset,
    Mask_Padding,
    Mask_AddLogChannel,
    Mask_LogTransform,
    Mask_ToTensor,
)
from peak_detection_2d.model.seg_model import UNET, train_one_epoch, evaluate
from peak_detection_2d.loss.custom_loss import (
    per_image_weighted_dice_metric,
    per_image_weighted_iou_metric,
)
from peak_detection_2d.loss.combo_loss import ComboLoss


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

torch.cuda.empty_cache()
gc.collect()

# data path
result_parent_dir = "/cmnfs/proj/ORIGINS/data/brain/FreshFrozenBrain/SingleShot/DDA/"
result_base_dir = "frame0_1830_ssDDA_P064428_Fresh1_5ug_R1_BD5_1_4921_ScanByScan_RTtol0.9_threshold_missabthres0.5_convergence_NoIntercept_pred_mzBinDigits2_imPeakWidth4_deltaMobilityThres80"
result_dir = os.path.join(result_parent_dir, result_base_dir)

# exp spec
TRAIN_MODEL = True
EVALUATE = True
RANDOM_STATE = 42
EPOCHS = 100
ES_PATIENCE = 10
INI_LR = 0.001
BATCH_SIZE = 256
FIRST_OUT_CHANNELS = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logging.info("Using device: %s", DEVICE)
LOSS_WEIGHT = {"bce": 1, "wdice": 4, "focal": 1}
CHANNEL_WEIGHT = [1, 0.5]
criterion = ComboLoss(**{"weights": LOSS_WEIGHT, "channel_weights": CHANNEL_WEIGHT})
loss_weight_str = "_".join(f"{key}{value}" for key, value in LOSS_WEIGHT.items())
channel_weight_str = "_".join(str(i) for i in CHANNEL_WEIGHT)
peak_selection_dir = os.path.join(
    result_dir,
    f"IMRT_fulloverlap_data_peak_selection_seg_model_1out{FIRST_OUT_CHANNELS}_lr{INI_LR}_bs{BATCH_SIZE}_comboloss_{loss_weight_str}_metric_wdice_channel{channel_weight_str}_resume",
)
writer = SummaryWriter(
    log_dir=os.path.join(result_dir, "runs"),
    comment=f"1out{FIRST_OUT_CHANNELS}_lr{INI_LR}_bs{BATCH_SIZE}_comboloss_{loss_weight_str}_metric_wdice_channel{channel_weight_str}_resume",
)
best_model_path = os.path.join(
    result_dir,
    "IMRT_fulloverlap_data_peak_selection_seg_model_1out32_lr0.005_bs256_comboloss_bce1_dice4_focal1_metric_wdice_channel1_0.5",
    "bst_model_0.7089.bin",
)
if not os.path.exists(peak_selection_dir):
    os.makedirs(peak_selection_dir)


model = UNET(
    in_channels=2,
    first_out_channels=FIRST_OUT_CHANNELS,
    exit_channels=1,
    downhill=4,
    padding=1,
).to(DEVICE)
if best_model_path:
    logging.info("Loading best model from %s", best_model_path)
    checkpoint = torch.load(best_model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint)
optimizer = torch.optim.Adam(model.parameters(), lr=INI_LR)
scheduler = ReduceLROnPlateau(
    optimizer, mode="max", factor=0.1, patience=3, min_lr=0.0000001
)

es = EarlyStopping(patience=ES_PATIENCE, mode="max")

# ================Data set===============#
hdf5_files = [
    os.path.join(os.path.join(result_dir, "peak_detection_mask_data"), file)
    for file in os.listdir(os.path.join(result_dir, "peak_detection_mask_data"))
    if file.endswith(".h5")
]

# Define transformations (if any)
transformation = Compose(
    [Mask_Padding((258, 258)), Mask_AddLogChannel(), Mask_ToTensor()]
)

# Create the dataset
dataset = MultiHDF5_MaskDataset(hdf5_files, transforms=transformation)

# Split the dataset into training and testing sets
train_val_dataset, test_dataset = dataset.split_dataset(
    train_ratio=0.9, seed=RANDOM_STATE
)
train_dataset, val_dataset = train_val_dataset.split_dataset(
    train_ratio=0.9, seed=RANDOM_STATE
)


train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=False
)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=2 * BATCH_SIZE, shuffle=False
)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=2 * BATCH_SIZE, shuffle=False
)
logging.info("Train dataset size: %d", len(train_dataset))
logging.info("Validation dataset size: %d", len(val_dataset))
logging.info("Test dataset size: %d", len(test_dataset))


# =================Train model================#
loss_tracking = {"train": [], "val": []}
metric = {"train": [], "val": []}
if TRAIN_MODEL:
    for epoch in range(EPOCHS):
        loss = train_one_epoch(
            train_dataloader,
            model,
            optimizer,
            criterion,
            use_image_as_input=True,
            device=DEVICE,
        )
        train_metric = evaluate(
            train_dataloader,
            model,
            metric=per_image_weighted_dice_metric,
            device=DEVICE,
            use_image_for_metric=True,
            channel=0,
        )
        val_metric = evaluate(
            val_dataloader,
            model,
            metric=per_image_weighted_dice_metric,
            device=DEVICE,
            use_image_for_metric=True,
            channel=0,
        )
        metric["train"].append(train_metric)
        metric["val"].append(val_metric)
        loss_tracking["train"].append(loss)

        print(
            f"EPOCH: {epoch}, TRAIN LOSS: {loss}, TRAIN Weighted DICE: {train_metric},"
            f" VAL Weighted DICE: {val_metric}"
        )
        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("Metric/train", train_metric, epoch)
        writer.add_scalar("Metric/val", val_metric, epoch)
        scheduler.step(val_metric)
        es(
            epoch_score=val_metric,
            epoch_num=epoch,
            loss=loss,
            optimizer=optimizer,
            model=model,
            model_path=os.path.join(
                result_dir,
                peak_selection_dir,
                f"bst_model_{np.round(val_metric,4)}.pt",
            ),
        )
        best_model_path = os.path.join(
            result_dir, peak_selection_dir, f"bst_model_{np.round(es.best_score,4)}.pt"
        )
        if es.early_stop:
            print("\n\n -------------- EARLY STOPPING -------------- \n\n")
            break


# Plot history
with open(os.path.join(peak_selection_dir, "loss.json"), "w", encoding="utf-8") as fp:
    json.dump(loss_tracking, fp)
with open(os.path.join(peak_selection_dir, "metric.json"), "w", encoding="utf-8") as fp:
    json.dump(metric, fp)

plot_history(loss_tracking, "loss", save_dir=peak_selection_dir)
plot_history(metric, "dice metric", save_dir=peak_selection_dir)

bst_model = UNET(2, FIRST_OUT_CHANNELS, 1, padding=1, downhill=4).to(DEVICE)
checkpoint = torch.load(best_model_path, map_location=DEVICE)
bst_model.load_state_dict(checkpoint["model_state_dict"])

# Get test metric
test_wdice, per_image_test_wdice = evaluate(
    test_dataloader,
    bst_model,
    metric=per_image_weighted_dice_metric,
    device=DEVICE,
    use_image_for_metric=True,
    save_all_loss=True,
    channel=0,
)

plot_per_image_metric_distr(
    per_image_test_wdice["losses"], "weighted_Dice", save_dir=peak_selection_dir
)

test_wiou, per_image_test_wiou = evaluate(
    test_dataloader,
    bst_model,
    metric=per_image_weighted_iou_metric,
    device=DEVICE,
    use_image_for_metric=True,
    save_all_loss=True,
    channel=0,
)
plot_per_image_metric_distr(
    per_image_test_wiou["losses"], "weighted_IoU", save_dir=peak_selection_dir
)

# Plot sample predictions
plot_sample_predictions(
    test_dataset,
    model=bst_model,
    n=10,
    metric_list=["mask_wiou", "wdice", "dice"],
    use_hint=False,
    zoom_in=False,
    label="mask",
    save_dir=os.path.join(peak_selection_dir, "sample_predictions"),
)
writer.close()
