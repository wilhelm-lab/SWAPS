from peak_detection_2d.loss.custom_loss import WeightedBoundingBoxIoULoss
import logging
import json
import os
import gc
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.transforms import Compose

from peak_detection_2d.utils import (
    plot_sample_predictions,
    plot_history,
)
from peak_detection_2d.dataset.dataset import MultiHDF5Dataset, ToTensor, Padding
from peak_detection_2d.model.model import (
    PeakDetectionNet,
    train_val_step,
    train_val_step_wiou,
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


torch.cuda.empty_cache()
gc.collect()
# experiment specific
result_parent_dir = "/cmnfs/proj/ORIGINS/data/brain/FreshFrozenBrain/SingleShot/DDA/"
result_base_dir = "frame0_1830_ssDDA_P064428_Fresh1_5ug_R1_BD5_1_4921_ScanByScan_RTtol0.9_threshold_missabthres0.5_convergence_NoIntercept_pred_mzBinDigits2_imPeakWidth4_deltaMobilityThres80"
result_dir = os.path.join(result_parent_dir, result_base_dir)
peak_selection_dir = os.path.join(
    result_dir, "peak_selection_model_1out64_lr01_wiou_l1"
)
if not os.path.exists(peak_selection_dir):
    os.makedirs(peak_selection_dir)

num_epoch = 100
patience = 10
inital_lr = 0.005
batch_size = 512

random_state = 42
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info("Using device: %s", device)
model = PeakDetectionNet(1, 64).to(device)

loss_func_val = WeightedBoundingBoxIoULoss(
    reduction="mean", add_diou=False, add_smooth_l1=False
)
loss_func_train = WeightedBoundingBoxIoULoss(
    reduction="mean", add_diou=False, add_smooth_l1=True
)
loss_func_l1 = nn.SmoothL1Loss(reduction="mean")
optimizer = torch.optim.Adam(model.parameters(), lr=inital_lr)
scheduler = ReduceLROnPlateau(
    optimizer, mode="min", factor=0.1, patience=3, min_lr=0.000001
)

loss_tracking = {"train": [], "val": []}
iou_tracking = {"train": [], "val": []}
best_loss = float("inf")


with open(os.path.join(result_dir, "param.json"), mode="r", encoding="utf-8") as file:
    config = json.load(file)

hdf5_files = [
    os.path.join(os.path.join(result_dir, "peak_detection_data"), file)
    for file in os.listdir(os.path.join(result_dir, "peak_detection_data"))
    if file.endswith(".h5")
]

# Define transformations (if any)
transformation = Compose([Padding((180, 180)), ToTensor(scale_label=False)])

# Create the dataset
dataset = MultiHDF5Dataset(hdf5_files, transforms=transformation)

# Split the dataset into training and testing sets
train_val_dataset, test_dataset = dataset.split_dataset(
    train_ratio=0.8, seed=random_state
)
train_dataset, val_dataset = train_val_dataset.split_dataset(
    train_ratio=0.8, seed=random_state
)
logging.info("Train dataset size: %d", len(train_dataset))
logging.info("Validation dataset size: %d", len(val_dataset))
logging.info("Test dataset size: %d", len(test_dataset))
# Example usage
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=False
)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=1024, shuffle=False
)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=1024, shuffle=False
)


for epoch in range(num_epoch):
    logging.info("Epoch %d/%d", epoch + 1, num_epoch)

    training_loss, trainig_iou = train_val_step_wiou(
        train_dataloader, model, loss_func_train, optimizer
    )
    loss_tracking["train"].append(training_loss)
    iou_tracking["train"].append(trainig_iou)

    with torch.inference_mode():
        val_loss, val_iou = train_val_step_wiou(
            val_dataloader, model, loss_func_val, None
        )
        loss_tracking["val"].append(val_loss)
        iou_tracking["val"].append(val_iou)
        if val_loss < best_loss:
            logging.info("Saving best model")
            torch.save(
                model.state_dict(), os.path.join(peak_selection_dir, "best_model.pt")
            )
            best_loss = val_loss
            current_patience = patience
        else:
            current_patience -= 1
            if current_patience == 0:
                logging.info("Early stopping")
                break
        scheduler.step(val_loss)
        logging.info(
            "Last learning rate: %s",
            scheduler.get_last_lr(),
        )

    logging.info("Training loss: %.6f, IoU: %.2f", training_loss, trainig_iou)
    logging.info("Validation loss: %.6f, IoU: %.6f", val_loss, val_iou)

# Plot history
with open(os.path.join(peak_selection_dir, "loss.json"), "w", encoding="utf-8") as fp:
    json.dump(loss_tracking, fp)
with open(os.path.join(peak_selection_dir, "iou.json"), "w", encoding="utf-8") as fp:
    json.dump(iou_tracking, fp)

plot_history(loss_tracking, "loss", save_dir=peak_selection_dir)
plot_history(iou_tracking, "iou", save_dir=peak_selection_dir)

# Plot sample predictions
plot_sample_predictions(
    test_dataset,
    seg_model=model,
    n=10,
    save_dir=os.path.join(peak_selection_dir, "sample_predictions"),
)
