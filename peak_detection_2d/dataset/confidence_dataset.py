import logging
import os
import random

import numpy as np

import h5py
import torch
from torchvision import tv_tensors

from torchvision.transforms.v2 import functional as F
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from peak_detection_2d.model import seg_model

Logger = logging.getLogger(__name__)


def prepare_2d_seg_output_and_confidence(
    dataloader,
    model,
    metric,
    use_image_for_metric: bool = True,
    device: str = "cpu",
    save_hdf5: str = None,
    **kwargs,
):
    model = model.to(device)
    model.eval()
    tk0 = tqdm(dataloader, total=len(dataloader))
    out_array = np.empty((0, 0, 0, 0))
    losses = np.empty((0))
    if not os.path.exists(os.path.dirname(save_hdf5)):
        os.makedirs(os.path.dirname(save_hdf5))
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
            # out_array.append(out.cpu().numpy())
            if out_array.size == 0:
                out_array = out.cpu().numpy()
            else:
                out_array = np.append(out_array, out.cpu().numpy(), axis=0)
            Logger.debug("out_array shape: %s", out_array.shape)
            losses = np.append(losses, b_loss.cpu().numpy())
    if save_hdf5 is not None:
        with h5py.File(save_hdf5, "w") as f:
            f.create_dataset("segmentation_mask", data=out_array)
            f.create_dataset("confidence_score", data=losses)
        Logger.info("Saved segmentation mask and confidence score to %s", save_hdf5)
        return None
    else:
        return out_array, losses


class ConfidenceDataset(torch.utils.data.Dataset):
    def __init__(self, hdf5_file, indices=None, transforms=None):
        self.hf = h5py.File(hdf5_file, "r")
        self.segmentation_outputs = self.hf["segmentation_mask"]
        self.confidence_scores = self.hf["confidence_score"]
        self.transforms = transforms
        # Use specified indices or all indices if not provided
        if indices is None:
            self.indices = list(range(len(self.confidence_scores)))
        else:
            self.indices = indices

    def _create_data_index(self):
        data_index = []
        for file_idx, filename in enumerate(self.hf):
            with h5py.File(filename, "r") as f:
                for group_name in f.keys():
                    data_index.append((file_idx, group_name))
        return data_index

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Logger.debug(
        #     "view data point, seg output: %s, mask: %s",
        #     self.segmentation_outputs[idx],
        #     self.confidence_scores[idx],
        # )
        segmentation_output = torch.from_numpy(self.segmentation_outputs[idx])
        confidence_score = torch.tensor(
            self.confidence_scores[idx], dtype=torch.float32
        )
        if self.transforms is not None:
            segmentation_output, confidence_score = self.transforms(
                (segmentation_output, confidence_score)
            )
        return segmentation_output, confidence_score

    def create_splits(self, test_size=0.2, val_size=0.1, random_state=None):
        num_samples = len(self.confidence_scores)
        indices = list(range(num_samples))
        if test_size is None:
            train_val_indices = indices
            test_indices = None
            test_size = 0
        else:
            train_val_indices, test_indices = train_test_split(
                indices, test_size=test_size, random_state=random_state
            )
        if val_size is None:
            train_indices = train_val_indices
            val_indices = None
        else:
            train_indices, val_indices = train_test_split(
                train_val_indices,
                test_size=val_size / (1 - test_size),
                random_state=random_state,
            )

        return (
            ConfidenceDataset(
                self.hf.filename, indices=train_indices, transforms=self.transforms
            ),
            ConfidenceDataset(
                self.hf.filename, indices=val_indices, transforms=self.transforms
            ),
            ConfidenceDataset(
                self.hf.filename, indices=test_indices, transforms=self.transforms
            ),
        )

    @staticmethod
    def combine_datasets(dataset1, dataset2, new_hdf5_file):
        # Combine indices
        combined_indices = sorted(dataset1.indices + dataset2.indices)

        # Combine segmentation outputs
        seg_outputs1 = dataset1.segmentation_outputs[sorted(dataset1.indices)]
        seg_outputs2 = dataset2.segmentation_outputs[sorted(dataset2.indices)]
        combined_seg_outputs = np.concatenate((seg_outputs1, seg_outputs2), axis=0)

        # Combine confidence scores
        conf_scores1 = dataset1.confidence_scores[sorted(dataset1.indices)]
        conf_scores2 = dataset2.confidence_scores[sorted(dataset2.indices)]
        combined_conf_scores = np.concatenate((conf_scores1, conf_scores2), axis=0)

        # Create a new HDF5 file to store combined data
        with h5py.File(new_hdf5_file, "w") as new_hf:
            new_hf.create_dataset("segmentation_mask", data=combined_seg_outputs)
            new_hf.create_dataset("confidence_score", data=combined_conf_scores)

        return ConfidenceDataset(new_hdf5_file)

    @staticmethod
    def save_dataset_to_hdf5(dataset, hdf5_file):
        with h5py.File(hdf5_file, "w") as hf:
            hf.create_dataset(
                "segmentation_mask",
                data=dataset.segmentation_outputs[sorted(dataset.indices)],
            )
            hf.create_dataset(
                "confidence_score",
                data=dataset.confidence_scores[sorted(dataset.indices)],
            )


class Conf_AsBinary:
    """Convert the confidence score to binary
    1 if confidence score is greater than 0.5
    0 otherwise"""

    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def __call__(self, data_item):
        segmentation_output = data_item[0]
        confidence_score = data_item[1]
        return (
            segmentation_output,
            torch.where(confidence_score > self.threshold, 1, 0).float(),
        )
