import logging

from yacs.config import CfgNode

import pandas as pd
import numpy as np
from sparse import SparseArray
import h5py
import torch
from torchvision import tv_tensors
from torchvision.transforms import Compose
from torchvision.transforms.v2 import functional as F
from .prepare_dataset import prepare_2d_act_and_mask_updated

Logger = logging.getLogger(__name__)


class MultiHDF5Dataset(torch.utils.data.Dataset):
    def __init__(self, hdf5_files, data_index=None, transforms=None):
        self.hdf5_files = hdf5_files
        self.transforms = transforms
        self.data_index = (
            data_index if data_index is not None else self._create_data_index()
        )

    def _create_data_index(self):
        data_index = []
        for file_idx, filename in enumerate(self.hdf5_files):
            with h5py.File(filename, "r") as f:
                for data_point_name in f.keys():
                    data_index.append((file_idx, data_point_name))
        return data_index

    def _load_data_point(self, file_idx, data_point_name):
        filename = self.hdf5_files[file_idx]
        with h5py.File(filename, "r") as f:
            loaded_data_point = f[data_point_name]
            returned_data_point = {}
            for data_key, data_value in loaded_data_point.items():
                if isinstance(data_value, h5py.Dataset) and data_value.shape == ():
                    returned_data_point[data_key] = data_value[()]
                else:
                    returned_data_point[data_key] = data_value[:]
        return returned_data_point

    def __getitem__(self, idx):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        file_idx, data_point_name = self.data_index[idx]
        data_point = self._load_data_point(file_idx, data_point_name)

        # Load images and masks
        img = data_point["data"]
        img[np.isnan(img)] = 0  # Replace NaN with 0
        boxes = data_point["bbox"]

        # There is only one class
        labels = torch.ones((1,), dtype=torch.int64)

        area = (boxes[3] - boxes[1]) * (boxes[2] - boxes[0])
        iscrowd = torch.zeros((1,), dtype=torch.int64)

        # Wrap sample and targets into torchvision tv_tensors:
        img = tv_tensors.Image(img)
        hint = torch.tensor(data_point["hint_idx"])
        target = {
            "boxes": boxes,
            "labels": labels,
            "pept_mz_rank": data_point["pept_mz_rank"],
            "area": area,
            "iscrowd": iscrowd,
        }
        # Logger.debug("Peptide mz rank %s", target["pept_mz_rank"])
        if self.transforms is not None:
            img, hint, target = self.transforms((img, hint, target))

        return img.to(device), hint.to(device), target.to(device)

    def __len__(self):
        return len(self.data_index)

    def split_dataset(self, train_ratio=0.8, seed=None):
        if seed is not None:
            np.random.seed(seed)
        indices = np.arange(len(self.data_index))
        np.random.shuffle(indices)
        split_idx = int(train_ratio * len(indices))
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]

        train_data_index = [self.data_index[i] for i in train_indices]
        test_data_index = [self.data_index[i] for i in test_indices]

        train_dataset = MultiHDF5Dataset(
            self.hdf5_files, data_index=train_data_index, transforms=self.transforms
        )
        test_dataset = MultiHDF5Dataset(
            self.hdf5_files, data_index=test_data_index, transforms=self.transforms
        )

        return train_dataset, test_dataset

    def merge_datasets(self, other_dataset):
        """
        Merges the current dataset with another MultiHDF5Dataset instance.
        The merged dataset will contain the combined data index of both datasets.

        Args:
            other_dataset (MultiHDF5Dataset): Another dataset to merge with this one.

        Returns:
            MultiHDF5Dataset: A new dataset instance containing the merged data.
        """
        if not isinstance(other_dataset, MultiHDF5Dataset):
            raise TypeError("other_dataset must be an instance of MultiHDF5Dataset")

        merged_hdf5_files = self.hdf5_files + other_dataset.hdf5_files

        # Adjust file indices in the other dataset's data index
        offset = len(self.hdf5_files)
        adjusted_other_data_index = [
            (file_idx + offset, name) for file_idx, name in other_dataset.data_index
        ]

        merged_data_index = self.data_index + adjusted_other_data_index

        return MultiHDF5Dataset(
            merged_hdf5_files, data_index=merged_data_index, transforms=self.transforms
        )


class PeptActPeakSelection_Infer_Dataset(torch.utils.data.Dataset):
    """Dataset for inference with peak selection, no target labels are included in data points"""

    def __init__(
        self,
        pept_act_coo_peptbatch: SparseArray,
        maxquant_dict: pd.DataFrame,
        hint_matrix: str = None,
        use_hint_channel: bool = True,
        data_index=None,
        transforms=None,
        add_label_mask=False,
    ):
        maxquant_dict[
            [
                "MS1_frame_idx_left_ref",
                "MS1_frame_idx_right_ref",
                "IM_search_idx_left",
                "IM_search_idx_right",
            ]
        ] = maxquant_dict[
            [
                "MS1_frame_idx_left_ref",
                "MS1_frame_idx_right_ref",
                "IM_search_idx_left",
                "IM_search_idx_right",
            ]
        ].astype(
            int
        )
        self.peptbatch_act = pept_act_coo_peptbatch
        self.use_hint_channel = use_hint_channel
        self.hint_matrix = hint_matrix
        self.maxquant_dict = maxquant_dict
        self.transforms = transforms
        self.data_index = (
            data_index if data_index is not None else self._create_data_index()
        )
        self.add_label_mask = add_label_mask

    def __len__(self):
        return len(self.data_index)

    def _create_data_index(self):
        data_index = self.maxquant_dict["mz_rank"].values
        Logger.debug("Creating data index %s", data_index)
        return data_index

    def __getitem__(self, idx):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        datapoint_dict = prepare_2d_act_and_mask_updated(
            pept_mz_rank=self.data_index[idx],
            peptbatch_act=self.peptbatch_act,
            maxquant_dict=self.maxquant_dict,
            hint_matrix=self.hint_matrix,
            add_label_mask=self.add_label_mask,
        )
        # Logger.debug("datapoint_dict %s", datapoint_dict)

        # Wrap sample and targets into torchvision tv_tensors:
        img = datapoint_dict["data"]
        hint = datapoint_dict["hint_channel"]
        mask = datapoint_dict["mask"]

        target = {
            "mask": mask,
            "pept_mz_rank": datapoint_dict["pept_mz_rank"],
            "target": datapoint_dict["target"],
            "ori_image_raw": img.clone(),
        }
        # Logger.debug("img shape %s", img.shape)
        # Logger.debug("hint shape %s", hint.shape)
        # Logger.debug("target %s", target)
        if self.transforms is not None:
            img, hint, target = self.transforms((img, hint, target))

        return img.to(device), hint.to(device), target


# TODO: stratified spliting of dataset by TD_pair_id, rather than mz_rank
class MultiHDF5_MaskDataset(torch.utils.data.Dataset):
    """Dataset for training as well as inference with peak selection, \
        target labels can be optionally included. Lazy loading from hdf5 files \
            generated by peak_selection.prepare_dataset.py"""

    def __init__(
        self,
        hdf5_files,
        use_hint_channel: bool,
        data_index=None,
        transforms=None,
    ):
        self.hdf5_files = hdf5_files
        self.transforms = transforms
        self.data_index = (
            data_index if data_index is not None else self._create_data_index()
        )
        self.use_hint_channel = use_hint_channel

    def _create_data_index(self):
        data_index = []
        for file_idx, filename in enumerate(self.hdf5_files):
            with h5py.File(filename, "r") as f:
                for group_name in f.keys():
                    data_index.append((file_idx, group_name))
        return data_index

    def _load_data_point(self, file_idx, group_name):
        filename = self.hdf5_files[file_idx]
        with h5py.File(filename, "r") as f:
            group = f[group_name]
            data_point = {}
            for dataset_name in group.keys():
                if (
                    isinstance(group[dataset_name], h5py.Dataset)
                    and group[dataset_name].shape == ()
                ):
                    data_point[dataset_name] = group[dataset_name][()]
                else:
                    data_point[dataset_name] = group[dataset_name][:]
        return data_point

    def __getitem__(self, idx):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        file_idx, group_name = self.data_index[idx]
        data_point = self._load_data_point(file_idx, group_name)

        # Load images and masks
        img = data_point["data"]
        # Logger.debug("Image shape %s", img.shape)
        img[np.isnan(img)] = 0  # Replace NaN with 0
        mask = data_point["mask"]
        # Logger.debug("Mask shape %s", mask.shape)
        if self.use_hint_channel:
            hint_channel = data_point["hint_channel"]
            # Logger.debug("Hint channel shape %s", hint_channel.shape)
            hint_channel = torch.tensor(hint_channel)
            # hint_channel = torch.unsqueeze(hint_channel, dim=0)
            hint = hint_channel
        else:
            hint = torch.tensor(data_point["hint_idx"])
        # There is only one class
        # labels = torch.ones((1,), dtype=torch.int64)

        # area = (mask[3] - mask[1]) * (mask[2] - mask[0])
        iscrowd = torch.zeros((1,), dtype=torch.int64)

        # Wrap sample and targets into torchvision tv_tensors:
        img = tv_tensors.Image(img)
        mask = tv_tensors.Image(mask)

        target = {
            "mask": mask,
            # "labels": labels,
            "pept_mz_rank": data_point["pept_mz_rank"],
            # "area": area,
            "iscrowd": iscrowd,
            "target": data_point["target"],
            "ori_image_raw": img.clone(),
        }
        # Logger.debug("Peptide mz rank %s", target["pept_mz_rank"])
        if self.transforms is not None:
            img, hint, target = self.transforms((img, hint, target))

        return img.to(device), hint.to(device), target

    def __len__(self):
        return len(self.data_index)

    def split_dataset(self, train_ratio=0.8, seed=None):
        if seed is not None:
            np.random.seed(seed)
        indices = np.arange(len(self.data_index))
        np.random.shuffle(indices)
        split_idx = int(train_ratio * len(indices))
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]

        train_data_index = [self.data_index[i] for i in train_indices]
        test_data_index = [self.data_index[i] for i in test_indices]

        train_dataset = MultiHDF5_MaskDataset(
            self.hdf5_files,
            data_index=train_data_index,
            transforms=self.transforms,
            use_hint_channel=self.use_hint_channel,
        )
        test_dataset = MultiHDF5_MaskDataset(
            self.hdf5_files,
            data_index=test_data_index,
            transforms=self.transforms,
            use_hint_channel=self.use_hint_channel,
        )

        return train_dataset, test_dataset


class Peak2dDataset(torch.utils.data.Dataset):
    def __init__(self, data_dict, transforms):
        self.data_dict = data_dict
        self.transforms = transforms

    def __getitem__(self, idx):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # load images and masks
        img = self.data_dict[idx]["data"]
        img[np.isnan(img)] = 0  # replace nan with 0
        boxes = self.data_dict[idx]["bbox"]

        # there is only one class
        labels = torch.ones((1,), dtype=torch.int64)

        area = (boxes[3] - boxes[1]) * (boxes[2] - boxes[0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((1,), dtype=torch.int64)

        # Wrap sample and targets into torchvision tv_tensors:
        img = tv_tensors.Image(img)
        hint = torch.tensor(self.data_dict[idx]["hint_idx"])
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["pept_mz_rank"] = self.data_dict[idx]["pept_mz_rank"]
        target["area"] = area
        target["iscrowd"] = iscrowd
        # Logger.debug("Peptide mz rank %s", target["pept_mz_rank"])
        if self.transforms is not None:
            img, hint, target = self.transforms((img, hint, target))

        return img.to(device), hint.to(device), target.to(device)

    def __len__(self):
        return len(self.data_dict)


class Padding:
    """Pad the image to the new size"""

    def __init__(self, new_size_im_rt=(180, 180)):
        self.new_im_width = new_size_im_rt[0]
        self.new_rt_height = new_size_im_rt[1]

    def __call__(self, data_hint_bbox):
        image = data_hint_bbox[0]
        hint = data_hint_bbox[1]
        label = data_hint_bbox[2]["boxes"]
        im0, rt0, im1, rt1 = label
        hint_rt, hint_im = hint
        batch_number, original_rt, original_im = image.size()
        if original_rt == self.new_rt_height and original_im == self.new_im_width:
            # Logger.debug(
            #     "Image size %s same as new size, returning image.", image.size()
            # )
            return image, hint, label
        delta_im_width = self.new_im_width - original_im
        delta_rt_height = self.new_rt_height - original_rt
        if delta_im_width > 0 and delta_rt_height > 0:
            Logger.debug(
                "Image size %s smaller than new size, deltas %s, %s, padding image.",
                image.size(),
                delta_im_width,
                delta_rt_height,
            )
            Logger.debug("Orginal label %s", label)
            image_new = F.pad(
                image,
                padding=(
                    delta_im_width // 2,
                    delta_rt_height // 2,
                    delta_im_width - delta_im_width // 2,
                    delta_rt_height - delta_rt_height // 2,
                ),
                fill=0,
            )
            label_new = (
                im0 + delta_im_width // 2,
                rt0 + delta_rt_height // 2,
                im1 + delta_im_width // 2,
                rt1 + delta_rt_height // 2,
            )
            # label_new, size_new = F.pad_bounding_boxes(
            #     torch.tensor(label, dtype=torch.float32).unsqueeze(0),
            #     format="xyxy",
            #     canvas_size=(original_rt, original_im),
            #     padding=(
            #         delta_im_width // 2,
            #         delta_im_width - delta_im_width // 2,
            #         delta_rt_height // 2,
            #         delta_rt_height - delta_rt_height // 2,
            #     ),
            # )
            # Logger.debug("Original label %s", label)
            # Logger.debug("label_new %s", label_new)
            # label_new = label_new[0]
        else:
            Logger.debug(
                "Image size %s larger than new size, deltas %s, %s, cropping image.",
                image.size(),
                delta_im_width,
                delta_rt_height,
            )
            image_new = F.center_crop(image, (self.new_im_width, self.new_rt_height))
            label_new, size_new = F.center_crop_bounding_boxes(
                torch.tensor(label, dtype=torch.float32).unsqueeze(0),
                format="xyxy",
                canvas_size=(original_rt, original_im),
                output_size=(self.new_rt_height, self.new_im_width),
            )
            # Logger.debug("label_new %s", label_new)
            label_new = label_new[0]

        return (
            image_new,
            (hint_rt + delta_rt_height // 2, hint_im + delta_im_width // 2),
            label_new,
        )


class Mask_LogTransform:
    """Log transform intensity in image"""

    def __init__(self) -> None:
        pass

    def __call__(self, data_hint_bbox):
        image = data_hint_bbox[0]
        hint = data_hint_bbox[1]
        label = data_hint_bbox[2]
        image = torch.log1p(image)
        return image, hint, label


class Mask_AddHintChannel:
    """Add hitn channel to image"""

    def __init__(self) -> None:
        pass

    def __call__(self, data_hint_bbox):
        image = data_hint_bbox[0]
        hint = data_hint_bbox[1]
        label = data_hint_bbox[2]
        # Logger.debug("Shape of image %s", image.size())
        # Logger.debug("Shape of hint %s", hint.size())
        # hint = torch.unsqueeze(hint, dim=0)
        image = torch.cat((image, hint), dim=0)
        return image, hint, label


class Mask_AddLogChannel:
    """Add log intensity channel to image"""

    def __init__(self) -> None:
        pass

    def __call__(self, data_hint_bbox):
        image = data_hint_bbox[0]
        hint = data_hint_bbox[1]
        label = data_hint_bbox[2]
        log_image = torch.log1p(image)
        image = torch.cat((image, log_image), dim=0)
        return image, hint, label


class Mask_Cropping:
    def __init__(self, crop_center_idx) -> None:
        self.crop_center_idx = crop_center_idx

    def __call__(self, data):
        data["input"] = F.center_crop(data["input"], self.crop_center_idx)

        return data


class Mask_Padding:
    """Pad the image to the new size"""

    def __init__(self, new_size_im_rt=(180, 180)):
        self.new_im_width = new_size_im_rt[0]
        self.new_rt_height = new_size_im_rt[1]

    def __call__(self, data_hint_bbox):
        image = data_hint_bbox[0]
        hint = data_hint_bbox[1]
        label = data_hint_bbox[2]["mask"]
        target = data_hint_bbox[2]
        # im0, rt0, im1, rt1 = label

        batch_number, original_rt, original_im = image.size()
        if original_rt == self.new_rt_height and original_im == self.new_im_width:
            # Logger.debug(
            #     "Image size %s same as new size, returning image.", image.size()
            # )
            return image, hint, label
        delta_im_width = self.new_im_width - original_im
        delta_rt_height = self.new_rt_height - original_rt
        # Logger.debug(
        #     "delta im width %s, delta rt height %s", delta_im_width, delta_rt_height
        # )
        image_new = F.center_crop(image, (self.new_im_width, self.new_rt_height))
        # Logger.debug("image new shape %s", image_new.size())
        label_new = F.center_crop(label, (self.new_im_width, self.new_rt_height))
        target["mask"] = label_new
        # Logger.debug("label new shape %s", label_new.size())
        if hint.size() == image.size():
            hint_new = F.center_crop(hint, (self.new_im_width, self.new_rt_height))
            # Logger.debug("hitn new shape %s", hint_new.size())
        else:
            hint_rt, hint_im = hint
            hint_new = (hint_rt + delta_rt_height // 2, hint_im + delta_im_width // 2)
            # Logger.debug("Correct hint coordination %s", hint_new)
        return (
            image_new,
            hint_new,
            target,
        )


class Mask_Resize:
    """Resize the image, hint channel and the mask"""

    def __init__(self, new_size_im_rt=(180, 180)):
        self.new_im_width = new_size_im_rt[0]
        self.new_rt_height = new_size_im_rt[1]

    def __call__(self, data_hint_bbox):
        image = data_hint_bbox[0]
        hint = data_hint_bbox[1]
        label = data_hint_bbox[2]["mask"]
        target = data_hint_bbox[2]
        # im0, rt0, im1, rt1 = label

        batch_number, original_rt, original_im = image.size()
        if original_rt == self.new_rt_height and original_im == self.new_im_width:
            return image, hint, label
        delta_im_width = self.new_im_width - original_im
        delta_rt_height = self.new_rt_height - original_rt
        image_new = F.resize(image, (self.new_im_width, self.new_rt_height))
        label_new = F.resize(label, (self.new_im_width, self.new_rt_height))
        target["ori_image_raw"] = F.resize(
            target["ori_image_raw"], (self.new_im_width, self.new_rt_height)
        )
        target["mask"] = label_new
        if hint.size() == image.size():
            hint = (
                hint * 1000
            )  # !! scale hint to 0-1000 otherwise single hint values in -1 to 1 will all be squeezed to 0
            hint_new = F.resize(hint, (self.new_im_width, self.new_rt_height))
        else:
            hint_rt, hint_im = hint
            hint_new = (hint_rt + delta_rt_height // 2, hint_im + delta_im_width // 2)
        return (
            image_new,
            hint_new,
            target,
        )


class Mask_MinMaxScale:
    def __init__(self, scale_log_channel: bool = True) -> None:
        self.scale_log_channel = scale_log_channel

    def __call__(self, data_hint_bbox):
        image = data_hint_bbox[0]
        target = data_hint_bbox[2]
        # target["ori_image_raw"] = image[0].detach().clone()
        image[0] = image[0] / image[0].max().item()
        hint = data_hint_bbox[1]
        if self.scale_log_channel:
            image[1] = image[1] / image[1].max().item()

        return image, hint, target


class Resize:
    """Resize the image and convert the label
    to the new shape of the image"""

    def __init__(self, new_size=(256, 256)):
        self.new_rt_width = new_size[0]
        self.new_im_height = new_size[1]

    def __call__(self, data_hint_bbox):
        # Logger.debug("data_hint_bbox %s", data_hint_bbox)
        image = data_hint_bbox[0]
        hint = data_hint_bbox[1]
        label = data_hint_bbox[2]["boxes"]

        # Logger.debug("label  %s", label)
        im0, rt0, im1, rt1 = label
        hint_rt, hint_im = hint
        # Logger.debug("image size %s", image.size())
        batch_number, original_rt_width, original_im_height = image.size()
        image_new = F.resize(image, (self.new_rt_width, self.new_im_height))  # type: ignore
        im0_new = im0 * self.new_im_height / original_im_height
        im1_new = im1 * self.new_im_height / original_im_height
        rt0_new = rt0 * self.new_rt_width / original_rt_width
        rt1_new = rt1 * self.new_rt_width / original_rt_width
        hint_rt_new = hint_rt * self.new_rt_width / original_rt_width
        hint_im_new = hint_im * self.new_im_height / original_im_height
        return (
            image_new,
            (hint_rt_new, hint_im_new),
            (im0_new, rt0_new, im1_new, rt1_new),
        )


class ToTensor:
    """Convert the image to a Pytorch tensor with
    the channel as first dimenstion and values
    between 0 to 1. Also convert the label to tensor
    with values between 0 to 1"""

    def __init__(self, scale_label=True):
        self.scale_label = scale_label

    def __call__(self, data_hint_bbox):
        # Logger.debug("data_hint_bbox %s", data_hint_bbox)
        image = data_hint_bbox[0]
        hint = data_hint_bbox[1]
        label = data_hint_bbox[2]
        x0, y0, x1, y1 = label
        hint_x, hint_y = hint
        # image = F.to_tensor(image)

        if self.scale_label:
            x0, y0, x1, y1 = (
                x0 / image.size(1),
                y0 / image.size(2),
                x1 / image.size(1),
                y1 / image.size(2),
            )
            hint_x, hint_y = hint_x / image.size(1), hint_y / image.size(2)
        label = torch.tensor([x0, y0, x1, y1], dtype=torch.float32)
        hint = torch.tensor([hint_x, hint_y], dtype=torch.float32)

        return image, hint, label


class Mask_ToTensor:
    """Convert the image to a Pytorch tensor with
    the channel as first dimenstion and values
    between 0 to 1. Also convert the label to tensor
    with values between 0 to 1"""

    def __init__(self):
        pass

    def __call__(self, data_hint_bbox):
        image = data_hint_bbox[0]
        hint = data_hint_bbox[1]
        # label = data_hint_bbox[2]["mask"]
        label = data_hint_bbox[2]
        image[0] = F.to_tensor(image[0])
        hint = F.to_tensor(hint)
        return (
            image,
            hint,
            label,
        )


class ToPILImage:
    """Convert a tensor image to PIL Image.
    Also convert the label to a tuple with
    values with the image units"""

    def __init__(self, unscale_label=True):
        self.unscale_label = unscale_label

    def __call__(self, data_hint_bbox):
        image = data_hint_bbox[0]
        label = data_hint_bbox[1]

        image = F.to_pil_image(image)
        w, h = image.size

        if self.unscale_label:
            x0, y0, x1, y1 = label
            x0, y0, x1, y1 = (
                x0 / w,
                y0 / h,
                x1 / w,
                y1 / h,
            )
            label = x0, y0, x1, y1

        return image, label


def build_transformation(dataset_cfg: CfgNode):
    transformation = []
    match dataset_cfg.RESHAPE_METHOD:
        case "resize":
            transformation.append(Mask_Resize(dataset_cfg.RESIZE_SHAPE))
        case "padding":
            transformation.append(Mask_Padding(dataset_cfg.PADDING_SHAPE))
    # TODO: so far normal is always there
    if (
        "normal" not in dataset_cfg.INPUT_CHANNELS
    ):  # do log transform if normal channel is not included
        transformation.append(Mask_LogTransform())
    else:  # if normal channel is included, optional to add log channel
        if "log" in dataset_cfg.INPUT_CHANNELS:
            transformation.append(Mask_AddLogChannel())
    if "hint" in dataset_cfg.INPUT_CHANNELS:
        transformation.append(Mask_AddHintChannel())
        # dataset_cfg.USE_HINT_CHANNEL = True
    # else:
    #     #dataset_cfg.USE_HINT_CHANNEL = False
    if dataset_cfg.MINMAX_SCALE:
        # raise NotImplementedError("TO_TENSOR is not implemented yet")
        transformation.append(Mask_MinMaxScale())
        # transformation.append(transforms.ToTensor())
    # dataset_cfg.N_CHANNEL = len(dataset_cfg.INPUT_CHANNELS)
    # match dataset_cfg.INPUT_CHANNELS:
    #     case ["normal"]:
    #         transformation = [
    #             Mask_Padding(dataset_cfg.PADDING_SHAPE),
    #             Mask_ToTensor(),
    #         ]
    #         dataset_cfg.N_CHANNEL = 1
    #     case ["log"]:
    #         transformation = [
    #             Mask_Padding(dataset_cfg.PADDING_SHAPE),
    #             Mask_LogTransform(),
    #             Mask_ToTensor(),
    #         ]
    #         dataset_cfg.N_CHANNEL = 1
    #     case ["normal", "log"]:  # normal is always the first channel if both are used
    #         transformation = [
    #             Mask_Padding(dataset_cfg.PADDING_SHAPE),
    #             Mask_AddLogChannel(),
    #             Mask_ToTensor(),
    #         ]
    #         dataset_cfg.N_CHANNEL = 2
    #     case _:
    #         raise ValueError(
    #             "Invalid input channels, please use ['normal'], ['log'] or ['normal',"
    #             " 'log']."
    #         )
    Logger.info("Transformation: %s", transformation)
    return Compose(transformation), dataset_cfg
