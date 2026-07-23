"""torch.utils.data.Dataset over Phase B's (image, hint, mask) HDF5s.

New, from-scratch replacement for `multi_output_PS_model`'s
`dataset/dataset.py` (`MultiHDF5_MaskDataset`, `PeptActPeakSelection_Infer_Dataset`,
`build_transformation`, `Mask_Resize`/`Mask_Padding`/`Mask_LogTransform`/
`Mask_AddHintChannel`/`Mask_AddLogChannel`/`Mask_ToTensor`), which assumed
the old one-HDF5-group-per-sample format with variable native shapes needing
runtime resize/pad. `prepare_dataset.build_training_dataset`'s
`{split}.h5` already has 3 fixed-shape `[N, H, W]` datasets
(images/hints/masks, see manifest.yaml's target_shape), so none of that
resize/pad machinery is needed here.
"""

import os
from typing import Optional

import h5py
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset


def load_target_shape(output_dir: str) -> tuple[int, int]:
    """Read manifest.yaml's target_shape -- the authoritative source, see
    prepare_dataset.build_training_dataset; never hardcode this."""
    manifest_path = os.path.join(output_dir, "manifest.yaml")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f)
    h, w = manifest["target_shape"]
    return int(h), int(w)


class PeakSegmentationDataset(Dataset):
    """One split's (image, hint, mask) triplets, stacked into an
    (IN_CHANNELS, H, W) model input: [raw intensity, log1p(raw intensity),
    hint channel] by default. Channel 0 must stay the raw intensity image --
    ComboLoss's weighted-dice term indexes images[:, 0, ...] as its per-pixel
    weight (see config/singleton_peak_detection.py).

    Opens its own h5py.File handle lazily on first access rather than in
    __init__, so each DataLoader worker process gets its own handle instead
    of one forked/pickled from the main process (standard PyTorch+h5py
    multi-worker practice).
    """

    def __init__(
        self,
        h5_path: str,
        metadata_path: Optional[str] = None,
        add_log_channel: bool = True,
    ):
        self.h5_path = h5_path
        self.add_log_channel = add_log_channel
        self.metadata = (
            pd.read_parquet(metadata_path) if metadata_path is not None else None
        )
        self._h5file: Optional[h5py.File] = None
        with h5py.File(h5_path, "r") as f:
            self._len = f["images"].shape[0]

    @classmethod
    def from_split(
        cls, output_dir: str, split: str, add_log_channel: bool = True
    ) -> "PeakSegmentationDataset":
        return cls(
            h5_path=os.path.join(output_dir, f"{split}.h5"),
            metadata_path=os.path.join(output_dir, f"{split}_metadata.parquet"),
            add_log_channel=add_log_channel,
        )

    def _file(self) -> h5py.File:
        if self._h5file is None:
            self._h5file = h5py.File(self.h5_path, "r")
        return self._h5file

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        f = self._file()
        image = f["images"][idx]
        hint = f["hints"][idx]
        mask = f["masks"][idx]

        channels = [image]
        if self.add_log_channel:
            channels.append(np.log1p(image))
        channels.append(hint)
        image_tensor = torch.from_numpy(np.stack(channels, axis=0)).float()
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)
        return image_tensor, mask_tensor

    def __getstate__(self):
        # h5py.File handles aren't picklable -- drop it, __getitem__ reopens
        # lazily in whichever process/worker actually needs it.
        state = self.__dict__.copy()
        state["_h5file"] = None
        return state
