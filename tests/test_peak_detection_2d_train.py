import copy
import os

import h5py
import numpy as np
import pandas as pd
import yaml

from swaps.peak_detection_2d.config.singleton_peak_detection import peak_detection_cfg
from swaps.peak_detection_2d.train import train


def _write_synthetic_split(data_dir, split, n, shape, seed):
    rng = np.random.default_rng(seed)
    h, w = shape
    with h5py.File(os.path.join(data_dir, f"{split}.h5"), "w") as f:
        f.create_dataset("images", data=rng.random((n, h, w)).astype(np.float32))
        f.create_dataset("hints", data=(rng.random((n, h, w)) > 0.99).astype(np.float32))
        f.create_dataset("masks", data=(rng.random((n, h, w)) > 0.7).astype(np.float32))
    meta = pd.DataFrame(
        {
            "row": np.arange(n),
            "mz_rank": np.arange(1000, 1000 + n),
            "source_experiment": ["synthetic_exp"] * n,
            "contributing_runs": ["run_a;run_b"] * n,
            "reference_raw_file": ["run_a"] * n,
            "was_rescaled": [False] * n,
        }
    )
    meta.to_parquet(os.path.join(data_dir, f"{split}_metadata.parquet"), index=False)


def _make_synthetic_dataset(tmp_path, shape=(32, 32)):
    data_dir = str(tmp_path / "dataset")
    os.makedirs(data_dir, exist_ok=True)
    _write_synthetic_split(data_dir, "train", n=6, shape=shape, seed=0)
    _write_synthetic_split(data_dir, "val", n=4, shape=shape, seed=1)
    with open(os.path.join(data_dir, "manifest.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump({"target_shape": list(shape)}, f)
    return data_dir


def test_train_runs_end_to_end_on_synthetic_data(tmp_path):
    data_dir = _make_synthetic_dataset(tmp_path)
    run_dir = str(tmp_path / "run")

    cfg = copy.deepcopy(peak_detection_cfg)
    cfg.DATASET.DATA_DIR = data_dir
    cfg.DATASET.TARGET_SHAPE = (32, 32)
    cfg.DATASET.TRAIN_BATCH_SIZE = 2
    cfg.DATASET.VAL_BATCH_SIZE = 2
    cfg.DATASET.NUM_WORKERS = 0
    cfg.MODEL.PARAMS.DOWNHILL = 2  # 32 -> 16 -> 8, safe for a tiny synthetic image
    cfg.MODEL.SOLVER.TOTAL_EPOCHS = 2
    cfg.MODEL.SOLVER.SCHEDULER.NAME = "unchange"
    cfg.MODEL.SOLVER.EARLY_STOPPING.PATIENCE = 10

    best_ckpt = train(cfg, run_dir)

    assert best_ckpt, "train() should return a non-empty best checkpoint path"
    assert os.path.exists(best_ckpt)
    assert os.path.isdir(os.path.join(run_dir, "checkpoints"))
    assert os.path.isdir(os.path.join(run_dir, "tensorboard"))


def test_train_respects_total_epochs_count(tmp_path):
    data_dir = _make_synthetic_dataset(tmp_path)
    run_dir = str(tmp_path / "run")

    cfg = copy.deepcopy(peak_detection_cfg)
    cfg.DATASET.DATA_DIR = data_dir
    cfg.DATASET.TARGET_SHAPE = (32, 32)
    cfg.DATASET.TRAIN_BATCH_SIZE = 2
    cfg.DATASET.VAL_BATCH_SIZE = 2
    cfg.DATASET.NUM_WORKERS = 0
    cfg.MODEL.PARAMS.DOWNHILL = 2
    cfg.MODEL.SOLVER.TOTAL_EPOCHS = 3
    cfg.MODEL.SOLVER.SCHEDULER.NAME = "unchange"
    cfg.MODEL.SOLVER.EARLY_STOPPING.PATIENCE = 10

    train(cfg, run_dir)

    checkpoints = os.listdir(os.path.join(run_dir, "checkpoints"))
    epoch_numbers = {int(f.split("_")[0].replace("epoch", "")) for f in checkpoints}
    assert max(epoch_numbers) <= 2  # epochs 0, 1, 2 -- zero-indexed, 3 total
