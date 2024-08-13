"""

calibrate model
Usage:
    calibrate.py --model_dir=<path> --model_name=<path>
    calibrate.py -h | --help

Options:
    -h --help               show this screen help
    --version              show version
"""


import os
from datetime import datetime
import logging
import torch
import numpy as np
import pandas as pd
from docopt import docopt
from sklearn.isotonic import IsotonicRegression
from pickle import dump, load
from .model.build_model import build_model
from .model.seg_model import (
    evaluate,
    inference_flatten_output,
    label_and_sum_intensity,
)
from .model.conf_model import inference_and_sum_intensity
from .loss.custom_loss import (
    per_image_weighted_iou_metric,
)
from .dataset.dataset import MultiHDF5_MaskDataset, build_transformation
from utils.config import get_cfg_defaults

from .loss.reliability_diagram import reliability_diagram


def calibrate(model_dir, model_name, cfg):
    #############################
    # Pre-calibrating
    #############################

    # PATHS
    torch.manual_seed(cfg.DATASET.RANDOM_STATE)
    results_dir = os.path.join(model_dir, "calibration")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Create writable timestamp for easier record keeping
    timestamp = datetime.now().isoformat(sep="T", timespec="auto")
    name_timestamp = timestamp.replace(":", "_")

    h5_path = os.path.join(
        cfg.DATASET.RAW_DATA_PATH,
        cfg.DATASET.ACTIVATION_PATH,
        cfg.DATASET.MODEL_DATA_PATH,
    )

    # Load data
    hdf5_files = [
        os.path.join(h5_path, f) for f in os.listdir(h5_path) if f.endswith(".h5")
    ]

    transformation = build_transformation(cfg.DATASET)
    if cfg.DATASET.INPUT_CHANNELS == ["log"]:
        cfg.ONLY_LOG_CHANNEL = True
    cfg.MODEL.PARAMS.IN_CHANNELS = cfg.DATASET.N_CHANNEL
    logging.info("Dataset channels: %d", cfg.MODEL.PARAMS.IN_CHANNELS)

    # Create the dataset
    dataset = MultiHDF5_MaskDataset(hdf5_files, transforms=transformation)
    # Split the dataset into training and testing sets
    train_val_dataset, test_dataset = dataset.split_dataset(
        train_ratio=cfg.DATASET.TRAIN_VAL_SIZE, seed=cfg.DATASET.RANDOM_STATE
    )
    train_dataset, val_dataset = train_val_dataset.split_dataset(
        train_ratio=cfg.DATASET.TRAIN_SIZE, seed=cfg.DATASET.RANDOM_STATE
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.DATASET.TRAIN_BATCH_SIZE, shuffle=False
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=128, shuffle=False
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False
    )
    logging.info("Train dataset size: %d", len(train_dataset))
    logging.info("Validation dataset size: %d", len(val_dataset))
    logging.info("Test dataset size: %d", len(test_dataset))

    # Build model using config dict node
    model = build_model(cfg.MODEL)
    model.to(device)

    model_path = os.path.join(model_dir, "model_backups", model_name)
    checkpoint = torch.load(model_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    logging.info("Model loaded from %s", model_path)

    # Calibrate with isotonic regression
    try:
        with open(os.path.join(results_dir, "iso_reg.pkl"), "rb") as f:
            iso_reg = load(f)
        logging.info("Isotonic regression loaded from %s", results_dir)
    except FileNotFoundError:
        logging.info("Isotonic regression not found, start training...")

        # Create training data from val set
        val_score, val_out_final, val_label, val_out = inference_flatten_output(
            data_loader=val_dataloader, model=model, device=device, get_labels=True
        )  # score is the prob of class 1
        logging.info("Finished calculating val set score, start isotonic regression...")
        iso_reg = IsotonicRegression().fit(val_out, val_label)
        with open(os.path.join(results_dir, "iso_reg.pkl"), "wb") as f:
            dump(iso_reg, f, protocol=5)
        del (val_score, val_out_final, val_label, val_out)
        logging.info("Finished isotonic regression, start evalutation...")

        # Test on test set, pixel-wise
        test_score, test_out_final, test_label, test_out = inference_flatten_output(
            data_loader=test_dataloader, model=model, device=device, get_labels=True
        )
        test_score_corrected = abs(0.5 - np.array(test_score)) * 2
        fig_sigmoid = reliability_diagram(
            true_labels=test_label,
            pred_labels=test_out_final,
            confidences=test_score_corrected,
            num_bins=10,
            return_fig=True,
            draw_bin_importance=False,
            dpi=100,
        )
        fig_sigmoid.savefig(
            os.path.join(results_dir, "reliability_diagram_sigmoid.png")
        )
        isoreg_score_test = iso_reg.predict(test_out)
        isoreg_score_test_corrected = abs(0.5 - isoreg_score_test) * 2
        fig = reliability_diagram(
            true_labels=test_label,
            pred_labels=test_out_final,
            confidences=isoreg_score_test_corrected,
            num_bins=10,
            return_fig=True,
            draw_bin_importance=False,
            dpi=100,
        )
        fig.savefig(os.path.join(results_dir, "reliability_diagram_isoreg.png"))
        del (test_out_final, test_label, test_out)

        logging.info(
            "Finished test set pixel-wise evaluation, start image-wise evaluation..."
        )

    # Image wise
    logging.info("Generating test set image-wise evaluation...")
    test_all_df = inference_and_sum_intensity(
        data_loader=test_dataloader,
        model=model,
        device=device,
        calib_model=iso_reg,
        calc_score="iso_reg",
        threshold=0.5,
        plot_calib_score_distribution=True,
        result_dir=results_dir,
    )
    # test_label_df = label_and_sum_intensity(data_loader=test_dataloader, device=device)
    # test_wiou, per_image_test_wiou = evaluate(
    #     test_dataloader,
    #     model,
    #     metric=per_image_weighted_iou_metric,
    #     device=device,
    #     use_image_for_metric=True,
    #     save_all_loss=True,
    #     channel=0,
    # )
    # per_image_test_wiou_df = pd.DataFrame(per_image_test_wiou)
    # test_all_df1 = pd.merge(
    #     left=test_pred_df,
    #     right=test_label_df,
    #     on="pept_mz_rank",
    #     suffixes=("_pred", "_label"),
    # )
    # test_all_df = pd.merge(
    #     left=test_all_df1,
    #     right=per_image_test_wiou_df,
    #     left_on="pept_mz_rank",
    #     right_on="ranks",
    # )
    # test_all_df["delta_log_intensity"] = abs(
    #     np.log10(test_all_df["sum_intensity_label"] + 1)
    #     - np.log10(test_all_df["sum_intensity_pred"] + 1)
    # )
    # test_all_df["log_label_intensity"] = np.log10(
    #     test_all_df["sum_intensity_label"] + 1
    # )
    # test_all_df["log_pred_intensity"] = np.log10(test_all_df["sum_intensity_pred"] + 1)
    # test_all_df.fillna(0.5, inplace=True)

    test_all_df.to_csv(os.path.join(results_dir, "test_all_df.csv"), index=False)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    try:
        arguments = docopt(
            __doc__, argv=None, help=False, version=None, options_first=False
        )
        print("Arguments parsed:")
        print(arguments)
        model_dir = arguments["--model_dir"]
        model_name = arguments["--model_name"]
        cfg = get_cfg_defaults()
        cfg_path = [
            os.path.join(model_dir, "results", f)
            for f in os.listdir(os.path.join(model_dir, "results"))
            if f.endswith(".yaml")
        ]
        cfg.merge_from_file(cfg_path[0])
        calibrate(model_dir, model_name, cfg)
    except Exception as e:
        print(f"Error: {e}")
        print(__doc__)
        raise
