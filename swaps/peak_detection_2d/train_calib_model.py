"""

train calibration model
Usage:
    train_calib_model --best_seg_model_path=<path> --path_cfg=<path>
    train_calib_model -h | --help

Options:
    -h --help               show this screen help
    --version              show version
"""


import os

from datetime import datetime
import logging

import torch
import numpy as np
from docopt import docopt
import json
from torch.utils.tensorboard import SummaryWriter

from peak_detection_2d.loss.build_metric import build_metric

from .model.build_model import build_model
from .model.conf_model import train_one_epoch, evaluate, inference_and_sum_intensity
from .solver.build_optimizer import (
    build_early_stopper,
    build_optimizer,
    build_scheduler,
)
from .loss.build_criterion import build_criterion
from .loss.custom_loss import (
    per_image_weighted_iou_metric,
)
from .dataset.dataset import MultiHDF5_MaskDataset, build_transformation
from .dataset.confidence_dataset import (
    ConfidenceDataset,
    prepare_2d_seg_output_and_confidence,
    Conf_AsBinary,
)
from torchvision.transforms import Compose
from utils.config import get_cfg_defaults
from .config.singleton_peak_detection import peak_detection_cfg
from .utils import (
    plot_confidence_distr,
)


def train_calib_model(
    cfg_confmodel,
    train_dataloader,
    val_dataloader,
    device: str = "cuda",
    peak_selection_dir: str = None,
):
    backup_dir = os.path.join(peak_selection_dir, "conf_model_backups")
    results_dir = os.path.join(peak_selection_dir, "conf_model_results")
    os.makedirs(backup_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=os.path.join(results_dir, "logs_tensorflow"))
    # Build model using config dict node
    model = build_model(cfg_confmodel)
    model.to(device)
    # Epochs
    total_epochs = cfg_confmodel.SOLVER.TOTAL_EPOCHS

    # Optimizer, scheduler, amp
    optimizer = build_optimizer(model, cfg_confmodel.SOLVER.OPTIMIZER)
    scheduler_type = cfg_confmodel.SOLVER.SCHEDULER.NAME
    scheduler = build_scheduler(
        optimizer,
        cfg_confmodel.SOLVER.SCHEDULER,
        steps_per_epoch=int(len(train_dataloader)),
        epochs=total_epochs,
    )
    criterion = build_criterion(cfg_confmodel.SOLVER.LOSS)
    eval_metric = build_metric(cfg_confmodel.EVAL)
    es = build_early_stopper(cfg_confmodel.SOLVER.EARLY_STOPPING)

    current_epoch = 0

    # # MultiGPU training
    # multi_gpu_training = cfg_model.MULTI_GPU_TRAINING
    if cfg_confmodel.RESUME_PATH != "":
        checkpoint = torch.load(cfg_confmodel.RESUME_PATH, map_location=device)
        current_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["model_state_dict"])
        logging.info("Model loaded from %s", cfg_confmodel.RESUME_PATH)
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            logging.info("Optimizer loaded from %s", cfg_confmodel.RESUME_PATH)
        if "scheduler_state_dict" in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            logging.info("Scheduler loaded from %s", cfg_confmodel.RESUME_PATH)

    metric = {"train": [], "val": []}
    loss_tracking = {"train": [], "val": []}
    for epoch in range(current_epoch, total_epochs):
        logging.info("Start epoch %s", epoch)

        loss = train_one_epoch(
            train_loader=train_dataloader,
            model=model,
            optimizer=optimizer,
            loss_fn=criterion,
            device=device,
            scheduler=scheduler,
        )

        ####################
        # Training and validation metrics
        ####################
        train_metric = evaluate(
            train_dataloader,
            model,
            metric=eval_metric,
            device=device,
        )

        val_metric = evaluate(
            val_dataloader,
            model,
            metric=eval_metric,
            device=device,
        )

        # Store training trace
        metric["train"].append(train_metric)
        metric["val"].append(val_metric)
        loss_tracking["train"].append(loss)

        print(
            f"EPOCH: {epoch}, TRAIN LOSS: {loss}, TRAIN {cfg_confmodel.EVAL.METRIC}:"
            f" {train_metric}, VAL {cfg_confmodel.EVAL.METRIC}: {val_metric}"
        )
        if tb_writer is not None:
            tb_writer.add_scalar("Loss/train", loss, epoch)
            tb_writer.add_scalar("Metric/train", train_metric, epoch)
            tb_writer.add_scalar("Metric/val", val_metric, epoch)
            tb_writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        ######################################
        # Update early stopper and scheduler, and saving model
        ######################################
        # Update scheudler here if not 'OneCycleLR'
        if scheduler is not None and scheduler_type != "one_cycle":
            if scheduler_type == "reduce_on_plateau":
                scheduler.step(val_metric)
            else:
                scheduler.step()
        es(
            epoch_score=val_metric,
            epoch_num=epoch,
            loss=loss,
            optimizer=optimizer,
            model=model,
            model_path=os.path.join(
                backup_dir,
                f"bst_model_{np.round(val_metric,4)}.pt",
            ),
            scheduler=scheduler,
        )
        if cfg_confmodel.SOLVER.EARLY_STOPPING.MODE == "min":
            factor = -1
        else:
            factor = 1
        best_model_path = os.path.join(
            backup_dir, f"bst_model_{np.round(factor*es.best_score,4)}.pt"
        )
        tb_writer.close()

        if es.early_stop:
            print("\n\n -------------- EARLY STOPPING -------------- \n\n")
            break
    with open(os.path.join(results_dir, "loss.json"), "w", encoding="utf-8") as fp:
        json.dump(loss_tracking, fp)
    with open(os.path.join(results_dir, "metric.json"), "w", encoding="utf-8") as fp:
        json.dump(metric, fp)

    return best_model_path


def get_image_dataset_and_prepare_conf_dataset(
    cfg_dataset,
    cfg_segmodel,
    device,
    best_model_path,
    peak_selection_dir,
    prepare_conf_dataset: bool = True,
):
    h5_path = os.path.join(
        cfg_dataset.RAW_DATA_PATH,
        cfg_dataset.ACTIVATION_PATH,
        cfg_dataset.MODEL_DATA_PATH,
    )
    # Load data
    hdf5_files = [
        os.path.join(h5_path, f) for f in os.listdir(h5_path) if f.endswith(".h5")
    ]

    transformation = build_transformation(cfg_dataset)
    if cfg_dataset.INPUT_CHANNELS == ["log"]:
        cfg_dataset.ONLY_LOG_CHANNEL = True
    logging.info("Dataset channels: %d", cfg_segmodel.PARAMS.IN_CHANNELS)

    # Create the dataset
    dataset = MultiHDF5_MaskDataset(hdf5_files, transforms=transformation)

    # Build model using config dict node
    model = build_model(cfg_segmodel)
    model.to(device)

    checkpoint = torch.load(best_model_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    logging.info("Model loaded from %s", best_model_path)

    # Split the dataset into training and testing sets
    train_val_dataset, test_image_dataset = dataset.split_dataset(
        train_ratio=cfg_dataset.TRAIN_VAL_SIZE, seed=cfg_dataset.RANDOM_STATE
    )
    train_image_dataset, val_image_dataset = train_val_dataset.split_dataset(
        train_ratio=cfg_dataset.TRAIN_SIZE, seed=cfg_dataset.RANDOM_STATE
    )
    train_image_dataloader = torch.utils.data.DataLoader(
        train_image_dataset, batch_size=128, shuffle=False
    )
    val_image_dataloader = torch.utils.data.DataLoader(
        val_image_dataset, batch_size=128, shuffle=False
    )
    test_image_dataloader = torch.utils.data.DataLoader(
        test_image_dataset, batch_size=128, shuffle=False
    )
    logging.info("Train dataset size: %d", len(train_image_dataset))
    logging.info("Validation dataset size: %d", len(val_image_dataset))
    logging.info("Test dataset size: %d", len(test_image_dataset))
    if prepare_conf_dataset:
        prepare_2d_seg_output_and_confidence(
            dataloader=train_image_dataloader,
            model=model,
            device=device,
            use_image_for_metric=True,
            metric=per_image_weighted_iou_metric,
            save_hdf5=os.path.join(
                peak_selection_dir, "confidence_dataset", "train_confidence_dataset.h5"
            ),
            channel=0,
        )

        prepare_2d_seg_output_and_confidence(
            dataloader=val_image_dataloader,
            model=model,
            device=device,
            use_image_for_metric=True,
            metric=per_image_weighted_iou_metric,
            save_hdf5=os.path.join(
                peak_selection_dir, "confidence_dataset", "val_confidence_dataset.h5"
            ),
            channel=0,
        )

        prepare_2d_seg_output_and_confidence(
            dataloader=test_image_dataloader,
            model=model,
            device=device,
            use_image_for_metric=True,
            metric=per_image_weighted_iou_metric,
            save_hdf5=os.path.join(
                peak_selection_dir, "confidence_dataset", "test_confidence_dataset.h5"
            ),
            channel=0,
        )
    return test_image_dataloader


def image_level_eval(
    best_conf_model_path,
    cfg_confmodel,
    best_seg_model_path,
    cfg_segmodel,
    test_dataloader,
    device,
    exp: bool = False,
    peak_selection_dir: str = None,
):
    results_dir = os.path.join(peak_selection_dir, "conf_model_results")
    # Build model using config dict node
    conf_model = build_model(cfg_confmodel)
    conf_checkpoint = torch.load(best_conf_model_path, map_location=device)
    conf_model.load_state_dict(conf_checkpoint["model_state_dict"])
    conf_model.to(device)

    seg_model = build_model(cfg_segmodel)
    seg_checkpoint = torch.load(best_seg_model_path, map_location=device)
    seg_model.load_state_dict(seg_checkpoint["model_state_dict"])
    seg_model.to(device)

    test_df = inference_and_sum_intensity(
        data_loader=test_dataloader,
        model=seg_model,
        device=device,
        calc_score="conf_model",
        conf_model=conf_model,
        exp=exp,
    )
    test_df.to_csv(os.path.join(results_dir, "test_results.csv"), index=False)
    plot_confidence_distr(test_df, save_dir=results_dir)


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
        best_seg_model_path = arguments["--best_seg_model_path"]
        calib_cfg_path = arguments["--path_cfg"]
        output_parent_path = os.path.dirname(os.path.dirname(best_seg_model_path))
        # get timestamp as folder name
        output_folder_name = datetime.now().strftime("%Y%m%d_%H%M%S%f")
        logging.info("Output folder name: %s", output_folder_name)
        cfg = get_cfg_defaults(peak_detection_cfg)
        seg_cfg_path = [
            os.path.join(output_parent_path, "results", f)
            for f in os.listdir(os.path.join(output_parent_path, "results"))
            if f.endswith(".yaml")
        ]
        if len(seg_cfg_path) > 0:
            cfg.merge_from_file(seg_cfg_path[0])
            print(f"merge with segmentaion model cfg file {seg_cfg_path[0]}")
        if calib_cfg_path is not None:
            cfg.merge_from_file(calib_cfg_path)
            print(f"merge with calibration model cfg file {calib_cfg_path}")
        cfg.OUTPUT_PARENT_PATH = output_parent_path
        cfg.OUTPUT_FOLDER_NAME = output_folder_name
        cfg.CONFIG_PATH = calib_cfg_path
        if cfg.DATASET.INPUT_CHANNELS == ["log"]:
            cfg.DATASET.ONLY_LOG_CHANNEL = True
            cfg.DATASET.N_CHANNEL = len(cfg.DATASET.INPUT_CHANNELS)
            cfg.MODEL.PARAMS.IN_CHANNELS = cfg.DATASET.N_CHANNEL
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        calib_exp_dir = os.path.join(cfg.OUTPUT_PARENT_PATH, cfg.OUTPUT_FOLDER_NAME)
        os.makedirs(calib_exp_dir, exist_ok=True)
        # Save configs
        cfg.dump(
            stream=open(
                os.path.join(calib_exp_dir, f"config_{output_folder_name}.yaml"),
                "w",
                encoding="utf-8",
            )
        )
        # Create dataset if not exist
        conf_dataset_paths = [
            os.path.join(
                cfg.OUTPUT_PARENT_PATH,
                "confidence_dataset",
                dataset_name,
            )
            for dataset_name in [
                "train_confidence_dataset.h5",
                "val_confidence_dataset.h5",
                "test_confidence_dataset.h5",
            ]
        ]
        if all(list(map(os.path.exists, conf_dataset_paths))):
            logging.info("All confidence datasets exist, start training")
            test_image_dataloader = get_image_dataset_and_prepare_conf_dataset(
                cfg.DATASET,
                cfg.MODEL,
                device=DEVICE,
                best_model_path=best_seg_model_path,
                peak_selection_dir=output_parent_path,
                prepare_conf_dataset=False,
            )
        else:
            logging.info("Some confidence datasets do not exist, prepare them")
            os.makedirs(
                os.path.join(output_parent_path, "confidence_dataset"), exist_ok=True
            )
            test_image_dataloader = get_image_dataset_and_prepare_conf_dataset(
                cfg.DATASET,
                cfg.MODEL,
                device=DEVICE,
                best_model_path=best_seg_model_path,
                peak_selection_dir=output_parent_path,
                prepare_conf_dataset=True,
            )
        transformation = None
        if cfg.CONFMODEL.BINARY_LABEL:
            transformation = Compose([Conf_AsBinary()])

        if cfg.CONFMODEL.DATASET_RESPLIT:
            try:
                train_conf_dataset = ConfidenceDataset(
                    os.path.join(
                        output_parent_path,
                        "confidence_dataset",
                        "newsplit_train_confidence_dataset.h5",
                    ),
                    transforms=transformation,
                )
                val_conf_dataset = ConfidenceDataset(
                    os.path.join(
                        output_parent_path,
                        "confidence_dataset",
                        "newsplit_val_confidence_dataset.h5",
                    ),
                    transforms=transformation,
                )
            except FileNotFoundError:
                logging.info("Resplit dataset not found, resplitting dataset...")
                ori_train_conf_dataset = ConfidenceDataset(
                    os.path.join(
                        output_parent_path,
                        "confidence_dataset",
                        "train_confidence_dataset.h5",
                    ),
                    transforms=transformation,
                )
                ori_val_conf_dataset = ConfidenceDataset(
                    os.path.join(
                        output_parent_path,
                        "confidence_dataset",
                        "val_confidence_dataset.h5",
                    ),
                    transforms=transformation,
                )
                (
                    train_conf_dataset,
                    val_conf_dataset,
                    _,
                ) = ori_train_conf_dataset.create_splits(
                    test_size=None,
                    val_size=(1 - cfg.DATASET.TRAIN_SIZE * cfg.DATASET.TRAIN_VAL_SIZE),
                )
                train_conf_dataset = ConfidenceDataset.combine_datasets(
                    train_conf_dataset,
                    ori_val_conf_dataset,
                    os.path.join(
                        output_parent_path,
                        "confidence_dataset",
                        "newsplit_train_confidence_dataset.h5",
                    ),
                )
                ConfidenceDataset.save_dataset_to_hdf5(
                    val_conf_dataset,
                    os.path.join(
                        output_parent_path,
                        "confidence_dataset",
                        "newsplit_val_confidence_dataset.h5",
                    ),
                )
                logging.info("Resplit dataset saved.")
                logging.info("Train dataset size: %d", len(train_conf_dataset))
                logging.info("Validation dataset size: %d", len(val_conf_dataset))
        else:
            train_conf_dataset = ConfidenceDataset(
                os.path.join(
                    output_parent_path,
                    "confidence_dataset",
                    "train_confidence_dataset.h5",
                ),
                transforms=transformation,
            )
            val_conf_dataset = ConfidenceDataset(
                os.path.join(
                    output_parent_path,
                    "confidence_dataset",
                    "val_confidence_dataset.h5",
                ),
                transforms=transformation,
            )
        # test confidence dataset remain unchanged
        test_conf_dataset = ConfidenceDataset(
            os.path.join(
                output_parent_path, "confidence_dataset", "test_confidence_dataset.h5"
            ),
            transforms=transformation,
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_conf_dataset, batch_size=cfg.DATASET.TRAIN_BATCH_SIZE
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_conf_dataset, batch_size=cfg.DATASET.VAL_BATCH_SIZE
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_conf_dataset, batch_size=cfg.DATASET.VAL_BATCH_SIZE
        )
        logging.info("Train dataset size: %d", len(train_conf_dataset))
        logging.info("Validation dataset size: %d", len(val_conf_dataset))
        logging.info("Test dataset size: %d", len(test_conf_dataset))

        # Train confidence model
        best_calib_model_path = train_calib_model(
            cfg_confmodel=cfg.CONFMODEL,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            device=DEVICE,
            peak_selection_dir=calib_exp_dir,
        )
        logging.info(
            "Seg model input channels: %s, first out channel: %s, dataset n"
            " channel: %s",
            cfg.MODEL.PARAMS.IN_CHANNELS,
            cfg.MODEL.PARAMS.FIRST_OUT_CHANNELS,
            cfg.DATASET.N_CHANNEL,
        )
        image_level_eval(
            best_conf_model_path=best_calib_model_path,
            cfg_confmodel=cfg.CONFMODEL,
            best_seg_model_path=best_seg_model_path,
            cfg_segmodel=cfg.MODEL,
            test_dataloader=test_image_dataloader,
            device=DEVICE,
            exp=cfg.DATASET.ONLY_LOG_CHANNEL,
            peak_selection_dir=calib_exp_dir,
        )
    except Exception as e:
        print(f"Error: {e}")
        print(__doc__)
        raise
