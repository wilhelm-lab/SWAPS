"""

train model
Usage:
    train.py --path_output=<path> --path_cfg=<path>
    train.py -h | --help

Options:
    -h --help               show this screen help
    --version              show version
"""

import os
from datetime import datetime
import logging
import gc
import json
import numpy as np
import pandas as pd
import torch

import torch.utils
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from optimization.custom_models import Logger
from result_analysis.result_analysis import SBSResult

# from .CLSMODEL.conf_model import inference_and_sum_intensity
from .model.build_model import build_model
from .model.seg_model import train_one_epoch, evaluate, inference_and_sum_intensity
from .solver.build_optimizer import (
    build_early_stopper,
    build_optimizer,
    build_scheduler,
)
from .loss.build_criterion import build_criterion
from .loss.custom_loss import (
    per_image_weighted_dice_metric,
    per_image_weighted_iou_metric,
)
from .dataset.dataset import MultiHDF5_MaskDataset, build_transformation

from .utils import (
    plot_sample_predictions,
    plot_per_image_metric_distr,
    plot_target_decoy_distr,
    plot_roc_auc,
    calc_fdr_and_thres,
)


def train(cfg_peak_selection, ps_exp_dir, random_state: int = 42, maxquant_dict=None):
    """Does not use CONFMODEL config"""
    #############################
    # Pre-training
    #############################
    if cfg_peak_selection.EVAL_ON_TEST:
        assert maxquant_dict is not None, "maxquant_dict must be provided for testing"
    torch.manual_seed(random_state)

    hdf5_files = cfg_peak_selection.TRAINING_DATA
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Make backup folders if they do not exist
    backup_dir = os.path.join(ps_exp_dir, "model_backups")
    if not os.path.exists(backup_dir):
        os.mkdir(backup_dir)

    # Make result folders if they do not exist
    ps_exp_results_dir = os.path.join(ps_exp_dir, "results")
    if not os.path.exists(ps_exp_results_dir):
        os.mkdir(ps_exp_results_dir)

    # to initialize Tensorboard
    writer = SummaryWriter(log_dir=os.path.join(ps_exp_dir, "logs_tensorflow"))

    # debug: load a smaller training file
    if cfg_peak_selection.DEBUG:
        hdf5_files = hdf5_files[:1]

    transformation, cfg_peak_selection.DATASET = build_transformation(
        cfg_peak_selection.DATASET
    )
    if cfg_peak_selection.DATASET.INPUT_CHANNELS == ["log"]:
        cfg_peak_selection.DATASET.ONLY_LOG_CHANNEL = True
    cfg_peak_selection.MODEL.PARAMS.IN_CHANNELS = len(
        cfg_peak_selection.DATASET.INPUT_CHANNELS
    )
    cfg_peak_selection.CLSMODEL.PARAMS.IN_CHANNELS = len(
        cfg_peak_selection.DATASET.INPUT_CHANNELS
    )
    logging.info("Dataset channels: %d", cfg_peak_selection.MODEL.PARAMS.IN_CHANNELS)

    # Save configs
    cfg_peak_selection.dump(
        stream=open(
            os.path.join(ps_exp_dir, "updated_peak_selection_config.yaml"),
            "w",
            encoding="utf-8",
        )
    )

    # Create the dataset

    use_hint_channel = "hint" in cfg_peak_selection.DATASET.INPUT_CHANNELS
    logging.info("Use hint channel: %s", use_hint_channel)
    dataset = MultiHDF5_MaskDataset(
        hdf5_files,
        use_hint_channel=use_hint_channel,
        transforms=transformation,
    )
    # sanity check
    image, hint, label = dataset[99]
    logging.info("Image shape in initial dataset: %s", image.shape)
    # Split the dataset into training and testing sets
    train_val_dataset, test_dataset = dataset.split_dataset(
        train_ratio=cfg_peak_selection.DATASET.TRAIN_VAL_SIZE,
        seed=random_state,
    )
    # sanity check
    image, hint, label = train_val_dataset[99]
    logging.info("Image shape in train_val_dataset: %s", image.shape)
    train_dataset, val_dataset = train_val_dataset.split_dataset(
        train_ratio=cfg_peak_selection.DATASET.TRAIN_SIZE,
        seed=random_state,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg_peak_selection.DATASET.TRAIN_BATCH_SIZE,
        shuffle=False,
    )
    _, train_eval_dataset = train_dataset.split_dataset(
        train_ratio=0.9, seed=random_state
    )
    train_eval_dataloader = torch.utils.data.DataLoader(
        train_eval_dataset,
        batch_size=cfg_peak_selection.DATASET.VAL_BATCH_SIZE,
        shuffle=False,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg_peak_selection.DATASET.VAL_BATCH_SIZE, shuffle=False
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg_peak_selection.DATASET.TEST_BATCH_SIZE,
        shuffle=False,
    )
    logging.info("Train dataset size: %d", len(train_dataset))
    logging.info("Train eval dataset size: %d", len(train_eval_dataset))
    logging.info("Validation dataset size: %d", len(val_dataset))
    logging.info("Test dataset size: %d", len(test_dataset))

    #############################
    # Segmentation training
    #############################

    if cfg_peak_selection.MODEL.KEEP_TRAINING:
        # Build model using config dict node
        model = build_model(cfg_peak_selection.MODEL)
        model.to(device)
        # Epochs
        total_epochs = cfg_peak_selection.MODEL.SOLVER.TOTAL_EPOCHS

        # Optimizer, scheduler, amp
        optimizer = build_optimizer(model, cfg_peak_selection.MODEL.SOLVER.OPTIMIZER)
        scheduler_type = cfg_peak_selection.MODEL.SOLVER.SCHEDULER.NAME
        scheduler = build_scheduler(
            optimizer,
            cfg_peak_selection.MODEL.SOLVER.SCHEDULER,
            steps_per_epoch=int(len(train_dataloader)),
            epochs=total_epochs,
        )
        criterion = build_criterion(cfg_peak_selection.MODEL.SOLVER.LOSS)
        es = build_early_stopper(cfg_peak_selection.MODEL.SOLVER.EARLY_STOPPING)

        current_epoch = 0
        if cfg_peak_selection.MODEL.RESUME_PATH != "":
            Logger.info("Loading model from %s", cfg_peak_selection.MODEL.RESUME_PATH)
            checkpoint = torch.load(
                cfg_peak_selection.MODEL.RESUME_PATH, map_location=device
            )
            current_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["model_state_dict"])
            logging.info("Model loaded from %s", cfg_peak_selection.MODEL.RESUME_PATH)
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                logging.info(
                    "Optimizer loaded from %s", cfg_peak_selection.MODEL.RESUME_PATH
                )
            if "scheduler_state_dict" in checkpoint and scheduler is not None:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                logging.info(
                    "Scheduler loaded from %s", cfg_peak_selection.MODEL.RESUME_PATH
                )
        ##########################################
        # Main training epoch loop starts here
        ##########################################

        # s_time = time.time()
        # parameters = list(model.parameters())
        loss_tracking = {"train": [], "val": []}
        metric = {"train": [], "val": []}
        for epoch in range(current_epoch, total_epochs):
            torch.cuda.empty_cache()
            gc.collect()
            logging.info("Start epoch %s", epoch)

            loss = train_one_epoch(
                train_loader=train_dataloader,
                model=model,
                optimizer=optimizer,
                loss_fn=criterion,
                # seg_cls_loss_weight=,
                use_image_as_input=True,
                device=device,
                scheduler=scheduler,
            )

            ###############################
            # Send images to Tensorboard
            # -- could also do this outside the loop with xb, yb = next(itr(DL))
            ###############################

            # if epoch == 0:
            #     # Get the std and mean of each channel
            #     std = torch.FloatTensor(cfg.DATASET.NORMALIZE_STD).view(3, 1, 1)
            #     m = torch.FloatTensor(cfg.DATASET.NORMALIZE_MEAN).view(3, 1, 1)

            #     # Un-normalize images, send mean and std to gpu for mixuped images
            #     imgs, imgs_mixup = ((inputs * std) + m) * 255, (
            #         (input_data * std.cuda()) + m.cuda()
            #     ) * 255
            #     imgs, imgs_mixup = imgs.type(torch.uint8), imgs_mixup.type(torch.uint8)
            #     img_grid = torchvision.utils.make_grid(imgs)
            #     img_grid_mixup = torchvision.utils.make_grid(imgs_mixup)

            #     img_grid = torchvision.utils.make_grid(imgs)
            #     img_grid_mixup = torchvision.utils.make_grid(imgs_mixup)

            #     writer_tensorboard.add_image("images no mixup", img_grid)
            #     writer_tensorboard.add_image("images with mixup", img_grid_mixup)

            ####################
            # Training and validation metrics
            ####################
            train_metric = evaluate(
                train_eval_dataloader,
                model,
                metric=per_image_weighted_dice_metric,
                device=device,
                use_image_for_metric=True,
                channel=0,
                exp=cfg_peak_selection.DATASET.ONLY_LOG_CHANNEL,
                threshold=cfg_peak_selection.MODEL.EVALUATION.THRESHOLD,
            )

            val_metric = evaluate(
                val_dataloader,
                model,
                metric=per_image_weighted_dice_metric,
                device=device,
                use_image_for_metric=True,
                channel=0,
                exp=cfg_peak_selection.DATASET.ONLY_LOG_CHANNEL,
                threshold=cfg_peak_selection.MODEL.EVALUATION.THRESHOLD,
            )

            # Store training trace
            metric["train"].append(train_metric)
            metric["val"].append(val_metric)
            loss_tracking["train"].append(loss)

            print(
                f"EPOCH: {epoch}, TRAIN LOSS: {loss}, TRAIN Weighted DICE and AUCROC: {train_metric},"
                f" VAL Weighted DICE and AUCROC: {val_metric}"
            )
            writer.add_scalar("Loss/seg/train", loss, epoch)
            writer.add_scalar("Metric/seg/train", train_metric, epoch)
            # writer.add_scalar("Metric/cls/train", train_metric, epoch)
            writer.add_scalar("Metric/seg/val", val_metric, epoch)
            # writer.add_scalar("Metric/cls/val", val_metric, epoch)
            writer.add_scalar("LR/seg", optimizer.param_groups[0]["lr"], epoch)

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
                    f"bst_seg_model_{np.round(val_metric,4)}.pt",
                ),
                scheduler=scheduler,
            )
            best_seg_model_path = os.path.join(
                backup_dir,
                f"bst_seg_model_{np.round(es.best_score,4)}.pt",
            )
            # # Add model to Tensorboard to inspect the details of the architecture
            # input_data = next(iter(train_dataloader))[0].float().to(device)
            # writer.add_graph(model, input_data)
            writer.close()

            if es.early_stop:
                print("\n\n -------------- EARLY STOPPING -------------- \n\n")
                break
        with open(
            os.path.join(ps_exp_results_dir, "loss.json"), "w", encoding="utf-8"
        ) as fp:
            json.dump(loss_tracking, fp)
        with open(
            os.path.join(ps_exp_results_dir, "metric.json"), "w", encoding="utf-8"
        ) as fp:
            json.dump(metric, fp)
        cfg_peak_selection.MODEL.RESUME_PATH = best_seg_model_path
        cfg_peak_selection.MODEL.KEEP_TRAINING = False
        # Save configs
        cfg_peak_selection.dump(
            stream=open(
                os.path.join(ps_exp_dir, "updated_peak_selection_config.yaml"),
                "w",
                encoding="utf-8",
            )
        )
    else:
        assert (
            cfg_peak_selection.MODEL.RESUME_PATH != ""
        ), "No previous seg model exists"
        best_seg_model_path = cfg_peak_selection.MODEL.RESUME_PATH
        Logger.info("Using previous model for segmentation")

    #############################
    # Classification training
    #############################
    if cfg_peak_selection.CLSMODEL.KEEP_TRAINING:
        # Build model using config dict node
        model = build_model(cfg_peak_selection.CLSMODEL)
        model.to(device)
        # Epochs
        total_epochs = cfg_peak_selection.CLSMODEL.SOLVER.TOTAL_EPOCHS

        # Optimizer, scheduler, amp
        optimizer = build_optimizer(model, cfg_peak_selection.CLSMODEL.SOLVER.OPTIMIZER)
        scheduler_type = cfg_peak_selection.CLSMODEL.SOLVER.SCHEDULER.NAME
        scheduler = build_scheduler(
            optimizer,
            cfg_peak_selection.CLSMODEL.SOLVER.SCHEDULER,
            steps_per_epoch=int(len(train_dataloader)),
            epochs=total_epochs,
        )
        criterion = nn.BCEWithLogitsLoss()
        es = build_early_stopper(cfg_peak_selection.CLSMODEL.SOLVER.EARLY_STOPPING)

        current_epoch = 0
        if cfg_peak_selection.CLSMODEL.RESUME_PATH != "":
            checkpoint = torch.load(
                cfg_peak_selection.CLSMODEL.RESUME_PATH, map_location=device
            )
            current_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["model_state_dict"])
            logging.info(
                "Model loaded from %s", cfg_peak_selection.CLSMODEL.RESUME_PATH
            )
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                logging.info(
                    "Optimizer loaded from %s", cfg_peak_selection.CLSMODEL.RESUME_PATH
                )
            if "scheduler_state_dict" in checkpoint and scheduler is not None:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                logging.info(
                    "Scheduler loaded from %s", cfg_peak_selection.CLSMODEL.RESUME_PATH
                )
        ##########################################
        # Main training epoch loop starts here
        ##########################################

        # s_time = time.time()
        # parameters = list(model.parameters())
        loss_tracking = {"train": [], "val": []}
        metric = {"train": [], "val": []}
        for epoch in range(current_epoch, total_epochs):
            torch.cuda.empty_cache()
            gc.collect()
            logging.info("Start epoch %s", epoch)

            loss = train_one_epoch(
                train_loader=train_dataloader,
                model=model,
                optimizer=optimizer,
                loss_fn=criterion,
                model_type="cls",
                # seg_cls_loss_weight=cfg_peak_selection.CLSMODEL.SOLVER.LOSS.SEG_CLS_WEIGHTS,
                use_image_as_input=True,
                device=device,
                scheduler=scheduler,
            )

            ####################
            # Training and validation metrics
            ####################
            train_metric = evaluate(
                train_eval_dataloader,
                model,
                metric=per_image_weighted_dice_metric,
                device=device,
                use_image_for_metric=True,
                channel=0,
                exp=cfg_peak_selection.DATASET.ONLY_LOG_CHANNEL,
                model_type="cls",
                threshold=cfg_peak_selection.CLSMODEL.EVALUATION.THRESHOLD,
            )

            val_metric = evaluate(
                val_dataloader,
                model,
                metric=per_image_weighted_dice_metric,
                device=device,
                use_image_for_metric=True,
                channel=0,
                exp=cfg_peak_selection.DATASET.ONLY_LOG_CHANNEL,
                model_type="cls",
                threshold=cfg_peak_selection.CLSMODEL.EVALUATION.THRESHOLD,
            )

            # Store training trace
            metric["train"].append(train_metric)
            metric["val"].append(val_metric)
            loss_tracking["train"].append(loss)

            print(
                f"EPOCH: {epoch}, TRAIN LOSS: {loss}, TRAIN Weighted DICE and AUCROC: {train_metric},"
                f" VAL Weighted DICE and AUCROC: {val_metric}"
            )
            writer.add_scalar("Loss/cls/train", loss, epoch)
            # writer.add_scalar("Metric/seg/train", train_metric[0], epoch)
            writer.add_scalar("Metric/cls/train", train_metric, epoch)
            # writer.add_scalar("Metric/seg/val", val_metric[0], epoch)
            writer.add_scalar("Metric/cls/val", val_metric, epoch)
            writer.add_scalar("LR/cls", optimizer.param_groups[0]["lr"], epoch)

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
                    f"bst_cls_model_{np.round(val_metric, 4)}.pt",
                ),
                scheduler=scheduler,
            )
            best_cls_model_path = os.path.join(
                backup_dir,
                f"bst_cls_model_{np.round(es.best_score,4)}.pt",
            )
            # # Add model to Tensorboard to inspect the details of the architecture
            # input_data = next(iter(train_dataloader))[0].float().to(device)
            # writer.add_graph(model, input_data)
            writer.close()

            if es.early_stop:
                print("\n\n -------------- EARLY STOPPING -------------- \n\n")
                break
        with open(
            os.path.join(ps_exp_results_dir, "loss.json"), "w", encoding="utf-8"
        ) as fp:
            json.dump(loss_tracking, fp)
        with open(
            os.path.join(ps_exp_results_dir, "metric.json"), "w", encoding="utf-8"
        ) as fp:
            json.dump(metric, fp)
        cfg_peak_selection.CLSMODEL.RESUME_PATH = best_cls_model_path
        cfg_peak_selection.CLSMODEL.KEEP_TRAINING = False
        # Save configs
        cfg_peak_selection.dump(
            stream=open(
                os.path.join(ps_exp_dir, "updated_peak_selection_config.yaml"),
                "w",
                encoding="utf-8",
            )
        )

    else:
        assert (
            cfg_peak_selection.CLSMODEL.RESUME_PATH != ""
        ), "No previous cls model exists"
        best_cls_model_path = cfg_peak_selection.CLSMODEL.RESUME_PATH
        Logger.info("Using previous model for classification")
    if cfg_peak_selection.EVAL_ON_TEST:
        testset_eval(
            best_seg_model_path=best_seg_model_path,
            best_cls_model_path=best_cls_model_path,
            cfg_seg_model=cfg_peak_selection.MODEL,
            cfg_cls_model=cfg_peak_selection.CLSMODEL,
            test_dataset=test_dataset,
            test_dataloader=test_dataloader,
            maxquant_result_ref=maxquant_dict,
            result_dir=ps_exp_results_dir,
            device=device,
            exp=cfg_peak_selection.DATASET.ONLY_LOG_CHANNEL,
            # threshold=cfg_peak_selection.CLSMODEL.EVALUATION.THRESHOLD,
        )
        test_pred_df = pd.read_csv(os.path.join(ps_exp_results_dir, "test_pred_df.csv"))
        test_pred_df_filtered = test_pred_df.loc[
            (test_pred_df["target_decoy_score"] > 0.33)
            & (test_pred_df["log_sum_intensity"] > 1)
            & (test_pred_df["Decoy"] == 0),
        ]
        sbs_ims_result = SBSResult(
            maxquant_ref_df=maxquant_dict,
            maxquant_merge_df=maxquant_dict,
            pept_act_sum_df_list=[test_pred_df_filtered],
            # sum_raw=test_pred_df,
            # sum_gaussian=train_label_df,
            ims=True,
            # other_cols=other_cols
        )
        sbs_ims_result.plot_intensity_corr(
            ref_col="Intensity",
            inf_col="sum_intensity",
            contour=False,
            save_dir=ps_exp_results_dir,
            # group_by="Leading razor protein",
        )
    # if cfg_peak_selection.REMOVE_CONFIG_AFTER_RUN:
    #     os.remove(cfg_peak_selection.CONFIG_PATH)
    #     logging.info("Training finished, config file removed.")
    return best_seg_model_path, best_cls_model_path


def testset_eval(
    best_seg_model_path,
    best_cls_model_path,
    cfg_seg_model,
    cfg_cls_model,
    test_dataset,
    test_dataloader,
    maxquant_result_ref,
    result_dir,
    device,
    exp: bool = False,
    # threshold: float = 0.5,
):
    # Plot history

    bst_seg_model = build_model(cfg_seg_model)
    checkpoint = torch.load(best_seg_model_path, map_location=device)
    Logger.info("best_seg_model_path: %s", best_seg_model_path)
    bst_seg_model.load_state_dict(checkpoint["model_state_dict"])

    bst_cls_model = build_model(cfg_cls_model)
    checkpoint = torch.load(best_cls_model_path, map_location=device)
    bst_cls_model.load_state_dict(checkpoint["model_state_dict"])
    test_pred_df = inference_and_sum_intensity(
        data_loader=test_dataloader,
        seg_model=bst_seg_model,
        cls_model=bst_cls_model,
        device=device,
        per_image_metric=[
            per_image_weighted_dice_metric,
            per_image_weighted_iou_metric,
        ],
        use_image_for_metric=[True, True],
        channel=None,
        exp=exp,
        threshold=cfg_seg_model.EVALUATION.THRESHOLD,
        calc_score=True,
    )

    test_pred_df_full = pd.merge(
        left=test_pred_df,
        right=maxquant_result_ref[["mz_rank", "Decoy"]],
        on="mz_rank",
        how="left",
    )
    test_pred_df_full["log_sum_intensity"] = np.log10(
        test_pred_df_full["sum_intensity"] + 1
    )

    test_pred_df_full.to_csv(os.path.join(result_dir, "test_pred_df.csv"), index=False)

    # Plot metric distribution
    plot_per_image_metric_distr(
        test_pred_df_full.loc[
            ~test_pred_df_full["Decoy"], "per_image_weighted_dice_metric"
        ],
        "Target_weighted_dice",
        save_dir=result_dir,
    )

    plot_per_image_metric_distr(
        test_pred_df_full.loc[
            test_pred_df_full["Decoy"], "per_image_weighted_dice_metric"
        ],
        "Decoy_weighted_dice",
        save_dir=result_dir,
    )
    plot_per_image_metric_distr(
        test_pred_df_full.loc[
            ~test_pred_df_full["Decoy"], "per_image_weighted_iou_metric"
        ],
        "Target_weighted_iou",
        save_dir=result_dir,
    )

    plot_per_image_metric_distr(
        test_pred_df_full.loc[
            test_pred_df_full["Decoy"], "per_image_weighted_iou_metric"
        ],
        "Decoy_weighted_iou",
        save_dir=result_dir,
    )

    # FDR eval
    plot_target_decoy_distr(
        test_pred_df_full,
        save_dir=result_dir,
        dataset_name="testset",
        main_plot_type="scatter",
        threshold=None,  # TODO: make this a parameter, or generate fdr as a func of threshold
    )
    plot_roc_auc(
        test_pred_df_full,
        save_dir=result_dir,
        dataset_name="testset",
    )
    pred_df_new = calc_fdr_and_thres(
        test_pred_df_full,
        score_col="target_decoy_score",
        filter_dict={"log_sum_intensity": [1, 100]},  # use log int 1 as threshold
        return_plot=True,
        save_dir=result_dir,
        dataset_name="testset",
    )
    pred_df_new.to_csv(
        os.path.join(result_dir, "test_pred_df_fdr_thres.csv"), index=False
    )

    # Plot sample predictions
    plot_sample_predictions(
        test_dataset,
        seg_model=bst_seg_model,
        cls_model=bst_cls_model,
        n=10,
        metric_list=["mask_wiou", "wdice", "dice"],
        use_hint=False,
        zoom_in=False,
        label="mask",
        save_dir=os.path.join(result_dir, "sample_predictions"),
        exp=exp,
    )
    # get indices of the top 10 worst performing images
    worst_performing_images = (
        test_pred_df_full["per_image_weighted_iou_metric"]
        .sort_values(ascending=True)
        .index[:10]
    )

    # Plot sample predictions
    plot_sample_predictions(
        test_dataset,
        seg_model=bst_seg_model,
        cls_model=bst_cls_model,
        sample_indices=worst_performing_images,
        metric_list=["mask_wiou", "wdice", "dice"],
        use_hint=False,
        zoom_in=False,
        label="mask",
        device=device,
        save_dir=os.path.join(result_dir, "sample_predictions_lowest_wiou"),
        exp=exp,
    )
    return None


# if __name__ == "__main__":
#     logging.basicConfig(
#         format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
#         level=logging.INFO,
#     )
#     try:
#         arguments = docopt(
#             __doc__, argv=None, help=False, version=None, options_first=False
#         )
#         print("Arguments parsed:")
#         print(arguments)
#         output_folder_name = arguments["--path_output"]
#         cfg_path = arguments["--path_cfg"]
#         cfg = get_cfg_defaults(peak_detection_cfg)
#         if cfg_path is not None:
#             cfg.merge_from_file(cfg_path)
#             print(f"merge with cfg file {cfg_path}")
#         cfg.OUTPUT_FOLDER_NAME = output_folder_name
#         cfg.CONFIG_PATH = cfg_path
#         train(cfg)
#     except Exception as e:
#         print(f"Error: {e}")
#         print(__doc__)
#         raise
