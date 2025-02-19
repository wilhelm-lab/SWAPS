import alphatims.bruker
import logging
import os
import pickle
import numpy as np
import pandas as pd
import fire
from joblib import Parallel, delayed
from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    f1_score,
    average_precision_score,
    auc,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.config import get_cfg_defaults
from optimization.inference import generate_id_partitions
from precursor_sampling.cls_model.dataset import (
    prepare_precursor_dataset,
    get_precursor_intensity_by_category,
    PrecursorDataset,
    calculate_precursor_averagine_fit,
    eval_precursor_fit,
)
from precursor_sampling.cls_model.model import predict, reconstruct
from precursor_sampling.utils.singleton_train_model import train_cfg
from peak_detection_2d.model.build_model import build_model
from peak_detection_2d.solver.build_optimizer import (
    build_early_stopper,
    build_optimizer,
    build_scheduler,
)
from peak_detection_2d.loss.build_criterion import build_criterion

# Clear existing logging handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
# Reconfigure logging
logging.basicConfig(
    level=logging.INFO,  # Set the desired logging level
    format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

cfg = get_cfg_defaults(train_cfg)
name_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def prepare_data(cfg_prepare_data):

    tims_data_path = cfg_prepare_data.INPUT.TIMS_DATA_PATH
    ok_output_dir = cfg_prepare_data.INPUT.OK_OUTPUT_DIR
    mq_001_dir = cfg_prepare_data.INPUT.MQ_001_DIR
    mq_100_dir = cfg_prepare_data.INPUT.MQ_100_DIR
    n_digits = cfg_prepare_data.N_DIGITS
    n_batch = cfg_prepare_data.N_BATCH
    out_file_dir = cfg_prepare_data.OUT_FILE_DIR
    os.makedirs(out_file_dir, exist_ok=True)
    filter_by_raw_file = os.path.basename(os.path.normpath(tims_data_path)).split(".d")[
        0
    ]
    cfg_prepare_data.dump(
        stream=open(
            os.path.join(out_file_dir, f"config_prepare_data.yaml"),
            "w",
            encoding="utf-8",
        )
    )
    data = alphatims.bruker.TimsTOF(tims_data_path)
    precursors = data.precursors

    precursors_with_label = get_precursor_intensity_by_category(
        mq_001_dir=mq_001_dir,
        precursors_df=precursors,
        filter_by_raw_file=filter_by_raw_file,
        mq_100_dir=mq_100_dir,
        ok_output_dir=ok_output_dir,
        plot_type=None,
    )
    logging.info(
        "Number of precursors with label: %s, identified %s, non-identified %s",
        len(precursors_with_label),
        precursors_with_label["Identified"].sum(),
        (~precursors_with_label["Identified"]).sum(),
    )
    if cfg_prepare_data.CALCULATE_PRECURSOR_METRICS:
        precursors_with_label.dropna(inplace=True)
    precursor_batches = generate_id_partitions(
        id_array=precursors_with_label["Id"].values, n_batch=n_batch, how="round_robin"
    )
    if cfg_prepare_data.CALCULATE_PRECURSOR_METRICS:
        Parallel(n_jobs=n_batch)(
            delayed(calculate_precursor_averagine_fit)(
                precursor_df=precursors_with_label.loc[
                    precursors_with_label["Id"].isin(batch)
                ],
                data=data,
                n_digits=3,
                save_path=os.path.join(out_file_dir, f"precursor_fit_{idx}.csv"),
            )
            for idx, batch in enumerate(precursor_batches)
        )
        file_paths = [
            os.path.join(out_file_dir, f"precursor_fit_{idx}.csv")
            for idx in range(n_batch)
        ]
        logging.info(
            "Precursor metrics calculation completed. Generated %s files: %s. Start evaluating the results.",
            n_batch,
            file_paths,
        )
        all_results = pd.DataFrame()
        for file in file_paths:
            if file.endswith(".csv"):
                all_results = pd.concat([all_results, pd.read_csv(file)])
        _ = eval_precursor_fit(
            precursor_df=precursors_with_label,
            fit_df=all_results,
            save_path=out_file_dir,
        )
    if cfg_prepare_data.PREPARE_TORCH_DATASET:
        Parallel(n_jobs=n_batch)(
            delayed(prepare_precursor_dataset)(
                precursor_df=precursors_with_label.loc[
                    precursors_with_label["Id"].isin(batch)
                ],
                data=data,
                n_digits=n_digits,
                save_path=os.path.join(
                    out_file_dir, f"data_dict_{filter_by_raw_file}_{idx}.pkl"
                ),
            )
            for idx, batch in enumerate(precursor_batches)
        )
        file_paths = [
            os.path.join(out_file_dir, f"data_dict_{filter_by_raw_file}_{idx}.pkl")
            for idx in range(n_batch)
        ]
        logging.info(
            "Data preparation completed. Generated %s files: %s", n_batch, file_paths
        )
    return file_paths


def train_and_test_model(cfg_model):
    # Create experiment directory
    name_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if cfg_model.ADD_TIMESTAMP_TO_RESULT_PATH:
        cfg_model.EXP_DIR = cfg_model.EXP_DIR + "_" + name_timestamp
        cfg_model.ADD_TIMESTAMP_TO_RESULT_PATH = (
            False  # in case of reuse of config file
        )
    if not os.path.exists(cfg_model.EXP_DIR):
        os.mkdir(cfg_model.EXP_DIR)
    backup_dir = os.path.join(cfg_model.EXP_DIR, "model_backups")
    if not os.path.exists(backup_dir):
        os.mkdir(backup_dir)
    ps_exp_results_dir = os.path.join(cfg_model.EXP_DIR, "results")
    if not os.path.exists(ps_exp_results_dir):
        os.mkdir(ps_exp_results_dir)
    cfg_model.dump(
        stream=open(
            os.path.join(
                cfg_model.EXP_DIR, f"config_train_model_{name_timestamp}.yaml"
            ),
            "w",
            encoding="utf-8",
        )
    )
    # Initialize Tensorboard
    writer = SummaryWriter(log_dir=os.path.join(cfg_model.EXP_DIR, "logs_tensorflow"))
    # Read test data
    test_data = {}
    for data in cfg_model.DATASET.TEST_DATA:
        with open(data, "rb") as f:
            test_data.update(pickle.load(f))
    # Set Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize and Train Model
    model = build_model(cfg_model)
    model.to(device)
    # Epochs, Optimizer, scheduler, criterion, early stopper

    if cfg_model.RESUME_PATH != "":
        logging.info("Loading model from %s", cfg_model.RESUME_PATH)
        checkpoint = torch.load(cfg_model.RESUME_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        logging.info("Model loaded from %s", cfg_model.RESUME_PATH)

    ##########################################
    # Main training epoch loop starts here
    ##########################################
    if cfg_model.KEEP_TRAINING:
        # Prepare trainging and validation Dataset and DataLoader
        train_data = {}
        val_data = {}
        for data in cfg_model.DATASET.TRAINING_DATA:
            with open(data, "rb") as f:
                train_data.update(pickle.load(f))
        for data in cfg_model.DATASET.VALIDATION_DATA:
            with open(data, "rb") as f:
                val_data.update(pickle.load(f))
        filter_class = 1 if cfg_model.TYPE == "precursor_reconstruction" else None
        train_dataset = PrecursorDataset(
            train_data,
            normalize=True,
            max_len=cfg_model.DATASET.MAX_LEN,
            sampling_strategy=cfg_model.DATASET.SAMPLING_STRATEGY,
            filter_class=filter_class,
            scale_factor=cfg_model.DATASET.SCALE_FACTOR,
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=cfg_model.DATASET.TRAINING_BATCH_SIZE,
            shuffle=True,
        )
        match cfg_model.TYPE:
            case "precursor_classification":
                val_dataset = PrecursorDataset(
                    val_data,
                    normalize=True,
                    max_len=cfg_model.DATASET.MAX_LEN,
                    sampling_strategy=cfg_model.DATASET.SAMPLING_STRATEGY,
                )

            case "precursor_reconstruction":
                val_dataset = PrecursorDataset(
                    val_data,
                    normalize=True,
                    max_len=cfg_model.DATASET.MAX_LEN,
                    sampling_strategy=None,
                    filter_class=1,
                    scale_factor=cfg_model.DATASET.SCALE_FACTOR,
                )
                val_neg_dataset = PrecursorDataset(
                    val_data,
                    normalize=True,
                    max_len=cfg_model.DATASET.MAX_LEN,
                    sampling_strategy=None,
                    filter_class=0,
                    scale_factor=cfg_model.DATASET.SCALE_FACTOR,
                )
                val_neg_dataloader = DataLoader(
                    val_neg_dataset,
                    batch_size=cfg_model.DATASET.VALIDATION_BATCH_SIZE,
                    shuffle=True,
                )
            case _:
                raise ValueError(f"Invalid model type: {cfg_model.TYPE}")
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=cfg_model.DATASET.VALIDATION_BATCH_SIZE,
            shuffle=True,
        )

        # Load optimizer, scheduler, criterion, early stopper
        total_epochs = cfg_model.SOLVER.TOTAL_EPOCHS
        current_epoch = 0
        optimizer = build_optimizer(model, cfg_model.SOLVER.OPTIMIZER)
        scheduler_type = cfg_model.SOLVER.SCHEDULER.NAME
        scheduler = build_scheduler(
            optimizer,
            cfg_model.SOLVER.SCHEDULER,
            steps_per_epoch=int(len(train_dataloader)),
            epochs=total_epochs,
        )
        criterion = build_criterion(cfg_model.SOLVER.LOSS, device=device)
        es = build_early_stopper(cfg_model.SOLVER.EARLY_STOPPING)
        if cfg_model.RESUME_PATH != "":
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                logging.info("Optimizer loaded from %s", cfg_model.RESUME_PATH)
            if "scheduler_state_dict" in checkpoint and scheduler is not None:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                logging.info("Scheduler loaded from %s", cfg_model.RESUME_PATH)
            current_epoch = checkpoint["epoch"]
        for epoch in range(current_epoch, total_epochs):
            logging.info("Start training epoch %s", epoch)
            model.train()
            total_train_loss = 0
            all_train_labels = []
            all_train_preds = []
            all_train_probs = []
            # Training Loop
            for features, labels in train_dataloader:
                features, labels = features.to(device), labels.to(device)
                match cfg_model.TYPE:
                    case "precursor_classification":
                        labels_for_loss = torch.stack(
                            [(1 - labels), labels], dim=1
                        ).float()  # Convert 0 to [1, 0] and 1 to [0, 1]
                        optimizer.zero_grad()
                        # Logger.debug("Features shape: %s", features.shape)
                        outputs = model(features)  # Output shape: (batch_size, 2)
                        # Logger.debug("Output shape: %s", outputs.shape)
                        loss = criterion(outputs, labels_for_loss)
                        loss.backward()
                        optimizer.step()
                        # writer.add_scalar("LR/", optimizer.param_groups[0]["lr"], epoch)
                        total_train_loss += loss.item()
                        probs = outputs[:, 1].float().cpu().detach().numpy()
                        preds = torch.argmax(outputs, dim=1).float().cpu().numpy()
                        labels_np = labels.cpu().numpy()

                        all_train_labels.extend(labels_np)
                        all_train_preds.extend(preds)
                        all_train_probs.extend(probs)
                    case "precursor_reconstruction":
                        labels_for_loss = features
                        optimizer.zero_grad()
                        outputs = model(features) * cfg_model.DATASET.SCALE_FACTOR
                        loss = criterion(outputs, labels_for_loss)
                        loss.backward()
                        optimizer.step()
                        total_train_loss += loss.item()

            train_loss = total_train_loss / len(train_dataloader)
            match cfg_model.TYPE:
                case "precursor_classification":
                    train_auc = roc_auc_score(all_train_labels, all_train_probs)
                    writer.add_scalar("Metric/train/auc", train_auc, epoch)
            writer.add_scalar("Loss/train", train_loss, epoch)

            # Validation Loop (if validation data is provided)
            match cfg_model.TYPE:
                case "precursor_classification":
                    val_loss, val_auc = None, None
                    model.eval()
                    total_val_loss = 0
                    all_val_labels = []
                    all_val_preds = []
                    all_val_probs = []
                    with torch.no_grad():
                        for features, labels in val_dataloader:
                            features, labels = features.to(device), labels.to(device)

                            labels_for_loss = torch.stack(
                                [(1 - labels), labels], dim=1
                            ).float()  # Convert 0 to [1, 0] and 1 to [0, 1]
                            outputs = model(features)  # Output shape: (batch_size, 2)
                            loss = criterion(outputs, labels_for_loss)

                            total_val_loss += loss.item()
                            preds = torch.argmax(outputs, dim=1).float().cpu().numpy()
                            probs = outputs[:, 1].float().cpu().detach().numpy()
                            labels_np = labels.cpu().numpy()

                            all_val_labels.extend(labels_np)
                            all_val_preds.extend(preds)
                            all_val_probs.extend(probs)

                    val_loss = total_val_loss / len(val_dataloader)

                    val_auc = roc_auc_score(all_val_labels, all_val_probs)
                    val_f1 = f1_score(all_val_labels, all_val_preds, average="micro")
                    val_ap = average_precision_score(
                        all_val_labels, all_val_probs, average="micro"
                    )
                    writer.add_scalar("Metric/val/auc", val_auc, epoch)
                    writer.add_scalar("Metric/val/f1", val_f1, epoch)
                    writer.add_scalar("Metric/val/ap", val_ap, epoch)
                    writer.add_scalar("Loss/val/", val_loss, epoch)
                    # log epoch results
                    logging.info(
                        f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Train AUC: {train_auc: .4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} | Val F1: {val_f1:.4f} | Val AP: {val_ap:.4f}"
                    )
                case "precursor_reconstruction":
                    val_neg_loss_all = reconstruct(
                        model,
                        val_neg_dataloader,
                        device,
                        scaling_factor=1,
                    )
                    val_pos_loss_all = reconstruct(
                        model,
                        val_dataloader,
                        device,
                        scaling_factor=1,
                    )
                    val_neg_loss = np.mean(val_neg_loss_all)
                    val_pos_loss = np.mean(val_pos_loss_all)
                    val_loss_diff = val_neg_loss - val_pos_loss
                    val_loss_rel_diff = val_loss_diff / val_pos_loss
                    writer.add_scalar("Loss/val/pos", val_pos_loss, epoch)
                    writer.add_scalar("Loss/val/neg", val_neg_loss, epoch)
                    writer.add_scalar("Loss/val/diff", val_loss_diff, epoch)
                    writer.add_scalar("Loss/val/rel_diff", val_loss_rel_diff, epoch)
                    logging.info(
                        f"Epoch {epoch+1} | Val Pos Loss: {val_pos_loss:.4f} | Val Neg Loss: {val_neg_loss:.4f} | Val Loss Diff: {val_loss_diff:.4f} | Val Loss Rel Diff: {val_loss_rel_diff:.4f}"
                    )

            ######################################
            # Update early stopper and scheduler, and saving model
            ######################################
            # Update scheudler here if not 'OneCycleLR'
            # Early Stopping Logic
            match cfg_model.SOLVER.EARLY_STOPPING.MONITOR:
                case "val_loss":
                    val_metric = val_loss
                case "val_auc":
                    val_metric = val_auc
                case "val_f1":
                    val_metric = val_f1
                case "val_ap":
                    val_metric = val_ap
                case "val_loss_diff":
                    val_metric = val_loss_diff
                case "val_loss_rel_diff":
                    val_metric = val_loss_rel_diff
                case "val_pos_loss":
                    val_metric = val_pos_loss
                case _:
                    raise ValueError(
                        f"Invalid metric: {cfg_model.SOLVER.EARLY_STOPPING.MONITOR}"
                    )
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
            best_seg_model_path = os.path.join(
                backup_dir,
                f"bst_model_{np.round(es.best_score,4)}.pt",
            )
            writer.close()

            if es.early_stop:
                logging.info("\n\n -------------- EARLY STOPPING -------------- \n\n")
                break

        cfg_model.RESUME_PATH = best_seg_model_path
        cfg_model.KEEP_TRAINING = False
        cfg_model.dump(
            stream=open(
                os.path.join(
                    cfg_model.EXP_DIR, f"config_train_model_{name_timestamp}.yaml"
                ),
                "w",
                encoding="utf-8",
            )
        )

    ##########################################
    # Evaluation on test data
    ##########################################
    if len(test_data) > 0:
        match cfg_model.TYPE:
            case "precursor_classification":
                test_dataset = PrecursorDataset(
                    test_data,
                    normalize=True,
                    max_len=cfg_model.DATASET.MAX_LEN,
                    # sampling_strategy=None, # No sampling strategy for test data
                    # filter_class=filter_class,
                    scale_factor=cfg_model.DATASET.SCALE_FACTOR,
                )
                test_dataloader = DataLoader(
                    test_dataset,
                    batch_size=cfg_model.DATASET.TEST_BATCH_SIZE,
                    shuffle=False,
                )
                probs, preds = predict(model, test_dataloader, device)
                labels = np.array([test_data[k]["label"] for k in test_data.keys()])
                # Compute precision-recall curve
                # logging.debug("labels shape: %s, probs shape: %s", labels.shape, probs.shape)
                precision, recall, _ = precision_recall_curve(labels, probs)
                pr_auc = auc(recall, precision)

                # Plot PR curve
                plt.figure(figsize=(6, 6))
                plt.plot(recall, precision, marker=".", label=f"PR AUC = {pr_auc:.4f}")
                plt.xlabel("Recall")
                plt.ylabel("Precision")
                plt.title("Precision-Recall Curve")
                plt.legend()
                plt.grid()
                # Save the plot
                plt.savefig(
                    os.path.join(ps_exp_results_dir, "pr_curve.png"),
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()
                # Write classification report and confusion matrix to file

                with open(
                    os.path.join(ps_exp_results_dir, "classification_report.txt"), "w"
                ) as f:
                    f.write(classification_report(y_true=labels, y_pred=preds))
                    tn, fp, fn, tp = confusion_matrix(
                        y_true=labels, y_pred=preds
                    ).ravel()
                    f.write(
                        f"\nConfusion Matrix:\nTN = {tn}, FP = {fp}, FN = {fn}, TP = {tp}"
                    )
            case "precursor_reconstruction":
                test_pos_dataset = PrecursorDataset(
                    test_data,
                    normalize=True,
                    max_len=cfg_model.DATASET.MAX_LEN,
                    sampling_strategy=None,
                    filter_class=1,
                    scale_factor=cfg_model.DATASET.SCALE_FACTOR,
                    return_ids=True,
                )
                test_neg_dataset = PrecursorDataset(
                    test_data,
                    normalize=True,
                    max_len=cfg_model.DATASET.MAX_LEN,
                    sampling_strategy=None,
                    filter_class=0,
                    scale_factor=cfg_model.DATASET.SCALE_FACTOR,
                    return_ids=True,
                )
                test_pos_dataloader = DataLoader(
                    test_pos_dataset,
                    batch_size=cfg_model.DATASET.TEST_BATCH_SIZE,
                    shuffle=False,
                )
                test_neg_dataloader = DataLoader(
                    test_neg_dataset,
                    batch_size=cfg_model.DATASET.TEST_BATCH_SIZE,
                    shuffle=False,
                )
                error_pos_df = reconstruct(
                    model,
                    test_pos_dataloader,
                    device,
                    scaling_factor=1,
                    error_cal=cfg_model.EVAL.ERROR_CAL,
                )
                error_neg_df = reconstruct(
                    model,
                    test_neg_dataloader,
                    device,
                    scaling_factor=1,
                    error_cal=cfg_model.EVAL.ERROR_CAL,
                )

                error_pos_df["label"] = True
                error_neg_df["label"] = False
                error_df = pd.concat([error_pos_df, error_neg_df])
                error_df.to_csv(
                    os.path.join(ps_exp_results_dir, "error_df.csv"), index=True
                )
                sns.kdeplot(error_df, x="error", common_norm=True, hue="label")
                plt.xlabel("Error")
                plt.title("Error Distribution By Class")
                # Save the plot
                plt.savefig(
                    os.path.join(ps_exp_results_dir, "error_distr.png"),
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()
                with open(
                    os.path.join(ps_exp_results_dir, "classification_report.txt"), "w"
                ) as f:
                    f.write(error_df.groupby("label")["error"].describe().to_string())


def main(config_path):
    cfg = get_cfg_defaults(train_cfg)
    # name_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if config_path is not None:
        cfg.merge_from_file(config_path)
        logging.info("merge with cfg file %s", config_path)
    if cfg.PREPARE_DATASET.ENABLE:
        prepare_data(cfg.PREPARE_DATASET)
    if cfg.MODEL.KEEP_TRAINING or (len(cfg.MODEL.DATASET.TEST_DATA) > 0):
        train_and_test_model(cfg.MODEL)
    else:
        logging.info("No training or evaluation specified.")


if __name__ == "__main__":
    fire.Fire(main)
