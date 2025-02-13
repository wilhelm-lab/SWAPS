import alphatims.bruker
import logging
import os
import pickle
import numpy as np
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

from utils.config import get_cfg_defaults

from precursor_sampling.explore_precursors import get_precursor_intensity_by_category
from precursor_sampling.cls_model.dataset import prepare_precursor_dataset
from precursor_sampling.cls_model.model import predict
from optimization.inference import generate_id_partitions

from precursor_sampling.singleton_train_model import train_cfg
from datetime import datetime
from precursor_sampling.cls_model.dataset import PrecursorDataset
import torch
from torch.utils.data import DataLoader
from peak_detection_2d.model.build_model import build_model
from torch.utils.tensorboard import SummaryWriter
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
    precursor_batches = generate_id_partitions(
        id_array=precursors_with_label["Id"].values, n_batch=n_batch, how="round_robin"
    )
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
        train_dataset = PrecursorDataset(
            train_data,
            normalize=True,
            max_len=cfg_model.DATASET.MAX_LEN,
            sampling_strategy=cfg_model.DATASET.SAMPLING_STRATEGY,
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=cfg_model.DATASET.TRAINING_BATCH_SIZE,
            shuffle=True,
        )

        val_dataset = PrecursorDataset(
            val_data,
            normalize=True,
            max_len=cfg_model.DATASET.MAX_LEN,
            sampling_strategy=cfg_model.DATASET.SAMPLING_STRATEGY,
        )
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
                labels_one_hot = torch.stack(
                    [(1 - labels), labels], dim=1
                ).float()  # Convert 0 to [1, 0] and 1 to [0, 1]
                optimizer.zero_grad()
                # Logger.debug("Features shape: %s", features.shape)
                outputs = model(features)  # Output shape: (batch_size, 2)
                # Logger.debug("Output shape: %s", outputs.shape)
                loss = criterion(outputs, labels_one_hot)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()
                probs = outputs[:, 1].float().cpu().detach().numpy()
                preds = torch.argmax(outputs, dim=1).float().cpu().numpy()
                labels_np = labels.cpu().numpy()

                all_train_labels.extend(labels_np)
                all_train_preds.extend(preds)
                all_train_probs.extend(probs)

            train_loss = total_train_loss / len(train_dataloader)
            train_auc = roc_auc_score(all_train_labels, all_train_probs)

            # Validation Loop (if validation data is provided)
            val_loss, val_auc = None, None
            model.eval()
            total_val_loss = 0
            all_val_labels = []
            all_val_preds = []
            all_val_probs = []
            with torch.no_grad():
                for features, labels in val_dataloader:
                    features, labels = features.to(device), labels.to(device)
                    labels_one_hot = torch.stack(
                        [(1 - labels), labels], dim=1
                    ).float()  # Convert 0 to [1, 0] and 1 to [0, 1]
                    outputs = model(features)  # Output shape: (batch_size, 2)
                    loss = criterion(outputs, labels_one_hot)

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

            # log epoch results
            logging.info(
                f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Train AUC: {train_auc: .4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} | Val F1: {val_f1:.4f} | Val AP: {val_ap:.4f}"
            )
            writer.add_scalar("Loss/val/", val_loss, epoch)
            writer.add_scalar("Metric/val/auc", val_auc, epoch)
            writer.add_scalar("Metric/val/f1", val_f1, epoch)
            writer.add_scalar("Metric/val/ap", val_ap, epoch)
            writer.add_scalar("Loss/train", loss, epoch)
            writer.add_scalar("Metric/train/auc", train_auc, epoch)
            writer.add_scalar("LR/", optimizer.param_groups[0]["lr"], epoch)

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
        test_dataset = PrecursorDataset(
            test_data, normalize=True, max_len=cfg_model.DATASET.MAX_LEN
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=cfg_model.DATASET.TEST_BATCH_SIZE, shuffle=False
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
            tn, fp, fn, tp = confusion_matrix(y_true=labels, y_pred=preds).ravel()
            f.write(f"\nConfusion Matrix:\nTN = {tn}, FP = {fp}, FN = {fn}, TP = {tp}")


def main(config_path):
    cfg = get_cfg_defaults(train_cfg)
    # name_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if config_path is not None:
        cfg.merge_from_file(config_path)
        logging.info("merge with cfg file %s", config_path)
    if cfg.PREPARE_DATASET.ENABLE:
        prepare_data(cfg.PREPARE_DATASET)
    train_and_test_model(cfg.MODEL)


if __name__ == "__main__":
    fire.Fire(main)
