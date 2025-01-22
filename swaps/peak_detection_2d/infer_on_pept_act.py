import os
import logging
import pandas as pd
import sparse
import torch

from peak_detection_2d.utils import plot_sample_predictions
from .model.build_model import build_model
from .dataset.dataset import PeptActPeakSelection_Infer_Dataset
from .model.seg_model import inference_and_sum_intensity
from .dataset.dataset import build_transformation

Logger = logging.getLogger(__name__)


def infer_on_pept_act(
    cfg,
    best_seg_model_path: str,
    best_cls_model_path: str,
    maxquant_dict: pd.DataFrame,
    ps_exp_dir: str,
    plot_samples: bool = False,
    add_label_mask: bool = False,
    dataset_name: str = "test",
    sigmoid_cls_score: bool = True,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hint_matrix = sparse.load_npz(
        os.path.join(
            cfg.RESULT_PATH, "peak_selection", "training_data", "hint_matrix.npz"
        )
    )
    transformation, _ = build_transformation(cfg.PEAK_SELECTION.DATASET)

    # Load models
    bst_seg_model = build_model(cfg.PEAK_SELECTION.MODEL)
    checkpoint = torch.load(best_seg_model_path, map_location=device)
    Logger.info("best_seg_model_path: %s", best_seg_model_path)
    bst_seg_model.load_state_dict(checkpoint["model_state_dict"])

    bst_cls_model = build_model(cfg.PEAK_SELECTION.CLSMODEL)
    checkpoint = torch.load(best_cls_model_path, map_location=device)
    Logger.info("best_cls_model_path: %s", best_cls_model_path)
    bst_cls_model.load_state_dict(checkpoint["model_state_dict"])

    pept_act_sum_ps_df_list = []
    use_hint_channel = "hint" in cfg.PEAK_SELECTION.DATASET.INPUT_CHANNELS
    for i in maxquant_dict["pept_batch_idx"].unique():
        Logger.info("Infering on pept batch %d ...", i)
        act_3d = sparse.load_npz(
            os.path.join(
                cfg.RESULT_PATH,
                "results",
                "activation",
                f"im_rt_pept_act_coo_peptbatch{i}.npz",
            )
        )
        infer_dataset = PeptActPeakSelection_Infer_Dataset(
            pept_act_coo_peptbatch=act_3d,
            maxquant_dict=maxquant_dict.loc[maxquant_dict["pept_batch_idx"] == i],
            hint_matrix=hint_matrix,
            transforms=transformation,
            use_hint_channel=use_hint_channel,
            data_index=None,
            add_label_mask=add_label_mask,
        )
        infer_dataloader = torch.utils.data.DataLoader(
            infer_dataset,
            batch_size=cfg.PEAK_SELECTION.DATASET.INFER_BATCH_SIZE,
            shuffle=False,
        )

        if plot_samples:
            plot_dir = os.path.join(ps_exp_dir, "inferred_samples_" + dataset_name)
            os.makedirs(plot_dir, exist_ok=True)
            plot_sample_predictions(
                infer_dataset,
                save_dir=plot_dir,
                seg_model=bst_seg_model,
                cls_model=bst_cls_model,
                device=device,
                n=maxquant_dict.loc[maxquant_dict["pept_batch_idx"] == i].shape[0],
                metric_list=["mask_wiou"],
                use_hint=False,
                zoom_in=False,
                label="mask",
                add_ps_channel=cfg.PEAK_SELECTION.CLSMODEL.PARAMS.USE_SEG_OUTPUT,
            )
        else:
            pept_act_sum_ps_df = inference_and_sum_intensity(
                seg_model=bst_seg_model,
                cls_model=bst_cls_model,
                data_loader=infer_dataloader,
                device=device,
                sigmoid_cls_score=sigmoid_cls_score,
                add_ps_channel=cfg.PEAK_SELECTION.CLSMODEL.PARAMS.USE_SEG_OUTPUT,
            )
            pept_act_sum_ps_df_list.append(pept_act_sum_ps_df)
    if not plot_samples:
        pept_act_sum_ps_df = pd.concat(pept_act_sum_ps_df_list, axis=0)
        pept_act_sum_ps_df.to_csv(
            os.path.join(ps_exp_dir, "pept_act_sum_ps.csv"), index=False
        )
