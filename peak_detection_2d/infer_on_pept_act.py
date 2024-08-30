import os
import pandas as pd
import sparse
import torch
from .model.build_model import build_model
from .dataset.dataset import PeptActPeakSelection_Infer_Dataset
from .model.seg_model import inference_and_sum_intensity
from .dataset.dataset import build_transformation


def infer_on_pept_act(
    cfg, best_model_path: str, maxquant_dict: pd.DataFrame, ps_exp_dir: str
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hint_matrix = sparse.load_npz(
        os.path.join(
            cfg.RESULT_PATH, "peak_selection", "training_data", "hint_matrix.npz"
        )
    )
    transformation, _ = build_transformation(cfg.PEAK_SELECTION.DATASET)
    model = build_model(cfg.PEAK_SELECTION.MODEL)
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    pept_act_sum_ps_df_list = []
    use_hint_channel = "hint" in cfg.PEAK_SELECTION.DATASET.INPUT_CHANNELS
    for i in range(cfg.OPTIMIZATION.N_BLOCKS_BY_PEPT):
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
        )
        infer_dataloader = torch.utils.data.DataLoader(
            infer_dataset,
            batch_size=cfg.PEAK_SELECTION.DATASET.INFER_BATCH_SIZE,
            shuffle=False,
        )
        pept_act_sum_ps_df = inference_and_sum_intensity(
            seg_model=model, data_loader=infer_dataloader, device=device
        )
        pept_act_sum_ps_df_list.append(pept_act_sum_ps_df)
    pept_act_sum_ps_df = pd.concat(pept_act_sum_ps_df_list, axis=0)
    pept_act_sum_ps_df.to_csv(
        os.path.join(ps_exp_dir, "pept_act_sum_ps.csv"), index=False
    )
