"""Model factory, ported from `multi_output_PS_model`'s `model/build_model.py`.

Only the `mask_segmentation` -> `UNET` branch is kept. Dropped, out of scope:
`mask_classification` (UNET with a cls head instead of a decoder),
`box_regression` (`PeakDetectionNet`), and `confidence_regression`
(`CNNEncoderRegressor`) -- and their imports (`model/model.py`,
`model/conf_model.py` are not ported at all).
"""

from yacs.config import CfgNode

from .seg_model import UNET


def build_model(model_cfg: CfgNode) -> UNET:
    if model_cfg.TYPE == "mask_segmentation" and model_cfg.NAME == "UNET":
        return UNET(
            in_channels=model_cfg.PARAMS.IN_CHANNELS,
            first_out_channels=model_cfg.PARAMS.FIRST_OUT_CHANNELS,
            exit_channels=model_cfg.PARAMS.EXIT_CHANNELS,
            downhill=model_cfg.PARAMS.DOWNHILL,
            padding=model_cfg.PARAMS.PADDING,
        )
    raise ValueError(
        f"Unsupported model_cfg.TYPE/NAME combination: "
        f"{model_cfg.TYPE!r}/{model_cfg.NAME!r} -- only mask_segmentation/UNET "
        "is supported in this build."
    )
