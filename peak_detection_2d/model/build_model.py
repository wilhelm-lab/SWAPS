from yacs.config import CfgNode
from .seg_model import UNET
from .conf_model import CNNEncoderRegressor, CNNRegression, ConfidenceModel
from .model import PeakDetectionNet


def build_model(model_cfg: CfgNode):
    if model_cfg.TYPE == "mask_segmentation":
        if model_cfg.NAME == "UNET":
            model = UNET(
                in_channels=model_cfg.PARAMS.IN_CHANNELS,
                first_out_channels=model_cfg.PARAMS.FIRST_OUT_CHANNELS,
                exit_channels=model_cfg.PARAMS.EXIT_CHANNELS,
                downhill=model_cfg.PARAMS.DOWNHILL,
                padding=model_cfg.PARAMS.PADDING,
                seg_head=True,
                cls_head=False,
            )
            return model
    elif model_cfg.TYPE == "mask_classification":
        if model_cfg.NAME == "UNET":
            model = UNET(
                in_channels=model_cfg.PARAMS.IN_CHANNELS,
                first_out_channels=model_cfg.PARAMS.FIRST_OUT_CHANNELS,
                exit_channels=model_cfg.PARAMS.EXIT_CHANNELS,
                downhill=model_cfg.PARAMS.DOWNHILL,
                padding=model_cfg.PARAMS.PADDING,
                seg_head=False,
                cls_head=True,
            )
            return model

    elif model_cfg.TYPE == "box_regression":
        if model_cfg.NAME == "PeakDetectionNet":
            model = PeakDetectionNet(
                in_channels=model_cfg.PARAMS.IN_CHANNELS,
                first_output_channels=model_cfg.PARAMS.OUT_CHANNELS,
            )
            return model
    elif model_cfg.TYPE == "confidence_regression":
        if model_cfg.NAME == "CNNEncoderRegressor":
            model = CNNEncoderRegressor(
                in_channels=model_cfg.PARAMS.IN_CHANNELS,
                first_out_channels=model_cfg.PARAMS.FIRST_OUT_CHANNELS,
                image_size=model_cfg.PARAMS.IMAGE_SIZE,
                downhill=model_cfg.PARAMS.DOWNHILL,
                dropout_rate=model_cfg.PARAMS.DROPOUT_RATE,
                sigmoid_output=model_cfg.PARAMS.SIGMOID_OUTPUT,
            )
            return model
