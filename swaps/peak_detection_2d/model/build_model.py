from yacs.config import CfgNode
from .seg_model import UNET, UNet_rec_input
from .conf_model import CNNEncoderRegressor, CNNRegression, ConfidenceModel
from .model import PeakDetectionNet
from precursor_sampling.cls_model.model import CNN1DModel, TCN, ResNet1D


def build_model(model_cfg: CfgNode):
    if model_cfg.TYPE == "mask_segmentation":
        if model_cfg.NAME == "UNET":
            model = UNet_rec_input(
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
            model = UNet_rec_input(
                in_channels=model_cfg.PARAMS.IN_CHANNELS,
                first_out_channels=model_cfg.PARAMS.FIRST_OUT_CHANNELS,
                exit_channels=model_cfg.PARAMS.EXIT_CHANNELS,
                downhill=model_cfg.PARAMS.DOWNHILL,
                padding=model_cfg.PARAMS.PADDING,
                seg_head=False,
                cls_head=True,
                drop_out=model_cfg.PARAMS.DROPOUT_RATE,
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
    elif model_cfg.TYPE == "precursor_classification":
        if model_cfg.NAME == "CNN":
            model = CNN1DModel()
            return model
        elif model_cfg.NAME == "TCN":
            model = TCN(num_classes=2)
            return model
        elif model_cfg.NAME == "ResNet1D":
            model = ResNet1D(num_classes=2)
            return model
