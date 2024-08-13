import os
import pandas as pd
from typing import Literal, Union
import logging
import pickle

Logger = logging.getLogger(__name__)
from deeplc import DeepLC
from deeplcretrainer import deeplcretrainer
import tensorflow as tf

from psm_utils.io.maxquant import MSMSReader
from psm_utils.io.peptide_record import to_dataframe
from utils.metrics import RT_metrics
from utils.tools import cleanup_maxquant


def update_rt_model(
    train_maxquant_df: pd.DataFrame,
    # for_train_filepath: str,
    train_dir: Union[str, None] = None,
    train_frac: float = 0.9,
    how_update_model: Literal["calib", "transfer"] = "transfer",
    seed: Union[int, None] = None,
    # train_suffix: Union[None, str] = None,
    # train_raw_file: str = None,
    keep_matched_precursors: bool = False,
    save_model_name: Union[str, None] = None,
):
    """
    transfer learn DeepLC model and predict RT

    :for_train_filepath: file containing data used for the training, calibration or transfer learning,
                         will be further split into train, val and test
    :to_pred_filepath: file containing data used for generating prediction, and further SBS analysis,
                        if None, use train data
    :save_dir: the directory to save all files generated, if not specified, use the parent dir of train
                and pred files, respectively
    :train_frac: the fraction of data used for training
    :how_update_model: whether to calibrate or transfer learn the model
    :seed: set random seed

    """
    Logger.info("Num GPUs Available: %s ", len(tf.config.list_physical_devices("GPU")))
    if seed is not None:
        tf.random.set_seed(seed)

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    # Load and prepare training data
    (train_evidence_tranfer_file, train_evidence_file_transfer_pred) = (
        prepare_maxquant_evidence(
            evidence=train_maxquant_df,
            usage="train",
            # maxquant_evidence_filepath=for_train_filepath,
            # suffix=train_suffix,
            save_dir=train_dir,
            # filter_raw_file=train_raw_file,
            keep_matched_precursors=keep_matched_precursors,
        )
    )

    train_maxquant_peprec, train_peprec_agg = _format_maxquant_as_deeplc_input(
        train_evidence_tranfer_file
    )

    # models
    prediction_dir = os.path.dirname(os.path.realpath(__file__))
    ori_model_paths = [
        "models/full_hc_train_pxd001468_1fd8363d9af9dcad3be7553c39396960.hdf5",
        "models/full_hc_train_pxd001468_8c22d89667368f2f02ad996469ba157e.hdf5",
        "models/full_hc_train_pxd001468_cb975cfdd4105f97efa0b3afffe075cc.hdf5",
    ]
    ori_model_paths = [os.path.join(prediction_dir, dm) for dm in ori_model_paths]

    # load DDA transfer learning data and split
    deeplc_train = train_peprec_agg.sample(frac=train_frac, random_state=seed)
    deeplc_test = train_peprec_agg.loc[
        train_peprec_agg.index.difference(deeplc_train.index)
    ]
    deeplc_train.fillna("", inplace=True)
    deeplc_test.fillna("", inplace=True)

    train_deeplc_file = os.path.join(train_dir, "deeplc_train.csv")
    deeplc_train.to_csv(
        train_deeplc_file, index=False
    )  # For training new models a file is needed

    match how_update_model:
        case "transfer":
            pred_col = "trans_pred_" + str(train_frac)

            # apply transfer learning
            models = deeplcretrainer.retrain(
                [train_deeplc_file],
                mods_transfer_learning=ori_model_paths,
                freeze_layers=True,
                n_epochs=10,
                freeze_after_concat=1,
                outpath=train_dir,
            )
            Logger.info("finished transfer learning, models are %s", models)
            # Make a DeepLC object with the models trained previously
            dlc = DeepLC(path_model=models, batch_num=1024000, pygam_calibration=False)
        case "calib":
            pred_col = "cal_pred_" + str(train_frac)

            # Call DeepLC with the downloaded models with GAM calibration
            dlc = DeepLC(
                path_model=ori_model_paths, batch_num=1024000, pygam_calibration=True
            )

    # Perform calibration, make predictions and calculate metrics
    dlc.calibrate_preds(seq_df=deeplc_train)

    if save_model_name is not None:
        model_path = os.path.join(
            train_dir, save_model_name + "_RT_model_deepLC_" + how_update_model + ".pkl"
        )
        with open(model_path, "wb") as outp:
            pickle.dump(dlc, outp, pickle.HIGHEST_PROTOCOL)
    deeplc_test[pred_col] = dlc.make_preds(seq_df=deeplc_test)

    rt_metric = RT_metrics(RT_obs=deeplc_test["tr"], RT_pred=deeplc_test[pred_col])
    Logger.info("MAE: %s", rt_metric.CalcMAE())
    delta_rt_95 = rt_metric.CalcDeltaRTwidth(95)
    Logger.info("Delta RT 95 percent: %s", delta_rt_95)
    Logger.info("Pearson Corr: %s", rt_metric.CalcPrsCorr())
    return delta_rt_95, models


def dict_add_rt_pred(
    updated_models,
    deeplc_train_path: str,
    maxquant_df: pd.DataFrame,
    # to_pred_filepath: Union[str, None],  # used for prediction
    save_dir: Union[str, None] = None,
    # pred_suffix: Union[None, str] = None,
    # pred_raw_file: str = None,
    keep_matched_precursors: bool = False,
    filter_by_rt_diff: Union[Literal["closest"], float, None] = "closest",
):
    """
    transfer learn DeepLC model and predict RT

    :for_train_filepath: file containing data used for the training, calibration or transfer learning,
                         will be further split into train, val and test
    :to_pred_filepath: file containing data used for generating prediction, and further SBS analysis,
                        if None, use train data
    :save_dir: the directory to save all files generated, if not specified, use the parent dir of train
                and pred files, respectively
    :train_frac: the fraction of data used for training
    :how_update_model: whether to calibrate or transfer learn the model
    :seed: set random seed

    """
    deeplc_train = pd.read_csv(deeplc_train_path)
    dlc = DeepLC(path_model=updated_models, batch_num=1024000, pygam_calibration=False)

    # Perform calibration, make predictions and calculate metrics
    dlc.calibrate_preds(seq_df=deeplc_train)

    # Logger.info("Num GPUs Available: %s ", len(tf.config.list_physical_devices("GPU")))
    # if seed is not None:
    #     tf.random.set_seed(seed)
    # if save_dir is None:
    #     train_dir = os.path.dirname(for_train_filepath)
    # else:
    #     train_dir = os.path.join(save_dir, "RT_tranfer_learn")
    #     if not os.path.exists(train_dir):
    #         os.makedirs(train_dir)

    # # Load and prepare training data
    # (train_evidence_tranfer_file, train_evidence_file_transfer_pred) = (
    #     prepare_maxquant_evidence(
    #         maxquant_evidence_filepath=for_train_filepath,
    #         suffix=train_suffix,
    #         save_dir=train_dir,
    #         filter_raw_file=train_raw_file,
    #         keep_matched_precursors=keep_matched_precursors,
    #     )
    # )

    # train_maxquant_peprec, train_peprec_agg = _format_maxquant_as_deeplc_input(
    #     train_evidence_tranfer_file
    # )
    # updated_model = pickle.load(open(model_path, "rb"))
    # dlc = DeepLC(path_model=updated_model, batch_num=1024000, pygam_calibration=False)
    # Load and prepare prediction data
    # if to_pred_filepath is None:
    #     pred_evidence_file_transfer_pred = train_evidence_file_transfer_pred

    #     pred_maxquant_peprec, pred_peprec_modpept_agg = (
    #         train_maxquant_peprec,
    #         train_peprec_agg,
    #     )
    # else:
    (
        pred_evidence_tranfer_file,
        pred_evidence_file_transfer_pred,
    ) = prepare_maxquant_evidence(
        evidence=maxquant_df,
        # maxquant_evidence_filepath=to_pred_filepath,
        # suffix=pred_suffix,
        usage="pred",
        save_dir=save_dir,
        # filter_raw_file=pred_raw_file,  # do not filter by raw file in pred
        keep_matched_precursors=keep_matched_precursors,
    )
    Logger.info("pred_evidence_transfer_file is %s", pred_evidence_tranfer_file)
    pred_maxquant_peprec, pred_peprec_modpept_agg = _format_maxquant_as_deeplc_input(
        pred_evidence_tranfer_file
    )

    # # models
    # prediction_dir = os.path.dirname(os.path.realpath(__file__))
    # ori_model_paths = [
    #     "models/full_hc_train_pxd001468_1fd8363d9af9dcad3be7553c39396960.hdf5",
    #     "models/full_hc_train_pxd001468_8c22d89667368f2f02ad996469ba157e.hdf5",
    #     "models/full_hc_train_pxd001468_cb975cfdd4105f97efa0b3afffe075cc.hdf5",
    # ]
    # ori_model_paths = [os.path.join(prediction_dir, dm) for dm in ori_model_paths]

    # # load DDA transfer learning data and split
    # deeplc_train = train_peprec_agg.sample(frac=train_frac, random_state=seed)
    # deeplc_test = train_peprec_agg.loc[
    #     train_peprec_agg.index.difference(deeplc_train.index)
    # ]
    # deeplc_train.fillna("", inplace=True)
    # deeplc_test.fillna("", inplace=True)

    # train_deeplc_file = os.path.join(train_dir, "deeplc_train.csv")
    # deeplc_train.to_csv(
    #     train_deeplc_file, index=False
    # )  # For training new models a file is needed

    # match how_update_model:
    #     case "transfer":
    #         pred_col = "trans_pred_" + str(train_frac)

    #         # apply transfer learning
    #         models = deeplcretrainer.retrain(
    #             [train_deeplc_file],
    #             mods_transfer_learning=ori_model_paths,
    #             freeze_layers=True,
    #             n_epochs=10,
    #             freeze_after_concat=1,
    #         )

    #         # Make a DeepLC object with the models trained previously
    #         dlc = DeepLC(path_model=models, batch_num=1024000, pygam_calibration=False)
    #     case "calib":
    #         pred_col = "cal_pred_" + str(train_frac)

    #         # Call DeepLC with the downloaded models with GAM calibration
    #         dlc = DeepLC(
    #             path_model=ori_model_paths, batch_num=1024000, pygam_calibration=True
    #         )

    # Perform calibration, make predictions and calculate metrics
    # dlc.calibrate_preds(seq_df=deeplc_train)

    # if save_model_name is not None:
    #     model_name = os.path.join(
    #         train_dir, save_model_name + "_RT_model_deepLC_" + how_update_model + ".pkl"
    #     )
    #     with open(model_name, "wb") as outp:
    #         pickle.dump(dlc, outp, pickle.HIGHEST_PROTOCOL)
    # deeplc_test[pred_col] = dlc.make_preds(seq_df=deeplc_test)

    # rt_metric = RT_metrics(RT_obs=deeplc_test["tr"], RT_pred=deeplc_test[pred_col])
    # Logger.info("MAE: %s", rt_metric.CalcMAE())
    # delta_rt_95 = rt_metric.CalcDeltaRTwidth(95)
    # Logger.info("Delta RT 95 percent: %s", delta_rt_95)
    # Logger.info("Pearson Corr: %s", rt_metric.CalcPrsCorr())

    # prepare final prediction output
    pred_peprec_modpept_agg["predicted_RT"] = dlc.make_preds(
        seq_df=pred_peprec_modpept_agg
    )
    pred_filtered = match_pred_to_input(
        MQ_peprec=pred_maxquant_peprec,
        peprec_RTpred=pred_peprec_modpept_agg,
        filter_by_RT_diff=filter_by_rt_diff,
    )
    pred_filtered.to_csv(pred_evidence_file_transfer_pred, sep="\t")
    return pred_filtered


def prepare_maxquant_evidence(
    evidence: pd.DataFrame,
    # maxquant_evidence_filepath: str,
    # suffix: Union[None, str] = None,
    usage: Literal["train", "pred"],
    save_dir=None,
    filter_raw_file: str = None,
    keep_matched_precursors: bool = False,
    cleanup_evidence: bool = False,
):
    # if save_dir is None:
    #     save_dir = os.path.dirname(maxquant_evidence_filepath)
    # maxquant_file_base = os.path.basename(maxquant_evidence_filepath)[:-4]
    # if suffix is not None:
    #     maxquant_file_base += "_" + suffix
    # maxquant_file_transfer_path = os.path.join(
    #     save_dir, maxquant_file_base + "_transfer.txt"
    # )
    # maxquant_file_transfer_pred_path = os.path.join(
    #     save_dir, maxquant_file_base + "_transfer_pred.txt"
    # )

    # evidence = pd.read_csv(maxquant_evidence_filepath, sep="\t")

    # fill in missing modified sequence or drop entries
    if keep_matched_precursors:
        evidence = fill_mbr_modified_sequence(evidence)
    # else:
    #     evidence = evidence.dropna(subset=["Modified sequence", "Retention time"])

    # filter by raw file
    if filter_raw_file is not None:
        Logger.info("Filter training data by raw file %s", filter_raw_file)
        evidence = evidence[evidence["Raw file"] == filter_raw_file]

    evidence = evidence.rename(columns={"MS/MS scan number": "Scan number"})
    if cleanup_evidence:
        evidence = cleanup_maxquant(
            maxquant_df=evidence, remove_decoys=True, how_duplicates="keep_highest_int"
        )
    maxquant_file_transfer_path = os.path.join(save_dir, usage + "_transfer.txt")
    maxquant_file_transfer_pred_path = os.path.join(
        save_dir, usage + "_transfer_pred.txt"
    )

    evidence.to_csv(maxquant_file_transfer_path, index=False, sep="\t")

    return (
        maxquant_file_transfer_path,
        maxquant_file_transfer_pred_path,
    )


def _format_maxquant_as_deeplc_input(maxquant_file: str):
    maxquant_df = pd.read_csv(maxquant_file, sep="\t")
    maxquant_msms_df = maxquant_df.copy()
    # if "Raw file" not in maxquant_df.columns:
    #     maxquant_msms_df["Raw file"] = "placeholder"
    # if "Reverse" not in maxquant_df.columns:
    #     maxquant_msms_df["Reverse"] = "placeholder"
    # if "Scan number" not in maxquant_df.columns:
    #     maxquant_msms_df["Scan number"] = "placeholder"
    for col in [
        "Raw file",
        "Reverse",
        "Proteins",
    ]:
        if col not in maxquant_df.columns:
            maxquant_msms_df[col] = "placeholder"
    for col in ["Scan number"]:
        if col not in maxquant_df.columns:
            maxquant_msms_df[col] = 0
    for col in ["m/z", "Retention time", "PEP", "Score"]:
        if col not in maxquant_df.columns:
            maxquant_msms_df[col] = 0.0
    msms_reader_path = os.path.join(
        os.path.dirname(maxquant_file), "for_msms_reader.txt"
    )
    maxquant_msms_df.to_csv(msms_reader_path, sep="\t", index=False)
    reader = MSMSReader(msms_reader_path)

    psm_list = reader.read_file()

    psm_list.add_fixed_modifications([("Carbamidomethyl", ["C"])])
    psm_list.apply_fixed_modifications()
    # Modify these to match the modifications in the data and library of deepLC model
    psm_list.rename_modifications(
        {"ox": "Oxidation", "ac": "Acetyl", "Oxidation (M)": "Oxidation"}
    )
    peprec = to_dataframe(psm_list)  # can be mapped to ori df

    # Only one RT for each (peptide seq, mod)
    peprec_modpept_agg = (
        peprec.groupby(by=["peptide", "modifications"])["observed_retention_time"]
        .median()
        .reset_index()
    )

    peprec_modpept_agg = peprec_modpept_agg.rename(
        columns={"peptide": "seq", "observed_retention_time": "tr"}
    )
    peprec_modpept_agg = peprec_modpept_agg[["seq", "modifications", "tr"]]
    maxquant_peprec = pd.concat([maxquant_df, peprec], axis=1)
    return maxquant_peprec, peprec_modpept_agg


def match_pred_to_input(
    MQ_peprec: pd.DataFrame,
    peprec_RTpred: pd.DataFrame,
    filter_by_RT_diff: Union[Literal["closest"], float, None] = None,
):
    """

    :peprec_RTpred: the column containing RT prediction should be named 'predicted_RT'
    :filtered_by_RT_diff: whether and how to filter results based on RT difference
        'closest': only keeping precursors that elute the closest to predicted RT
        float: specify a threshold, all entries with difference larger will be discarded,
                and for the ones kept, intensity will be aggregated by sum
        None: do not filter
    """
    MQ_RTpred = pd.merge(
        left=MQ_peprec,
        right=peprec_RTpred,
        left_on=["peptide", "modifications"],
        right_on=["seq", "modifications"],
        how="left",
    )
    if filter_by_RT_diff is not None:
        MQ_RTpred["RT_diff"] = abs(
            MQ_RTpred["Retention time"] - MQ_RTpred["predicted_RT"]
        )

        if filter_by_RT_diff == "closest":
            n_before = MQ_RTpred.shape[0]
            MQ_RTpred = MQ_RTpred.loc[
                MQ_RTpred.groupby(["Modified sequence", "Charge"])["RT_diff"].idxmin()
            ]
            n_after = MQ_RTpred.shape[0]
            Logger.info("Removed %s entries.", n_before - n_after)

        elif isinstance(filter_by_RT_diff, float):
            Logger.debug("Filter by threshold %s", filter_by_RT_diff)
            n_before = MQ_RTpred.shape[0]
            MQ_RTpred = MQ_RTpred.loc[MQ_RTpred["RT_diff"] <= filter_by_RT_diff]
            n_after = MQ_RTpred.shape[0]
            Logger.info(
                "Removed %s entries from RT difference threshold", n_before - n_after
            )
            column_map = {col: "first" for col in MQ_RTpred.columns}
            column_map["Intensity"] = "sum"
            MQ_RTpred = MQ_RTpred.groupby(
                ["Modified sequence", "Charge"], as_index=False
            ).agg(column_map)
            n_after_after = MQ_RTpred.shape[0]
            Logger.info(
                "Removed %s entries from RT difference threshold",
                n_after - n_after_after,
            )

        else:
            raise ValueError(
                "filter_by_RT_diff should be either a float or str closest!"
            )

    return MQ_RTpred


def fill_mbr_modified_sequence(
    evidence_df: pd.DataFrame,
    MSMS_type: str = "TIMS-MULTI-MSMS",
    MATCH_type: str = "TIMS-MULTI-MATCH",
) -> pd.DataFrame:
    """
    Fill the 'Modified sequence' where 'Type' is 'TIMS-MULTI-MATCH' with the 'Modified sequence' where 'Type' is 'TIMS-MULTI-MSMS'
    Works on Maxquant evidence.txt when Match between run is enabled and the 'Modified sequence' is missing in the 'TIMS-MULTI-MATCH' rows
    :param evidence_df: pd.DataFrame
    :param MSMS_type: str, default 'TIMS-MULTI-MSMS' Needs to be changed for other equipment
    :param MATCH_type: str, default 'TIMS-MULTI-MATCH' Needs to be changed for other equipment
    :return: pd.DataFrame
    """

    # Create a mapping from 'Mod. peptide ID' to 'Modified sequence' where 'Type' is 'TIMS-MULTI-MSMS'
    mapping = (
        evidence_df[evidence_df["Type"] == MSMS_type]
        .set_index("Mod. peptide ID")["Modified sequence"]
        .to_dict()
    )

    # Fill the 'Modified sequence' where 'Type' is 'TIMS-MULTI-MATCH'
    evidence_df.loc[
        (evidence_df["Type"] == MATCH_type) & (evidence_df["Modified sequence"].isna()),
        "Modified sequence",
    ] = evidence_df["Mod. peptide ID"].map(mapping)

    return evidence_df
