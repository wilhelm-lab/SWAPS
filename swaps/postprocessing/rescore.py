import os
from typing import Literal, Optional
import pandas as pd
import numpy as np
import mokapot
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.base import BaseEstimator

Logger = logging.getLogger(__name__)


def prepare_mokapot_input(
    df,
    feature_cols: list,
    scannr_col: Optional[str] = None,
    psmid_col: Optional[str] = None,
    normalize_features: bool = True,
    decoy_col: str = "decoy",
    peptide_col: str = "modified_sequence",
    protein_col: str = "proteins",
    filename_col: Optional[str] = None,
):
    """
    Prepares input dataframe for mokapot analysis by standardizing column names and format.
    This function takes a pandas DataFrame containing PSM data and reformats it to be compatible
    with mokapot's expected input format. It handles scan numbers, PSM IDs, labels for target/decoy,
    peptide sequences, protein information, and filename data.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing PSM data
    feature_cols : list
        List of column names containing features to be used for classification
    scannr_col : str, optional
        Column name containing MS/MS scan numbers (default: "MS/MS scan number"),
        this column is not necessarily unique
    psmid_col : str, optional
        Column name containing PSM IDs (default: None), this column should be unique
    normalize_features : bool, optional
        Whether to normalize feature columns (default: True)
    decoy_col : str, optional
        Column name indicating decoy status (default: "decoy")
    peptide_col : str, optional
        Column name containing peptide sequences (default: "modified_sequence")
    protein_col : str, optional
        Column name containing protein identifiers (default: "proteins")
    filename_col : str, optional
        Column name containing raw file names (default: "Raw file")

    Returns
    -------
    pandas.DataFrame
        Reformatted DataFrame with standardized column names ready for mokapot input:
        - specid: Unique identifier for each PSM
        - scannr: Scan number
        - label: Target/decoy label (1 for target, -1 for decoy)
        - [feature columns]: Selected features for classification
        - peptide: Peptide sequence
        - proteins: Protein identifiers
        - filename: Source file name
    """

    df_pin = df.copy()
    df_pin.dropna(subset=feature_cols, inplace=True)
    if scannr_col is not None:
        df_pin["scannr"] = df_pin[scannr_col]
    else:
        df_pin["scannr"] = (
            df_pin["mz_rank"].astype(str) + "_" + df_pin["label"].astype(str)
        )
    if psmid_col is not None:
        df_pin["specid"] = df_pin[psmid_col]
    else:
        df_pin["specid"] = (
            df_pin["mz_rank"].astype(str) + "_" + df_pin["label"].astype(str)
        )
    df_pin.drop(columns=["label"], errors="ignore", inplace=True)
    # df_pin["spectra"] = np.arange(len(df_pin))
    df_pin["label"] = 1
    df_pin.loc[df_pin[decoy_col], "label"] = -1
    id_col = ["scannr", "specid", "label", "proteins"]
    if peptide_col is not None:
        df_pin["peptide"] = df_pin[peptide_col]
        id_col += ["peptide"]
    df_pin["proteins"] = df_pin[protein_col]

    if filename_col is not None:
        df_pin["filename"] = df_pin[filename_col]
        id_col.append("filename")
    else:
        df_pin["filename"] = "file"
        id_col.append("filename")
    df_pin = df_pin[id_col + feature_cols]
    if normalize_features:
        for col in tqdm(feature_cols, desc="Normalizing features"):
            col_z = col + "_z"
            df_pin[col_z] = (df_pin[col] - df_pin[col].mean()) / df_pin[col].std()
        feature_cols = [col + "_z" for col in feature_cols]
        df_pin = df_pin[id_col + feature_cols]
    Logger.info("Prepared mokapot input: %s", df_pin.head(5))
    return df_pin


def brew_with_mokapot(
    peptide_info_dataframe: pd.DataFrame,
    train_fdr: float = 0.1,
    test_fdr: float = 0.1,
    model: Optional[BaseEstimator] = None,
    # level: Literal["pfm", "protein"] = "pfm",
    work_dir: Optional[str] = None,
    direction: Optional[str] = None,
    **kwargs,
):
    """
    Wrapper function to run mokapot rescoring on a given peptide information DataFrame.
    This function prepares the input data, runs mokapot for rescoring, and returns the results.
    Parameters
    ----------
    peptide_info_dataframe : pandas.DataFrame
        DataFrame containing peptide information and features for mokapot input
    train_fdr : float, optional
        FDR threshold for training (default: 0.1)
    test_fdr : float, optional
        FDR threshold for testing (default: 0.1)
    work_dir : str, optional
        Directory to save temporary files (default: current working directory)
    **kwargs : additional keyword arguments
        Additional arguments to pass to the prepare_mokapot_input function
    Returns
    -------
    tuple
        A tuple containing:
        - mokapot.results.PSMResults: Results of the mokapot rescoring
        - mokapot.model.Model: The trained mokapot model
    """
    if work_dir is None:
        work_dir = os.getcwd()
    else:
        os.makedirs(work_dir, exist_ok=True)
    # Prepare mokapot input
    mokapot_input = prepare_mokapot_input(
        peptide_info_dataframe,
        **kwargs,
    )

    # mokapot_input['peptide'] = mokapot_input['proteins']
    Logger.info("Mokapot input columns: %s", mokapot_input.columns.tolist())
    # Use a temporary file to write the input .pin
    mokapot_input.to_csv(
        os.path.join(work_dir, "mokapot_input.pin"), sep="\t", index=False
    )

    # Read the .pin file and run mokapot
    psms_pin = mokapot.read_pin(os.path.join(work_dir, "mokapot_input.pin"))
    if model is None:
        mokapot_model = mokapot.model.PercolatorModel(
            train_fdr=train_fdr, direction=direction
        )
    else:
        mokapot_model = mokapot.model.Model(model, train_fdr=train_fdr)
    result, model = mokapot.brew(
        psms_pin, model=mokapot_model, test_fdr=test_fdr, folds=5
    )

    # Clean up the temporary file
    # os.remove(os.path.join(work_dir, "mokapot_input.pin"))
    result.plot_qvalues()
    plt.savefig(
        os.path.join(work_dir, "mokapot_qvalues.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()

    result.to_txt(work_dir, decoys=True)

    # merge with mokapot results
    psms = result.confidence_estimates["psms"]
    psms_decoy = result.decoy_confidence_estimates["psms"]
    psms = pd.concat([psms, psms_decoy], ignore_index=True)
    # psms["peak_label"] = psms["specid"].str.split("_").str[1].astype(int)

    sns.histplot(data=psms, x="mokapot score", bins=100, hue="label", multiple="dodge")
    plt.savefig(
        os.path.join(work_dir, "mokapot_score_distr.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()

    sns.histplot(
        data=psms,
        x="mokapot score",
        hue="label",
        bins=100,
        multiple="dodge",
        log_scale=(True, False),
    )
    plt.savefig(
        os.path.join(work_dir, "mokapot_score_distr_log.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    return result, model, psms


def merge_peaks_result_and_dict(
    peaks_result,
    dict_ref,
    ms1_scan_gap,
    remove_bias: bool = False,
    log_int_bias=1,
    keep_closet_int: Optional[int] = None,
    keep_closet_rt: Optional[int] = None,
):
    """
    Merges the peaks result DataFrame with a reference dictionary DataFrame based on the 'peptide' column.
    This function combines information from both DataFrames, ensuring that all relevant data is retained.

    Parameters
    ----------
    peaks_result : pandas.DataFrame
        DataFrame containing peak results with a 'peptide' column
    dict_ref : pandas.DataFrame
        Reference dictionary DataFrame containing additional information with a 'peptide' column
    ms1_scan_gap : float
        The gap between MS1 scans, used for calculating retention time index length
    log_int_bias : float, optional
        Bias added to log transformed intensity sum to ensure linearity for log_diff feature
    Returns
    -------
    pandas.DataFrame
        Merged DataFrame containing combined information from both input DataFrames
    """
    # dict_ref["rt_idx_length"] = (
    #     dict_ref["MS1_frame_idx_right_ref"] - dict_ref["MS1_frame_idx_left_ref"]
    # )
    if "Retention length" not in dict_ref.columns:
        dict_ref["Retention length"] = (
            (
                dict_ref["Retention time finish_ref"]
                - dict_ref["Retention time start_ref"]
            )
            * 60.0
            / ms1_scan_gap
        )  # convert to index length
    dict_ref["rt_idx_length_calc"] = dict_ref["Retention length"] / ms1_scan_gap
    peaks_result_merged_dict = pd.merge(
        peaks_result,
        dict_ref[
            [
                "mz_rank",
                "IM_search_idx_center",
                # "MS1_frame_idx_center",
                "MS1_frame_idx_center_ref",
                "MS1_frame_idx_left_ref",
                "MS1_frame_idx_right_ref",
                "Decoy",
                "Sequence",
                "Charge",
                "Proteins",
                "Ion mobility length",
                # "rt_idx_length",
                "rt_idx_length_calc",
                "Intensity",
                "source",
            ]
        ],
        left_on="mz_rank",
        right_on="mz_rank",
        how="left",
    )
    peaks_result_merged_dict.rename(
        columns={
            "bbox-0": "rt_min_index",
            "bbox-2": "rt_max_index",
            "bbox-1": "im_min_index",
            "bbox-3": "im_max_index",
            "centroid-0": "pixel_apex_rt",
            "centroid-1": "pixel_apex_im",
        },
        inplace=True,
    )
    peaks_result_merged_dict["log_Intensity_MQ"] = np.log10(
        1 + peaks_result_merged_dict["Intensity"]
    )
    peaks_result_merged_dict["rt_diff"] = (
        peaks_result_merged_dict["rt_apex_index"]
        - peaks_result_merged_dict["MS1_frame_idx_center_ref"]
    )
    peaks_result_merged_dict["rt_pixel_diff"] = (
        peaks_result_merged_dict["pixel_apex_rt"]
        - peaks_result_merged_dict["MS1_frame_idx_center_ref"]
    )
    # peaks_result_merged_dict["rt_start_diff"] = (
    #     peaks_result_merged_dict["rt_min_index"]
    #     - peaks_result_merged_dict["MS1_frame_idx_left_ref"]
    # )
    # peaks_result_merged_dict["rt_end_diff"] = (
    #     peaks_result_merged_dict["rt_max_index"]
    #     - peaks_result_merged_dict["MS1_frame_idx_right_ref"]
    # )
    peaks_result_merged_dict["im_diff"] = (
        peaks_result_merged_dict["im_apex_index"]
        - peaks_result_merged_dict["IM_search_idx_center"]
    )
    peaks_result_merged_dict["im_length_diff"] = (
        peaks_result_merged_dict["im_length"]
        - peaks_result_merged_dict["Ion mobility length"]
    )
    peaks_result_merged_dict["rt_length_diff"] = (
        peaks_result_merged_dict["rt_length"]
        - peaks_result_merged_dict["rt_idx_length_calc"]
    )
    peaks_result_merged_dict["log_int_diff"] = (
        np.log10(1 + peaks_result_merged_dict["intensity_sum"])
        - peaks_result_merged_dict["log_Intensity_MQ"]
        + log_int_bias
    )

    for col in peaks_result_merged_dict.columns:
        if "diff" in col:
            if remove_bias and (
                col
                in [
                    "rt_length_diff",
                    "im_length_diff",
                    "rt_diff",
                    "im_diff",
                    "rt_pixel_diff",
                ]
            ):
                Logger.info(
                    "Column %s: median %.4f",
                    col,
                    peaks_result_merged_dict[col].median(),
                )
                peaks_result_merged_dict[col] = (
                    peaks_result_merged_dict[col]
                    - peaks_result_merged_dict[col].median()
                )
            # Take absolute value for diff features
            peaks_result_merged_dict[col] = abs(peaks_result_merged_dict[col])
    if keep_closet_rt is not None:
        Logger.info(
            "Keeping only the %d closest retention times per peptide, rows before filtering: %d",
            keep_closet_rt,
            peaks_result_merged_dict.shape[0],
        )
        peaks_result_merged_dict.sort_values(
            by=["rt_diff"], inplace=True, ascending=True
        )
        peaks_result_merged_dict = (
            peaks_result_merged_dict.groupby("mz_rank")
            .head(keep_closet_rt)
            .reset_index(drop=True)
        )
        Logger.info("Rows after filtering: %d", peaks_result_merged_dict.shape[0])
    if keep_closet_int is not None:
        Logger.info(
            "Keeping only the %d closest intensities per peptide, rows before filtering: %d",
            keep_closet_int,
            peaks_result_merged_dict.shape[0],
        )
        peaks_result_merged_dict.sort_values(
            by=["log_int_diff"], inplace=True, ascending=True
        )
        peaks_result_merged_dict = (
            peaks_result_merged_dict.groupby("mz_rank")
            .head(keep_closet_int)
            .reset_index(drop=True)
        )
        Logger.info("Rows after filtering: %d", peaks_result_merged_dict.shape[0])
    return peaks_result_merged_dict


def combine_matches_target_decoy(matches_target, matches_decoy, dict_ref):

    ## Target Decoy Competition
    matches_target["IsTarget"] = True
    matches_decoy["IsTarget"] = False
    matches_target["decoy_mz_rank"] = -1
    tdc_df = pd.concat([matches_target, matches_decoy], ignore_index=True)
    tdc_df = pd.merge(tdc_df, dict_ref[["mz_rank", "Sequence", "Proteins"]])
    tdc_df.loc[~tdc_df["IsTarget"], "Sequence"] = (
        "DECOY_" + tdc_df.loc[tdc_df["IsTarget"], "Sequence"]
    )
    tdc_df.loc[~tdc_df["IsTarget"], "Proteins"] = (
        "DECOY_" + tdc_df.loc[tdc_df["IsTarget"], "Proteins"]
    )
    tdc_df["Decoy"] = ~tdc_df["IsTarget"]
    tdc_df["label"] = tdc_df["IsTarget"]
    tdc_df["Sequence_with_runs"] = tdc_df["Sequence"] + "_" + tdc_df["matched_run"]
    return tdc_df


def normalize_shift_by_runs(
    matches_target, matches_decoy, cols_to_scale=["rt_shift", "im_shift"]
):
    """
    Normalize rt_shift and im_shift per run-pair using target statistics,
    apply normalization to both target and decoy,
    and return them separately.
    """

    # Compute mean/std per run-pair from target only
    stats = matches_target.groupby(["reference_run", "matched_run"])[cols_to_scale].agg(
        ["mean", "std"]
    )

    # Flatten MultiIndex columns
    stats.columns = [f"{col}_{agg}" for col, agg in stats.columns]

    def apply_normalization(df):
        # Merge stats onto the dataframe
        df_norm = df.merge(
            stats,
            how="left",
            left_on=["reference_run", "matched_run"],
            right_index=True,
        )

        for col in cols_to_scale:
            mean_col = f"{col}_mean"
            std_col = f"{col}_std"

            # Avoid division by zero
            df_norm[f"{col}_scaled"] = (df_norm[col] - df_norm[mean_col]) / df_norm[
                std_col
            ].replace(0, 1)
            df_norm[f"{col}_abs_scaled"] = df_norm[f"{col}_scaled"].abs()

        # Drop temporary columns
        df_norm = df_norm.drop(
            columns=[
                c for c in df_norm.columns if c.endswith("_mean") or c.endswith("_std")
            ]
        )
        return df_norm

    # Apply normalization separately
    matches_target_norm = apply_normalization(matches_target)
    matches_decoy_norm = apply_normalization(matches_decoy)

    return matches_target_norm, matches_decoy_norm
