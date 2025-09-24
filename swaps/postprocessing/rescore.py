import os
from typing import Literal, Optional
import pandas as pd
import numpy as np
import mokapot
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm

Logger = logging.getLogger(__name__)


def prepare_mokapot_input(
    df,
    feature_cols: list,
    scannr_col: Optional[str] = None,
    psmid_col: Optional[str] = None,
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
    Logger.info("Prepared mokapot input: %s", df_pin.head(5))
    return df_pin


def brew_with_mokapot(
    peptide_info_dataframe: pd.DataFrame,
    train_fdr: float = 0.1,
    test_fdr: float = 0.1,
    # level: Literal["pfm", "protein"] = "pfm",
    work_dir: Optional[str] = None,
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
    mokapot_model = mokapot.model.PercolatorModel(train_fdr=train_fdr)
    result, model = mokapot.brew(
        psms_pin, model=mokapot_model, test_fdr=test_fdr, folds=3
    )

    # Clean up the temporary file
    # os.remove(os.path.join(work_dir, "mokapot_input.pin"))
    result.plot_qvalues()
    plt.savefig(os.path.join(work_dir, "mokapot_qvalues.png"))
    plt.close()

    result.to_txt(work_dir, decoys=True)
    return result, model


def merge_peaks_result_and_dict(peaks_result, dict_ref, ms1_scan_gap):
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

    Returns
    -------
    pandas.DataFrame
        Merged DataFrame containing combined information from both input DataFrames
    """
    dict_ref["rt_idx_length"] = (
        dict_ref["MS1_frame_idx_right_ref"] - dict_ref["MS1_frame_idx_left_ref"]
    )
    dict_ref["rt_idx_length_calc"] = dict_ref["Retention length"] / ms1_scan_gap
    peaks_result_merged_dict = pd.merge(
        peaks_result,
        dict_ref[
            [
                "mz_rank",
                "IM_search_idx_center",
                "MS1_frame_idx_center_ref",
                "Decoy",
                "Sequence",
                "Charge",
                "Proteins",
                "Ion mobility length",
                "rt_idx_length",
                "rt_idx_length_calc",
                "Intensity",
                "source",
            ]
        ],
        left_on="mz_rank",
        right_on="mz_rank",
        how="left",
    )
    peaks_result_merged_dict["log_Intensity_MQ"] = np.log10(
        1 + peaks_result_merged_dict["Intensity"]
    )
    peaks_result_merged_dict["rt_diff"] = abs(
        peaks_result_merged_dict["rt_apex_index"]
        - peaks_result_merged_dict["MS1_frame_idx_center_ref"]
    )
    peaks_result_merged_dict["im_diff"] = abs(
        peaks_result_merged_dict["im_apex_index"]
        - peaks_result_merged_dict["IM_search_idx_center"]
    )
    peaks_result_merged_dict["im_length_diff"] = abs(
        peaks_result_merged_dict["im_length"]
        - peaks_result_merged_dict["Ion mobility length"]
    )
    peaks_result_merged_dict["rt_length_diff"] = abs(
        peaks_result_merged_dict["rt_length"]
        - peaks_result_merged_dict["rt_idx_length_calc"]
    )
    peaks_result_merged_dict["log_int_diff"] = abs(
        np.log10(1 + peaks_result_merged_dict["intensity_sum"])
        - peaks_result_merged_dict["log_Intensity_MQ"]
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
    return peaks_result_merged_dict
