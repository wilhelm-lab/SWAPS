import pandas as pd
import logging
import numpy as np
from typing import Optional
Logger = logging.getLogger(__name__)
sage_rename_dict = {
    "psm_id": "id",
    "peptide": "Modified sequence",
    "proteins": "Proteins",
    "filename": "Raw file",
    "scannr": "MS/MS scan number",
    "rank": "Combinatorics",
    "expmass": "Mass",
    "charge": "Charge",
    "peptide_len": "Length",
    "missed_cleavages": "Missed cleavages",
    "precursor_ppm": "Mass error [ppm]",
    "hyperscore": "Score",
    "delta_next": "Delta score",
    "rt": "Retention time",
    "ion_mobility": "1/K0",
    "delta_mobility": "Match K0 difference",
    "matched_peaks": "Number of isotopic peaks",  # FIXME: this is not correct
    "matched_intensity_pct": "Fraction of total spectrum",
    "sage_discriminant_score": "Match score",
    "posterior_error": "PEP",
    "spectrum_q": "Match q-value",
    "ms2_intensity": "Intensity",
}

fragpipe_rename_dict = {
    "Peptide Sequence": "Sequence",
    "Modified Sequence": "Modified sequence",
    "Peptide Length": "Length",
    "M/Z": "m/z",
    "Protein": "Proteins",
    "Apex Retention Time": "Calibrated retention time",
    "Retention Time Start": "Calibrated retention time start",
    "Retention Time End": "Calibrated retention time finish",
    "Apex Ion Mobility": "1/K0",
    "Ion Mobility Start": "1/K0 start",
    "Ion Mobility End": "1/K0 finish",
}
def sage_parser(
    sage_output: pd.DataFrame, sage_rename_dict: dict = sage_rename_dict
) -> pd.DataFrame:
    """
    Parse the SAGE output DataFrame and rename columns based on the provided dictionary.

    Args:
        sage_output (pd.DataFrame): The SAGE output DataFrame.
        sage_rename_dict (dict): A dictionary mapping old column names to new column names.

    Returns:
        pd.DataFrame: The parsed DataFrame with renamed columns.
    """
    sage_output["filename"] = sage_output["filename"].str[:-2]
    sage_output["Is_Decoy"] = sage_output["proteins"].str.contains("rev_")
    sage_output["m/z"] = sage_output["expmass"] / sage_output["charge"]
    sage_output["calc_m/z"] = sage_output["calcmass"] / sage_output["charge"]
    Logger.info(
        "Before filtering at 0.01 spectrum and peptide q-value: %s", len(sage_output)
    )
    sage_output = sage_output.loc[
        (sage_output["peptide_q"] <= 0.01) & (sage_output["spectrum_q"] <= 0.01)
    ]
    Logger.info(
        "After filtering at 0.01 peptide and spectrum q-value: %s", len(sage_output)
    )
    # Rename columns based on the provided dictionary
    sage_output.rename(columns=sage_rename_dict, inplace=True)

    sage_output["Type"] = "TIMS-MULTI-MSMS"
    sage_output["Calibrated retention time"] = sage_output["Retention time"]
    sage_output["Calibrated retention time start"] = sage_output["Retention time"]
    sage_output["Calibrated retention time finish"] = sage_output["Retention time"]
    sage_output["Ion mobility length"] = 80  # FIXME: very dirty fix
    sage_output.loc[sage_output["Is_Decoy"], "Reverse"] = "+"
    sage_output["Sequence"] = sage_output["Modified sequence"].str.replace(
        r"\[.*?\]", "", regex=True
    )
    sage_output['Retention time calibration'] = 0
    return sage_output

def get_median_from_reps(df, rep_cols):
    """
    Efficiently calculate the median of non-zero values across specified replicate columns for each row.
    Returns 0 if all values are zero.
    """
    # Replace zeros with NaN so they’re ignored in median
    # df[df[rep_cols] < 10e-6]=np.nan
    arr = df[rep_cols]
    arr = arr.replace(0, np.nan)
    # Compute row-wise median, skipping NaN (i.e., zeros)
    med = arr.median(axis=1, skipna=True)
    
    # Replace NaN (all zeros) with 0
    return med.fillna(np.nan)

def fragpipe_parser(
    fragpipe_output: pd.DataFrame,
    exp_name: Optional[str] = None,
    remove_not_quantified: bool = True,
) -> pd.DataFrame:
    """
    Parse the FragPipe output DataFrame and rename columns based on the provided dictionary.

    Args:
        fragpipe_output (pd.DataFrame): The FragPipe output DataFrame.
        exp_name (Optional[str]): The experiment name to filter columns. If None, prepares a reference dictionary.
    Returns:
        pd.DataFrame: The parsed DataFrame with renamed columns.
    """
    fragpipe_output_copy = fragpipe_output.copy(deep=True)
    fragpipe_output_copy['Raw file'] = exp_name
    mbr_feature_cols = [
            "Apex Retention Time",
            "Apex Scan Number",
            "Retention Time Start", 
            "Retention Time End",
            "Apex Ion Mobility",
            "Ion Mobility Start",
            "Ion Mobility End",
            "Intensity",
            "Traced Scans",
            "Ion Mobility FWHM",
            "Retention Time FWHM",
            "Spectral Count",
        ]

    for col_kw in ["Localization"]:
        fragpipe_output_copy = fragpipe_output_copy.drop(
            columns=[col for col in fragpipe_output_copy.columns if col_kw in col]
        )
    if exp_name is None: # Prepare reference dictionary
        Logger.info("Preparing reference dictionary...")
        match_type_col = [col for col in fragpipe_output_copy.columns if "Match Type" in col]
        for col_kw in mbr_feature_cols:
            cols_to_check = [col for col in fragpipe_output_copy.columns if col_kw in col]
            if len(cols_to_check) == 0:
                Logger.warning("No columns found for keyword: %s", col_kw)
            for i, col in enumerate(cols_to_check):
                # only keep values from MS/MS IDs
                non_msms_mask = fragpipe_output_copy[fragpipe_output_copy[match_type_col[i]] != "MS/MS"].index
                fragpipe_output_copy.loc[non_msms_mask, col] = np.nan
                # Logger.info("non MS/MS rows count for column %s: %s, na values: %s", col, len(non_msms_mask), fragpipe_output_copy[col].isna().sum())
            fragpipe_output_copy[col_kw] = get_median_from_reps(
                fragpipe_output_copy, cols_to_check
            )
            fragpipe_output_copy = fragpipe_output_copy.drop(columns=cols_to_check)
    else:
        Logger.info("Preparing experiment dictionary for experiment: %s", exp_name)
        exp_name = exp_name.replace(".d", "")
        exp_name = exp_name.replace("-", "_")

        exp_col = [col for col in fragpipe_output_copy.columns if exp_name in col]
        fragpipe_exp_values = fragpipe_output_copy[exp_col].copy(deep=True)
        for col in mbr_feature_cols + ["Match Type"]:
            cols_to_check = [c for c in fragpipe_output_copy.columns if col in c]
            fragpipe_output_copy = fragpipe_output_copy.drop(columns=cols_to_check)
        exp_rt_apex = [col for col in exp_col if "Apex Retention Time" in col]
        exp_rt_start = [col for col in exp_col if "Retention Time Start" in col]
        exp_rt_end = [col for col in exp_col if "Retention Time End" in col]
        exp_im_apex = [col for col in exp_col if "Apex Ion Mobility" in col]
        exp_im_start = [col for col in exp_col if "Ion Mobility Start" in col]
        exp_im_end = [col for col in exp_col if "Ion Mobility End" in col]
        exp_intensity = [col for col in exp_col if "Intensity" in col]
        exp_match_type = [col for col in exp_col if "Match Type" in col]
        fragpipe_output_copy["Calibrated retention time"] = fragpipe_exp_values[exp_rt_apex]
        fragpipe_output_copy["Calibrated retention time start"] = fragpipe_exp_values[exp_rt_start]
        fragpipe_output_copy["Calibrated retention time finish"] = fragpipe_exp_values[exp_rt_end]
        fragpipe_output_copy["1/K0 start"] = fragpipe_exp_values[exp_im_start]
        fragpipe_output_copy["1/K0 finish"] = fragpipe_exp_values[exp_im_end]
        fragpipe_output_copy["1/K0 length"] = fragpipe_output_copy["1/K0 finish"] - fragpipe_output_copy["1/K0 start"]
        fragpipe_output_copy["1/K0"] = fragpipe_exp_values[exp_im_apex]
        fragpipe_output_copy["Intensity"] = fragpipe_exp_values[exp_intensity]
        fragpipe_output_copy['Match Type'] = fragpipe_exp_values[exp_match_type]
        fragpipe_output_copy = fragpipe_output_copy.loc[fragpipe_output_copy['Match Type'] == "MS/MS"]


        # Logger.info("fragpipe_output_copy columns after filtering: %s", fragpipe_output_copy.columns)
    fragpipe_output_copy.rename(columns=fragpipe_rename_dict, inplace=True)
    fragpipe_output_copy["Type"] = "TIMS-MULTI-MSMS"
    fragpipe_output_copy["Reverse"] = np.nan
    fragpipe_output_copy["Is_Decoy"] = False
    fragpipe_output_copy.loc[fragpipe_output_copy["Is_Decoy"], "Reverse"] = "+"
    fragpipe_output_copy['Retention time calibration'] = 0
    fragpipe_output_copy['Retention time'] = fragpipe_output_copy["Calibrated retention time"]
    fragpipe_output_copy['Ion mobility length'] = 80 # FIXME: very dirty fix
    # Convert Retention time to minute
    for rt_col in ["Calibrated retention time", "Calibrated retention time start", "Calibrated retention time finish", "Retention time"]:
        fragpipe_output_copy[rt_col] = fragpipe_output_copy[rt_col] / 60.0
    if remove_not_quantified:
        initial_count = len(fragpipe_output_copy)
        fragpipe_output_copy = fragpipe_output_copy.loc[fragpipe_output_copy["Intensity"] > 0]
        Logger.info("Removed %s not quantified entries", initial_count - len(fragpipe_output_copy))
    return fragpipe_output_copy