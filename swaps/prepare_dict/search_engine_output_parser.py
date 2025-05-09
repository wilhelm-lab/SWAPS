import pandas as pd
import logging

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

    return sage_output
