import re
import pandas as pd
import logging
import numpy as np

Logger = logging.getLogger(__name__)

_MAXQUANT_MOD_RE = re.compile(r"^(\d+)([A-Za-z])\(([-+0-9.]+)\)$")
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
    "Peptide": "Sequence",
    "Peptide Sequence": "Sequence",
    "Modified Sequence": "Modified sequence",
    "Modified Peptide": "Modified sequence",
    "Peptide Length": "Length",
    "Observed M/Z": "m/z",
    "M/Z": "m/z",
    "Protein": "Proteins",
    "Apex Retention Time": "Calibrated retention time",
    "Retention Time Start": "Calibrated retention time start",
    "Retention Time End": "Calibrated retention time finish",
    # "Apex Ion Mobility": "1/K0",
    "Ion Mobility": "1/K0",
    "Ion Mobility Start": "1/K0 start",
    "Ion Mobility End": "1/K0 finish",
    "Assigned Modifications": "Modifications",
    "Retention": "Retention time",
    "Hyperscore": "Score",
}


def sage_parser(
    sage_output: pd.DataFrame,
    sage_rename_dict: dict = sage_rename_dict,
    rt_window: float = 0.0,
    im_window: float = 0.0,
) -> pd.DataFrame:
    """
    Parse the SAGE output DataFrame and rename columns based on the provided dictionary.

    Args:
        sage_output (pd.DataFrame): The SAGE output DataFrame.
        sage_rename_dict (dict): A dictionary mapping old column names to new column names.
        rt_window (float): RT elution window in minutes to add as ``Retention length``.
            0 (default) triggers auto-calculation: 1% of the maximum observed RT.
        im_window (float): IM elution window in 1/K0 units to add as ``1/K0 length``.
            0 (default) triggers auto-calculation: 0.1 1/K0 units.

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
    ].copy()
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

    # RT elution window: SAGE reports apex only, so we synthesise a window.
    if rt_window == 0.0:
        rt_window = sage_output["Retention time"].max() * 0.01
        Logger.warning(
            "SAGE_RT_WINDOW not set; defaulting to 1%% of max RT = %.4f min", rt_window
        )
    sage_output["Retention length"] = rt_window

    # IM elution window: SAGE reports apex only, so we use a fixed default.
    if im_window == 0.0:
        im_window = 0.1
        Logger.warning(
            "SAGE_IM_WINDOW not set; defaulting to %.2f 1/K0 units", im_window
        )
    sage_output["1/K0 length"] = im_window

    sage_output.loc[sage_output["Is_Decoy"], "Reverse"] = "+"
    sage_output["Sequence"] = sage_output["Modified sequence"].str.replace(
        r"\[.*?\]", "", regex=True
    )
    # Extract modifications from the modified sequence, e.g. "PEPTM[+15.9949]IK"
    # → "M+15.9949". Produces "Unmodified" for unmodified peptides so the column
    # is never NaN (required by cleanup_maxquant/pivot_psm_by_mz_rank groupby).
    import re as _re
    def _extract_mods(seq):
        hits = _re.findall(r"([A-Z])\[([+-]?\d+\.?\d*)\]", str(seq))
        if not hits:
            return "Unmodified"
        return ";".join(f"{aa}{mass}" for aa, mass in hits)
    sage_output["Modifications"] = sage_output["Modified sequence"].apply(_extract_mods)
    sage_output["Retention time calibration"] = 0
    return sage_output


def fragpipe_psm_parser(fragpipe_output: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the FragPipe output DataFrame and rename columns based on the provided dictionary.

    Args:
        fragpipe_output (pd.DataFrame): The FragPipe output DataFrame.
        exp_name (Optional[str]): The experiment name to filter columns. If None, prepares a reference dictionary.
    Returns:
        pd.DataFrame: The parsed DataFrame with renamed columns.
    """
    fragpipe_output_copy = fragpipe_output.copy(deep=True)
    fragpipe_output_copy["Modified Peptide"] = fragpipe_output_copy[
        "Modified Peptide"
    ].fillna(fragpipe_output_copy["Peptide"])
    fragpipe_output_copy.rename(columns=fragpipe_rename_dict, inplace=True)
    fragpipe_output_copy["Raw file"] = fragpipe_output["Spectrum"].str.split(".").str[0]
    fragpipe_output_copy["Reverse"] = np.where(
        fragpipe_output_copy["Is Decoy"],
        "+",
        np.nan,
    )
    fragpipe_output_copy["Type"] = "TIMS-MULTI-MSMS"
    # Convert Retention time to minute
    for rt_col in [
        "Calibrated retention time",
        "Calibrated retention time start",
        "Calibrated retention time finish",
        "Retention time",
    ]:
        fragpipe_output_copy[rt_col] = fragpipe_output_copy[rt_col] / 60.0

    fragpipe_output_copy["Retention length"] = (
        fragpipe_output_copy["Calibrated retention time finish"]
        - fragpipe_output_copy["Calibrated retention time start"]
    )
    fragpipe_output_copy["1/K0 length"] = (
        fragpipe_output_copy["1/K0 finish"] - fragpipe_output_copy["1/K0 start"]
    )
    fragpipe_output_copy["Modifications"] = fragpipe_output_copy[
        "Modifications"
    ].fillna("-")
    return fragpipe_output_copy


def maxquant_mods_to_fragpipe_modseq(sequence: str, modifications: str) -> str | None:
    """MaxQuant/SWAPS dict Sequence+Modifications -> FragPipe bracketed modified-
    sequence string, e.g. ("AAAEVGAPFIEIHTGCYADAK", "16C(57.0215)") ->
    "AAAEVGAPFIEIHTGC[57.0215]YADAK". Bridges dict_ref rows onto the same key
    space as FragPipe's own "Modified Sequence" column (psm.tsv/combined_ion.tsv)
    for joins that don't go through the search-engine parsers above. Returns
    None if a modification token cannot be parsed, so callers can flag failures
    instead of silently mis-mapping.
    """
    if modifications is None or modifications == "-" or pd.isna(modifications):
        return sequence
    nterm_mass = None
    residue_mods = {}
    for part in modifications.split(","):
        part = part.strip()
        if part.startswith("N-term"):
            nterm_mass = part[part.index("(") + 1 : part.index(")")]
            continue
        m = _MAXQUANT_MOD_RE.match(part)
        if not m:
            return None
        pos, mass = int(m.group(1)), m.group(3)
        residue_mods[pos] = mass
    chars = []
    for i, ch in enumerate(sequence, start=1):
        chars.append(ch)
        if i in residue_mods:
            chars.append(f"[{residue_mods[i]}]")
    s = "".join(chars)
    if nterm_mass is not None:
        s = f"n[{nterm_mass}]" + s
    return s


