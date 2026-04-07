"""
IonQuant-style piecewise intensity normalization.

Corrects nonlinear, intensity-dependent experimental errors across runs by
dividing the log-intensity axis into bins and applying a per-bin median
log-ratio correction relative to a reference run.

Reference
---------
IonQuant: https://doi.org/10.1016/j.mcpro.2021.100133
"""

import logging

import numpy as np
import pandas as pd

Logger = logging.getLogger(__name__)

_DEFAULT_N_BINS = 10


def piecewise_normalize(df: pd.DataFrame, n_bins: int = _DEFAULT_N_BINS) -> pd.DataFrame:
    """
    Apply IonQuant-style piecewise normalization to a quantification matrix.

    The algorithm:
      1. Identify the reference run as the column with the most quantified ions.
      2. For every other run, find ions present in both the run and the reference.
      3. Compute log2-ratios of those overlapping ions.
      4. Bin the overlapping ions by log2-intensity into *n_bins* equal-width
         ranges and compute the median log2-ratio in each bin.
      5. Subtract the per-bin median correction from all quantified ions in
         the run (bins are extrapolated at the boundaries for out-of-range ions).

    Parameters
    ----------
    df : pd.DataFrame
        Rows = ions, columns contain the keyword ``"Intensity"`` for run
        intensity values.  Zero and NaN values are treated as unquantified.
    n_bins : int
        Number of equal-width intensity bins for the piecewise correction.
        Default is 10 (matching the IonQuant default).

    Returns
    -------
    pd.DataFrame
        Copy of *df* with intensity columns normalised; the reference run is
        returned unchanged.

    Raises
    ------
    ValueError
        If no columns containing ``"Intensity"`` are found.
    """
    intensity_cols = [c for c in df.columns if "Intensity" in c]
    if not intensity_cols:
        raise ValueError("No columns containing 'Intensity' found in the DataFrame.")

    result = df.copy()

    # Treat zeros as missing
    result[intensity_cols] = result[intensity_cols].replace(0.0, np.nan)

    # Reference run: the column with the most non-NaN ions
    ion_counts = result[intensity_cols].notna().sum()
    ref_col = ion_counts.idxmax()
    Logger.info("Reference run: %s  (%d ions)", ref_col, ion_counts[ref_col])

    log_ref = np.log2(result[ref_col].to_numpy(dtype=float))

    for col in intensity_cols:
        if col == ref_col:
            continue

        log_run = np.log2(result[col].to_numpy(dtype=float))

        # Ions quantified in both this run and the reference
        overlap = np.isfinite(log_ref) & np.isfinite(log_run)
        n_overlap = overlap.sum()

        if n_overlap < n_bins:
            Logger.warning(
                "%s: only %d overlapping ions (need >= %d); skipping normalization.",
                col,
                n_overlap,
                n_bins,
            )
            continue

        log_run_ov = log_run[overlap]
        log_ratio_ov = log_run_ov - log_ref[overlap]

        # Equal-width bin edges over the overlapping log-intensities
        lo, hi = log_run_ov.min(), log_run_ov.max()
        edges = np.linspace(lo, hi, n_bins + 1)
        # Interior edges define n_bins bins; values outside [lo, hi] are
        # assigned to the nearest boundary bin by np.digitize behaviour.
        interior = edges[1:-1]

        bin_idx = np.digitize(log_run_ov, interior)  # values in [0, n_bins-1]

        median_corrections = np.full(n_bins, np.nan)
        for b in range(n_bins):
            mask = bin_idx == b
            if mask.any():
                median_corrections[b] = np.median(log_ratio_ov[mask])

        # Fill any empty bins by linear interpolation / boundary extrapolation
        median_corrections = _fill_nan_bins(median_corrections)

        # Apply correction to every quantified ion in this run
        quantified = np.isfinite(log_run)
        if not quantified.any():
            continue

        log_run_q = log_run[quantified]
        bin_idx_all = np.clip(np.digitize(log_run_q, interior), 0, n_bins - 1)
        corrected_log = log_run_q - median_corrections[bin_idx_all]

        corrected = np.full(len(log_run), np.nan)
        corrected[quantified] = np.exp2(corrected_log)
        result[col] = corrected

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fill_nan_bins(arr: np.ndarray) -> np.ndarray:
    """Fill NaN entries by linear interpolation; clip to boundary values outside."""
    arr = arr.copy()
    nans = np.isnan(arr)
    if not nans.any():
        return arr
    if nans.all():
        arr[:] = 0.0
        return arr
    idx = np.arange(len(arr))
    # np.interp clips at boundary values for out-of-range indices (desired)
    arr[nans] = np.interp(idx[nans], idx[~nans], arr[~nans])
    return arr
