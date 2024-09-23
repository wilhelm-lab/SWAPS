import os
import logging
from typing import List, Literal
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from pyteomics import mzml
import numpy as np
import pandas as pd
from Bio.SeqIO.FastaIO import SimpleFastaParser
from scipy.sparse import csr_matrix
from scipy.signal import find_peaks, peak_widths
from scipy.spatial import cKDTree

Logger = logging.getLogger(__name__)


def plot_colortable(colors, *, ncols=4, sort_colors=True):

    cell_width = 212
    cell_height = 22
    swatch_width = 48
    margin = 12

    # Sort colors by hue, saturation, value and name.
    if sort_colors is True:
        names = sorted(
            colors, key=lambda c: tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(c)))
        )
    else:
        names = list(colors)

    n = len(names)
    nrows = math.ceil(n / ncols)

    width = cell_width * ncols + 2 * margin
    height = cell_height * nrows + 2 * margin
    dpi = 72

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.subplots_adjust(
        margin / width,
        margin / height,
        (width - margin) / width,
        (height - margin) / height,
    )
    ax.set_xlim(0, cell_width * ncols)
    ax.set_ylim(cell_height * (nrows - 0.5), -cell_height / 2.0)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_off()

    for i, name in enumerate(names):
        row = i % nrows
        col = i // nrows
        y = row * cell_height

        swatch_start_x = cell_width * col
        text_pos_x = cell_width * col + swatch_width + 7

        ax.text(
            text_pos_x,
            y,
            name,
            fontsize=14,
            horizontalalignment="left",
            verticalalignment="center",
        )

        ax.add_patch(
            Rectangle(
                xy=(swatch_start_x, y - 9),
                width=swatch_width,
                height=18,
                facecolor=colors[name],
                edgecolor="0.7",
            )
        )

    return fig


def cleanup_maxquant(
    maxquant_df: pd.DataFrame,
    remove_decoys: bool = True,
    how_duplicates: Literal[
        "keep_all", "keep_highest_int", "keep_one"
    ] = "keep_highest_int",
):
    """clean up the maxquant experiment file, remove decoys and duplicates"""
    if remove_decoys:
        n_pre_clean = maxquant_df.shape[0]
        maxquant_df = maxquant_df.loc[maxquant_df["Reverse"] != "+", :]
        n_post_clean = maxquant_df.shape[0]
        Logger.info(
            "Removing %s decoys from file, %s entries left.",
            n_pre_clean - n_post_clean,
            n_post_clean,
        )

    match how_duplicates:
        case "keep_all":
            pass
        case "keep_highest_int":
            n_pre_clean = maxquant_df.shape[0]
            maxquant_df = maxquant_df.sort_values(
                by=["Modified sequence", "Charge", "Intensity"], ascending=False
            ).drop_duplicates(subset=["Modified sequence", "Charge"], keep="first")
            n_post_clean = maxquant_df.shape[0]
            Logger.info(
                "Removing %s duplicate entries from experiment file, %s entries left.",
                n_pre_clean - n_post_clean,
                n_post_clean,
            )
        case "keep_one":
            maxquant_df = maxquant_df.drop_duplicates(
                subset=["Modified sequence", "Charge"]
            )
        case _:
            raise ValueError(f"Unknown option {how_duplicates}")
    return maxquant_df


def _merge_activation_results(
    processed_scan_dict: dict, ref_id: pd.Series, n_ms1scans: int
):
    """Merge the activation results."""
    activation = pd.DataFrame(index=ref_id, columns=range(n_ms1scans))
    precursor_scan_cos_dist = pd.DataFrame(index=ref_id, columns=range(n_ms1scans))
    precursor_collinear_sets = pd.DataFrame(index=ref_id, columns=range(n_ms1scans))
    scan_record_list = []
    for scan_idx, result_dict_scan in processed_scan_dict.items():
        if result_dict_scan["activation"] is not None:
            activation.loc[result_dict_scan["activation"]["precursor"], scan_idx] = (
                result_dict_scan["activation"]["activation"]
            )
        if result_dict_scan["precursor_cos_dist"] is not None:
            precursor_scan_cos_dist.loc[
                result_dict_scan["precursor_cos_dist"]["precursor"], scan_idx
            ] = result_dict_scan["precursor_cos_dist"]["cos_dist"]
        if result_dict_scan["precursor_collinear_sets"] is not None:
            precursor_collinear_sets.loc[
                result_dict_scan["precursor_collinear_sets"]["precursor"], scan_idx
            ] = result_dict_scan["precursor_collinear_sets"]["collinear_candidates"]
        scan_record_list.append(result_dict_scan["scans_record"])
    scan_record = pd.DataFrame(
        scan_record_list,
        columns=[
            "Scan",
            "Time",
            "CandidatePrecursorByRT",
            "FilteredPrecursor",
            "NumberHighlyCorrDictCandidate",
            "BestAlpha",
            "Cosine Dist",
            "IntensityExplained",
        ],
    )
    return activation, precursor_scan_cos_dist, scan_record, precursor_collinear_sets


def _perc_fmt(x, total):
    return f"{x:.1f}%\n{total * x / 100:.0f}"


def ExtractPeak(
    x: np.ndarray,
    y: np.ndarray,
    rel_height: float = 0.75,
    distance=None,
    prominence=None,
    return_summary: bool = False,
):
    peaks, _ = find_peaks(y, height=0, prominence=prominence, distance=distance)
    (peakWidth, peakHeight, left, right) = peak_widths(y, peaks, rel_height=rel_height)
    left = np.round(left, decimals=0).astype(int)
    right = np.round(right, decimals=0).astype(int)
    left_mz = x[left]
    right_mz = x[right]
    peak_intensity = [y[i : j + 1].sum() for (i, j) in zip(left, right)]
    peak_result = pd.DataFrame(
        {
            "apex_mzidx": peaks,
            "apex_mz": x[peaks],
            "start_mzidx": left,
            "start_mz": left_mz,
            "end_mzidx": right,
            "end_mz": right_mz,
            "peak_width": right_mz - left_mz,
            "peak_height": peakHeight,
            "peak_intensity_sum": peak_intensity,
        }
    )
    if return_summary:
        return (
            peak_result,
            peak_result.shape[0],
            peak_result.peak_width.mean(),
            peak_result.peak_width.std(),
        )
    else:
        return peak_result


def match_time_to_scan(
    df: pd.DataFrame, time_cols: List[str], ms1scans_no_array: pd.DataFrame
):
    """Match the time values in the dataframe to the scan numbers in the ms1scans_no_array"""
    # Build a KDTree from the starttime values
    tree = cKDTree(ms1scans_no_array[["starttime"]])

    for time_col in time_cols:
        time_scan_col = time_col + "_scan"

        # Query the tree for the nearest neighbor to each time value
        _, indices = tree.query(df[[time_col]].values)

        # Use the indices to get the corresponding scan numbers
        df[time_scan_col] = ms1scans_no_array.iloc[indices]["scan_number"].values

    return df


def load_mzml(msconvert_file: str):
    """
    read data from mzml format

    :msconvert_file: filepath to mzml
    """
    if msconvert_file.endswith(".pkl"):
        msconvert_file_base = msconvert_file[:-3]
        Logger.info("Reading pickle file")
        df_ms1 = pd.read_pickle(msconvert_file)
    elif msconvert_file.endswith(".mzML"):
        msconvert_file_base = msconvert_file[:-4]
        Logger.info("Reading mzML file")
        ind, mslev, bpmz, bpint, starttime, mzarray, intarray = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        with mzml.read(msconvert_file) as reader:
            for each_dict in reader:
                if each_dict["ms level"] == 1:
                    ind.append(each_dict["index"])
                    bpmz.append(each_dict["base peak m/z"])
                    bpint.append(each_dict["base peak intensity"])
                    mzarray.append(each_dict["m/z array"])
                    intarray.append(each_dict["intensity array"])
                    v_dict = each_dict["scanList"]
                    v_dict = v_dict["scan"][0]
                    starttime.append(v_dict["scan start time"])

        mslev = [1] * len(ind)
        mzarray = [x.tolist() for x in mzarray]
        intarray = [x.tolist() for x in intarray]
        col_set = ["ind", "mslev", "bpmz", "bpint", "starttime", "mzarray", "intarray"]
        df_ms1 = pd.DataFrame(
            list(zip(ind, mslev, bpmz, bpint, starttime, mzarray, intarray)),
            columns=col_set,
        )
        Logger.info("Saving data to pickle file")
        df_ms1.to_pickle(msconvert_file[:-5] + ".pkl")

    else:
        raise ValueError("File format not supported")
    if not os.path.isfile(msconvert_file_base + "_MS1Scans_NoArray.csv"):
        ms1cans_no_array = df_ms1.iloc[:, 1:5].copy()
        ms1cans_no_array.to_csv(msconvert_file_base + "_MS1Scans_NoArray.csv", index=0)
    return df_ms1


def write_df_to_fasta(df: pd.DataFrame, id_col: str, seq_col: str, fasta_path: str):
    with open(fasta_path, "w", encoding="utf-8") as f:
        for idx, row in df[[id_col, seq_col]].iterrows():
            f.write(f">{row[id_col]}\n{row[seq_col]}\n")


def write_fasta_to_df(fasta_path: str):
    with open(fasta_path) as fasta_file:
        sequence = []
        for title, seq in SimpleFastaParser(fasta_file):
            # identifiers.append(title.split(None, 1)[0])  # First word is ID
            sequence.append(seq)

    df = pd.Series(sequence)
    return df


def save_sparse_csr(filename, array):
    # note that .npz extension is added automatically
    np.savez(
        filename,
        data=array.data,
        indices=array.indices,
        indptr=array.indptr,
        shape=array.shape,
    )


def load_sparse_csr(filename):
    # here we need to add .npz extension manually
    loader = np.load(filename + ".npz")
    return csr_matrix(
        (loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"]
    )


def jaccard_similarity_from_peak_results(row) -> float:
    seq1 = range(row[["start_scan", "end_scan"]][0], row[["start_scan", "end_scan"]][1])
    seq2 = range(
        row[
            [
                "Calibrated retention time start_scan",
                "Calibrated retention time finish_scan",
            ]
        ][0],
        row[
            [
                "Calibrated retention time start_scan",
                "Calibrated retention time finish_scan",
            ]
        ][1],
    )
    intersection = len(set(seq1).intersection(set(seq2)))
    union = len(set(seq1).union(set(seq2)))
    return intersection / union
