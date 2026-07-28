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
from scipy.signal import find_peaks, peak_widths
from scipy.spatial import cKDTree  # type: ignore
from pathlib import Path

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
        "keep_all", "keep_highest_int", "keep_one", "keep_highest_score"
    ] = "keep_highest_int",
    id_cols=["Modified sequence", "Charge"],
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
                by=id_cols + ["Intensity"], ascending=False
            ).drop_duplicates(subset=id_cols, keep="first")
            n_post_clean = maxquant_df.shape[0]
            Logger.info(
                "Removing %s duplicate entries from experiment file, %s entries left.",
                n_pre_clean - n_post_clean,
                n_post_clean,
            )
        case "keep_one":
            maxquant_df = maxquant_df.drop_duplicates(subset=id_cols)
        case "keep_highest_score":
            n_pre_clean = maxquant_df.shape[0]
            maxquant_df["Score"].fillna(
                -1, inplace=True
            )  # Fill NaN scores with 0 for comparison
            maxquant_df = maxquant_df.sort_values(
                by=id_cols + ["Score"], ascending=False
            ).drop_duplicates(subset=id_cols, keep="first")
            n_post_clean = maxquant_df.shape[0]
            Logger.info(
                "Removing %s duplicate entries from experiment file, %s entries left.",
                n_pre_clean - n_post_clean,
                n_post_clean,
            )
        case _:
            raise ValueError(f"Unknown option {how_duplicates}")
    return maxquant_df


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


def load_mzml(msconvert_file: str, unify_format: bool = False) -> pd.DataFrame:
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
        if unify_format:
            df_ms1.rename(
                mapper={
                    "starttime": "Time_minute",
                    "ind": "Id",
                    "bpint": "MaxIntensity",
                },
                axis=1,
                inplace=True,
            )
            df_ms1["MS1_frame_idx"] = range(len(df_ms1))
        Logger.info("Saving data to pickle file")
        df_ms1.to_pickle(msconvert_file[:-5] + ".pkl")

    else:
        raise ValueError("File format not supported")
    if not os.path.isfile(msconvert_file_base + "_MS1Scans_NoArray.csv"):
        ms1cans_no_array = df_ms1.iloc[:, 1:5].copy()
        ms1cans_no_array.to_csv(
            path_or_buf=msconvert_file_base + "_MS1Scans_NoArray.csv", index=False
        )
    return df_ms1


def get_dot_d_paths(root_path, exclude_list):
    found_paths = []
    root = Path(root_path)

    for item in root.iterdir():
        # Skip excluded names
        if item.name in exclude_list:
            continue

        # If it's a .d directory, add it and STOP going deeper in this branch
        if item.is_dir() and item.suffix == ".d":
            found_paths.append(str(item.absolute()))

        # If it's a regular directory, recurse deeper
        elif item.is_dir():
            found_paths.extend(get_dot_d_paths(item, exclude_list))
    Logger.info(
        "Found %d .d directories in %s after excluding %s",
        len(found_paths),
        root_path,
        exclude_list,
    )
    return found_paths


def report_snap_log_collection(
    snap_log_collection: dict, quant_dir: str
) -> pd.DataFrame | None:
    no_seg_count = sum(
        1 for v in snap_log_collection.values() if v.get("no_seg_log") is not None
    )
    discard_count = sum(
        1 for v in snap_log_collection.values() if v.get("discard_record")
    )
    Logger.info(
        "snap_log_collection: total mz_ranks=%d | no_seg_log=%d | discard_record=%d",
        len(snap_log_collection),
        no_seg_count,
        discard_count,
    )

    rows = []
    for mz_rank, v in snap_log_collection.items():
        jal = v.get("jump_anchor_log")
        if not jal:
            continue
        for run_name, entry in jal.items():
            anchor = entry["anchor"]
            nlp = entry["nearest_labeled_pixel"]
            rows.append(
                {
                    "mz_rank": mz_rank,
                    "run_name": run_name,
                    "dist_to_label": entry["dist_to_label"],
                    "rt_dist": abs(anchor[0] - nlp[0]),
                    "im_dist": abs(anchor[1] - nlp[1]),
                }
            )

    if not rows:
        Logger.info("jump_anchor_log: no jump events recorded")
        return None

    df_jump = pd.DataFrame(rows)
    arr_total = df_jump["dist_to_label"].to_numpy()
    arr_rt = df_jump["rt_dist"].to_numpy()
    arr_im = df_jump["im_dist"].to_numpy()
    Logger.info(
        "jump_anchor_log: n=%d | total dist mean=%.2f median=%.2f max=%.2f"
        " | rt dist mean=%.2f median=%.2f max=%.2f"
        " | im dist mean=%.2f median=%.2f max=%.2f",
        len(arr_total),
        arr_total.mean(),
        np.median(arr_total),
        arr_total.max(),
        arr_rt.mean(),
        np.median(arr_rt),
        arr_rt.max(),
        arr_im.mean(),
        np.median(arr_im),
        arr_im.max(),
    )

    for arr, label, fname in [
        (arr_total, "Total jump distance (px)", "jump_anchor_dist_total.png"),
        (arr_rt, "RT jump distance (rows)", "jump_anchor_dist_rt.png"),
        (arr_im, "IM jump distance (cols)", "jump_anchor_dist_im.png"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(arr, bins=50, edgecolor="none")
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.set_title(f"{label} (n={len(arr)})")
        fig.tight_layout()
        fig.savefig(os.path.join(quant_dir, fname), dpi=300)
        plt.close(fig)

    return df_jump
