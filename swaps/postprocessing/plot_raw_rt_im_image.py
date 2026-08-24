"""Standalone diagnostic: plot the raw RT x IM intensity image around a
candidate's monoisotopic m/z, read straight from the raw .d file via
AlphaTims.

Unlike match_features.py's illustration images (built from the *activation*
matrix produced by scan-wise activation, see helper.get_pept_act_from_parquet),
this sums raw scan intensities directly -- no per-run RT/IM alignment,
watershed segmentation, or template matching. The RT/IM window is dict_ref's
own reference search window (RT_search_left/right, IM_search_left/right --
the same columns get_rt_im_range() in prepare_dict.py derives from the search
engine's own RT/IM apex + length statistics), optionally padded by delta_im
on each side.
"""

import argparse
import logging
import os
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.config import get_cfg_defaults, merge_cfg_from_file
from utils.singleton_swaps_optimization import swaps_optimization_cfg
from utils.tools import get_dot_d_paths
from utils.ims_utils import load_dotd_data, export_im_and_ms1scans

Logger = logging.getLogger(__name__)


def _raw_file_dot_d_paths(cfg) -> dict[str, str]:
    """raw_file name (as used elsewhere, e.g. activation dir names) -> .d path.

    Mirrors sbs_runner_ims.py's own raw_file_list derivation.
    """
    return {
        os.path.basename(dot_d_path).split(".")[0]: dot_d_path
        for data_path in cfg.DATA_PATH
        for dot_d_path in get_dot_d_paths(data_path, cfg.EXCLUDE_DATASET_NAME)
    }


def _resolve_iso_mz(row: pd.Series, isotope_rank: int) -> float:
    """isotope_rank=1 -> monoisotopic (dict_ref["m/z"], the theoretical value
    computed straight from the peptide sequence). isotope_rank>1 -> the
    isotope_rank-th most abundant isotopologue from dict_ref's own IsoMZ/
    IsoAbundance envelope. IsoMZ is sorted by mass, NOT abundance (fine
    isotopic structure interleaves composition types, e.g. a 13C-pair vs a
    15N/2H peak can land in either mass order regardless of abundance), so
    resolving "rank" requires re-sorting by IsoAbundance descending rather
    than indexing IsoMZ directly.
    """
    if isotope_rank < 1:
        raise ValueError(f"isotope_rank must be >= 1, got {isotope_rank}")
    if isotope_rank == 1:
        return float(row["m/z"])
    iso_mz = np.asarray(row["IsoMZ"], dtype=float)
    iso_abundance = np.asarray(row["IsoAbundance"], dtype=float)
    if isotope_rank > len(iso_mz):
        raise ValueError(
            f"isotope_rank={isotope_rank} exceeds the number of isotopes "
            f"available ({len(iso_mz)}) for mz_rank={row.name}."
        )
    order = np.argsort(-iso_abundance)
    return float(iso_mz[order[isotope_rank - 1]])


def _windowed_axes(
    ms1scans: pd.DataFrame,
    mobility_values_df: pd.DataFrame,
    rt_min: float,
    rt_max: float,
    im_min: float,
    im_max: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """MS1 frames / mobility bins within [rt_min,rt_max] x [im_min,im_max],
    ascending order -- the shared row/col axis definition both the raw image
    grid (_raw_rt_im_image) and anchor placement (_local_anchor) bin into.
    """
    rt_window = ms1scans.loc[ms1scans["Time_minute"].between(rt_min, rt_max)]
    im_window = mobility_values_df.loc[
        mobility_values_df["mobility_values"].between(im_min, im_max)
    ]
    return rt_window, im_window


def _local_anchor(
    rt_window: pd.DataFrame, im_window: pd.DataFrame, run_rt: float, run_im: float
) -> Optional[tuple[int, int]]:
    """Local (row, col) position of a run's own MS/MS-observed RT/IM within
    the raw image's row/col axes, clamped to the image bounds -- mirrors
    helper.get_pept_act_from_parquet's rt_exp_center/im_exp_center fallback
    behavior for an exp value that falls outside its own search window.
    """
    if len(rt_window) == 0 or len(im_window) == 0:
        return None
    rt_array = rt_window["Time_minute"].to_numpy()
    im_array = im_window["mobility_values"].to_numpy()
    row = int(np.clip(np.searchsorted(rt_array, run_rt), 0, len(rt_array) - 1))
    col = int(np.clip(np.searchsorted(im_array, run_im), 0, len(im_array) - 1))
    return row, col


def _raw_rt_im_image(
    data,
    ms1scans: pd.DataFrame,
    mobility_values_df: pd.DataFrame,
    target_mz: float,
    rt_min: float,
    rt_max: float,
    im_min: float,
    im_max: float,
    ppm_tol: float,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Sum raw intensities into an RT (rows) x IM (cols) grid, MS1 only,
    filtered to the given m/z (+/- ppm_tol), rt_min/max (minutes) and
    im_min/max (1/K0) window. Row/col order follows ms1scans/
    mobility_values_df (ascending RT, ascending 1/K0), matching the
    origin="lower" convention match_features.py's imshow calls use.
    """
    mz_tol_da = target_mz * ppm_tol * 1e-6
    mz_lo, mz_hi = target_mz - mz_tol_da, target_mz + mz_tol_da

    rt_window, im_window = _windowed_axes(
        ms1scans, mobility_values_df, rt_min, rt_max, im_min, im_max
    )
    frame_ids = rt_window["Id"].to_numpy()
    im_values = im_window["mobility_values"].to_numpy()
    extent = (
        (im_values[0], im_values[-1], rt_min, rt_max)
        if len(im_values)
        else (im_min, im_max, rt_min, rt_max)
    )
    img = np.zeros((len(frame_ids), len(im_values)))
    if len(frame_ids) == 0 or len(im_values) == 0:
        return img, extent

    hits = data[
        {
            "rt_values": slice(rt_min * 60, rt_max * 60),
            "mobility_values": slice(im_min, im_max),
            "precursor_indices": [0],
            "mz_values": slice(mz_lo, mz_hi),
        },
        "df",
    ]
    if len(hits) == 0:
        return img, extent

    row_of_frame = pd.Series(np.arange(len(frame_ids)), index=frame_ids)
    rows = row_of_frame.reindex(hits["frame_indices"].to_numpy()).to_numpy()
    valid = ~np.isnan(rows)
    cols = np.clip(
        np.searchsorted(im_values, hits["mobility_values"].to_numpy()[valid]),
        0,
        len(im_values) - 1,
    )
    np.add.at(
        img, (rows[valid].astype(int), cols), hits["intensity_values"].to_numpy()[valid]
    )
    return img, extent


def _plot_mz_rank_grid(
    mz_rank: int,
    target_mz: float,
    isotope_rank: int,
    panels: list[tuple[str, np.ndarray, tuple[float, float, float, float]]],
    output_dir: str,
    log_transform: bool,
    max_cols: int = 5,
) -> None:
    """One figure per mz_rank, one panel per raw file -- mirrors
    match_features.py's _visualize_consensus_bundle/_make_grid_fig grid-montage
    convention: shared vmin/vmax across panels, imshow(aspect="auto",
    origin="lower"), unused grid cells hidden.
    """
    import math

    n = len(panels)
    cols = min(max_cols, n)
    n_rows = math.ceil(n / cols)
    display_imgs = [np.log1p(img) if log_transform else img for _, img, _ in panels]
    vmin = min(img.min() for img in display_imgs)
    vmax = max(img.max() for img in display_imgs)

    fig, axes = plt.subplots(
        n_rows, cols, figsize=(4 * cols, 4 * n_rows), squeeze=False, constrained_layout=True
    )
    im = None
    for i, ((raw_file, _, extent), display_img) in enumerate(zip(panels, display_imgs)):
        row, col = divmod(i, cols)
        ax = axes[row, col]
        im = ax.imshow(
            display_img, aspect="auto", origin="lower", extent=extent, vmin=vmin, vmax=vmax
        )
        ax.set_title(raw_file, fontsize=4)
        ax.set_xlabel("IM (1/K0)")
        ax.set_ylabel("RT (min)")

    for empty in range(n, n_rows * cols):
        row, col = divmod(empty, cols)
        axes[row, col].set_visible(False)

    fig.colorbar(
        im,
        ax=axes.ravel().tolist(),
        shrink=0.6,
        label="log1p(intensity)" if log_transform else "Intensity",
    )
    iso_label = "monoisotopic" if isotope_rank == 1 else f"isotope rank {isotope_rank}"
    fig.suptitle(f"mz_rank={mz_rank}  m/z={target_mz:.4f}  ({iso_label})")
    suffix = "" if isotope_rank == 1 else f"_iso{isotope_rank}"
    out_path = os.path.join(output_dir, f"mz{mz_rank}{suffix}_raw_grid.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    Logger.info("Saved %s", out_path)


def plot_raw_rt_im_images(
    swaps_dir: str,
    mz_ranks: Sequence[int],
    output_dir: str,
    config_path: Optional[str] = None,
    ppm_tol: Optional[float] = None,
    delta_im: float = 0.0,
    isotope_rank: int = 1,
    log_transform: bool = False,
) -> None:
    """For each mz_rank, plot one grid figure (one panel per configured raw
    file) of the raw RT x IM intensity image -- mirrors match_features.py's
    per-peptide grid-montage illustrations.

    Parameters
    ----------
    swaps_dir : RESULT_PATH of a (completed or in-progress) sbs_runner_ims
        run; must contain dict_ref.pkl, and (unless config_path is given)
        effective_config.yaml.
    mz_ranks : candidate mz_rank values to plot (dict_ref["mz_rank"]).
    output_dir : directory PNGs are written to, one per mz_rank.
    config_path : overrides swaps_dir/effective_config.yaml as the config
        source for DATA_PATH/EXCLUDE_DATASET_NAME/PREPARE_DICT.PPM_TOL.
    ppm_tol : m/z tolerance around the target m/z; defaults to
        cfg.PREPARE_DICT.PPM_TOL.
    delta_im : extra +/- padding (1/K0 units) added on each side of
        dict_ref's own IM_search_left/right window.
    isotope_rank : which isotopologue to plot, ranked by abundance -- 1 (the
        default) is the monoisotopic peak (dict_ref["m/z"]); 2, 3, ... step
        down through progressively less abundant isotopes from dict_ref's
        IsoMZ/IsoAbundance envelope (see _resolve_iso_mz).
    log_transform : plot log1p(intensity) instead of raw intensity.
    """
    os.makedirs(output_dir, exist_ok=True)
    cfg = get_cfg_defaults(swaps_optimization_cfg)
    merge_cfg_from_file(
        cfg, config_path or os.path.join(swaps_dir, "effective_config.yaml")
    )
    ppm_tol = float(ppm_tol) if ppm_tol is not None else float(cfg.PREPARE_DICT.PPM_TOL)

    dict_ref = pd.read_pickle(os.path.join(swaps_dir, "dict_ref_with_activation.pkl"))
    dict_ref_by_mz = dict_ref.drop_duplicates("mz_rank").set_index("mz_rank")
    mz_ranks = list(mz_ranks)
    missing = [mz for mz in mz_ranks if mz not in dict_ref_by_mz.index]
    if missing:
        Logger.warning("mz_rank(s) not found in dict_ref, skipping: %s", missing)

    raw_file_paths = _raw_file_dot_d_paths(cfg)
    if not raw_file_paths:
        raise ValueError(f"No .d files found under cfg.DATA_PATH: {list(cfg.DATA_PATH)}")

    panels_by_mz: dict[int, list[tuple[str, np.ndarray, tuple]]] = {
        mz: [] for mz in mz_ranks if mz in dict_ref_by_mz.index
    }
    target_mz_by_mz: dict[int, float] = {
        mz: _resolve_iso_mz(dict_ref_by_mz.loc[mz], isotope_rank) for mz in panels_by_mz
    }

    for raw_file, dot_d_path in raw_file_paths.items():
        Logger.info("Loading %s", dot_d_path)
        data, _ = load_dotd_data(dot_d_path, swaps_result_dir=cfg.EXPORT_DATA_HDF5_DIR)
        ms1scans, mobility_values_df = export_im_and_ms1scans(
            data, swaps_result_dir=None
        )

        for mz_rank in panels_by_mz:
            row = dict_ref_by_mz.loc[mz_rank]
            target_mz = target_mz_by_mz[mz_rank]
            rt_min, rt_max = float(row["RT_search_left"]), float(row["RT_search_right"])
            im_min = float(row["IM_search_left"]) - delta_im
            im_max = float(row["IM_search_right"]) + delta_im

            img, extent = _raw_rt_im_image(
                data,
                ms1scans,
                mobility_values_df,
                target_mz,
                rt_min,
                rt_max,
                im_min,
                im_max,
                ppm_tol,
            )
            if img.shape[0] == 0 or img.shape[1] == 0:
                Logger.warning(
                    "Empty RT/IM window for mz_rank=%s, raw_file=%s -- skipping.",
                    mz_rank,
                    raw_file,
                )
                continue
            panels_by_mz[mz_rank].append((raw_file, img, extent))

    for mz_rank, panels in panels_by_mz.items():
        if not panels:
            Logger.warning(
                "No raw images for mz_rank=%s across any raw file -- skipping.", mz_rank
            )
            continue
        _plot_mz_rank_grid(
            mz_rank,
            target_mz_by_mz[mz_rank],
            isotope_rank,
            panels,
            output_dir,
            log_transform,
        )


def plot_raw_rt_im_matched(
    swaps_dir: str,
    mz_ranks: Sequence[int],
    output_dir: str,
    config_path: Optional[str] = None,
    ppm_tol: Optional[float] = None,
    delta_im: float = 0.0,
    isotope_rank: int = 1,
    log_transform_display: bool = False,
    template_frac: float = 0.3,
) -> None:
    """Run match_features.py's own template-matching alignment + consensus
    watershed segmentation on the RAW (pre-activation) RT x IM images
    (_raw_rt_im_image) instead of the pipeline's normal activation-matrix
    images -- a standalone experiment, match_features.py itself is untouched.

    align_images_to_reference/segment_consensus_from_aligned/
    _visualize_consensus_bundle (all in match_features.py) take only
    explicit arguments (no cfg/globals -- verified by inspection), so this
    is a drop-in swap of the image source.

    Per-run anchors come from dict_ref's own MS/MS-observed RT/IM for that
    run ("{run}_RT"/"{run}_1K0", pivoted from the search engine's evidence
    table in prepare_dict.py). A run with no MS/MS observation for this
    candidate has RT/IM stored as exactly 0.0 (pivot_psm_by_mz_rank's
    fillna sentinel, confirmed empirically -- a real observed RT is never
    exactly 0 given these short-gradient runs) and gets anchor=None,
    relying on template matching alone -- mirroring match_features_batch's
    own Reference/Quant_Only (anchored) vs Match/Not_Match (unanchored)
    policy, without needing that per-run role classification here.

    The reference run is the anchored run with the highest total raw
    intensity (falls back to the highest-intensity run overall if none are
    anchored) -- a simple stand-in for "best-quality run" (CLAUDE.md).

    Parameters mirror plot_raw_rt_im_images; template_frac is
    align_images_to_reference's own template-crop-fraction parameter.
    """
    from .match_features import (
        align_images_to_reference,
        segment_consensus_from_aligned,
        _visualize_consensus_bundle,
    )

    os.makedirs(output_dir, exist_ok=True)
    cfg = get_cfg_defaults(swaps_optimization_cfg)
    merge_cfg_from_file(
        cfg, config_path or os.path.join(swaps_dir, "effective_config.yaml")
    )
    ppm_tol = float(ppm_tol) if ppm_tol is not None else float(cfg.PREPARE_DICT.PPM_TOL)

    dict_ref = pd.read_pickle(os.path.join(swaps_dir, "dict_ref_with_activation.pkl"))
    dict_ref_by_mz = dict_ref.drop_duplicates("mz_rank").set_index("mz_rank")
    mz_ranks = list(mz_ranks)
    missing = [mz for mz in mz_ranks if mz not in dict_ref_by_mz.index]
    if missing:
        Logger.warning("mz_rank(s) not found in dict_ref, skipping: %s", missing)

    raw_file_paths = _raw_file_dot_d_paths(cfg)
    if not raw_file_paths:
        raise ValueError(f"No .d files found under cfg.DATA_PATH: {list(cfg.DATA_PATH)}")

    target_mz_by_mz = {
        mz: _resolve_iso_mz(dict_ref_by_mz.loc[mz], isotope_rank)
        for mz in mz_ranks
        if mz in dict_ref_by_mz.index
    }
    images_by_mz: dict[int, list[np.ndarray]] = {mz: [] for mz in target_mz_by_mz}
    anchors_by_mz: dict[int, list[Optional[tuple[int, int]]]] = {
        mz: [] for mz in target_mz_by_mz
    }
    labels_by_mz: dict[int, list[str]] = {mz: [] for mz in target_mz_by_mz}

    for raw_file, dot_d_path in raw_file_paths.items():
        Logger.info("Loading %s", dot_d_path)
        data, _ = load_dotd_data(dot_d_path, swaps_result_dir=cfg.EXPORT_DATA_HDF5_DIR)
        ms1scans, mobility_values_df = export_im_and_ms1scans(
            data, swaps_result_dir=None
        )

        for mz_rank, target_mz in target_mz_by_mz.items():
            row = dict_ref_by_mz.loc[mz_rank]
            rt_min, rt_max = float(row["RT_search_left"]), float(row["RT_search_right"])
            im_min = float(row["IM_search_left"]) - delta_im
            im_max = float(row["IM_search_right"]) + delta_im

            rt_window, im_window = _windowed_axes(
                ms1scans, mobility_values_df, rt_min, rt_max, im_min, im_max
            )
            img, _ = _raw_rt_im_image(
                data,
                ms1scans,
                mobility_values_df,
                target_mz,
                rt_min,
                rt_max,
                im_min,
                im_max,
                ppm_tol,
            )
            if img.shape[0] == 0 or img.shape[1] == 0:
                Logger.warning(
                    "Empty RT/IM window for mz_rank=%s, raw_file=%s -- skipping.",
                    mz_rank,
                    raw_file,
                )
                continue

            run_rt = float(row.get(f"{raw_file}_RT", 0.0))
            run_im = float(row.get(f"{raw_file}_1K0", 0.0))
            anchor = (
                _local_anchor(rt_window, im_window, run_rt, run_im)
                if run_rt > 0.0
                else None
            )

            images_by_mz[mz_rank].append(img)
            anchors_by_mz[mz_rank].append(anchor)
            labels_by_mz[mz_rank].append(raw_file)

    for mz_rank, images in images_by_mz.items():
        if not images:
            Logger.warning(
                "No raw images for mz_rank=%s across any raw file -- skipping.", mz_rank
            )
            continue
        anchors = anchors_by_mz[mz_rank]
        labels = labels_by_mz[mz_rank]
        anchored_idx = [i for i, a in enumerate(anchors) if a is not None]
        candidate_pool = anchored_idx or list(range(len(images)))
        reference_idx = max(candidate_pool, key=lambda i: images[i].sum())

        alignment_state = align_images_to_reference(
            images, reference_idx=reference_idx, anchors=anchors, template_frac=template_frac
        )
        segmentation_state = segment_consensus_from_aligned(
            alignment_state,
            denoise_kwargs={},
            watershed_kwargs={},
            consensus_image_indices=anchored_idx or None,
        )
        suffix = "" if isotope_rank == 1 else f"_iso{isotope_rank}"
        filename = f"mz{mz_rank}{suffix}_raw_matched.png"
        _visualize_consensus_bundle(
            alignment_state,
            segmentation_state,
            fig_dir=output_dir,
            filename=filename,
            labels=labels,
            log_transform_display=log_transform_display,
        )
        Logger.info(
            "Saved %s (reference=%s, %d/%d runs anchored)",
            filename,
            labels[reference_idx],
            len(anchored_idx),
            len(images),
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot the raw (pre-activation, pre-alignment, pre-segmentation) "
            "RT x IM intensity image for a list of dict_ref mz_ranks, one "
            "grid figure per mz_rank (one panel per configured raw file)."
        )
    )
    parser.add_argument("swaps_dir", help="RESULT_PATH containing dict_ref.pkl")
    parser.add_argument(
        "mz_ranks", help="Comma-separated mz_rank values, e.g. 1023,4587"
    )
    parser.add_argument("output_dir", help="Directory to write PNGs to")
    parser.add_argument(
        "--config-path",
        default=None,
        help="Overrides swaps_dir/effective_config.yaml",
    )
    parser.add_argument(
        "--ppm-tol", type=float, default=None, help="Overrides cfg.PREPARE_DICT.PPM_TOL"
    )
    parser.add_argument(
        "--delta-im",
        type=float,
        default=0.0,
        help="Extra +/- padding (1/K0 units) added to dict_ref's IM_search_left/right window",
    )
    parser.add_argument(
        "--isotope-rank",
        type=int,
        default=1,
        help=(
            "Which isotopologue to plot, ranked by abundance: 1 (default) is "
            "monoisotopic; 2, 3, ... step down through progressively less "
            "abundant isotopes from dict_ref's IsoMZ/IsoAbundance envelope."
        ),
    )
    parser.add_argument(
        "--log-transform", action="store_true", help="Plot log1p(intensity)"
    )
    parser.add_argument(
        "--matched",
        action="store_true",
        help=(
            "Instead of a plain per-run grid, run match_features.py's own "
            "template-matching alignment + consensus watershed segmentation "
            "on the raw images (plot_raw_rt_im_matched) and save one "
            "consensus-bundle figure per mz_rank."
        ),
    )
    parser.add_argument(
        "--template-frac",
        type=float,
        default=0.3,
        help="--matched only: align_images_to_reference's template-crop fraction (default 0.3)",
    )
    args = parser.parse_args()

    mz_ranks = [int(x) for x in args.mz_ranks.split(",") if x.strip()]
    if args.matched:
        plot_raw_rt_im_matched(
            args.swaps_dir,
            mz_ranks,
            args.output_dir,
            config_path=args.config_path,
            ppm_tol=args.ppm_tol,
            delta_im=args.delta_im,
            isotope_rank=args.isotope_rank,
            log_transform_display=args.log_transform,
            template_frac=args.template_frac,
        )
    else:
        plot_raw_rt_im_images(
            args.swaps_dir,
            mz_ranks,
            args.output_dir,
            config_path=args.config_path,
            ppm_tol=args.ppm_tol,
            delta_im=args.delta_im,
            isotope_rank=args.isotope_rank,
            log_transform=args.log_transform,
        )


if __name__ == "__main__":
    main()
