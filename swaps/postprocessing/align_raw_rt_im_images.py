"""Standalone diagnostic: run match_features.py's own template-matching
alignment + consensus averaging + watershed segmentation on RAW RT x IM
images (plot_raw_rt_im_image._raw_rt_im_image) instead of the pipeline's
usual activation-matrix images.

Reuses match_features.py's align_images_to_reference / segment_consensus_
from_aligned / _visualize_consensus_bundle completely unmodified -- only the
image SOURCE differs. Anchors follow the same policy as match_features_batch's
own (nested, non-importable) _positional_anchors closure: only the reference
run and any Quant_Only run get a real (row, col) anchor (their own MS/MS-
observed apex, dict_ref's "{run}_MS1_frame_idx_exp"/"{run}_mobility_values_
index_exp"); every other run is aligned by template matching alone.
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
import yaml

from utils.config import get_cfg_defaults, merge_cfg_from_file
from utils.singleton_swaps_optimization import swaps_optimization_cfg
from utils.ims_utils import load_dotd_data, export_im_and_ms1scans

from .plot_raw_rt_im_image import _raw_file_dot_d_paths, _raw_rt_im_image, _resolve_iso_mz
from .match_features import (
    align_images_to_reference,
    segment_consensus_from_aligned,
    _visualize_consensus_bundle,
    _make_grid_fig,
    _denoise_kwargs_for_stage,
    _log_transform_enabled,
)

Logger = logging.getLogger(__name__)


def _visualize_match_score_maps(
    alignment_state,
    labels: list[str],
    output_dir: str,
    filename: str,
    max_cols: int = 5,
) -> None:
    """Plot the raw skimage.feature.match_template correlation surface for
    every non-reference run that went through template matching (i.e. one
    panel per alignment_state.match_score_maps[k], titled with the run label
    match_score_label_indices[k] resolves to), with a marker at
    match_score_peaks[k] -- the (row, col) top-left position the search
    picked as its best match. Diagnostic for "why did template matching lock
    onto the wrong feature": a flat/multi-modal surface, or a competing peak
    elsewhere with a similar or higher score, explains a misalignment that
    the aligned-image panels alone don't show directly.
    """
    import math

    maps = alignment_state.match_score_maps
    peaks = alignment_state.match_score_peaks
    label_indices = alignment_state.match_score_label_indices
    n = len(maps)
    if n == 0:
        Logger.warning(
            "No match-score maps to plot for %s (alignment skipped for every run).",
            filename,
        )
        return

    fig, axes = _make_grid_fig(n, max_cols)
    im = None
    for k in range(n):
        row, col = divmod(k, max_cols)
        ax = axes[row, col]
        im = ax.imshow(maps[k], aspect="auto", origin="upper", cmap="viridis")
        peak_row, peak_col = peaks[k]
        ax.plot(
            peak_col,
            peak_row,
            marker="+",
            color="red",
            markersize=12,
            markeredgewidth=2,
        )
        run_idx = label_indices[k]
        ax.set_title(
            f"{labels[run_idx]}\nscore={alignment_state.max_scores[run_idx]:.3f}",
            fontsize=6,
        )
        ax.set_xlabel("IM (template top-left col)")
        ax.set_ylabel("RT (template top-left row)")

    n_rows = math.ceil(n / max_cols)
    for empty in range(n, n_rows * max_cols):
        row, col = divmod(empty, max_cols)
        axes[row, col].set_visible(False)

    fig.colorbar(
        im, ax=axes.ravel().tolist(), shrink=0.6, label="Normalized cross-correlation"
    )
    fig.suptitle("Template-match correlation surfaces (red + = chosen shift)")
    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    Logger.info("Saved %s", out_path)


def _reference_match_quant_roles(row: pd.Series) -> tuple[str, set[str]]:
    """(reference_raw_file, quant_only_raw_files) -- same role-selection
    logic as match_features_batch's own (nested) _reference_match_quant_files
    closure in match_features.py:917."""
    str_values = row[row.map(lambda x: isinstance(x, str))]
    reference_matches = str_values.index[str_values == "Reference"]
    if len(reference_matches) == 0:
        raise ValueError(f"No raw file has 'Reference' status for mz_rank={row.name}")
    reference_raw_file = str(reference_matches[0])
    quant_only_raw_files = set(str_values.index[str_values == "Quant_Only"].tolist())
    return reference_raw_file, quant_only_raw_files


def _local_anchor(
    row: pd.Series,
    run_name: str,
    frame_ids: np.ndarray,
    im_values: np.ndarray,
    ms1scans: pd.DataFrame,
    mobility_values_df: pd.DataFrame,
) -> Optional[tuple[int, int]]:
    """This run's own MS/MS-observed apex, as a (row, col) pixel coordinate
    local to a raw image whose rows/cols are frame_ids/im_values (same
    construction as _raw_rt_im_image) -- mirrors what
    helper.get_pept_act_from_parquet returns as (rt_exp_center, im_exp_center),
    resolved via real RT/IM values since here the crop is dict_ref's generic
    RT_search/IM_search window rather than a per-run frame-index crop.

    When the exp position falls outside this run's own window (real per-run
    RT/IM drift can push a run's own apex beyond the cross-run RT_search/
    IM_search band), falls back to the window's own center -- same fallback
    get_pept_act_from_parquet uses (helper.py:296-301) -- rather than
    clipping to the nearest edge, which would otherwise pin the anchor to a
    window boundary far from the real peak.
    """
    exp_frame_idx = row.get(f"{run_name}_MS1_frame_idx_exp")
    exp_im_idx = row.get(f"{run_name}_mobility_values_index_exp")
    if exp_frame_idx is None or exp_im_idx is None:
        return None
    if pd.isna(exp_frame_idx) or pd.isna(exp_im_idx):
        return None
    if len(frame_ids) == 0 or len(im_values) == 0:
        return None
    exp_frame_id = ms1scans.loc[int(exp_frame_idx), "Id"]
    exp_mobility = mobility_values_df.loc[int(exp_im_idx), "mobility_values"]
    row_idx = np.searchsorted(frame_ids, exp_frame_id)
    row_idx = int(row_idx) if 0 <= row_idx < len(frame_ids) else len(frame_ids) // 2
    col_idx = np.searchsorted(im_values, exp_mobility)
    col_idx = int(col_idx) if 0 <= col_idx < len(im_values) else len(im_values) // 2
    return (row_idx, col_idx)


def align_and_segment_raw_images(
    swaps_dir: str,
    mz_ranks: Sequence[int],
    output_dir: str,
    config_path: Optional[str] = None,
    ppm_tol: Optional[float] = None,
    delta_im: float = 0.0,
    isotope_ranks: Sequence[int] = (1,),
    template_frac: float = 0.3,
    align_in_log_space: bool = True,
    top_intensity_frac: float = 1.0,
    log_transform: bool = False,
    plot_match_scores: bool = True,
) -> None:
    """For each mz_rank: build a raw RT x IM image per raw file (same
    construction as plot_raw_rt_im_image.plot_raw_rt_im_images), summing one
    image per requested isotope rank (same RT/IM window, different m/z
    window each) into a single per-run image, then run match_features.py's
    own alignment/consensus/watershed pipeline on those (summed) raw images
    and visualize with its own _visualize_consensus_bundle.

    Parameters mirror plot_raw_rt_im_image.plot_raw_rt_im_images (swaps_dir,
    mz_ranks, output_dir, config_path, ppm_tol, delta_im, log_transform);
    isotope_ranks generalizes plot_raw_rt_im_image's single isotope_rank to a
    list -- e.g. (1, 2) sums the monoisotopic and second-most-abundant
    isotopologue images per run before alignment (see _resolve_iso_mz for how
    each rank resolves to a target m/z). template_frac/align_in_log_space/
    top_intensity_frac are passed straight through to align_images_to_reference:
    template_frac is the fraction of the reference image's own size used as
    the template-match search patch (default 0.3, matches match_features_
    batch's own default); align_in_log_space runs the template-matching
    correlation itself on log2(1+x)-transformed images (search space only --
    returned aligned_images/consensus stay linear) and defaults to True,
    matching sbs_runner_ims.py's own MATCH_FEATURES_KWARGS.align_in_log_space
    default; top_intensity_frac (default 1.0 = no filtering, matching
    match_features_batch's own MATCH_FEATURES_KWARGS.template_match_top_
    intensity_frac default) restricts the template-match search to only the
    top `frac * (reference image's own pixel count)` pixels by intensity in
    both the template and every run's search image -- useful on raw images,
    where a persistent low-intensity background feature (absent from the
    sparse activation matrix this alignment code normally runs on) can
    otherwise dominate the correlation over a real, but more localized,
    elution peak; see _resolve_top_intensity_pixel_count/
    _keep_top_n_intensity_pixels in match_features.py.
    plot_match_scores (default True) additionally saves each mz_rank's raw
    match_template correlation surfaces (see _visualize_match_score_maps) --
    a diagnostic for why a run's alignment locked onto a particular shift.
    """
    isotope_ranks = list(isotope_ranks)
    if not isotope_ranks:
        raise ValueError("isotope_ranks must contain at least one rank.")
    os.makedirs(output_dir, exist_ok=True)
    cfg = get_cfg_defaults(swaps_optimization_cfg)
    merge_cfg_from_file(
        cfg, config_path or os.path.join(swaps_dir, "effective_config.yaml")
    )
    ppm_tol = float(ppm_tol) if ppm_tol is not None else float(cfg.PREPARE_DICT.PPM_TOL)

    # Same conversion sbs_runner_ims.py uses (build_consensus_feature_bundle/
    # match_features_batch's own denoise_kwargs/watershed_kwargs source) --
    # smooth (gaussian blur) -> clean (remove small objects) -> log2(1+x),
    # applied to the consensus MEAN (not per-image), matching production's
    # "log of mean" order exactly. Without this, segment_consensus_from_aligned
    # falls back to its own bare defaults (no smoothing/cleaning/log, and a
    # much larger h_rel), which is why watershed was oversegmenting earlier.
    processing_kwargs = yaml.safe_load(cfg.MATCH_FEATURES_KWARGS.dump())
    denoise_cfg = dict(processing_kwargs.get("denoise") or {})
    consensus_denoise_kwargs = {
        **_denoise_kwargs_for_stage(denoise_cfg, "consensus"),
        "log_transform": _log_transform_enabled(denoise_cfg),
    }
    watershed_kwargs = dict(processing_kwargs.get("peak_consensus_kwargs") or {})

    dict_ref = pd.read_pickle(os.path.join(swaps_dir, "dict_ref_with_activation.pkl"))
    dict_ref_by_mz = dict_ref.drop_duplicates("mz_rank").set_index("mz_rank")
    mz_ranks = [mz for mz in mz_ranks if mz in dict_ref_by_mz.index]
    missing = set(mz_ranks) - set(dict_ref_by_mz.index)
    if missing:
        Logger.warning("mz_rank(s) not found in dict_ref, skipping: %s", sorted(missing))

    roles_by_mz: dict[int, tuple[str, set[str]]] = {}
    target_mzs_by_mz: dict[int, list[float]] = {}
    for mz_rank in mz_ranks:
        row = dict_ref_by_mz.loc[mz_rank]
        try:
            roles_by_mz[mz_rank] = _reference_match_quant_roles(row)
        except ValueError as e:
            Logger.warning("%s -- skipping.", e)
            continue
        target_mzs_by_mz[mz_rank] = [
            _resolve_iso_mz(row, rank) for rank in isotope_ranks
        ]
    mz_ranks = [mz for mz in mz_ranks if mz in roles_by_mz]

    raw_file_paths = _raw_file_dot_d_paths(cfg)
    if not raw_file_paths:
        raise ValueError(f"No .d files found under cfg.DATA_PATH: {list(cfg.DATA_PATH)}")

    # (raw_file, img, extent, anchor) per mz_rank, collected across raw files
    panels_by_mz: dict[int, list[tuple]] = {mz: [] for mz in mz_ranks}

    for raw_file, dot_d_path in raw_file_paths.items():
        Logger.info("Loading %s", dot_d_path)
        data, _ = load_dotd_data(dot_d_path, swaps_result_dir=cfg.EXPORT_DATA_HDF5_DIR)
        ms1scans, mobility_values_df = export_im_and_ms1scans(
            data, swaps_result_dir=None
        )

        for mz_rank in mz_ranks:
            row = dict_ref_by_mz.loc[mz_rank]
            target_mzs = target_mzs_by_mz[mz_rank]
            rt_min, rt_max = float(row["RT_search_left"]), float(row["RT_search_right"])
            im_min = float(row["IM_search_left"]) - delta_im
            im_max = float(row["IM_search_right"]) + delta_im

            # One raw image per requested isotope rank (same RT/IM window,
            # own m/z window each), summed into a single per-run image before
            # alignment -- extent is identical across ranks since the window
            # doesn't depend on m/z, so it's safe to keep just the last one.
            img = None
            extent = None
            for target_mz in target_mzs:
                rank_img, extent = _raw_rt_im_image(
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
                img = rank_img if img is None else img + rank_img
            if img.shape[0] == 0 or img.shape[1] == 0:
                Logger.warning(
                    "Empty RT/IM window for mz_rank=%s, raw_file=%s -- skipping.",
                    mz_rank,
                    raw_file,
                )
                continue

            reference_raw_file, quant_only_raw_files = roles_by_mz[mz_rank]
            anchor = None
            if raw_file == reference_raw_file or raw_file in quant_only_raw_files:
                frame_ids = ms1scans.loc[
                    ms1scans["Time_minute"].between(rt_min, rt_max), "Id"
                ].to_numpy()
                im_values = mobility_values_df.loc[
                    mobility_values_df["mobility_values"].between(im_min, im_max),
                    "mobility_values",
                ].to_numpy()
                anchor = _local_anchor(
                    row, raw_file, frame_ids, im_values, ms1scans, mobility_values_df
                )

            panels_by_mz[mz_rank].append((raw_file, img, extent, anchor))

    for mz_rank, panels in panels_by_mz.items():
        if not panels:
            Logger.warning(
                "No raw images for mz_rank=%s across any raw file -- skipping.", mz_rank
            )
            continue
        reference_raw_file, _ = roles_by_mz[mz_rank]
        # Reference run first -- matches match_features_batch's own
        # _consensus_raw_files = [reference_raw_file] + match_raw_files
        # ordering, and align_images_to_reference's reference_idx=0 default.
        panels = sorted(panels, key=lambda p: p[0] != reference_raw_file)
        labels = [p[0] for p in panels]
        images = [p[1] for p in panels]
        anchors = [p[3] for p in panels]

        alignment_state = align_images_to_reference(
            images,
            reference_idx=0,
            anchors=anchors,
            template_frac=template_frac,
            align_in_log_space=align_in_log_space,
            top_intensity_frac=top_intensity_frac,
        )
        segmentation_state = segment_consensus_from_aligned(
            alignment_state,
            denoise_kwargs=consensus_denoise_kwargs,
            watershed_kwargs=watershed_kwargs,
        )

        iso_suffix = (
            "" if isotope_ranks == [1] else "_iso" + "+".join(map(str, isotope_ranks))
        )
        topfrac_suffix = (
            ""
            if top_intensity_frac >= 1.0
            else f"_top{top_intensity_frac}".replace(".", "p")
        )
        run_suffix = iso_suffix + topfrac_suffix
        filename = f"mz{mz_rank}{run_suffix}_raw_align_consensus.png"
        _visualize_consensus_bundle(
            alignment_state,
            segmentation_state,
            fig_dir=output_dir,
            filename=filename,
            labels=labels,
            log_transform_display=log_transform,
        )
        Logger.info("Saved %s", os.path.join(output_dir, filename))

        if plot_match_scores:
            _visualize_match_score_maps(
                alignment_state,
                labels,
                output_dir,
                f"mz{mz_rank}{run_suffix}_match_scores.png",
            )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run match_features.py's own template-matching alignment + "
            "consensus averaging + watershed segmentation on raw RT x IM "
            "images (instead of the activation matrix), one figure per "
            "mz_rank."
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
        "--isotope-ranks",
        default="1",
        help=(
            "Comma-separated isotopologue ranks to sum before alignment, "
            "ranked by abundance: 1 (default) is monoisotopic; 2, 3, ... "
            "step down through progressively less abundant isotopes from "
            "dict_ref's IsoMZ/IsoAbundance envelope. e.g. '1,2' sums the "
            "monoisotopic and second-most-abundant isotope images per run."
        ),
    )
    parser.add_argument(
        "--template-frac",
        type=float,
        default=0.3,
        help="Fraction of the reference image used as the template-match search patch",
    )
    parser.add_argument(
        "--no-align-in-log-space",
        dest="align_in_log_space",
        action="store_false",
        help=(
            "Run template-matching correlation on linear intensity instead of "
            "log2(1+x) (default: log space, matching sbs_runner_ims.py's own "
            "MATCH_FEATURES_KWARGS.align_in_log_space default)."
        ),
    )
    parser.add_argument(
        "--top-intensity-frac",
        type=float,
        default=1.0,
        help=(
            "Restrict template-matching to only the top fraction (by "
            "intensity) of pixels in the template and every search image; "
            "1.0 (default) = no filtering. Useful on raw images, where a "
            "persistent low-intensity background can otherwise dominate the "
            "correlation over a real, more localized elution peak."
        ),
    )
    parser.add_argument(
        "--log-transform", action="store_true", help="Display log2(1+intensity)"
    )
    parser.add_argument(
        "--no-plot-match-scores",
        dest="plot_match_scores",
        action="store_false",
        help=(
            "Skip saving the per-run match_template correlation surfaces "
            "(default: saved alongside the alignment figure, as "
            "mz{rank}_match_scores.png)."
        ),
    )
    args = parser.parse_args()

    mz_ranks = [int(x) for x in args.mz_ranks.split(",") if x.strip()]
    isotope_ranks = [int(x) for x in args.isotope_ranks.split(",") if x.strip()]
    align_and_segment_raw_images(
        args.swaps_dir,
        mz_ranks,
        args.output_dir,
        config_path=args.config_path,
        ppm_tol=args.ppm_tol,
        delta_im=args.delta_im,
        isotope_ranks=isotope_ranks,
        template_frac=args.template_frac,
        align_in_log_space=args.align_in_log_space,
        top_intensity_frac=args.top_intensity_frac,
        log_transform=args.log_transform,
        plot_match_scores=args.plot_match_scores,
    )


if __name__ == "__main__":
    main()
