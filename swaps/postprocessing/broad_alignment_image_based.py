"""Image-based broad-alignment calibration: full-image phase-correlation RT/IM
registration between every raw-file pair (swaps.postprocessing.rt_im_image_registration),
as an alternative to broad_alignment.py's calibration-peptide/template-matching approach.

Writes a SEPARATE table, broad_alignment_shift_table_global_raw_image.parquet, alongside
(not replacing) broad_alignment.py's broad_alignment_shift_table.parquet -- both producer
(sbs_runner_ims.py) and consumer (match_features.py) are repointed at this new file when
the image-based method is used; the peptide-based code path and any pre-existing table are
left untouched.

Per raw-file pair: build (or reuse a cached) collapsed (rt, im) intensity image for each
run, sweep window_width x stride (see PIECEWISE_METHOD_BUILDERS in
rt_im_image_registration.py) plus the single global shift, and keep whichever candidate
minimizes the L1 registration residual as that pair's shift curve -- this correctly lets a
plain global shift win over any piecewise curve when the pair doesn't have enough
per-window signal to support local fitting (see rt_im_image_registration.py's
PairImages.use_global_prior / windowing_base_img). The reverse-direction pair's table rows
are derived by negating the forward curve (evaluated over the reverse pair's own native
frame range) instead of recomputing -- phase_cross_correlation(ref, mov) and
phase_cross_correlation(mov, ref) are shift-negations of each other by construction.

Unlike broad_alignment.py's calibration (which only needs dict_ref + Stage-2 activation
parquets), this reopens every raw .d file via AlphaTims to build the collapsed image --
Stage 2's activation output is a sparse per-candidate matrix, not a full-image heatmap.
That per-file load is parallelized and disk-cached (load_rt_im_image), so it only happens
once per raw file regardless of how many pairs reference it.

The table is keyed by MS1 frame index (not RT minutes): dict_ref already carries a
per-raw-file frame-index RT column, MS1_frame_idx_center_ref_<raw_file> (added by
dict_add_rt_index during Stage 2), which match_features.py looks up directly -- shift_rt/
shift_im themselves are already frame/pixel-index offsets on the same grid used everywhere
else (align_images_to_reference), so no unit conversion is needed either way.
"""

import argparse
import itertools
import logging
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import pandas as pd
import threadpoolctl

from utils.tools import get_dot_d_paths

from .broad_alignment import _TABLE_COLUMNS
from .rt_im_image_registration import load_rt_im_image, raw_file_rt_im_range_bounds, run_pair

logger = logging.getLogger(__name__)


def _limit_worker_blas_threads() -> None:
    """ProcessPoolExecutor initializer: OpenBLAS defaults to one thread pool PER
    PROCESS sized to the whole machine's core count (32 here), so N worker processes
    each independently try to use all 32 cores -- real oversubscription (measured load
    average exceeded core count, each worker showing 90-325% CPU instead of ~100%).
    Each pairwise registration only touches small already-cached 2D arrays, so a single
    BLAS thread per worker is plenty; capping this is what makes raising max_workers
    toward the full core count actually help instead of making things worse."""
    threadpoolctl.threadpool_limits(1)


def _resolve_raw_file_paths(raw_file_list: list[str], data_paths: list[str], exclude_dataset_names: list[str]) -> dict[str, str]:
    """Maps each raw_file_list entry (a RESULT_PATH subdirectory name) to its full
    .d path, using the exact same `os.path.basename(p).split(".")[0]` key derivation
    sbs_runner_ims.py uses to build raw_file_list in the first place -- guarantees the
    keys here match raw_file_list entries exactly, not rt_im_image_registration's own
    (differently-derived) raw_file_label helper."""
    dot_d_paths = []
    for data_path in data_paths:
        dot_d_paths.extend(get_dot_d_paths(data_path, exclude_dataset_names))
    by_name = {os.path.basename(p).split(".")[0]: p for p in dot_d_paths}
    missing = [rf for rf in raw_file_list if rf not in by_name]
    if missing:
        raise FileNotFoundError(f"Could not resolve .d path for raw files {missing} under data_paths={data_paths}")
    return {rf: by_name[rf] for rf in raw_file_list}


# No real RT/IM shift can plausibly exceed this many frames/scans -- a curve-fit
# artifact (e.g. a low-inlier-count spline overshooting between its knots, Runge's
# phenomenon-style) can occasionally produce an astronomically large finite value that
# doesn't show up in the aggregate L1 residual (only a few pixels along the curve are
# affected) but overflows pyarrow's int64 conversion when writing the table. Clip
# defensively rather than let one degenerate pair crash the whole calibration.
_MAX_ABS_SHIFT = 100_000


def _pair_table_rows(
    reference_run: str, matched_run: str, rt_range: np.ndarray, curve_values: np.ndarray, im_shift: float, is_fallback: bool = False
) -> list[dict]:
    """One row per frame index -- the curve already has per-frame resolution, so no
    RT-binning/sparse-bin-fallback logic is needed (unlike broad_alignment.py's
    peptide-sample-driven _build_pair_bins)."""
    clipped_curve = np.clip(curve_values, -_MAX_ABS_SHIFT, _MAX_ABS_SHIFT)
    used_indices = np.asarray(rt_range)
    n_clipped = int(np.sum(clipped_curve[used_indices] != curve_values[used_indices]))
    if n_clipped:
        logger.warning(
            "%s vs %s: clipped %d/%d out-of-range shift_rt values to +/-%d (likely a curve-fit overshoot artifact)",
            reference_run, matched_run, n_clipped, len(rt_range), _MAX_ABS_SHIFT,
        )
    shift_im_int = int(round(float(np.clip(im_shift, -_MAX_ABS_SHIFT, _MAX_ABS_SHIFT))))
    return [
        {
            "reference_run": reference_run,
            "matched_run": matched_run,
            "rt_bin_index": int(frame),
            "rt_bin_start": float(frame),
            "rt_bin_end": float(frame) + 1.0,
            "shift_rt": int(round(clipped_curve[frame])),
            "shift_im": shift_im_int,
            "n_samples": 1,
            "is_fallback": is_fallback,
        }
        for frame in rt_range
    ]


def _select_optimal_curve(pair, residuals_df: pd.DataFrame, curves_by_combo: dict) -> tuple[np.ndarray, float, str, float]:
    """Picks the single lowest-L1-residual candidate across Global shift (param-
    independent) and every (window_width, stride, method) piecewise curve. Returns
    (curve_values_total, im_shift, method_label, l1_residual)."""
    best_residual = pair.resid_after
    best_curve = np.full(pair.rt_range.shape, pair.global_rt_shift, dtype=float)
    best_im_shift = pair.global_im_shift
    best_label = "Global shift"

    piecewise = residuals_df[~residuals_df["method"].isin(["Unregistered", "Global shift"])]
    for _, row in piecewise.iterrows():
        if row["l1_residual"] >= best_residual:
            continue
        curve = curves_by_combo[(int(row["window_width"]), int(row["stride"]))][row["method"]]
        best_residual = float(row["l1_residual"])
        best_curve = curve
        best_im_shift = pair.global_im_shift
        best_label = f"{row['method']} (ww={row['window_width']}, stride={row['stride']})"
    return best_curve, best_im_shift, best_label, best_residual


def _process_pair(
    ref_run: str, mov_run: str, raw_file_paths: dict, cache_dir: str, pairs_dir: str,
    window_widths: tuple, strides: tuple, upsample_factor: int,
    rt_range_minutes: tuple[float, float] | None = None,
    im_range: tuple[float, float] | None = None,
) -> list[dict]:
    """One pair's full param sweep + optimal-curve selection + forward/reverse table
    rows. Module-level (not a closure) so it's picklable for ProcessPoolExecutor --
    processes, not threads, since run_param_combo does matplotlib plotting internally,
    which isn't safe to run concurrently across threads sharing pyplot's global state.
    Returns only the (small) row dicts, not the pair's full images/curves, to keep
    inter-process serialization cheap."""
    pair_out_dir = os.path.join(pairs_dir, f"{ref_run}__vs__{mov_run}")
    residuals_df, pair, curves_by_combo = run_pair(
        raw_file_paths[ref_run], raw_file_paths[mov_run], cache_dir, pair_out_dir,
        list(window_widths), list(strides), upsample_factor=upsample_factor, return_curves=True,
        rt_range_minutes=rt_range_minutes, im_range=im_range,
    )

    best_curve, best_im_shift, best_label, best_residual = _select_optimal_curve(pair, residuals_df, curves_by_combo)
    logger.info("Pair %s vs %s: chose %s (L1 residual=%.0f)", ref_run, mov_run, best_label, best_residual)

    forward_rt_range = np.arange(pair.ref_native_len)
    rows = _pair_table_rows(ref_run, mov_run, forward_rt_range, best_curve, best_im_shift, is_fallback=False)

    reverse_rt_range = np.arange(pair.mov_native_len)
    reverse_curve = -best_curve[: pair.mov_native_len]
    reverse_im_shift = -best_im_shift
    rows.extend(_pair_table_rows(mov_run, ref_run, reverse_rt_range, reverse_curve, reverse_im_shift, is_fallback=True))
    return rows


def _shared_rt_im_range(dict_ref_path: str | None) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    """Global RT (minutes)/IM (1/K0) range across the WHOLE dict_ref -- not
    per-peptide -- so every raw file's collapsed rt/im image is built from the
    same physical range (see load_rt_im_image's docstring). None/None when no
    dict_ref_path is given, keeping each run's old, unrestricted, whole-native
    image (e.g. for standalone use without a dict_ref on hand)."""
    if dict_ref_path is None:
        return None, None
    dict_ref = pd.read_pickle(dict_ref_path)
    rt_range_minutes = (
        float(dict_ref["RT_search_left"].min()),
        float(dict_ref["RT_search_right"].max()),
    )
    im_range = (
        float(dict_ref["IM_search_left"].min()),
        float(dict_ref["IM_search_right"].max()),
    )
    logger.info(
        "Shared rt/im image range from %s: rt=%.4f-%.4f min, im=%.4f-%.4f",
        dict_ref_path, *rt_range_minutes, *im_range,
    )
    return rt_range_minutes, im_range


def calibrate_broad_alignment_image_based(
    result_dir: str,
    raw_file_list: list[str],
    data_paths: list[str],
    exclude_dataset_names: list[str],
    dict_ref_path: str | None = None,
    output_path: str | None = None,
    window_widths: tuple[int, ...] = (40, 60, 120, 240),
    strides: tuple[int, ...] = (5, 20),
    upsample_factor: int = 10,
    max_workers: int | None = None,
) -> pd.DataFrame:
    """Build and persist the image-based broad-alignment shift table for one dataset.

    Writes broad_alignment_rt_im_cache/<raw_file>.npy (per-file, shared across every
    pair) and broad_alignment_image_based_pairs/<ref>__vs__<mov>/ww<W>_stride<S>/
    (per-pair-per-combo diagnostic plots + residuals.csv, from rt_im_image_registration's
    run_param_combo) under result_dir, alongside the final table.

    `dict_ref_path` (a dict_ref[_with_activation].pkl), when given, restricts phase-
    correlation SHIFT ESTIMATION (not the collapsed image itself, which always stays the
    whole native run) to the shared RT/IM range spanned by the whole dict_ref (min
    RT_search_left/max RT_search_right, min IM_search_left/max IM_search_right), mapped
    onto each run's own frame/scan index via nearest-match -- mirrors dict_add_rt_index/
    dict_add_im_index's per-peptide treatment, applied here to the whole collapsed image's
    correlation input instead (see rt_im_image_registration.raw_file_rt_im_range_bounds /
    prepare_pair_images). Out-of-range pixels are masked (zeroed) only for that purpose;
    curves still cover the full native frame range. Omitting it keeps every step exactly
    as before this parameter existed.
    """
    raw_file_paths = _resolve_raw_file_paths(raw_file_list, data_paths, exclude_dataset_names)
    cache_dir = os.path.join(result_dir, "broad_alignment_rt_im_cache")
    pairs_dir = os.path.join(result_dir, "broad_alignment_image_based_pairs")
    rt_range_minutes, im_range = _shared_rt_im_range(dict_ref_path)

    # Deliberately capped low and independent of max_workers/N_CPU: each raw .d file's
    # AlphaTims TimsTOF object (+ HDF backing) peaks at ~15GB RSS while building the
    # collapsed image (measured directly), so loading too many concurrently risks OOM
    # regardless of CPU count -- matches the existing build_mz_sorted_activation
    # precedent (sbs_runner_ims.py), which caps at 5 for the same reason, but tighter
    # given the larger per-file footprint here. The pairwise step below is unrelated: it
    # only ever touches small already-cached numpy arrays, so it can safely use the full
    # requested max_workers.
    effective_cache_workers = min(2, len(raw_file_list))
    logger.info("Building rt/im image cache for %d raw files (max_workers=%d)...", len(raw_file_list), effective_cache_workers)
    with ThreadPoolExecutor(max_workers=effective_cache_workers) as executor:
        futures = {}
        for rf in raw_file_list:
            futures[executor.submit(load_rt_im_image, raw_file_paths[rf], cache_dir)] = rf
            if rt_range_minutes is not None:
                # Range bounds only need cheap frame/mobility metadata (no
                # bin_intensities), but reopening the .d file is still worth
                # doing up front, in parallel, rather than lazily inside
                # _process_pair -- primed here alongside the image build.
                futures[executor.submit(raw_file_rt_im_range_bounds, raw_file_paths[rf], cache_dir, rt_range_minutes, im_range)] = rf
        for future in futures:
            rf = futures[future]
            exc = future.exception()
            if exc:
                raise RuntimeError(f"Failed to build rt/im image or range bounds for raw file {rf}") from exc

    pair_list = list(itertools.combinations(raw_file_list, 2))
    effective_max_workers = max_workers if max_workers and max_workers > 0 else min(8, len(pair_list))
    logger.info("Registering %d raw-file pairs (max_workers=%d, 1 BLAS thread/worker)...", len(pair_list), effective_max_workers)
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=effective_max_workers, initializer=_limit_worker_blas_threads) as executor:
        futures = {
            executor.submit(
                _process_pair, ref_run, mov_run, raw_file_paths, cache_dir, pairs_dir,
                tuple(window_widths), tuple(strides), upsample_factor,
                rt_range_minutes, im_range,
            ): (ref_run, mov_run)
            for ref_run, mov_run in pair_list
        }
        for future in futures:
            ref_run, mov_run = futures[future]
            exc = future.exception()
            if exc:
                raise RuntimeError(f"Failed to register pair {ref_run} vs {mov_run}") from exc
            rows.extend(future.result())

    table = pd.DataFrame(rows, columns=_TABLE_COLUMNS)
    resolved_output_path = output_path or os.path.join(result_dir, "broad_alignment_shift_table_global_raw_image.parquet")
    table.to_parquet(resolved_output_path, index=False)
    logger.info("Image-based broad_alignment shift table written to %s (%d rows)", resolved_output_path, len(table))
    return table


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate the image-based broad-alignment shift table.")
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--raw-file", action="append", required=True, dest="raw_file_list")
    parser.add_argument("--data-path", action="append", required=True, dest="data_paths")
    parser.add_argument("--exclude-dataset-name", action="append", default=[], dest="exclude_dataset_names")
    parser.add_argument(
        "--dict-ref-path",
        default=None,
        help="dict_ref[_with_activation].pkl -- when given, restricts every raw file's "
        "collapsed image to the shared RT/IM range spanned by the whole dict_ref.",
    )
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--window-widths", default="40,60,120,240")
    parser.add_argument("--strides", default="5,20")
    parser.add_argument("--upsample-factor", type=int, default=10)
    parser.add_argument("--max-workers", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")  # must precede any pyplot import in this process AND its
    # ProcessPoolExecutor children (fork start method inherits this setting) --
    # sbs_runner_ims.py already sets this when this module runs as part of the
    # pipeline; standalone invocation needs it set explicitly here instead.

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    _args = _parse_args()
    calibrate_broad_alignment_image_based(
        result_dir=_args.result_dir,
        raw_file_list=_args.raw_file_list,
        data_paths=_args.data_paths,
        exclude_dataset_names=_args.exclude_dataset_names,
        dict_ref_path=_args.dict_ref_path,
        output_path=_args.output_path,
        window_widths=tuple(int(w) for w in _args.window_widths.split(",")),
        strides=tuple(int(s) for s in _args.strides.split(",")),
        upsample_factor=_args.upsample_factor,
        max_workers=_args.max_workers,
    )
