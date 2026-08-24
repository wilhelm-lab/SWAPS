"""Full-image RT/IM registration between two raw runs, via collapsed
(m/z-summed) intensity heatmaps and phase correlation.

Ported from the exploratory prototype in
notebooks/dev_swaps2.0_100_braod_alignment_based_on_full_raw_file.ipynb.
This is a research/param-sweep module, not wired into sbs_runner_ims.py --
see broad_alignment.py for the production calibration-peptide-based RT/IM
shift correction actually used by match_features.py.

Pipeline per raw-file pair: collapse each run's m/z dimension into an
(rt, im) intensity image (`load_rt_im_image`, cached to disk, always the
whole native run) -> equalize shapes by zero-padding, not cropping, so no
real signal is discarded (`prepare_pair_images`) -> optionally restrict
SHIFT ESTIMATION (not the image itself) to each run's own physically
in-shared-RT/IM-range pixels, nearest-matched per run
(`raw_file_rt_im_range_bounds`, `_zero_outside_bounds`) -> single global
translation via phase correlation -> sliding-window local phase correlation
(`sliding_window_shifts`) -> a continuous rt_shift(rt) curve fit through the
window estimates, several outlier-handling variants
(`PIECEWISE_METHOD_BUILDERS`) -> per-variant L1 residual against the
reference image (real pixel values, full native range), discounting pixels
with no real (non-padding) source data on either side AND, when a shared
range was given, out-of-range pixels on either side too -- same rationale as
restricting shift estimation: a curve/method shouldn't look better or worse
because of a match (or mismatch) outside calibration scope
(`valid_pixels_after_shift`).
"""

import logging
import os
from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.ndimage import map_coordinates
from scipy.ndimage import shift as nd_shift
from skimage.registration import phase_cross_correlation
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Raw file -> collapsed (rt, im) intensity image, with disk caching
# ---------------------------------------------------------------------------


def raw_file_label(raw_file_path: str) -> str:
    return os.path.basename(os.path.normpath(raw_file_path)).removesuffix(".d")


def _nearest_index(values: np.ndarray, target: float) -> int:
    """Index of the value in sorted ascending `values` nearest to `target` --
    the same "nearest" tie-breaking pd.merge_asof(direction="nearest") uses
    in dict_add_rt_index/dict_add_im_index (prepare_dict.py), just for a
    single scalar lookup instead of merging a whole column."""
    idx = int(np.searchsorted(values, target))
    if idx <= 0:
        return 0
    if idx >= len(values):
        return len(values) - 1
    before, after = values[idx - 1], values[idx]
    return idx - 1 if (target - before) <= (after - target) else idx


def _range_cache_suffix(rt_range_minutes, im_range) -> str:
    parts = []
    if rt_range_minutes is not None:
        parts.append(f"_rt{rt_range_minutes[0]:.4f}-{rt_range_minutes[1]:.4f}")
    if im_range is not None:
        parts.append(f"_im{im_range[0]:.4f}-{im_range[1]:.4f}")
    return "".join(parts)


def load_rt_im_image(raw_file_path: str, cache_dir: str) -> np.ndarray:
    """Collapse the m/z dimension of one raw run into an (rt, im) intensity
    heatmap, MS1 frames only. Building this from the raw .d file is slow, so
    the result is cached to `cache_dir` and reused across all pairs/param
    combos that reference this raw file.

    Always the WHOLE native run -- no RT/IM range restriction. To keep
    phase-correlation shift estimation from being biased by physically
    out-of-scope pixels (e.g. a persistent wash/void-volume peak past the
    last real search window) while still fitting a shift curve over the full
    native frame range, see `raw_file_rt_im_range_bounds` +
    `prepare_pair_images`'s `ref_range_bounds`/`mov_range_bounds` -- those
    zero out-of-range pixels for correlation purposes only, rather than
    cropping the image itself.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{raw_file_label(raw_file_path)}.npy")
    if os.path.exists(cache_path):
        logger.info("Loading cached rt/im image %s", cache_path)
        return np.load(cache_path)

    import alphatims.bruker  # deferred: heavy import, only needed on cache miss

    logger.info("Building rt/im image for %s (cache miss)", raw_file_path)
    data = alphatims.bruker.TimsTOF(raw_file_path)
    # `precursor_indices` is a per-quad-segment MS1/MS2 id array, NOT raw
    # indices -- select MS1-only raw indices via slicing instead:
    # [frame, scan, precursor==0, tof]
    ms1_indices = data[:, :, 0, :, "raw"]
    rt_im_full = data.bin_intensities(ms1_indices, axis=("rt_values", "mobility_values"))
    # rt_im_full rows are indexed by *all* frame ids (frame_max_index),
    # including interleaved MS2/PASEF fragment frames (all-zero here) -- drop
    # them so the rt axis only holds real MS1 frames.
    ms1_frame_ids = data.frames.loc[data.frames["MsMsType"] == 0, "Id"].values
    rt_im = rt_im_full[ms1_frame_ids, :]

    np.save(cache_path, rt_im)
    return rt_im


def raw_file_rt_im_range_bounds(
    raw_file_path: str,
    cache_dir: str,
    rt_range_minutes: tuple[float, float],
    im_range: tuple[float, float],
) -> tuple[int, int, int, int]:
    """(rt_lo, rt_hi, im_lo, im_hi), INCLUSIVE on both ends: this run's own
    MS1-frame-index/mobility-scan-index bounds nearest-matching the given
    SHARED physical RT (minutes)/IM (1/K0) range -- the same approach
    dict_add_rt_index/dict_add_im_index (prepare_dict.py) already use to give
    every run's *per-peptide* activation crop the same RT/IM range despite
    each run having its own frame timing/scan grid, applied here to the whole
    collapsed image instead. These bounds index directly into
    load_rt_im_image's output for this same raw file (native, unpadded rows/
    cols) -- see prepare_pair_images's `ref_range_bounds`/`mov_range_bounds`
    for how they're used to mask phase correlation without cropping.

    Only needs frame/mobility metadata (export_im_and_ms1scans), not the
    slow, memory-heavy bin_intensities collapse load_rt_im_image does --
    cached separately (and much cheaper to rebuild) from the image itself.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(
        cache_dir,
        f"{raw_file_label(raw_file_path)}_range_bounds{_range_cache_suffix(rt_range_minutes, im_range)}.npy",
    )
    if os.path.exists(cache_path):
        return tuple(int(v) for v in np.load(cache_path))

    import alphatims.bruker  # deferred: heavy import, only needed on cache miss

    from utils.ims_utils import export_im_and_ms1scans

    logger.info("Computing rt/im range bounds for %s (cache miss)", raw_file_path)
    data = alphatims.bruker.TimsTOF(raw_file_path)
    ms1scans, mobility_values_df = export_im_and_ms1scans(data, swaps_result_dir=None)
    rt_values = ms1scans["Time_minute"].to_numpy()
    im_values = mobility_values_df["mobility_values"].to_numpy()
    bounds = (
        _nearest_index(rt_values, rt_range_minutes[0]),
        _nearest_index(rt_values, rt_range_minutes[1]),
        _nearest_index(im_values, im_range[0]),
        _nearest_index(im_values, im_range[1]),
    )
    np.save(cache_path, np.array(bounds, dtype=int))
    return bounds


# ---------------------------------------------------------------------------
# Shape equalization (zero-padding, not cropping) + validity masking
# ---------------------------------------------------------------------------


def pad_to(img: np.ndarray, shape: tuple) -> np.ndarray:
    out = np.zeros(shape, dtype=img.dtype)
    out[: img.shape[0], : img.shape[1]] = img
    return out


def _zero_outside_bounds(img: np.ndarray, bounds: tuple[int, int, int, int] | None) -> np.ndarray:
    """Copy of `img` with everything outside [rt_lo, rt_hi] x [im_lo, im_hi]
    (inclusive) zeroed -- used ONLY as phase_cross_correlation's input, so
    physically out-of-shared-range pixels can't pull a shift estimate toward
    matching content that isn't actually in scope for calibration (e.g. a
    persistent wash/void-volume peak past the last real search window).
    `img` itself is untouched by this: residuals, diagnostic plots, and the
    final registered/output image all keep using the REAL pixel values over
    the full native index range -- only shift ESTIMATION is restricted.
    bounds=None (no shared range given) returns `img` unchanged."""
    if bounds is None:
        return img
    rt_lo, rt_hi, im_lo, im_hi = bounds
    masked = np.zeros_like(img)
    masked[rt_lo : rt_hi + 1, im_lo : im_hi + 1] = img[rt_lo : rt_hi + 1, im_lo : im_hi + 1]
    return masked


def valid_pixels_after_shift(
    shift_rt,
    common_shape: tuple,
    ref_native_len: int,
    mov_native_len: int,
    ref_range_bounds: tuple[int, int, int, int] | None = None,
    mov_range_bounds: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """2D boolean mask (shape=common_shape): a pixel counts toward the L1
    residual only if BOTH (1) the reference pixel there is real data, not
    the zero-padding used to equalize rt length (ref rows >= ref_native_len
    are padding) -- and, if `ref_range_bounds` is given, also within ref's
    own shared-range bounds -- and (2) the pixel it draws from in the
    (possibly shifted) moving image is also real -- not padding (row
    >= mov_native_len or < 0, which nd_shift/map_coordinates would otherwise
    silently fill with cval=0) -- and, if `mov_range_bounds` is given, also
    within mov's own shared-range bounds. Any of those would show up as a
    hard 0-vs-real contrast, or an out-of-calibration-scope match, that has
    nothing to do with genuine registration quality -- so the SAME masking
    that keeps out-of-range pixels from influencing shift ESTIMATION
    (prepare_pair_images's ref_corr_img/mov_corr_img) also keeps them from
    influencing which curve/method looks best.

    `shift_rt` may be a scalar (global shift) or a per-row array (a fitted
    curve) -- both bounds-less args (n_rows-only) reduce to the original
    row-only mask when ref_range_bounds/mov_range_bounds are both None."""
    n_rows, n_cols = common_shape
    row_idx = np.arange(n_rows)
    col_idx = np.arange(n_cols)
    src_row = row_idx - np.asarray(shift_rt)

    ref_row_valid = row_idx < ref_native_len
    mov_row_valid = (src_row >= 0) & (src_row < mov_native_len)
    ref_col_valid = np.ones(n_cols, dtype=bool)
    mov_col_valid = np.ones(n_cols, dtype=bool)

    if ref_range_bounds is not None:
        rt_lo, rt_hi, im_lo, im_hi = ref_range_bounds
        ref_row_valid = ref_row_valid & (row_idx >= rt_lo) & (row_idx <= rt_hi)
        ref_col_valid = (col_idx >= im_lo) & (col_idx <= im_hi)
    if mov_range_bounds is not None:
        rt_lo, rt_hi, im_lo, im_hi = mov_range_bounds
        mov_row_valid = mov_row_valid & (src_row >= rt_lo) & (src_row <= rt_hi)
        mov_col_valid = (col_idx >= im_lo) & (col_idx <= im_hi)

    row_valid = ref_row_valid & mov_row_valid
    col_valid = ref_col_valid & mov_col_valid
    return row_valid[:, None] & col_valid[None, :]


def masked_diff(ref: np.ndarray, img: np.ndarray, valid_mask: np.ndarray = None) -> np.ndarray:
    d = (ref - img).astype(float)
    if valid_mask is not None:
        d = d.copy()
        d[~valid_mask] = np.nan
    return d


# ---------------------------------------------------------------------------
# Global registration
# ---------------------------------------------------------------------------


@dataclass
class PairImages:
    ref_img: np.ndarray
    mov_img: np.ndarray
    ref_corr_img: np.ndarray
    mov_corr_img: np.ndarray
    global_registered_corr_img: np.ndarray
    ref_range_bounds: tuple | None
    mov_range_bounds: tuple | None
    ref_native_len: int
    mov_native_len: int
    common_shape: tuple
    rt_range: np.ndarray
    rt_grid: np.ndarray
    im_grid: np.ndarray
    global_rt_shift: float
    global_im_shift: float
    global_error: float
    global_registered_img: np.ndarray
    global_valid_mask: np.ndarray
    unregistered_valid_mask: np.ndarray
    resid_before: float
    resid_after: float
    use_global_prior: bool

    @property
    def windowing_base_img(self) -> np.ndarray:
        """The REAL-pixel-value image sliding-window correlation's output
        shift gets applied against downstream (residuals/warping): ref_img
        pre-registered by the global shift when that shift already beat
        doing nothing, otherwise the raw mov_img unchanged. Correlation
        INPUT should come from windowing_base_img_masked instead -- see
        there."""
        return self.global_registered_img if self.use_global_prior else self.mov_img

    @property
    def windowing_base_img_masked(self) -> np.ndarray:
        """Same role as windowing_base_img, but built from the range-masked
        (out-of-shared-range pixels zeroed) images -- what
        sliding_window_shifts actually correlates against, so those pixels
        never influence a window's measured shift, mirroring how the global
        shift above is estimated from ref_corr_img/mov_corr_img rather than
        ref_img/mov_img."""
        return self.global_registered_corr_img if self.use_global_prior else self.mov_corr_img

    @property
    def windowing_rt_offset(self) -> float:
        """Added back onto every window's measured rt_shift so the fitted
        curve represents the TOTAL shift from mov_img's own native
        coordinates -- windows measured against windowing_base_img(_masked)
        only see the residual left after the global pre-shift, not the full
        shift."""
        return self.global_rt_shift if self.use_global_prior else 0.0


def prepare_pair_images(
    ref_rt_im: np.ndarray,
    mov_rt_im: np.ndarray,
    upsample_factor: int = 10,
    ref_range_bounds: tuple[int, int, int, int] | None = None,
    mov_range_bounds: tuple[int, int, int, int] | None = None,
) -> PairImages:
    """`ref_range_bounds`/`mov_range_bounds` (see
    raw_file_rt_im_range_bounds), when given, restrict phase-correlation
    SHIFT ESTIMATION (both the global step here and every sliding window --
    see PairImages.windowing_base_img_masked) to each run's own in-shared-
    range pixels, by zeroing everything else out in a masked COPY of the
    image used only as correlation input. The real ref_img/mov_img -- and
    therefore every residual, diagnostic plot, and the final registered
    image -- stay on the full native index range, unmasked. Default None
    keeps every step exactly as before this parameter existed (masked copies
    equal the real images)."""
    ref_native_len = ref_rt_im.shape[0]
    mov_native_len = mov_rt_im.shape[0]
    common_shape = (max(ref_native_len, mov_native_len), ref_rt_im.shape[1])
    rt_range = np.arange(common_shape[0])
    rt_grid, im_grid = np.meshgrid(rt_range, np.arange(common_shape[1]), indexing="ij")

    ref_img = np.log1p(pad_to(ref_rt_im, common_shape))
    mov_img = np.log1p(pad_to(mov_rt_im, common_shape))
    ref_corr_img = _zero_outside_bounds(ref_img, ref_range_bounds)
    mov_corr_img = _zero_outside_bounds(mov_img, mov_range_bounds)

    # normalization=None (raw cross-correlation), not skimage's default
    # normalization="phase" -- phase-only whitening treats every frequency as
    # equally important, including the spurious high-frequency edge the
    # zero-padding above introduces wherever the two runs' native lengths
    # differ, and is noise-sensitive on this sparse data generally (same
    # reason sliding_window_shifts already uses normalization=None). Verified
    # directly on a real pair with a ~6% length mismatch: normalization="phase"
    # gave wildly different (and implausible) answers padded vs cropped
    # (181.8 vs 64.1 frames); normalization=None agreed closely either way
    # (0.4 vs 0.0) and landed on a physically-plausible near-zero shift.
    shift_estimate, error, _ = phase_cross_correlation(ref_corr_img, mov_corr_img, upsample_factor=upsample_factor, normalization=None)
    rt_shift, im_shift = shift_estimate
    registered_img = nd_shift(mov_img, shift_estimate)
    registered_corr_img = nd_shift(mov_corr_img, shift_estimate)

    rt_valid_mask = valid_pixels_after_shift(
        rt_shift, common_shape, ref_native_len, mov_native_len, ref_range_bounds, mov_range_bounds
    )
    unregistered_valid_mask = valid_pixels_after_shift(
        0.0, common_shape, ref_native_len, mov_native_len, ref_range_bounds, mov_range_bounds
    )
    resid_before = float(np.abs((ref_img - mov_img)[unregistered_valid_mask]).sum())
    resid_after = float(np.abs((ref_img - registered_img)[rt_valid_mask]).sum())
    use_global_prior = resid_after < resid_before

    return PairImages(
        ref_img=ref_img,
        mov_img=mov_img,
        ref_corr_img=ref_corr_img,
        mov_corr_img=mov_corr_img,
        global_registered_corr_img=registered_corr_img,
        ref_range_bounds=ref_range_bounds,
        mov_range_bounds=mov_range_bounds,
        ref_native_len=ref_native_len,
        mov_native_len=mov_native_len,
        common_shape=common_shape,
        rt_range=rt_range,
        rt_grid=rt_grid,
        im_grid=im_grid,
        global_rt_shift=float(rt_shift),
        global_im_shift=float(im_shift),
        global_error=float(error),
        global_registered_img=registered_img,
        global_valid_mask=rt_valid_mask,
        unregistered_valid_mask=unregistered_valid_mask,
        resid_before=resid_before,
        resid_after=resid_after,
        use_global_prior=use_global_prior,
    )


# ---------------------------------------------------------------------------
# Sliding-window local shifts
# ---------------------------------------------------------------------------


def sliding_window_shifts(pair: PairImages, window_width: int, starts=None, stride: int = None, upsample_factor: int = 1) -> pd.DataFrame:
    """rt/im shift per rt-window, measured against `pair.windowing_base_img`
    (the global-pre-registered image when that shift already helped, else
    the raw mov_img) -- so each same-index window pair is comparing content
    that's already roughly corresponding, instead of relying on same-index
    alignment holding across the raw, un-corrected global offset. The
    resulting `rt_shift` is therefore a RESIDUAL on top of
    `pair.windowing_rt_offset`, not the total shift from mov_img's own
    coordinates -- callers must add that offset back in before using it to
    warp mov_img directly.

    Pass either a fixed `stride` (uniform spacing) or an explicit `starts`
    list (any spacing, e.g. denser near the rt edges). Correlation runs on
    the range-masked images (windowing_base_img_masked/ref_corr_img) -- see
    PairImages -- so a window straddling the shared-range boundary only
    "sees" its in-range pixels; a window entirely outside the shared range
    is skipped by the max()<1 check below exactly like an all-padding window
    already is."""
    if starts is None:
        starts = range(0, pair.common_shape[0] - window_width + 1, stride)
    mov_base_img = pair.windowing_base_img_masked
    rows = []
    for start in starts:
        end = start + window_width
        ref_win = pair.ref_corr_img[start:end, :]
        mov_win = mov_base_img[start:end, :]
        if ref_win.max() < 1 or mov_win.max() < 1:
            continue
        (win_rt_shift, win_im_shift), win_error, _ = phase_cross_correlation(
            ref_win, mov_win, upsample_factor=upsample_factor, normalization=None
        )
        rows.append(
            {
                "rt_center": start + window_width / 2,
                "rt_shift": win_rt_shift,
                "im_shift": win_im_shift,
                "error": win_error,
                "window_width": window_width,
            }
        )
    return pd.DataFrame(rows)


def dense_edge_starts(n_rows: int, window_width: int, base_stride: int, edge_stride: int, edge_width: int) -> list:
    """Window start positions: edge_stride (denser) within edge_width of
    either rt boundary, base_stride in the middle -- more redundancy right
    where the boundary has the least support to fall back on if a window
    gets rejected as an outlier."""
    max_start = n_rows - window_width
    starts = []
    pos = 0
    while pos < max_start:
        starts.append(pos)
        near_edge = pos < edge_width or pos > max_start - edge_width
        pos += edge_stride if near_edge else base_stride
    starts.append(max_start)
    return starts


# ---------------------------------------------------------------------------
# Outlier rejection + continuous rt_shift(rt) curve fitting
# ---------------------------------------------------------------------------


def ransac_inlier_mask(x, y, poly_degree=2, min_samples=6, residual_threshold=None) -> np.ndarray:
    """RANSAC outlier rejection against a low-degree polynomial baseline.
    residual_threshold=None lets RANSAC pick it from the target MAD (sklearn default).
    Falls back to trusting every window when there are fewer windows than
    min_samples -- e.g. a pair with very little rt overlap (mismatched
    gradient lengths) can leave only a handful of non-empty windows, and
    RANSACRegressor errors outright if min_samples > n_samples; random-subset
    outlier rejection isn't meaningful with that few points anyway."""
    if len(x) < min_samples:
        logger.warning(
            "Only %d sliding windows (< min_samples=%d) -- skipping RANSAC, trusting all windows", len(x), min_samples
        )
        return np.ones(len(x), dtype=bool)
    ransac = RANSACRegressor(
        make_pipeline(PolynomialFeatures(poly_degree), LinearRegression()),
        min_samples=min_samples,
        residual_threshold=residual_threshold,
        random_state=0,
    )
    ransac.fit(x.reshape(-1, 1), y)
    return ransac.inlier_mask_


def ransac_inlier_mask_force_edges(x, y, poly_degree=2, min_samples=6, residual_threshold=None, n_edge=1) -> np.ndarray:
    """Same RANSAC-vs-global-polynomial rejection, but always keep the n_edge
    lowest- and highest-rt windows as inliers regardless of the verdict."""
    inlier_mask = ransac_inlier_mask(x, y, poly_degree, min_samples, residual_threshold).copy()
    order = np.argsort(x)
    edge_idx = np.concatenate([order[:n_edge], order[-n_edge:]])
    inlier_mask[edge_idx] = True
    return inlier_mask


def rescue_confident_outliers(errors, inlier_mask, confidence_percentile=50) -> np.ndarray:
    """Un-reject any rejected point whose own error is better than the
    confidence_percentile of the current inlier set's errors."""
    inlier_error_threshold = np.percentile(errors[inlier_mask], confidence_percentile)
    rescued_mask = inlier_mask.copy()
    rescued_mask[(~inlier_mask) & (errors <= inlier_error_threshold)] = True
    return rescued_mask


def _constant_curve(value: float):
    """Callable curve_fn(rt) that returns `value` everywhere -- the only
    sane fallback when there's just one (x, y) point to build a curve from."""
    def curve_fn(rt):
        return np.full(np.shape(rt), value, dtype=float)
    return curve_fn


def _linear_curve_through(x, y, inlier_mask):
    order = np.argsort(x[inlier_mask])
    xs, ys = x[inlier_mask][order], y[inlier_mask][order]
    if len(xs) == 1:
        return _constant_curve(ys[0])
    return interp1d(xs, ys, kind="linear", bounds_error=False, fill_value=(ys[0], ys[-1]))


def _sanitize_curve_values(
    curve_values: np.ndarray, x: np.ndarray, y: np.ndarray, inlier_mask: np.ndarray, rt_range: np.ndarray, method_name: str
) -> np.ndarray:
    """Compares `curve_values` against `baseline` -- piecewise-LINEAR
    interpolation through the SAME inliers -- which by construction can
    never overshoot beyond its neighboring points, so any large deviation
    from it is exactly the failure mode to catch: a curve oscillating
    between/around a gap in RANSAC-surviving windows rather than tracking
    the real local trend. A fixed multiple of the run's own frame count (an
    earlier version of this check) was tried first and found too loose --
    measured directly on real data, a spline spiked ~580 units away from a
    baseline sitting near 0 (inlier spread only ~78 units, typical deviation
    elsewhere on the SAME curve well under 10) while comfortably inside a
    1000-frames-or-more cap; comparing against each curve's own baseline
    instead scales with what's actually plausible for THIS pair/inlier set,
    not an unrelated absolute size. tolerance is generous relative to the
    inliers' own spread (5x) with a floor (50 frames) so legitimate smooth-
    ing wiggle (observed: single digits) is never mistaken for instability.
    Checked against the SAME domain (`rt_range`) the curve is actually
    evaluated over in production, applied to every method uniformly (not
    just the spline builder that's most prone to this -- the exactly-linear
    methods trivially pass since their own curve IS this baseline)."""
    baseline = _linear_curve_through(x, y, inlier_mask)(rt_range)
    data_spread = float(np.ptp(y[inlier_mask])) if inlier_mask.any() else float(np.ptp(y))
    tolerance = max(50.0, 5 * data_spread)
    deviation = np.abs(curve_values - baseline)
    if deviation.max() > tolerance:
        logger.warning(
            "%s: curve deviates %.0f frames from its own linear-interpolation baseline "
            "(tolerance %.0f) -- falling back to linear interpolation through the same inliers",
            method_name, deviation.max(), tolerance,
        )
        return baseline
    return curve_values


def build_piecewise_linear_rt_shift_raw(shifts_df: pd.DataFrame):
    """No outlier rejection -- every window's own measured rt_shift,
    connected directly with straight-line segments in rt order."""
    x = shifts_df["rt_center"].to_numpy()
    y = shifts_df["rt_shift"].to_numpy()
    inlier_mask = np.ones(len(x), dtype=bool)
    return _linear_curve_through(x, y, inlier_mask), inlier_mask


def build_piecewise_linear_rt_shift(shifts_df: pd.DataFrame, poly_degree=2, min_samples=6, residual_threshold=None):
    """RANSAC-reject outlier windows, then linearly interpolate between the
    surviving inliers."""
    x = shifts_df["rt_center"].to_numpy()
    y = shifts_df["rt_shift"].to_numpy()
    inlier_mask = ransac_inlier_mask(x, y, poly_degree, min_samples, residual_threshold)
    return _linear_curve_through(x, y, inlier_mask), inlier_mask


def build_piecewise_linear_rt_shift_edge_forced(shifts_df: pd.DataFrame, poly_degree=2, min_samples=6, residual_threshold=None, n_edge=1):
    x = shifts_df["rt_center"].to_numpy()
    y = shifts_df["rt_shift"].to_numpy()
    inlier_mask = ransac_inlier_mask_force_edges(x, y, poly_degree, min_samples, residual_threshold, n_edge)
    return _linear_curve_through(x, y, inlier_mask), inlier_mask


def build_piecewise_linear_rt_shift_confidence_rescued(
    shifts_df: pd.DataFrame, poly_degree=2, min_samples=6, residual_threshold=None, n_edge=1, confidence_percentile=50
):
    """Edge-forcing + a rescue pass on top for any other confident interior
    point RANSAC still rejected."""
    x = shifts_df["rt_center"].to_numpy()
    y = shifts_df["rt_shift"].to_numpy()
    errors = shifts_df["error"].to_numpy()
    inlier_mask = ransac_inlier_mask_force_edges(x, y, poly_degree, min_samples, residual_threshold, n_edge)
    inlier_mask = rescue_confident_outliers(errors, inlier_mask, confidence_percentile)
    return _linear_curve_through(x, y, inlier_mask), inlier_mask


def fit_robust_rt_shift_curve(shifts_df: pd.DataFrame, poly_degree=3, min_samples=6, residual_threshold=None, spline_smoothing=1):
    """RANSAC-reject outlier windows, then fit a confidence-weighted
    smoothing spline through the inliers."""
    x = shifts_df["rt_center"].to_numpy()
    y = shifts_df["rt_shift"].to_numpy()
    inlier_mask = ransac_inlier_mask(x, y, poly_degree, min_samples, residual_threshold)

    xs, ys = x[inlier_mask], y[inlier_mask]
    weights = 1 / (shifts_df["error"].to_numpy()[inlier_mask] + 1e-3)
    order = np.argsort(xs)
    xs, ys, weights = xs[order], ys[order], weights[order]
    if len(xs) == 1:
        return _constant_curve(ys[0]), inlier_mask
    # UnivariateSpline requires strictly more points than its degree k --
    # degrade to a lower-order (down to linear) spline instead of crashing
    # when there are too few inliers for a full cubic fit.
    k = min(3, len(xs) - 1)
    spline = UnivariateSpline(xs, ys, w=weights, k=k, s=spline_smoothing, ext=3)
    # UnivariateSpline can become numerically ill-conditioned (FITPACK warns
    # "s too small") when the RANSAC-surviving inliers have a large gap --
    # ext=3 only bounds extrapolation OUTSIDE [xs.min(), xs.max()]; it does
    # nothing for oscillation BETWEEN sparse points inside that range. Caught
    # centrally, for every method (not just this one), by run_param_combo's
    # _sanitize_curve_values -- see there for why a fixed multiple of the
    # data's own spread wasn't a tight enough bound on its own.
    return spline, inlier_mask


# Ordered so plotting/reporting is stable and reproducible across runs.
PIECEWISE_METHOD_BUILDERS: dict[str, Callable] = {
    "linear (no rejection)": build_piecewise_linear_rt_shift_raw,
    "linear (plain RANSAC)": build_piecewise_linear_rt_shift,
    "spline (RANSAC deg=3)": fit_robust_rt_shift_curve,
    "linear (edges forced)": build_piecewise_linear_rt_shift_edge_forced,
    "linear (edges forced + confidence-rescued)": build_piecewise_linear_rt_shift_confidence_rescued,
}

_LINE_STYLES = {
    "spline (RANSAC deg=3)": dict(color="black", linestyle=":"),
    "linear (plain RANSAC)": dict(color="tab:blue", linestyle="--"),
    "linear (no rejection)": dict(color="gray", linestyle="-."),
    "linear (edges forced)": dict(color="tab:orange", linestyle="--"),
    "linear (edges forced + confidence-rescued)": dict(color="black", linewidth=2),
}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_DIFF_CMAP = plt.cm.coolwarm.copy()
_DIFF_CMAP.set_bad(color="lightgray")  # excluded (invalid) rows render as gray, not a fake color


def plot_diff_grid(ref_img: np.ndarray, variants: list, out_path: str, n_cols: int = 4) -> None:
    """variants: list of (title, img, valid_mask_or_None, l1_residual)."""
    n_rows = int(np.ceil(len(variants) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 5 * n_rows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).flatten()

    for ax, (title, img, mask, resid) in zip(axes, variants):
        ax.imshow(masked_diff(ref_img, img, mask).T, aspect="auto", origin="lower", cmap=_DIFF_CMAP, vmin=-3, vmax=3)
        ax.set_title(f"{title}\nL1 residual={resid:.0f}", fontsize=10)
        ax.set_xlabel("MS1 frame index (rt)")
    for ax in axes[len(variants):]:
        ax.axis("off")
    for row in range(n_rows):
        axes[row * n_cols].set_ylabel("scan index (im)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_windows_confidence(
    shifts_df: pd.DataFrame,
    curves: dict,
    inlier_mask_linear: np.ndarray,
    inlier_mask_rescued: np.ndarray,
    rt_range: np.ndarray,
    reference_value: float,
    out_path: str,
    reference_label: str = "single global rt_shift",
) -> None:
    """Scatter of every window's measured rt_shift, colored by phase-
    correlation confidence, with each fitted curve overlaid and outlier
    windows circled by which stage of the rejection cascade let them back in.
    `shifts_df`/`curves` may be on a residual scale (see
    PairImages.windowing_base_img) -- `reference_value` should already be on
    that same scale (see run_param_combo)."""
    fig, ax = plt.subplots(figsize=(14, 5))
    sc = ax.scatter(shifts_df["rt_center"], shifts_df["rt_shift"], c=shifts_df["error"], cmap="viridis_r", s=40, zorder=3)

    still_rejected = ~inlier_mask_rescued
    rescued_only = inlier_mask_rescued & ~inlier_mask_linear
    ax.scatter(
        shifts_df.loc[rescued_only, "rt_center"], shifts_df.loc[rescued_only, "rt_shift"],
        facecolors="none", edgecolors="tab:orange", s=140, linewidths=2, zorder=4,
    )
    ax.scatter(
        shifts_df.loc[still_rejected, "rt_center"], shifts_df.loc[still_rejected, "rt_shift"],
        color="tab:red", marker="x", s=140, linewidths=2, zorder=4,
    )

    for label, values in curves.items():
        ax.plot(rt_range, values, label=label, **_LINE_STYLES.get(label, {}))
    ax.axhline(reference_value, color="gray", linestyle=":", alpha=0.5, label=reference_label)

    fig.colorbar(sc, ax=ax, label="phase-correlation error (lower = more confident)")
    ax.set_xlabel("rt (MS1 frame index)")
    ax.set_ylabel("rt_shift")

    # Explicit proxy handles for the two outlier-category scatters: an
    # ax.scatter(..., label=...) call with zero points can silently drop its
    # own legend entry, so build these two by hand instead of relying on
    # ax.get_legend_handles_labels() to have picked them up.
    handles, labels = ax.get_legend_handles_labels()
    handles += [
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="none", markeredgecolor="tab:orange", markersize=11, markeredgewidth=2),
        Line2D([], [], marker="x", linestyle="none", color="tab:red", markersize=11, markeredgewidth=2),
    ]
    labels += ["rescued (rejected by plain RANSAC)", "still rejected (all methods)"]
    ax.legend(handles, labels, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_group_summary(residuals_df: pd.DataFrame, out_path: str, n_cols: int = 3) -> None:
    """One figure per raw-file group/list, faceted by pair: x = param combo
    (window_width/stride), y = L1 residual, color = curve-fitting method.
    Unregistered/Global-shift baselines are drawn as reference lines."""
    piecewise_df = residuals_df[~residuals_df["method"].isin(["Unregistered", "Global shift"])].copy()
    piecewise_df["param"] = piecewise_df["window_width"].astype(str) + "/" + piecewise_df["stride"].astype(str)
    pairs = sorted(piecewise_df["pair"].unique())
    methods = list(PIECEWISE_METHOD_BUILDERS.keys())
    palette = dict(zip(methods, sns.color_palette("tab10", len(methods))))

    n_rows = int(np.ceil(len(pairs) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows), squeeze=False)
    axes = axes.flatten()

    combined_handles, combined_labels = None, None
    for ax, pair_name in zip(axes, pairs):
        sub = piecewise_df[piecewise_df["pair"] == pair_name]
        sns.stripplot(data=sub, x="param", y="l1_residual", hue="method", hue_order=methods, palette=palette, dodge=True, jitter=0.15, size=7, ax=ax)

        baseline_rows = residuals_df[residuals_df["pair"] == pair_name]
        global_resid = baseline_rows.loc[baseline_rows["method"] == "Global shift", "l1_residual"]
        unreg_resid = baseline_rows.loc[baseline_rows["method"] == "Unregistered", "l1_residual"]
        if len(global_resid):
            ax.axhline(global_resid.iloc[0], color="gray", linestyle=":", label="Global shift (baseline)")
        if len(unreg_resid):
            ax.axhline(unreg_resid.iloc[0], color="lightgray", linestyle="--", label="Unregistered (baseline)")

        ref_label, mov_label = pair_name.split("__vs__", 1)
        ax.set_title(f"{ref_label}\nvs {mov_label}", fontsize=6)
        ax.set_xlabel("window_width/stride")
        ax.set_ylabel("L1 residual")
        ax.tick_params(axis="x", rotation=30)

        if combined_handles is None:
            combined_handles, combined_labels = ax.get_legend_handles_labels()
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    for ax in axes[len(pairs):]:
        ax.axis("off")

    if combined_handles:
        fig.legend(combined_handles, combined_labels, loc="lower center", ncol=min(len(combined_labels), 4), bbox_to_anchor=(0.5, -0.05), fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-(pair, param-combo) and per-pair orchestration
# ---------------------------------------------------------------------------


def run_param_combo(
    pair: PairImages,
    ref_label: str,
    mov_label: str,
    window_width: int,
    stride: int,
    out_dir: str,
    edge_stride: int = 5,
    edge_width: int = 150,
    upsample_factor: int = 10,
    return_curves: bool = False,
):
    """Returns just `residuals_df` by default. With `return_curves=True`,
    also returns `curves_total: dict[method_name, np.ndarray]` -- each
    method's full per-frame TOTAL shift (already offset back to mov_img's
    own coordinates) -- since that's never persisted to disk (only the
    summary CSV/plots are). This bypasses the residuals.csv-exists shortcut
    (curves aren't cached, but recomputing them is cheap: the expensive part,
    rt_im loading, is still cached separately via load_rt_im_image)."""
    os.makedirs(out_dir, exist_ok=True)
    residuals_path = os.path.join(out_dir, "residuals.csv")
    if not return_curves and os.path.exists(residuals_path):
        logger.info("Skipping %s (residuals.csv already exists)", out_dir)
        return pd.read_csv(residuals_path)

    starts = dense_edge_starts(pair.common_shape[0], window_width, base_stride=stride, edge_stride=edge_stride, edge_width=edge_width)
    shifts_df = sliding_window_shifts(pair, window_width, starts=starts, upsample_factor=upsample_factor)

    # shifts_df["rt_shift"] is a RESIDUAL on top of pair.windowing_rt_offset
    # whenever the global shift was used as a windowing prior (see
    # PairImages.windowing_base_img) -- curves stay on that same residual
    # scale for plotting, but every curve must be offset back to mov_img's
    # own coordinates before it's used to warp mov_img or compute a residual.
    x = shifts_df["rt_center"].to_numpy()
    y = shifts_df["rt_shift"].to_numpy()
    curves, curves_total, inlier_masks, variants, rows = {}, {}, {}, [], []
    for method_name, builder in PIECEWISE_METHOD_BUILDERS.items():
        curve_fn, inlier_mask = builder(shifts_df)
        curve_values_residual = curve_fn(pair.rt_range)
        curve_values_residual = _sanitize_curve_values(curve_values_residual, x, y, inlier_mask, pair.rt_range, method_name)
        curve_values_total = curve_values_residual + pair.windowing_rt_offset
        registered_img = map_coordinates(
            pair.mov_img, [pair.rt_grid - curve_values_total[:, None], pair.im_grid - pair.global_im_shift],
            order=1, mode="constant", cval=0.0,
        )
        valid_mask = valid_pixels_after_shift(
            curve_values_total, pair.common_shape, pair.ref_native_len, pair.mov_native_len,
            pair.ref_range_bounds, pair.mov_range_bounds,
        )
        resid = float(np.abs((pair.ref_img - registered_img)[valid_mask]).sum())

        curves[method_name] = curve_values_residual
        curves_total[method_name] = curve_values_total
        inlier_masks[method_name] = inlier_mask
        variants.append((f"Piecewise ({method_name})", registered_img, valid_mask, resid))
        rows.append(
            dict(method=method_name, window_width=window_width, stride=stride, l1_residual=resid,
                 n_windows=len(shifts_df), n_inliers=int(inlier_mask.sum()), used_global_prior=pair.use_global_prior)
        )

    rows.append(dict(method="Unregistered", window_width=window_width, stride=stride, l1_residual=pair.resid_before,
                      n_windows=len(shifts_df), n_inliers=None, used_global_prior=pair.use_global_prior))
    rows.append(dict(method="Global shift", window_width=window_width, stride=stride, l1_residual=pair.resid_after,
                      n_windows=len(shifts_df), n_inliers=None, used_global_prior=pair.use_global_prior))

    all_variants = [
        ("Unregistered", pair.mov_img, pair.unregistered_valid_mask, pair.resid_before),
        ("Global shift", pair.global_registered_img, pair.global_valid_mask, pair.resid_after),
    ] + variants
    plot_diff_grid(pair.ref_img, all_variants, os.path.join(out_dir, "diff_grid.png"))
    reference_label = (
        "global rt_shift (used as prior; 0 = matches it exactly)" if pair.use_global_prior else "single global rt_shift (not used, hurt L1)"
    )
    plot_windows_confidence(
        shifts_df, curves,
        inlier_masks["linear (plain RANSAC)"], inlier_masks["linear (edges forced + confidence-rescued)"],
        pair.rt_range, pair.global_rt_shift - pair.windowing_rt_offset,
        os.path.join(out_dir, "windows_confidence_kept_rejected.png"),
        reference_label=reference_label,
    )

    residuals_df = pd.DataFrame(rows)
    residuals_df["ref_file"] = ref_label
    residuals_df["mov_file"] = mov_label
    residuals_df.to_csv(residuals_path, index=False)
    if return_curves:
        return residuals_df, curves_total
    return residuals_df


def run_pair(
    ref_path: str,
    mov_path: str,
    cache_dir: str,
    pair_out_dir: str,
    window_widths: list,
    strides: list,
    upsample_factor: int = 10,
    return_curves: bool = False,
    rt_range_minutes: tuple[float, float] | None = None,
    im_range: tuple[float, float] | None = None,
):
    """Returns just `residuals_df` by default. With `return_curves=True`,
    also returns `(pair, curves_by_combo)` where `pair` is the PairImages
    (global_rt_shift/global_im_shift/resid_after/rt_range/native lengths) and
    `curves_by_combo: dict[(window_width, stride), dict[method_name, np.ndarray]]`.

    `rt_range_minutes`/`im_range`, when given (together), restrict shift
    ESTIMATION to each run's own in-shared-range pixels (nearest-matched via
    raw_file_rt_im_range_bounds) -- see prepare_pair_images's docstring for
    exactly what that does and doesn't affect. Both images are always loaded
    whole (load_rt_im_image never crops), so the fitted curve still covers
    the full native frame range either way; default None skips range
    computation entirely, unchanged from before this parameter existed."""
    if (rt_range_minutes is None) != (im_range is None):
        raise ValueError("rt_range_minutes and im_range must be given together")
    ref_label = raw_file_label(ref_path)
    mov_label = raw_file_label(mov_path)
    ref_rt_im = load_rt_im_image(ref_path, cache_dir)
    mov_rt_im = load_rt_im_image(mov_path, cache_dir)

    ref_bounds = mov_bounds = None
    if rt_range_minutes is not None:
        ref_bounds = raw_file_rt_im_range_bounds(ref_path, cache_dir, rt_range_minutes, im_range)
        mov_bounds = raw_file_rt_im_range_bounds(mov_path, cache_dir, rt_range_minutes, im_range)

    pair = prepare_pair_images(
        ref_rt_im, mov_rt_im, upsample_factor=upsample_factor,
        ref_range_bounds=ref_bounds, mov_range_bounds=mov_bounds,
    )

    combo_dfs = []
    curves_by_combo = {}
    for window_width in window_widths:
        for stride in strides:
            combo_out_dir = os.path.join(pair_out_dir, f"ww{window_width}_stride{stride}")
            if return_curves:
                combo_df, curves_total = run_param_combo(
                    pair, ref_label, mov_label, window_width, stride, combo_out_dir,
                    upsample_factor=upsample_factor, return_curves=True,
                )
                curves_by_combo[(window_width, stride)] = curves_total
            else:
                combo_df = run_param_combo(pair, ref_label, mov_label, window_width, stride, combo_out_dir, upsample_factor=upsample_factor)
            combo_dfs.append(combo_df)

    residuals_df = pd.concat(combo_dfs, ignore_index=True)
    residuals_df["pair"] = f"{ref_label}__vs__{mov_label}"
    residuals_df.to_csv(os.path.join(pair_out_dir, "residuals_summary.csv"), index=False)
    if return_curves:
        return residuals_df, pair, curves_by_combo
    return residuals_df
