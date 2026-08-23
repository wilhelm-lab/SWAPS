"""Full-image RT/IM registration between two raw runs, via collapsed
(m/z-summed) intensity heatmaps and phase correlation.

Ported from the exploratory prototype in
notebooks/dev_swaps2.0_100_braod_alignment_based_on_full_raw_file.ipynb.
This is a research/param-sweep module, not wired into sbs_runner_ims.py --
see broad_alignment.py for the production calibration-peptide-based RT/IM
shift correction actually used by match_features.py.

Pipeline per raw-file pair: collapse each run's m/z dimension into an
(rt, im) intensity image (`load_rt_im_image`, cached to disk) -> equalize
shapes by zero-padding, not cropping, so no real signal is discarded
(`prepare_pair_images`) -> single global translation via phase correlation
-> sliding-window local phase correlation (`sliding_window_shifts`) -> a
continuous rt_shift(rt) curve fit through the window estimates, several
outlier-handling variants (`PIECEWISE_METHOD_BUILDERS`) -> per-variant L1
residual against the reference image, discounting rows with no real
(non-padding) source data on either side (`valid_rows_after_shift`).
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


def load_rt_im_image(raw_file_path: str, cache_dir: str) -> np.ndarray:
    """Collapse the m/z dimension of one raw run into an (rt, im) intensity
    heatmap, MS1 frames only. Building this from the raw .d file is slow, so
    the result is cached to `cache_dir` and reused across all pairs/param
    combos that reference this raw file."""
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


# ---------------------------------------------------------------------------
# Shape equalization (zero-padding, not cropping) + validity masking
# ---------------------------------------------------------------------------


def pad_to(img: np.ndarray, shape: tuple) -> np.ndarray:
    out = np.zeros(shape, dtype=img.dtype)
    out[: img.shape[0], : img.shape[1]] = img
    return out


def valid_rows_after_shift(shift_values, n_rows: int, ref_native_len: int, mov_native_len: int) -> np.ndarray:
    """A row counts toward the L1 residual only if BOTH: (1) the reference
    pixel there is real data, not the zero-padding used to equalize rt
    length (ref rows >= ref_native_len are padding), and (2) the row it
    draws from in the (possibly shifted) moving image is also real -- not
    padding (>= mov_native_len) and not an out-of-[0, n_rows) sample that
    nd_shift/map_coordinates would otherwise silently fill with cval=0. Any
    of those would show up as a hard 0-vs-real contrast that has nothing to
    do with registration quality."""
    idx = np.arange(n_rows)
    src = idx - np.asarray(shift_values)
    ref_valid = idx < ref_native_len
    mov_valid = (src >= 0) & (src < mov_native_len)
    return ref_valid & mov_valid


def masked_diff(ref: np.ndarray, img: np.ndarray, valid_mask: np.ndarray = None) -> np.ndarray:
    d = (ref - img).astype(float)
    if valid_mask is not None:
        d = d.copy()
        d[~valid_mask, :] = np.nan
    return d


# ---------------------------------------------------------------------------
# Global registration
# ---------------------------------------------------------------------------


@dataclass
class PairImages:
    ref_img: np.ndarray
    mov_img: np.ndarray
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
        """The image sliding-window correlation should compare against
        ref_img: pre-registered by the global shift when that shift already
        beat doing nothing (so each same-index window pair is much closer to
        genuinely corresponding content, cutting down peak-hopping between
        similar-looking peaks), otherwise the raw mov_img unchanged."""
        return self.global_registered_img if self.use_global_prior else self.mov_img

    @property
    def windowing_rt_offset(self) -> float:
        """Added back onto every window's measured rt_shift so the fitted
        curve represents the TOTAL shift from mov_img's own native
        coordinates -- windows measured against windowing_base_img only see
        the residual left after the global pre-shift, not the full shift."""
        return self.global_rt_shift if self.use_global_prior else 0.0


def prepare_pair_images(ref_rt_im: np.ndarray, mov_rt_im: np.ndarray, upsample_factor: int = 10) -> PairImages:
    ref_native_len = ref_rt_im.shape[0]
    mov_native_len = mov_rt_im.shape[0]
    common_shape = (max(ref_native_len, mov_native_len), ref_rt_im.shape[1])
    rt_range = np.arange(common_shape[0])
    rt_grid, im_grid = np.meshgrid(rt_range, np.arange(common_shape[1]), indexing="ij")

    ref_img = np.log1p(pad_to(ref_rt_im, common_shape))
    mov_img = np.log1p(pad_to(mov_rt_im, common_shape))

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
    shift_estimate, error, _ = phase_cross_correlation(ref_img, mov_img, upsample_factor=upsample_factor, normalization=None)
    rt_shift, im_shift = shift_estimate
    registered_img = nd_shift(mov_img, shift_estimate)

    rt_valid_mask = valid_rows_after_shift(rt_shift, common_shape[0], ref_native_len, mov_native_len)
    unregistered_valid_mask = valid_rows_after_shift(0.0, common_shape[0], ref_native_len, mov_native_len)
    resid_before = float(np.abs((ref_img - mov_img)[unregistered_valid_mask]).sum())
    resid_after = float(np.abs((ref_img - registered_img)[rt_valid_mask]).sum())
    use_global_prior = resid_after < resid_before

    return PairImages(
        ref_img=ref_img,
        mov_img=mov_img,
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
    list (any spacing, e.g. denser near the rt edges)."""
    if starts is None:
        starts = range(0, pair.common_shape[0] - window_width + 1, stride)
    mov_base_img = pair.windowing_base_img
    rows = []
    for start in starts:
        end = start + window_width
        ref_win = pair.ref_img[start:end, :]
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
    # observed in practice reaching ~1e13, not just an ordinary overshoot --
    # even though every actual data point is fit closely. ext=3 only bounds
    # extrapolation OUTSIDE [xs.min(), xs.max()]; it does nothing for
    # oscillation BETWEEN sparse points inside that range. Validate the fit
    # stays within a generous but bounded multiple of the data's own spread
    # across its whole domain, falling back to the always-stable
    # piecewise-linear interpolation (which by construction can't overshoot
    # beyond its neighboring points) when it doesn't.
    probe = np.linspace(xs.min(), xs.max(), max(200, len(xs) * 5))
    data_span = max(float(np.ptp(ys)), 1.0)
    if np.abs(spline(probe) - np.median(ys)).max() > 1000 * data_span:
        logger.warning(
            "spline fit is numerically unstable (likely a gap in RANSAC-surviving windows) "
            "-- falling back to linear interpolation through the same inliers"
        )
        return _linear_curve_through(x, y, inlier_mask), inlier_mask
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
    curves, curves_total, inlier_masks, variants, rows = {}, {}, {}, [], []
    for method_name, builder in PIECEWISE_METHOD_BUILDERS.items():
        curve_fn, inlier_mask = builder(shifts_df)
        curve_values_residual = curve_fn(pair.rt_range)
        curve_values_total = curve_values_residual + pair.windowing_rt_offset
        registered_img = map_coordinates(
            pair.mov_img, [pair.rt_grid - curve_values_total[:, None], pair.im_grid - pair.global_im_shift],
            order=1, mode="constant", cval=0.0,
        )
        valid_mask = valid_rows_after_shift(curve_values_total, pair.common_shape[0], pair.ref_native_len, pair.mov_native_len)
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
):
    """Returns just `residuals_df` by default. With `return_curves=True`,
    also returns `(pair, curves_by_combo)` where `pair` is the PairImages
    (global_rt_shift/global_im_shift/resid_after/rt_range/native lengths) and
    `curves_by_combo: dict[(window_width, stride), dict[method_name, np.ndarray]]`."""
    ref_label = raw_file_label(ref_path)
    mov_label = raw_file_label(mov_path)
    ref_rt_im = load_rt_im_image(ref_path, cache_dir)
    mov_rt_im = load_rt_im_image(mov_path, cache_dir)
    pair = prepare_pair_images(ref_rt_im, mov_rt_im, upsample_factor=upsample_factor)

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
