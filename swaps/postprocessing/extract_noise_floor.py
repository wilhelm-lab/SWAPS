import os
import logging
from typing import Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde, norm
from scipy.signal import find_peaks
from skimage.morphology import remove_small_objects
from sklearn.mixture import GaussianMixture

from .helper import load_peptide_batch_df_from_partquet, get_pept_act_from_parquet

def extract_noise_pixels_no_smooth(
    image: np.ndarray,
    threshold: float = 0,
    min_size: int = 3,
    visualize: bool = False,
    save_path: Optional[str] = None,
    save_svg: bool = True,
    log_non_noise: bool = False,
):
    '''Matches 03_HYE.ipynb's current cells 44/45: no smoothing, near-zero
    threshold, min_size=3 -- operates directly on the raw activation image.

    visualize: if True, plot the original image (log2(1+x) scale) with the
    selected noise pixels marked, colored white-to-red by each pixel's own
    log2(1+x) value (whiter = lower, redder = higher).
    save_path: if given (with visualize=True), save the figure as PNG there,
    plus a sibling .svg unless save_svg=False (matches direct_lfq._save_fig's
    PNG+SVG convention).
    log_non_noise: if True, also return the non-zero pixels that survived
    remove_small_objects (i.e. the real-signal pixels, not noise) as a third
    and fourth return value (non_noise_vals, n_non_noise).'''
    non_zero = image > threshold
    if non_zero.any():
        kept_mask = remove_small_objects(non_zero, min_size=min_size, connectivity=2)
    else:
        kept_mask = np.zeros_like(non_zero)
    removed_mask = non_zero & ~kept_mask
    noise_vals, n_noise = image[removed_mask], int(removed_mask.sum())

    if visualize:
        fig, ax = plt.subplots()
        ax.imshow(np.log2(1 + image), origin="lower", aspect="auto", cmap="viridis")
        ys, xs = np.where(removed_mask)
        white_red = LinearSegmentedColormap.from_list("white_red", ["white", "red"])
        noise_log_vals = np.log2(1 + noise_vals)
        sc = ax.scatter(xs, ys, c=noise_log_vals, cmap=white_red, s=18,
                         edgecolors="black", linewidths=0.4,
                         label=f"noise px\n(n={n_noise})")
        if n_noise:
            cbar = fig.colorbar(sc, ax=ax, shrink=0.9)
            cbar.set_label("noise pixel value (log2(1+x))")
        # ax.set_title(f"threshold={threshold}, min_size={min_size}")
        ax.set_xlabel("IM axis"); ax.set_ylabel("RT axis")
        # ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.15))
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
            if save_svg:
                fig.savefig(os.path.splitext(save_path)[0] + ".svg", bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    if log_non_noise:
        non_noise_vals, n_non_noise = image[kept_mask], int(kept_mask.sum())
        return noise_vals, n_noise, non_noise_vals, n_non_noise
    return noise_vals, n_noise


def _transform_nonzero_pixels(vals: np.ndarray):
    '''log2(1+x) transform when the pooled value range spans >50x (matches
    dev_swaps2.0_110_background_extraction_mix_of_gaussian.ipynb's
    convention), else identity. Returns (transform_fn, use_log2).'''
    positive = vals[vals > 0]
    use_log2 = bool((vals.max() / positive.min()) > 50)
    transform = (lambda v: np.log2(1 + v)) if use_log2 else (lambda v: v)
    return transform, use_log2


def _fit_two_component_gmm(vals: np.ndarray, random_state: int = 2024):
    '''Fits a 2-component GaussianMixture to (optionally log2(1+x)-
    transformed) pooled pixel values, seeded at the pooled distribution's
    two local density maxima instead of sklearn's default single
    k-means-seeded EM run.

    That default was verified (dev_swaps2.0_110_background_extraction_mix_of_gaussian.ipynb,
    both nanoflow_5min and microflow_30min HYE benchmarks) to reliably
    converge to a broad, overlapping local optimum on this skewed,
    heavy-tailed per-mode pixel-intensity data -- confirmed by sweeping
    GaussianMixture's n_init up to 20 with the default k-means init and
    landing on the identical (lower-likelihood) solution every time.
    Seeding each component at one of the histogram's visible peaks (found
    via a KDE + scipy.signal.find_peaks, falling back to a 25th/75th
    percentile split if the data isn't visibly bimodal) finds a
    higher-likelihood fit that also tracks the two visible modes.

    Returns (gmm, gmm_input, transform, use_log2).'''
    transform, use_log2 = _transform_nonzero_pixels(vals)
    gmm_input = transform(vals).reshape(-1, 1)

    xs = np.linspace(gmm_input.min(), gmm_input.max(), 500)
    density = gaussian_kde(gmm_input.ravel())(xs)
    peak_idx, _ = find_peaks(density)
    if len(peak_idx) >= 2:
        top2 = peak_idx[np.argsort(density[peak_idx])[-2:]]
        peak_locs = np.sort(xs[top2])
    else:
        peak_locs = np.quantile(gmm_input.ravel(), [0.25, 0.75])

    nearest_peak = np.argmin(np.abs(gmm_input - peak_locs.reshape(1, -1)), axis=1)
    means_init = np.array([[gmm_input[nearest_peak == k].mean()] for k in range(2)])
    precisions_init = np.array([[[1 / gmm_input[nearest_peak == k].var()]] for k in range(2)])
    weights_init = np.array([(nearest_peak == k).mean() for k in range(2)])

    gmm = GaussianMixture(
        n_components=2, random_state=random_state,
        means_init=means_init, precisions_init=precisions_init, weights_init=weights_init,
    ).fit(gmm_input)
    return gmm, gmm_input, transform, use_log2


def _save_gmm_fit_plot(gmm_input: np.ndarray, gmm: GaussianMixture, order: np.ndarray,
                        use_log2: bool, save_path: str, save_svg: bool = True) -> None:
    '''Overlays the fitted 2-component mixture (each weighted component
    density + their sum) on a histogram of gmm_input, saved as PNG
    (+ sibling .svg unless save_svg=False).'''
    means = gmm.means_.ravel()[order]
    stds = np.sqrt(gmm.covariances_.ravel()[order])
    weights = gmm.weights_[order]

    fig, ax = plt.subplots(figsize=(8, 5))
    xs = np.linspace(gmm_input.min(), gmm_input.max(), 500)
    ax.hist(gmm_input.ravel(), bins=100, density=True, color="lightgray", edgecolor="white",
            label="non-zero pixels")
    mixture_pdf = np.zeros_like(xs)
    for i, (w, m, s) in enumerate(zip(weights, means, stds)):
        component_pdf = w * norm.pdf(xs, loc=m, scale=s)
        mixture_pdf += component_pdf
        ax.plot(xs, component_pdf, linewidth=2, label=f"component {i} (w={w:.2f})")
    ax.plot(xs, mixture_pdf, color="black", linestyle="--", linewidth=2, label="mixture")
    ax.set_xlabel("log2(1 + pixel intensity)" if use_log2 else "pixel intensity")
    ax.set_ylabel("density")
    ax.set_title("floor_extraction: 2-component Gaussian mixture fit to sampled non-zero pixels")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if save_svg:
        fig.savefig(os.path.splitext(save_path)[0] + ".svg", bbox_inches="tight")
    plt.close(fig)


def estimate_dataset_noise_level(
    dict_ref: pd.DataFrame,
    raw_file_list: list,
    result_dir: str,
    n_candidates: int = 100,
    seed: int = 789,
    save_plot_path: Optional[str] = None,
) -> tuple:
    '''Dataset-level background pixel-intensity mean for floor_extraction.

    Samples n_candidates random mz_ranks from dict_ref (no organism/species
    filtering -- dataset-agnostic), then pools every non-zero pixel from
    every (mz_rank, run) image available for those mz_ranks across all
    raw_file_list runs (no noise/signal segmentation -- every non-zero
    pixel counts) -- same sampling/pooling scheme as
    dev_swaps2.0_110_background_extraction_mix_of_gaussian.ipynb.

    Fits a 2-component Gaussian mixture to the pooled pixel values via
    _fit_two_component_gmm (see its docstring for why the fit is seeded at
    the histogram's local maxima rather than left at sklearn's default
    init). The low-intensity ("background") component's raw-space mean is
    then estimated as the posterior-responsibility-weighted average of the
    raw (untransformed) pixel values -- not the naive back-transform of the
    fitted log-space mean, which is biased upward by the log2(1+x)
    transform for pixels close to 0 (verified empirically in the same
    notebook: back-transformed fitted mean vs. responsibility-weighted raw
    mean differ by ~10-20%).

    Meant to run once per dataset (shared across every FDR.METHOD variant),
    not per-image -- see MATCH_FEATURES_KWARGS.floor_extraction in
    singleton_swaps_optimization.py.

    save_plot_path: if given, saves the 2-component fit overlaid on the
    pooled pixel histogram as PNG there (+ sibling .svg).

    Returns (noise_mean, diagnostics) where diagnostics is a dict with
    n_images, n_pixels, and the fitted component weights/means/stds (in
    the same, possibly log2(1+x)-transformed, space the fit ran in) for
    logging.'''
    rng = np.random.default_rng(seed)
    dict_ref_by_mz = dict_ref.set_index("mz_rank") if dict_ref.index.name != "mz_rank" else dict_ref
    sample_mz_ranks = [
        int(m) for m in rng.choice(
            dict_ref["mz_rank"].values, size=min(n_candidates, len(dict_ref)), replace=False
        )
    ]

    nonzero_pixel_values = []
    n_images = 0
    for rf in raw_file_list:
        act_dir = os.path.join(result_dir, rf, "activation")
        act_df = load_peptide_batch_df_from_partquet(act_dir, sample_mz_ranks)
        act_by_mz = {int(mz): sub for mz, sub in act_df.groupby("mz_rank", sort=False)}
        for mz_rank in sample_mz_ranks:
            sub = act_by_mz.get(mz_rank)
            if sub is None or sub.empty:
                continue
            image, _, _ = get_pept_act_from_parquet(sub, mz_rank, dict_ref_by_mz, rf)
            nonzero_vals = image[image > 0].astype(float)
            if nonzero_vals.size:
                nonzero_pixel_values.append(nonzero_vals)
                n_images += 1

    if not nonzero_pixel_values:
        raise ValueError(
            "floor_extraction: no non-zero pixels found in any sampled "
            "candidate -- cannot estimate noise level. Check n_candidates_sampled."
        )
    all_nonzero_vals = np.concatenate(nonzero_pixel_values)

    gmm, gmm_input, _, use_log2 = _fit_two_component_gmm(all_nonzero_vals)
    order = np.argsort(gmm.means_.ravel())
    component_probs = gmm.predict_proba(gmm_input)[:, order]
    weights = gmm.weights_[order]
    means = gmm.means_.ravel()[order]
    stds = np.sqrt(gmm.covariances_.ravel()[order])

    noise_responsibility = component_probs[:, 0]
    noise_mean = float(np.sum(noise_responsibility * all_nonzero_vals) / np.sum(noise_responsibility))

    diagnostics = {
        "n_images": n_images,
        "n_pixels": int(len(all_nonzero_vals)),
        "component_weights": weights.tolist(),
        "component_means": means.tolist(),
        "component_stds": stds.tolist(),
        "use_log2": use_log2,
        "noise_mean": noise_mean,
    }
    logging.info(
        "floor_extraction: noise_mean=%.4f (responsibility-weighted raw pixel mean of "
        "the low-intensity GMM component; weights=%s, means=%s%s) from %d non-zero "
        "pixels across %d images",
        noise_mean, np.round(weights, 3).tolist(), np.round(means, 3).tolist(),
        " [log2(1+x) space]" if use_log2 else "", diagnostics["n_pixels"], n_images,
    )

    if save_plot_path:
        _save_gmm_fit_plot(gmm_input, gmm, order, use_log2, save_plot_path)

    return noise_mean, diagnostics


def apply_floor_extraction(df: pd.DataFrame, noise_mean: float) -> pd.DataFrame:
    '''In-place-swap background correction: intensity_sum becomes
    (intensity_sum - noise_mean * area).clip(lower=0); the pre-correction
    value is preserved as intensity_sum_without_floor_correction. Every
    existing downstream consumer that reads intensity_sum by name
    (build_pivot/DirectLFQ, calc_quant_corr, FDR.INT_THRES, ...) picks up
    the corrected value with no other code changes.'''
    df = df.copy()
    background_estimate = noise_mean * df["area"]
    df["intensity_sum_without_floor_correction"] = df["intensity_sum"]
    df["floor_background_estimate"] = background_estimate
    df["intensity_sum"] = (df["intensity_sum"] - background_estimate).clip(lower=0)
    return df