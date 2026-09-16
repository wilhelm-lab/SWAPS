import os
import logging
from typing import Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from skimage.morphology import remove_small_objects

from .helper import load_peptide_batch_df_from_partquet, get_pept_act_from_parquet

def extract_noise_pixels_no_smooth(
    image: np.ndarray,
    threshold: float = 0,
    min_size: int = 3,
    visualize: bool = False,
    save_path: Optional[str] = None,
    save_svg: bool = True,
):
    '''Matches 03_HYE.ipynb's current cells 44/45: no smoothing, near-zero
    threshold, min_size=3 -- operates directly on the raw activation image.

    visualize: if True, plot the original image (log2(1+x) scale) with the
    selected noise pixels marked, colored white-to-red by each pixel's own
    log2(1+x) value (whiter = lower, redder = higher).
    save_path: if given (with visualize=True), save the figure as PNG there,
    plus a sibling .svg unless save_svg=False (matches direct_lfq._save_fig's
    PNG+SVG convention).'''
    non_zero = image > threshold
    if non_zero.any():
        kept_mask = remove_small_objects(non_zero, min_size=min_size, connectivity=2)
        removed_mask = non_zero & ~kept_mask
    else:
        removed_mask = np.zeros_like(non_zero)
    noise_vals, n_noise = image[removed_mask], int(removed_mask.sum())

    if visualize:
        fig, ax = plt.subplots(figsize=(6.2, 4.5))
        ax.imshow(np.log2(1 + image), origin="lower", aspect="auto", cmap="viridis")
        ys, xs = np.where(removed_mask)
        white_red = LinearSegmentedColormap.from_list("white_red", ["white", "red"])
        noise_log_vals = np.log2(1 + noise_vals)
        sc = ax.scatter(xs, ys, c=noise_log_vals, cmap=white_red, s=18,
                         edgecolors="black", linewidths=0.4,
                         label=f"noise px\n(n={n_noise})")
        if n_noise:
            cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
            cbar.set_label("noise pixel value (log2(1+x))")
        ax.set_title(f"threshold={threshold}, min_size={min_size}")
        ax.set_xlabel("IM axis"); ax.set_ylabel("RT axis")
        ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.25, 1.05))
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            if save_svg:
                fig.savefig(os.path.splitext(save_path)[0] + ".svg", bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    return noise_vals, n_noise


def estimate_dataset_noise_median(
    dict_ref: pd.DataFrame,
    raw_file_list: list,
    result_dir: str,
    n_candidates: int = 100,
    seed: int = 789,
    threshold: float = 0.0,
    min_size: int = 3,
) -> tuple:
    '''Dataset-level background noise-pixel median for floor_extraction.

    Samples n_candidates random mz_ranks from dict_ref (no organism/species
    filtering -- dataset-agnostic) and extracts "noise" pixels from each
    run's own raw activation image via extract_noise_pixels_no_smooth,
    pooling every removed pixel's own intensity across the sample. Meant to
    run once per dataset (shared across every FDR.METHOD variant), not
    per-image -- see MATCH_FEATURES_KWARGS.floor_extraction in
    singleton_swaps_optimization.py.

    Returns (noise_median, diagnostics) where diagnostics is a dict with
    n_images, n_noise_pixels, noise_mean for logging.'''
    rng = np.random.default_rng(seed)
    dict_ref_by_mz = dict_ref.set_index("mz_rank") if dict_ref.index.name != "mz_rank" else dict_ref
    sample_mz_ranks = [
        int(m) for m in rng.choice(
            dict_ref["mz_rank"].values, size=min(n_candidates, len(dict_ref)), replace=False
        )
    ]

    noise_pixel_values = []
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
            noise_vals, n_noise = extract_noise_pixels_no_smooth(
                image, threshold=threshold, min_size=min_size
            )
            if n_noise:
                noise_pixel_values.append(noise_vals)
            n_images += 1

    if not noise_pixel_values:
        raise ValueError(
            "floor_extraction: no noise pixels extracted from any sampled "
            "candidate -- cannot estimate noise_median. Check n_candidates_sampled, "
            "threshold, min_size."
        )
    noise_pixel_values = np.concatenate(noise_pixel_values)
    diagnostics = {
        "n_images": n_images,
        "n_noise_pixels": int(len(noise_pixel_values)),
        "noise_mean": float(np.mean(noise_pixel_values)),
    }
    noise_median = float(np.median(noise_pixel_values))
    logging.info(
        "floor_extraction: noise_median=%.4f from %d noise pixels across %d images "
        "(mean=%.4f)",
        noise_median, diagnostics["n_noise_pixels"], n_images, diagnostics["noise_mean"],
    )
    return noise_median, diagnostics


def apply_floor_extraction(df: pd.DataFrame, noise_median: float, noise_fraction: float) -> pd.DataFrame:
    '''In-place-swap background correction: intensity_sum becomes
    (intensity_sum - noise_median * area * noise_fraction).clip(lower=0);
    the pre-correction value is preserved as
    intensity_sum_without_floor_correction. Every existing downstream
    consumer that reads intensity_sum by name (build_pivot/DirectLFQ,
    calc_quant_corr, FDR.INT_THRES, ...) picks up the corrected value with
    no other code changes.'''
    df = df.copy()
    background_estimate = noise_median * df["area"] * noise_fraction
    df["intensity_sum_without_floor_correction"] = df["intensity_sum"]
    df["floor_background_estimate"] = background_estimate
    df["intensity_sum"] = (df["intensity_sum"] - background_estimate).clip(lower=0)
    return df