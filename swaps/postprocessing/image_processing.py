import logging
from typing import Any, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter, uniform_filter, gaussian_filter1d
from skimage.morphology import remove_small_objects, h_maxima
from skimage.feature import match_template

from skimage.segmentation import watershed
from skimage.measure import regionprops_table
from skimage.filters import sobel
import cv2
import matplotlib.pyplot as plt

from mahotas.features import zernike_moments
from scipy.spatial.distance import cosine

Logger = logging.getLogger(__name__)


def detect_2d_peak_with_watershed(
    image,
    int_threshold=0.5,
    h_rel: float = 0.15,
    norm_percentile: int = 95,
    coordinates: Optional[np.ndarray] = None,
    seed_radius: int = 0,
    use_competing_peaks: bool = True,  # new: enable/disable the feature
    # min_distance_to_true_seed: int = 15,  # new: minimum distance from competing peaks to the true seed
    visualize: bool = False,
):
    """
    Detect peaks in a 2D image using the watershed algorithm.

    Parameters:
    - pept_act_log: 2D numpy array
        The input image in which to detect peaks. Usually log10 transformed intensity.
    - log_threshold: float
        Threshold for log-transformed intensity to create the signal mask.
    - h_rel: float
        Prominence threshold for h-maxima: a peak must rise ≥ h_rel × norm_percentile
        of the signal above its surrounding saddle.
    - norm_percentile: int
        Percentile of in-mask intensity used to normalise the image before h-maxima.
    - visualize: bool
        If True, show a step-by-step matplotlib figure of each stage.
    Returns:
    - labels: 2D numpy array
        Labeled regions corresponding to detected peaks.
    """

    def _viz_step(ax, data, title, cmap="viridis", points=None, point_styles=None):
        """Helper to draw one panel. points is a list of (coords_array, kwargs) tuples."""
        ax.imshow(data, origin="lower", cmap=cmap, aspect="auto")
        ax.set_title(title, fontsize=8)
        ax.axis("off")
        if points:
            for coords, kwargs in points:
                if len(coords):
                    ax.scatter(coords[:, 1], coords[:, 0], **kwargs)

    # 2. Compute distance (to background) transform inside signal
    mask_signal = image > int_threshold
    _coords_provided = coordinates is not None
    # snapped_seed is initialised after coordinates are resolved below.
    snapped_seed = coordinates[0] if coordinates is not None else None
    if not mask_signal.any():
        image[~mask_signal] = 0
        return (
            np.empty((0, 2), dtype=int),
            np.zeros_like(image, dtype=int),
            np.zeros_like(image, dtype=float),
            np.zeros_like(image, dtype=int),
            np.empty((0, 2), dtype=int),
        )

    # Normalise once; shared by both auto-detect and competing-peaks branches.
    _pN = np.percentile(image[mask_signal], norm_percentile)
    _norm_image = image / _pN if _pN > 0 else image

    if coordinates is None:
        # Auto-detect mode: peaks must rise ≥ h_rel × p(norm_percentile) above saddle.
        hmax = h_maxima(_norm_image, h=h_rel) & mask_signal
        if hmax.any():
            peak_labels, n_peaks = ndi.label(hmax)
            coordinates = np.array(
                [
                    np.unravel_index(
                        np.argmax(np.where(peak_labels == k, image, -np.inf)),
                        image.shape,
                    )
                    for k in range(1, n_peaks + 1)
                ],
                dtype=int,
            )
        else:
            coordinates = np.empty((0, 2), dtype=int)
        # In multi-peak mode snapped_seed is not meaningful; initialise to
        # first detected peak as a safe fallback for the return value.
        if coordinates.size > 0:
            snapped_seed = coordinates[0]

    if coordinates.size == 0:
        return (
            coordinates,
            np.zeros_like(image, dtype=int),
            np.zeros_like(image, dtype=float),
            np.zeros_like(image, dtype=int),
            coordinates,
        )

    # By this point coordinates is a non-empty array; ensure snapped_seed is
    # resolved (it is None only when coordinates was None on entry).
    if snapped_seed is None:
        snapped_seed = coordinates[0]

    mask = np.zeros(image.shape, dtype=bool)

    # Visualization state collected throughout
    _viz = {}

    # --- new: inject competing background seeds when true seed is provided ---
    if use_competing_peaks and coordinates.shape[0] == 1:
        # gradient = sobel(image)

        _hmax_bg = h_maxima(_norm_image, h=h_rel) & mask_signal
        if _hmax_bg.any():
            _lbl_bg, _n_bg = ndi.label(_hmax_bg)
            bg_peaks = np.array(
                [
                    np.unravel_index(
                        np.argmax(np.where(_lbl_bg == k, image, -np.inf)),
                        image.shape,
                    )
                    for k in range(1, _n_bg + 1)
                ],
                dtype=int,
            )
        else:
            bg_peaks = np.empty((0, 2), dtype=int)

        if len(bg_peaks) > 0:
            true_seed = coordinates[0]

            # Label connected components in the signal mask
            connected_components, _ = ndi.label(mask_signal)
            true_seed_component = connected_components[tuple(true_seed)]

            # Filter peaks to only those in the same connected component
            same_component_mask = np.array(
                [
                    connected_components[tuple(p)] == true_seed_component
                    for p in bg_peaks
                ]
            )

            if same_component_mask.any():
                same_component_peaks = bg_peaks[same_component_mask]
                component_mask = connected_components == true_seed_component

                # Run a provisional watershed using all local maxima in the same
                # connected component. We only accept a snapped seed if its
                # watershed region actually contains the true seed.
                candidate_mask = np.zeros_like(mask, dtype=bool)
                candidate_mask[tuple(same_component_peaks.T)] = True
                candidate_markers, _ = ndi.label(candidate_mask)
                candidate_labels = watershed(
                    -image,
                    candidate_markers,
                    mask=component_mask,
                    compactness=0.001,
                )
                true_seed_label = candidate_labels[tuple(true_seed)]

                if true_seed_label > 0:
                    snapped_peak_mask = np.array(
                        [
                            candidate_markers[tuple(p)] == true_seed_label
                            for p in same_component_peaks
                        ]
                    )
                    if snapped_peak_mask.any():
                        snapped_seed = same_component_peaks[
                            np.flatnonzero(snapped_peak_mask)[0]
                        ]
                    else:
                        snapped_seed = true_seed
                else:
                    # Fallback: if the true seed is not captured by any
                    # candidate watershed region, keep the original seed.
                    snapped_seed = true_seed
            else:
                # No peak in same component, fall back to true seed
                same_component_peaks = np.empty(
                    (0, bg_peaks.shape[1]), dtype=bg_peaks.dtype
                )
                snapped_seed = true_seed

            mask[tuple(snapped_seed)] = True  # foreground anchor (snapped)

            for bg_peak in same_component_peaks:
                if np.array_equal(bg_peak, snapped_seed):
                    continue
                mask[tuple(bg_peak)] = True  # background competitors

            if visualize:
                _viz["true_seed"] = true_seed
                _viz["bg_peaks"] = bg_peaks
                _viz["connected_components"] = connected_components
                _viz["same_component_peaks"] = same_component_peaks
                _viz["snapped_seed"] = snapped_seed
                competitors = np.array(
                    [
                        p
                        for p in same_component_peaks
                        if not np.array_equal(p, snapped_seed)
                    ]
                )
                _viz["competitors"] = competitors

    else:
        # Multi-peak / auto-detect mode: use every detected coordinate
        # directly as a watershed seed (original intended behaviour).
        mask[tuple(coordinates.T)] = True

    # -------------------------------------------------------------------------

    markers, _ = ndi.label(mask)
    if seed_radius > 0:
        dist, (ri, ci) = ndi.distance_transform_edt(~mask, return_indices=True)
        markers = np.where(dist <= seed_radius, markers[ri, ci], 0)

    labels_multi_markers = watershed(
        -image, markers, mask=mask_signal, compactness=0.001
    )

    # --- new: when competing peaks were used, only return true seed's region ---
    if use_competing_peaks and coordinates.shape[0] == 1:
        true_marker = markers[tuple(snapped_seed)]
        labels = np.where(labels_multi_markers == true_marker, true_marker, 0)
    # --------------------------------------------------------------------------
    else:
        labels = labels_multi_markers

    # ---- stepwise visualization ----
    if visualize:
        n_cols = 6
        fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

        # Step 1: raw image + input coordinates
        _viz_step(
            axes[0],
            image,
            "1. Image + input seed",
            points=[
                (
                    coordinates,
                    dict(c="red", s=40, marker="x", label="input seed", zorder=5),
                )
            ],
        )

        # Step 2: signal mask
        _viz_step(
            axes[1],
            mask_signal.astype(np.uint8),
            "2. Signal mask (> threshold)",
            cmap="gray",
        )

        # Step 3: bg peaks + true seed (competing peaks mode)
        if _viz:
            bg_all = _viz["bg_peaks"]
            ts = _viz["true_seed"][np.newaxis]
            _viz_step(
                axes[2],
                image,
                "3. All bg peaks (same cpt filtered)",
                points=[
                    (
                        bg_all,
                        dict(
                            c="orange", s=30, marker="o", label="all bg peaks", zorder=4
                        ),
                    ),
                    (
                        _viz["same_component_peaks"],
                        dict(
                            c="cyan", s=40, marker="^", label="same component", zorder=5
                        ),
                    ),
                    (ts, dict(c="red", s=60, marker="x", label="true seed", zorder=6)),
                ],
            )
            # Step 4: connected components + snapped seed
            _viz_step(
                axes[3],
                _viz["connected_components"],
                "4. Connected components + snapped seed",
                cmap="tab20",
                points=[
                    (
                        _viz["same_component_peaks"],
                        dict(c="cyan", s=40, marker="^", zorder=4),
                    ),
                    (
                        _viz["snapped_seed"][np.newaxis],
                        dict(
                            c="lime", s=80, marker="*", label="snapped seed", zorder=6
                        ),
                    ),
                ],
            )
        else:
            _viz_step(axes[2], image, "3. (no competing peaks)", cmap="gray")
            _viz_step(axes[3], image, "4. (no competing peaks)", cmap="gray")

        # Step 5: markers (seeds passed to watershed)
        snapped = (
            _viz.get("snapped_seed", coordinates[0])
            if coordinates.shape[0] == 1
            else None
        )
        seed_pts = np.array([snapped]) if snapped is not None else coordinates
        _viz_step(
            axes[4],
            markers,
            "5. Markers for watershed",
            cmap="tab20",
            points=[
                (
                    seed_pts,
                    dict(c="lime", s=80, marker="*", label="snapped seed", zorder=5),
                )
            ],
        )

        # Step 6: final labels vs all-marker watershed
        _viz_step(
            axes[5],
            labels_multi_markers,
            "6. All-marker watershed  |  final (outline)",
            cmap="tab20",
        )
        # overlay final region as contour
        from skimage.segmentation import find_boundaries

        boundary = find_boundaries(labels > 0, mode="outer")
        axes[5].contour(boundary, levels=[0.5], colors="white", linewidths=1)

        for ax in axes:
            ax.legend(fontsize=6, loc="upper right", markerscale=0.8)

        fig.suptitle(
            f"detect_2d_peak_with_watershed  |  int_threshold={int_threshold}  "
            f"h_rel={h_rel}  norm_percentile={norm_percentile}  "
            f"seed_radius={seed_radius}",
            fontsize=9,
        )
        fig.tight_layout()
        plt.show()
    # --------------------------------

    # When one coordinate was explicitly provided, snapped_seed is the peak that lies
    # inside the kept label (it may differ from the original coordinate after bg-competition
    # snapping). In auto-detect mode every detected coordinate is already a watershed marker
    # and therefore guaranteed to be inside its own label.
    if _coords_provided:
        peaks_out = np.atleast_2d(snapped_seed)
    else:
        peaks_out = coordinates

    # Invariant: every non-zero label in the returned labels must contain at least one
    # returned peak. Violated peaks would cause silent discards in the snapping step.
    if peaks_out.shape[0] > 0 and labels.max() > 0:
        peak_label_vals = labels[peaks_out[:, 0], peaks_out[:, 1]]
        for lbl in np.unique(labels[labels > 0]):
            assert (
                lbl in peak_label_vals
            ), f"Watershed label {lbl} has no corresponding peak in returned coordinates"

    return peaks_out, labels, image, labels_multi_markers, snapped_seed


def calculate_peak_property_from_labels_and_image(
    labels,
    image_2d,
    grad_mag: Optional[np.ndarray] = None,
    image_res_2d: Optional[np.ndarray] = None,
    min_peak_area=10,
    min_peak_sum_intensity=1000,
    return_dy: bool = False,
):
    """
    Calculate properties of detected peaks from coordinates of the local maximum and labels.

    Parameters:
    - labels: 2D numpy array
        The labeled regions corresponding to detected peaks.
    - image_2d: 2D numpy array
        The original image from which to calculate peak properties.
    - image_2d_log: 2D numpy array
        The log-transformed image from which to calculate peak properties.
    - min_peak_area: int
        Minimum area of peaks to be considered valid.
    - min_peak_sum_intensity: float
        Minimum sum intensity of peaks to be considered valid.

    Returns:
    - df: pd.DataFrame
        A DataFrame containing peak properties, including row-based smoothness.
    """
    num_labels = labels.max()
    if num_labels == 0:
        # Logger.debug("No peaks detected after watershed.")
        return None

    # Region properties from skimage (fast, C-based)
    props = regionprops_table(
        labels,
        intensity_image=image_2d,
        properties=(
            "label",
            "centroid",
            "area",
            "area_filled",
            "bbox",
            "intensity_max",
            "intensity_mean",
            "intensity_std",
            # "intensity_median",
            "solidity",
        ),
    )
    df = pd.DataFrame(props)
    # Filter out small/weak peaks
    df["intensity_sum"] = df["intensity_mean"] * df["area"]
    df = df[
        (df["area"] >= min_peak_area) & (df["intensity_sum"] >= min_peak_sum_intensity)
    ]
    if df.empty:
        Logger.debug(
            "All detected peaks are filtered out by min area %s and min intensity sum %s",
            min_peak_area,
            min_peak_sum_intensity,
        )
        return None
    # Derived properties

    df["intensity_cv"] = df["intensity_std"] / (df["intensity_mean"] + 1e-8)
    df["im_length"] = df["bbox-3"] - df["bbox-1"]
    df["rt_length"] = df["bbox-2"] - df["bbox-0"]
    df["image_total_intensity"] = image_2d.sum()
    # Logger.info("Unique label values after filtering: %s", df["label"].values)
    if image_res_2d is not None:
        res_act_ratio = np.log1p(image_res_2d / (image_2d + 1e-6))
        props_res = regionprops_table(
            labels,
            intensity_image=res_act_ratio,
            properties=(
                "label",
                "intensity_max",
                "intensity_min",
                "intensity_mean",
                "intensity_std",
            ),
        )
        df_res = pd.DataFrame(props_res)
        df = df.merge(df_res, on="label", how="left", suffixes=("", "_res"))
        # df.drop(columns=["label_res"], inplace=True)

    # Compute row-based smoothness if gradient magnitude is provided
    if grad_mag is not None:
        row_smoothness_df = compute_row_smoothness_and_apex_index(
            labels=labels,
            dy=grad_mag,
            pept_act=image_2d,
            label_values=df["label"].values,
            apply_gaussian_smoothing=False,
        )

        df = df.merge(row_smoothness_df, on="label", how="left")
        if return_dy:
            return df, grad_mag
    else:
        return df


def compute_row_smoothness_and_apex_index(
    labels,
    dy,
    pept_act,
    label_values,
    apply_gaussian_smoothing: bool = True,
    sigma: float = 1.0,
):
    """
    Compute row-wise smoothness metrics for each labeled region.

    Parameters
    ----------
    labels : 2D numpy array
        Labeled peak regions.
    dy : 2D numpy array
        Gradient along rows (RT dimension).
    label_values : list or array-like
        Unique label values to compute metrics for.
    apply_gaussian_smoothing : bool, optional
        Whether to apply Gaussian smoothing to column profiles before computing second derivatives. Default is True.
    sigma : float, optional
        Standard deviation for Gaussian kernel if smoothing is applied. Default is 1.0.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - label
        - within_row_consistency
        - across_row_score_smoothness
        - row_smoothness (geometric mean)
    """
    records = []

    for lbl in label_values:
        mask = labels == lbl
        rows = np.where(np.any(mask, axis=1))[0]
        cols = np.where(np.any(mask, axis=0))[0]
        if len(rows) == 0 or len(cols) == 0:
            records.append(
                {
                    "label": lbl,
                    "within_row_consistency": np.nan,
                    "across_row_score_smoothness": np.nan,
                    "row_smoothness": np.nan,
                }
            )
            continue
        rt_apex = rows[np.argmax(pept_act[rows, :][:, cols].sum(axis=1))]
        im_apex = cols[np.argmax(pept_act[:, cols][rows, :].sum(axis=0))]
        # 1️⃣ within-row score: stability of dy along each row
        std_vals = [np.std(dy[r, mask[r, :]]) for r in rows]
        within_row_consistency = 1 / (1 + np.mean(std_vals))

        # 2️⃣ across-row score: smooth Gaussian-like shape per column
        col_scores = []
        for c in cols:
            col_mask = mask[:, c]
            col_vals = dy[:, c][col_mask]
            if len(col_vals) < 3:
                continue  # need at least 3 points to compute second derivative
            if apply_gaussian_smoothing:
                col_vals = gaussian_filter1d(col_vals, sigma=sigma)
            # col_vals_smooth = gaussian_filter1d(col_vals, sigma=1.0)
            second_deriv = np.diff(col_vals, n=2)
            col_score = 1 / (1 + np.mean(np.abs(second_deriv)))
            col_scores.append(col_score)

        if len(col_scores) == 0:
            across_row_score_smoothness = np.nan
        else:
            across_row_score_smoothness = np.mean(col_scores)

        # 3️⃣ geometric mean as combined score
        geom_mean = np.sqrt(within_row_consistency * across_row_score_smoothness)

        records.append(
            {
                "label": lbl,
                "within_row_consistency": within_row_consistency,
                "across_row_score_smoothness": across_row_score_smoothness,
                "row_smoothness": geom_mean,
                "rt_apex_index": rt_apex,
                "im_apex_index": im_apex,
            }
        )

    return pd.DataFrame(records)


def smooth_and_denoise_image(
    image: np.ndarray,
    smooth: dict | None = None,
    clean: dict | None = None,
    log_transform: bool = False,
) -> np.ndarray:
    """Apply denoising steps independently to a 2D image.

    Each step is optional (None / False skips it). Steps run in order:
    smooth → clean → log_transform.

    Parameters
    ----------
    image : 2D array
    smooth : dict or None
        ``{"filter": "gaussian"|"uniform",
           "gaussian_kwargs": {"sigma": ..., "mode": ...},
           "uniform_kwargs": {"size": ...}}``
    clean : dict or None
        ``{"threshold": float, "remove_kwargs": {"min_size": int}}``
    log_transform : bool
        Apply ``log2(1 + x)`` after smooth/clean.
    """
    result = image
    if smooth is not None:
        smooth = dict(smooth)
        filter_type = smooth.get("filter", "gaussian")
        g_kw = dict(smooth.get("gaussian_kwargs") or {})
        u_kw = dict(smooth.get("uniform_kwargs") or {})
        if "sigma" not in g_kw:
            g_kw["sigma"] = 2
            g_kw["mode"] = "nearest"
        if "size" not in u_kw:
            u_kw["size"] = (1, 5)
        if filter_type == "gaussian":
            result = gaussian_filter(result, **g_kw)
        elif filter_type == "uniform":
            blurred = uniform_filter(result, **u_kw)
            result = np.maximum(result, blurred)
    if clean is not None:
        clean = dict(clean)
        threshold = clean.get("threshold", 10)
        r_kw = dict(clean.get("remove_kwargs") or {})
        if "min_size" not in r_kw:
            r_kw["min_size"] = 5
        cleaned_mask = remove_small_objects(result >= threshold, **r_kw)
        result = result * cleaned_mask
    if log_transform:
        result = np.log2(1 + result)
    return result


def get_orb_peak_descriptor(
    img, peak_coords, patch_size=100
):  # This doesn't work well when image is noisy or only one smooth peak exists
    """
    Computes the ORB descriptor for a specific peak.
    Returns the descriptor (feature vector).

    Parameters
    ----------
    img : 2D array
        Input image (should be in uint8 format).
    peak_coords : tuple
        (y, x) coordinates of the peak for which to compute the descriptor.
    patch_size : int, optional
        Size of the patch around the peak to consider for descriptor computation. Default is 31.
    """
    # 1. Normalize and convert to 8-bit once
    img_8bit = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

    # 2. Initialize ORB
    orb = cv2.ORB_create()
    y, x = peak_coords

    # 3. Create the KeyPoint at the peak
    kp = [cv2.KeyPoint(x=float(x), y=float(y), size=patch_size)]

    # 4. Compute the descriptor
    _, des = orb.compute(img_8bit, kp)

    return des


def get_sift_descriptor(img, peak_coords, patch_size=31):
    """
    Computes a SIFT descriptor for a specific peak coordinate.
    """
    # 1. SIFT works best on 8-bit images.
    # Normalization ensures intensity differences don't break the gradient math.
    img_8bit = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

    # 2. Initialize SIFT
    sift = cv2.SIFT_create()

    y, x = peak_coords

    # 3. Create a KeyPoint.
    # 'size' determines the area the descriptor looks at.
    # 'angle=0' is used because your images are already aligned.
    kp = [cv2.KeyPoint(x=float(x), y=float(y), size=patch_size, angle=0)]

    # 4. Compute the descriptor
    _, des = sift.compute(img_8bit, kp)

    return des


def get_roi_descriptor(roi, radius=None, degree=8):
    roi = roi.astype(np.float32)

    # 1. Percentile clip BEFORE normalization — kills hot pixels
    p1, p99 = np.percentile(roi, 1), np.percentile(roi, 99)
    roi_clipped = np.clip(roi, p1, p99)

    # 2. Check for flat patch AFTER clipping
    if roi_clipped.max() - roi_clipped.min() < 1e-6:
        n_moments = len(
            zernike_moments(np.zeros((3, 3), dtype=np.uint8), 1, degree=degree)
        )
        # Logger.debug(
        #     f"Flat ROI detected (shape={roi.shape}), returning zero zernike descriptor."
        # )
        return np.zeros(n_moments, dtype=np.float64)

    # 3. Normalize clipped patch
    roi_norm = (roi_clipped - roi_clipped.min()) / (
        roi_clipped.max() - roi_clipped.min()
    )
    roi_uint8 = (roi_norm * 255).astype(np.uint8)

    # 4. Radius based on min dimension so circle fits within patch
    if radius is None:
        radius = min(roi.shape) // 2

    # 5. Center: mahotas cm is (row, col)
    cm = (roi.shape[0] // 2, roi.shape[1] // 2)

    moments = zernike_moments(roi_uint8, radius, cm=cm, degree=degree)
    return moments


def fast_intensity_weighted_ncc(image, template, power=2.0):
    image = image.astype(np.float64)
    template = template.astype(np.float64)
    th, tw = template.shape

    # Weight map — fixed for this template
    weight = (template / 255.0) ** power
    weight /= weight.sum()

    t_mean = np.sum(weight * template)
    t_centered = template - t_mean
    t_var = np.sum(weight * t_centered**2)

    # Precompute per-pixel weighted image using 2D correlation
    # weight * I(patch) mean → cross-correlate weight map with image
    w_sum_map = cv2.filter2D(image, -1, weight, anchor=(0, 0))[: -(th - 1), : -(tw - 1)]
    w_i2_map = cv2.filter2D(image**2, -1, weight, anchor=(0, 0))[
        : -(th - 1), : -(tw - 1)
    ]
    w_cross_map = cv2.filter2D(image, -1, weight * t_centered, anchor=(0, 0))[
        : -(th - 1), : -(tw - 1)
    ]

    i_mean_map = w_sum_map  # E_w[I]
    i_var_map = w_i2_map - i_mean_map**2  # E_w[I²] - E_w[I]²

    # Numerator: Σ w · (I - μ_I)(T - μ_T)
    # = Σ w·I·(T-μ_T) - μ_I · Σ w·(T-μ_T)
    # Note: Σ w·(T-μ_T) = 0 by definition of mean, so:
    numerator = w_cross_map - i_mean_map * np.sum(weight * t_centered)
    denominator = np.sqrt(t_var * np.clip(i_var_map, 0, None)) + 1e-8

    return (numerator / denominator).astype(np.float32)
