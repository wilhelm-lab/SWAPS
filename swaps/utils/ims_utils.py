import logging
from typing import Optional, Literal
import sparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt

from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import regionprops_table
from skimage.filters import sobel
from scipy.ndimage import gaussian_filter1d, uniform_filter, gaussian_filter
from skimage.morphology import remove_small_objects
import alphatims.bruker

Logger = logging.getLogger(__name__)


def load_dotd_data(dotd_file_path: str, swaps_result_dir: str = ""):
    """
    Load .d file data and save hdf5 if not already exists.
    :param: dotd_file_path: str, path to the .d file
    :param: swaps_result_dir: str, path to the directory to save the hdf5 file, optional, default .d directory
    :return: data: alphatims.bruker.TimsTOF, data object
    :return: hdf_file_name: str, path to the saved hdf5 file
    """
    data = alphatims.bruker.TimsTOF(dotd_file_path)
    if swaps_result_dir == "":
        Logger.info("No output directory provided, using the directory of the .d file")
        swaps_result_dir = os.path.dirname(dotd_file_path)
    os.makedirs(os.path.join(swaps_result_dir), exist_ok=True)
    hdf_path = os.path.join(swaps_result_dir, f"{data.sample_name}.hdf")
    if not os.path.isfile(hdf_path):
        hdf_file_name = data.save_as_hdf(
            directory=swaps_result_dir,
            file_name=f"{data.sample_name}.hdf",
            overwrite=False,
        )
        Logger.info("HDF file saved as %s", hdf_file_name)
    else:
        hdf_file_name = hdf_path
        Logger.info("HDF file %s already exists", hdf_file_name)
    return data, hdf_file_name


def export_im_and_ms1scans(
    data: alphatims.bruker.TimsTOF, swaps_result_dir: Optional[str] = None
):
    """
    Export IM and MS1 scans to csv files.
    :param: data: alphatims.bruker.TimsTOF, data object
    :param: swaps_result_dir: str, path to the directory to save the csv files, optional, default None then no export
    :return: ms1scans: pd.DataFrame, MS1 scans
    :return: mobility_values_df: pd.DataFrame, mobility values
    """
    # ms1scans
    ms1scans = data.frames.loc[data.frames.MsMsType == 0].copy()
    ms1scans["Time_minute"] = ms1scans["Time"] / 60
    ms1scans["MS1_frame_idx"] = (
        ms1scans["Time"].rank(axis=0, method="first", ascending=True).astype(int) - 1
    )  # 0-based index
    ms1scans.set_index("MS1_frame_idx", inplace=True, drop=False)
    Logger.info(
        "Double check MS1 frame index range: %s - %s",
        ms1scans["MS1_frame_idx"].min(),
        ms1scans["MS1_frame_idx"].max(),
    )

    # mobilty values
    mobility_values = np.sort(data.mobility_values)
    mobility_values_df = pd.DataFrame(
        mobility_values, columns=["mobility_values"]
    ).reset_index()
    mobility_values_df = mobility_values_df.rename(
        columns={"index": "mobility_values_index"}
    )
    Logger.info(
        "Double check mobility values index range: %s - %s",
        mobility_values_df["mobility_values_index"].min(),
        mobility_values_df["mobility_values_index"].max(),
    )
    ms1scans = pd.DataFrame(ms1scans)
    # export if swaps_result_dir is not None
    if swaps_result_dir is not None:
        os.makedirs(os.path.join(swaps_result_dir), exist_ok=True)
        ms1scans.to_csv(os.path.join(swaps_result_dir, "ms1scans.csv"))
        mobility_values_df.to_csv(os.path.join(swaps_result_dir, "mobility_values.csv"))
    return ms1scans, mobility_values_df




def detect_2d_peak_with_watershed(
    image,
    int_threshold=0.5,
    min_distance=15,
    threshold_rel=0.2,
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
    - min_distance: int
        Minimum distance between detected peaks.
    - threshold_rel: float
        Minimum intensity of peaks, calculated as max(image) * threshold_rel.
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
    distance = image
    mask_signal = distance > int_threshold
    snapped_seed = coordinates[0]
    if not mask_signal.any():
        distance[~mask_signal] = 0
        return (
            np.empty((0, 2), dtype=int),
            np.zeros_like(image, dtype=int),
            np.zeros_like(image, dtype=float),
            np.zeros_like(image, dtype=int),
            np.empty((0, 2), dtype=int),
        )

    if coordinates is None:
        # --- original behavior unchanged ---
        coordinates = peak_local_max(
            distance,
            min_distance=min_distance,
            threshold_rel=threshold_rel,
            labels=mask_signal,
        )

    if coordinates.size == 0:
        return (
            coordinates,
            np.zeros_like(image, dtype=int),
            np.zeros_like(image, dtype=float),
            np.zeros_like(image, dtype=int),
            coordinates,
        )

    mask = np.zeros(image.shape, dtype=bool)
    # mask[tuple(coordinates.T)] = True

    # Visualization state collected throughout
    _viz = {}

    # --- new: inject competing background seeds when true seed is provided ---
    if use_competing_peaks and coordinates.shape[0] == 1:
        # gradient = sobel(image)

        bg_peaks = peak_local_max(
            image,
            min_distance=min_distance,
            threshold_rel=threshold_rel,
            labels=mask_signal,
        )

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
                distances = np.linalg.norm(same_component_peaks - true_seed, axis=1)
                closest_idx = np.argmin(distances)
                snapped_seed = same_component_peaks[closest_idx]
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
            f"min_distance={min_distance}  threshold_rel={threshold_rel}  "
            f"seed_radius={seed_radius}",
            fontsize=9,
        )
        fig.tight_layout()
        plt.show()
    # --------------------------------

    return coordinates, labels, image, labels_multi_markers, snapped_seed


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


def detect_2d_peak_and_calculate_peak_property(
    im_rt_pept_act_coo,
    ref_dict: pd.DataFrame,
    start_mz_idx: int,
    end_mz_idx: int,
    im_rt_pept_res_coo: Optional[sparse.COO] = None,
    filter: Literal["gaussian", "uniform", None] = "uniform",
    detect_kwargs=None,
    calc_kwargs=None,
):
    """
    Detect 2D peaks and calculate their properties for a range of mz ranks.

    :param im_rt_pept_act_coo: 3D sparse COO array of ion intensity data.
    :type im_rt_pept_act_coo: np.ndarray
    :param start_idx: int
        Starting m/z rank index to process.
    :param end_idx: int
        Ending m/z rank index to process.
    :param detect_kwargs: dict, optional
        Keyword arguments for detect_2d_peak_with_watershed.
    :param calc_kwargs: dict, optional
        Keyword arguments for calculate_peak_property_from_coords_and_labels.
    :return: DataFrame containing peak properties for all detected peaks.
    :rtype: pd.DataFrame
    """
    dict_ref_filtered = ref_dict[
        (ref_dict["mz_rank"] >= start_mz_idx) & (ref_dict["mz_rank"] < end_mz_idx)
    ]
    im_min = max(dict_ref_filtered["IM_search_idx_left"].min() - 25, 0)
    im_max = dict_ref_filtered["IM_search_idx_right"].max() + 25

    all_peak_properties_in_chunk = []

    detect_kwargs = detect_kwargs or {}
    calc_kwargs = calc_kwargs or {}

    for g in dict_ref_filtered["RT_group"].unique():
        group_df = dict_ref_filtered[dict_ref_filtered["RT_group"] == g]
        mz_idx = group_df["mz_rank"].unique()
        if len(mz_idx) == 0:
            Logger.info("No mz ranks found for RT group %s, skipping.", g)
            continue
        rt_min = group_df["MS1_frame_idx_left_ref"].min()
        rt_max = group_df["MS1_frame_idx_right_ref"].max() + 1
        Logger.info("Processing RT group %s with RT range %s - %s", g, rt_min, rt_max)
        im_rt_pept_act_coo_dense = np.atleast_3d(
            im_rt_pept_act_coo[rt_min:rt_max, im_min:im_max, mz_idx].todense()
        )  # Convert to dense for easier indexing
        if im_rt_pept_res_coo is not None:
            im_rt_pept_res_coo_dense = np.abs(
                np.atleast_3d(
                    im_rt_pept_res_coo[rt_min:rt_max, im_min:im_max, mz_idx].todense()
                )
            )  # sometimes residue is negative
        else:
            im_rt_pept_res_coo_dense = None
        for rel_idx, mz_rank in enumerate(mz_idx):
            pept_act = im_rt_pept_act_coo_dense[:, :, rel_idx]

            # pept_act = im_rt_pept_act_coo[rt_start:rt_end, im_start:im_end, pept_idx].todense()
            cleaned_mask = remove_small_objects(
                pept_act >= 10, min_size=9
            )  # TODO: hardcoded threshold
            match filter:
                case "gaussian":
                    pept_act_smoothed = gaussian_filter(pept_act, sigma=1.0)
                case "uniform":
                    blurred = uniform_filter(pept_act, size=9)
                    pept_act_smoothed = np.maximum(pept_act, blurred)
                case None:
                    pept_act_smoothed = pept_act
            pept_act_smoothed = pept_act_smoothed * cleaned_mask
            pept_act_smoothed_log = np.log10(1 + pept_act_smoothed)

            # Calculate gradient mask
            dy, dx = np.gradient(pept_act_smoothed)
            grad_mag = np.sqrt(dx**2 + dy**2)
            # grad_mag_smooth = gaussian_filter(grad_mag, sigma=2.0)

            # Detect peaks with flexible kwargs
            coordinates, labels, distance, labels_multi_markers = (
                detect_2d_peak_with_watershed(
                    pept_act_smoothed_log,
                    int_threshold=1,
                    threshold_rel=0.2,
                    min_distance=10,
                )
            )
            # Calculate peak properties with flexible kwargs
            peak_properties = calculate_peak_property_from_labels_and_image(
                labels, pept_act, grad_mag, **calc_kwargs
            )

            if isinstance(peak_properties, pd.DataFrame) and not peak_properties.empty:
                peak_properties["mz_rank"] = mz_rank
                peak_properties["rt_apex_index"] += rt_min
                peak_properties["im_apex_index"] += im_min
                all_peak_properties_in_chunk.append(peak_properties)
            else:
                Logger.debug("No peaks detected for mz_rank %s", mz_rank)
                continue

    if len(all_peak_properties_in_chunk) > 1:
        all_peak_properties_in_chunk_df = pd.concat(
            all_peak_properties_in_chunk, ignore_index=True
        )
        Logger.info(
            "Returning %s images/mzrank with %s peaks in chunk %s - %s",
            len(all_peak_properties_in_chunk),
            all_peak_properties_in_chunk_df.shape[0],
            start_mz_idx,
            end_mz_idx,
        )
        return all_peak_properties_in_chunk_df
    else:
        Logger.warning("No peaks returned in chunk %s - %s", start_mz_idx, end_mz_idx)
        return None
