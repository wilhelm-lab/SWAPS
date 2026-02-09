import logging
from typing import Tuple, Literal
import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt, uniform_filter
from skimage.registration import phase_cross_correlation
from skimage.morphology import remove_small_objects
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import shift as ndi_shift
import numpy as np
import cv2

from ..utils.ims_utils import (
    detect_2d_peak_with_watershed,
    calculate_peak_property_from_labels_and_image,
)

Logger = logging.getLogger(__name__)


def quantify_from_coords(
    pept_act_image,
    anchor,
    reference_image: np.ndarray | None = None,
    smooth_kwargs: dict | None = None,
    peak_kwargs: dict | None = None,
    align_kwargs: dict | None = None,
    patch_size: int | None = None,
):
    assert (
        anchor[0] < pept_act_image.shape[0] and anchor[1] < pept_act_image.shape[1]
    ), "Anchor coordinates are out of bounds of the image dimensions."
    anchor = np.array([(anchor[0].astype(int), anchor[1].astype(int))])
    smooth_kwargs = {} if smooth_kwargs is None else dict(smooth_kwargs)
    peak_kwargs = {} if peak_kwargs is None else dict(peak_kwargs)
    align_kwargs = {} if align_kwargs is None else dict(align_kwargs)
    if "int_threshold" not in peak_kwargs:
        peak_kwargs["int_threshold"] = 1
    if "threshold_rel" not in peak_kwargs:
        peak_kwargs["threshold_rel"] = 0.2
    if "min_distance" not in peak_kwargs:
        peak_kwargs["min_distance"] = 10

    pept_act_image_smoothed = smooth_and_denoise_image(pept_act_image, **smooth_kwargs)

    if reference_image is not None:
        pept_act_image_smoothed_aligned, shift, phasediff = align_images(
            aligned_image=pept_act_image_smoothed,
            reference_image=reference_image,
            **align_kwargs,
        )
        pept_act_image_aligned = ndi_shift(pept_act_image, shift)
    else:
        pept_act_image_smoothed_aligned = pept_act_image_smoothed
        pept_act_image_aligned = pept_act_image
    _, labels, _ = detect_2d_peak_with_watershed(
        pept_act_image_smoothed_aligned,
        **peak_kwargs,
        coordinates=anchor,
    )
    peak_properties = calculate_peak_property_from_labels_and_image(
        labels, pept_act_image_aligned, min_peak_sum_intensity=500
    )
    if peak_properties is None:
        return pept_act_image_smoothed_aligned, None
    else:
        peak_properties["orb_des"] = None
        peak_properties.at[0, "orb_des"] = get_sift_descriptor(
            pept_act_image_aligned, anchor[0], patch_size=patch_size
        )
        if reference_image is not None:
            peak_properties["shift_rt"] = shift[0]
            peak_properties["shift_im"] = shift[1]
        else:
            peak_properties["shift_rt"] = 0
            peak_properties["shift_im"] = 0
        return pept_act_image_smoothed_aligned, peak_properties


def compare_peak_properties(peak_properties_a, peak_properties_b):
    return {
        "orb_similarity": compare_sift_descriptors(
            peak_properties_a["orb_des"].values[0],
            peak_properties_b["orb_des"].values[0],
        ),
        "rt_shift": abs(
            peak_properties_a["shift_rt"].values[0]
            - peak_properties_b["shift_rt"].values[0]
        ),
        "im_shift": abs(
            peak_properties_a["shift_im"].values[0]
            - peak_properties_b["shift_im"].values[0]
        ),
        "rt_length_diff": abs(
            peak_properties_a["rt_length"].values[0]
            - peak_properties_b["rt_length"].values[0]
        ),
        "im_length_diff": abs(
            peak_properties_a["im_length"].values[0]
            - peak_properties_b["im_length"].values[0]
        ),
        "rt_length_diff_rel": abs(
            peak_properties_a["rt_length"].values[0]
            - peak_properties_b["rt_length"].values[0]
        )
        / peak_properties_a["rt_length"].values[0],
        "im_length_diff_rel": abs(
            peak_properties_a["im_length"].values[0]
            - peak_properties_b["im_length"].values[0]
        )
        / peak_properties_a["im_length"].values[0],
        "int_max_diff_rel": abs(
            peak_properties_a["intensity_max"].values[0]
            - peak_properties_b["intensity_max"].values[0]
        )
        / peak_properties_a["intensity_max"].values[0],
        "int_sum_diff_rel": abs(
            peak_properties_a["intensity_sum"].values[0]
            - peak_properties_b["intensity_sum"].values[0]
        )
        / peak_properties_a["intensity_sum"].values[0],
        "area_diff_rel": abs(
            peak_properties_a["area"].values[0] - peak_properties_b["area"].values[0]
        )
        / peak_properties_a["area"].values[0],
    }


def smooth_and_denoise_image(
    image,
    smooth_filter: Literal["gaussian", "uniform"] = "gaussian",
    log_transform: bool = True,
    threshold: float = 10,
    gaussian_kwargs: dict | None = None,
    uniform_kwargs: dict | None = None,
    remove_kwargs: dict | None = None,
):
    """Smooth image with filters and denoise by remove small objects

    Parameters
    ----------
    image : 2D array
        Input image to be smoothed.
    smooth_filter : str, optional
        Type of filter to use. Options are "gaussian" or "uniform". Default is "gaussian".
    threshold : float, optional
        Threshold used to create a mask before removing small objects.
    gaussian_kwargs : dict, optional
        Keyword arguments for scipy.ndimage.gaussian_filter.
    uniform_kwargs : dict, optional
        Keyword arguments for scipy.ndimage.uniform_filter.
    remove_kwargs : dict, optional
        Keyword arguments for skimage.morphology.remove_small_objects.
    """
    gaussian_kwargs = {} if gaussian_kwargs is None else dict(gaussian_kwargs)
    uniform_kwargs = {} if uniform_kwargs is None else dict(uniform_kwargs)
    remove_kwargs = {} if remove_kwargs is None else dict(remove_kwargs)

    if "sigma" not in gaussian_kwargs:
        gaussian_kwargs["sigma"] = 2
    if "size" not in uniform_kwargs:
        uniform_kwargs["size"] = (1, 10)
    if "min_size" not in remove_kwargs:
        remove_kwargs["min_size"] = 5

    match smooth_filter:
        case "gaussian":
            image_smoothed = gaussian_filter(image, **gaussian_kwargs)
        case "uniform":
            blurred = uniform_filter(image, **uniform_kwargs)
            image_smoothed = np.maximum(image, blurred)
    # remove small objects after smoothing
    cleaned_mask = remove_small_objects(image_smoothed >= threshold, **remove_kwargs)
    image_smoothed = image_smoothed * cleaned_mask

    # log transform smoothed and cleaned up
    if log_transform:
        image_smoothed = np.log10(1 + image_smoothed)
    return image_smoothed


def align_images(reference_image, aligned_image, mask_threshold=25, upsample_factor=10):
    """Align two images using phase cross-correlation and return the aligned image and the calculated shift.
    Parameters
    ----------
    reference_image : 2D array
        Reference image to align to.
    image_b : 2D array
        Image to be aligned.
    Returns
    -------
    aligned_image_b : 2D array
        Aligned version of image_b.
    shift : tuple
        Calculated shift applied to image_b.
    """
    mask1 = reference_image > np.percentile(reference_image, mask_threshold)
    mask2 = aligned_image > np.percentile(aligned_image, mask_threshold)
    shift, _, phasediff = phase_cross_correlation(
        reference_image,
        aligned_image,
        reference_mask=mask1,
        moving_mask=mask2,
        upsample_factor=upsample_factor,
    )

    aligned_image_b = ndi_shift(aligned_image, shift)

    return aligned_image_b, shift, phasediff


def compare_windowed_cosine(img1, img2, peak_coords, window_size=21):
    y, x = peak_coords
    r = window_size // 2

    # Pad to handle peaks near the boundary
    pad1 = np.pad(img1, r, mode="edge")
    pad2 = np.pad(img2, r, mode="edge")

    # Adjust coords for padding
    py, px = y + r, x + r

    # Extract and flatten patches
    patch1 = pad1[py - r : py + r + 1, px - r : px + r + 1].ravel()
    patch2 = pad2[py - r : py + r + 1, px - r : px + r + 1].ravel()

    # Cosine Similarity Formula
    norm = np.linalg.norm(patch1) * np.linalg.norm(patch2)
    return np.dot(patch1, patch2) / norm if norm != 0 else 0.0


def compare_gaussian_weighted(img1, img2, peak_coords, sigma=15):
    y, x = peak_coords
    yy, xx = np.indices(img1.shape)

    # Create Gaussian weight centered at coordinates
    dist_sq = (yy - y) ** 2 + (xx - x) ** 2
    weights = np.exp(-dist_sq / (2 * sigma**2))

    # Normalize to 0-1 for SSIM consistency
    i1_n = cv2.normalize(img1.astype(float), None, 0, 1, cv2.NORM_MINMAX)
    i2_n = cv2.normalize(img2.astype(float), None, 0, 1, cv2.NORM_MINMAX)

    # Structural Similarity weighted by the Gaussian mask
    return ssim(i1_n * weights, i2_n * weights, data_range=1.0)


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


def compare_orb_descriptors(des1, des2):
    """
    Compares two pre-computed descriptors using Hamming distance.
    Returns a similarity score between 0.0 and 1.0.
    """
    if des1 is None or des2 is None:
        return 0.0

    # Hamming distance: count bit differences
    # Lower distance = Higher similarity
    dist = cv2.norm(des1, des2, cv2.NORM_HAMMING)

    # ORB descriptors are 256 bits (32 bytes)
    return 1.0 - (dist / 256.0)


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


def compare_sift_descriptors(des1, des2):
    if des1 is None or des2 is None:
        return 0.0

    # SIFT descriptors must be float32 for NORM_L2
    # This line prevents the "Assertion failed" error
    d1 = des1.astype(np.float32)
    d2 = des2.astype(np.float32)

    # Use L2 (Euclidean) distance for SIFT
    # NORM_HAMMING is only for binary descriptors like ORB
    dist = cv2.norm(d1, d2, cv2.NORM_L2)

    # Convert distance to a 0-1 similarity score
    # SIFT distances for a match are usually < 200
    similarity = np.exp(-dist / 100.0)
    return similarity
