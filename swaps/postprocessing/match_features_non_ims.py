import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Union

import numpy as np
import pandas as pd
import duckdb
from scipy.ndimage import gaussian_filter1d
from scipy.signal import correlate, correlation_lags, find_peaks
from .image_processing import detect_1d_peak_with_watershed
from sklearn.metrics import auc
import tqdm

import os
from cowarp import warp
from .helper import load_peptide_batch_df_from_partquet, get_pept_act_from_parquet, get_rt_window

Logger = logging.getLogger(__name__)
_COWARP_FALLBACK_LOGGED = False
_WORKER_CONTEXT: dict[str, Any] = {}


def extract_elution_peak_from_act_row(
    activation_row: Union[pd.Series, np.array],
    ms1scans_no_array: pd.DataFrame,
    rt_search_center: float | None = None,
    n_peaks: int = 1,
    return_peak_result: bool = False,
    peak_width_thres=(2, None),
    peak_height_thres=(None, None),
    pcm_id: int | None = 0,
    **kwargs,  # find_peaks parameters
):
    """Extract elution peak from activation row by match to search center

    :param activation_row: activation row
    :param MS1ScansNoArray: MS1ScansNoArray
    :param ref_RT_apex: reference retention time apex
    :param ref_RT_start: reference retention time start
    :param ref_RT_end: reference retention time end
    :param n_peaks: number of peaks (closest) to select
    :param return_peak_result: return peak result
    :param peak_width_thres: peak width threshold
    :param peak_height_thres: peak height threshold
    :param PCM_id: precursor charge multiplicity id
    :param kwargs: find_peaks parameters

    """
    # Logger.debug("row name %s", activation_row.name)
    peaks, peak_properties = find_peaks(
        activation_row,
        width=peak_width_thres,
        height=peak_height_thres,
        rel_height=1,  # critical for peak_width, do not change
        **kwargs,
    )
    left = np.round(peak_properties["left_ips"], decimals=0).astype(int)
    right = np.round(peak_properties["right_ips"], decimals=0).astype(int)
    Logger.debug("Reference retention time %s", rt_search_center)
    peak_result = pd.DataFrame(
        {
            "id": np.repeat(pcm_id, len(left)).astype(int),
            "apex_scan": peaks,
            "apex_time": ms1scans_no_array["starttime"][peaks].values,
            "start_scan": left,
            "start_time": ms1scans_no_array["starttime"][left].values,
            "end_scan": right,
            "end_time": ms1scans_no_array["starttime"][right].values,
            "peak_width": right - left,
            "peak_height": peak_properties["peak_heights"],
            "peak_intensity_auc": [
                auc(
                    x=ms1scans_no_array["starttime"][i - 1 : j + 1],
                    y=activation_row[i - 1 : j + 1],
                )
                for (i, j) in zip(left, right)
            ],
        }
    )
    peak_result["rt_search_center_diff"] = abs(
        peak_result["apex_time"] - rt_search_center
    )
    peak_result["closest_to_search_center"] = False
    peak_result.loc[
        peak_result.nsmallest(n_peaks, "rt_search_center_diff").index,
        "closest_to_search_center",
    ] = True
    peak_result["matched"] = np.nan
    sum_intensity = peak_result.loc[
        peak_result["closest_to_search_center"], "peak_intensity_auc"
    ].sum()

    if return_peak_result:
        return peak_result, sum_intensity
    else:
        return sum_intensity


def extract_1d_curve_(
    df_act: pd.DataFrame,
    mz: int,
    dict_ref_row: pd.Series | None,
    run_name: str,
) -> tuple[np.ndarray, int, float, int, int]:
    """Extract a complete 1D RT-elution curve with zero-filled gaps (vectorized)."""
    if df_act.empty or dict_ref_row is None:
        return np.array([], dtype=np.float32), 0, 0.0, 0, 0

    rt_start_idx, rt_end_idx, rt_exp_center = get_rt_window(dict_ref_row, run_name)

    df_grouped = df_act.groupby("frame_idx", as_index=False)["activation"].sum()
    
    rt_range_len = rt_end_idx - rt_start_idx + 1
    activation_1d = np.zeros(rt_range_len, dtype=np.float32)
    
    # Filter and index directly without iterating
    mask = (df_grouped["frame_idx"] >= rt_start_idx) & (df_grouped["frame_idx"] <= rt_end_idx)
    valid_frames = df_grouped.loc[mask]
    
    if not valid_frames.empty:
        indices = valid_frames["frame_idx"].values.astype(np.int32) - rt_start_idx
        activation_1d[indices] = valid_frames["activation"].values.astype(np.float32)
    
    intensity_sum = float(activation_1d.sum())
    return activation_1d, rt_exp_center, intensity_sum, rt_start_idx, rt_end_idx


def compute_rt_alignment_shift(
    activation_curve_ref: np.ndarray,
    activation_curve_target: np.ndarray,
    rt_exp_center_ref: int,
    rt_exp_center_target: int,
    max_lag: int = 20,
    alignment_method: str = "xcorr",
    gaussian_sigma: float = 0.0,
) -> int:
    """Compute frame shift of target relative to reference."""
    if gaussian_sigma > 0 and alignment_method == "xcorr":
        activation_curve_ref = smooth_curve_with_gaussian(activation_curve_ref, gaussian_sigma)
        activation_curve_target = smooth_curve_with_gaussian(activation_curve_target, gaussian_sigma)

    if alignment_method == "xcorr":
        xcorr_shift = estimate_rt_shift_xcorr(
            activation_curve_ref,
            activation_curve_target,
            max_lag=max_lag,
        )
        if xcorr_shift is not None:
            return int(xcorr_shift)

    if rt_exp_center_ref > 0 and rt_exp_center_target > 0:
        return int(rt_exp_center_ref - rt_exp_center_target)

    apex_ref = int(np.argmax(activation_curve_ref)) if activation_curve_ref.max() > 0 else len(activation_curve_ref) // 2
    apex_target = int(np.argmax(activation_curve_target)) if activation_curve_target.max() > 0 else len(activation_curve_target) // 2
    return apex_ref - apex_target


def estimate_rt_shift_xcorr(
    activation_curve_ref: np.ndarray,
    activation_curve_target: np.ndarray,
    max_lag: int = 20,
) -> int | None:
    """Estimate a small constant lag using cross-correlation."""
    ref = np.asarray(activation_curve_ref, dtype=np.float32)
    tgt = np.asarray(activation_curve_target, dtype=np.float32)

    if ref.size == 0 or tgt.size == 0:
        return None

    # Fast zero-fill check
    if np.max(np.abs(ref)) == 0 or np.max(np.abs(tgt)) == 0:
        return None

    ref = ref - ref.mean()
    tgt = tgt - tgt.mean()

    # Early exit for flat signals
    ref_norm = np.linalg.norm(ref)
    tgt_norm = np.linalg.norm(tgt)
    if ref_norm < 1e-6 or tgt_norm < 1e-6:
        return None

    ref = ref / ref_norm
    tgt = tgt / tgt_norm

    corr = correlate(tgt, ref, mode="full", method="fft")
    lags = correlation_lags(tgt.size, ref.size, mode="full")

    # Filter by max_lag constraint
    lag_mask = np.abs(lags) <= int(max_lag)
    if not np.any(lag_mask):
        return None

    best_lag = int(lags[lag_mask][np.argmax(corr[lag_mask])])
    # scipy correlate(tgt, ref) at lag k computes sum_n tgt[n+k]*ref[n], so lag k means
    # "shift tgt LEFT by k to align with ref" → the array shift to apply is -k.
    return -best_lag


def _pad_or_crop_curve(curve: np.ndarray, target_len: int) -> np.ndarray:
    curve = np.asarray(curve, dtype=np.float32)
    if curve.size == target_len:
        return curve
    if curve.size > target_len:
        return curve[:target_len]
    padded = np.zeros(target_len, dtype=np.float32)
    padded[: curve.size] = curve
    return padded


def shift_curve_zero_padded(curve: np.ndarray, shift: int, target_len: int | None = None) -> np.ndarray:
    """Shift a curve by an integer lag while keeping the output length fixed."""
    curve = np.asarray(curve, dtype=np.float32)
    if target_len is None:
        target_len = int(curve.size)
    
    if curve.size == 0 or target_len == 0:
        return np.zeros(target_len, dtype=np.float32)

    shifted = np.zeros(target_len, dtype=np.float32)

    if shift >= 0:
        length = min(curve.size, target_len - shift)
        if length > 0:
            shifted[shift:shift + length] = curve[:length]
    else:
        src_start = -shift
        length = min(curve.size - src_start, target_len)
        if length > 0:
            shifted[:length] = curve[src_start:src_start + length]
    
    return shifted


def smooth_curve_with_gaussian(curve: np.ndarray, sigma: float) -> np.ndarray:
    """Smooth a 1D curve with a Gaussian kernel"""
    curve = np.asarray(curve, dtype=np.float32)
    if curve.size == 0 or sigma <= 0:
        return curve
    if curve.size < 3:
        return curve
    return gaussian_filter1d(curve, sigma=float(sigma), mode="nearest", truncate=3.0).astype(np.float32)


def _choose_off_target_shift_1d(
    consensus_curve: np.ndarray,
    apex_idx: int,
    min_offset_frac: float = 0.35,
    max_overlap_fraction: float = 0.05,
    rep: int = 0,
) -> int | None:
    """Choose an integer RT shift that places the apex off the true peak region."""
    n = len(consensus_curve)
    if n == 0 or apex_idx < 0 or apex_idx >= n:
        return None
    apex_height = float(consensus_curve[apex_idx])
    if apex_height <= 0:
        return None
    base_mask = consensus_curve >= apex_height / 2.0
    base_area = int(base_mask.sum())
    if base_area == 0:
        return None

    frac_values = [min_offset_frac, min(0.5, min_offset_frac + 0.15)]
    candidates: list[int] = []
    for frac in frac_values:
        dr = max(1, int(round(n * frac)))
        candidates.extend([dr, -dr])
    seen: set[int] = set()
    unique_candidates = [c for c in candidates if c not in seen and not seen.add(c)]  # type: ignore[func-returns-value]

    evaluated: list[tuple[float, float, int]] = []
    valid: list[tuple[float, float, int]] = []
    for shift in unique_candidates:
        shifted_apex = apex_idx + shift
        if shifted_apex < 0 or shifted_apex >= n:
            continue
        shifted_mask = np.zeros(n, dtype=bool)
        if shift >= 0:
            shifted_mask[shift:] = base_mask[: n - shift]
        else:
            shifted_mask[: n + shift] = base_mask[-shift:]
        overlap = float(np.logical_and(base_mask, shifted_mask).sum()) / max(base_area, 1)
        record = (overlap, -float(abs(shift)), shift)
        evaluated.append(record)
        if overlap <= max_overlap_fraction:
            valid.append(record)

    pool = valid if valid else evaluated
    if not pool:
        return None
    return sorted(pool, key=lambda x: (x[0], x[1]))[rep % len(pool)][2]


def align_1d_curve_to_reference(
    reference_curve: np.ndarray,
    target_curve: np.ndarray,
    rt_exp_center_ref: int,
    rt_exp_center_target: int,
    max_lag: int = 20,
    alignment_method: str = "xcorr",
    cowarp_kwargs: dict | None = None,
    gaussian_sigma: float = 0.0,
) -> tuple[np.ndarray, int, float]:
    """Align one curve to a reference curve and return aligned curve, integer shift and score."""
    reference_curve = np.asarray(reference_curve, dtype=np.float32)
    target_curve = np.asarray(target_curve, dtype=np.float32)
    target_len = int(reference_curve.size)

    if alignment_method == "cowarp":
        if reference_curve.size > 0 and target_curve.size > 0:
            cowarp_error = None
            try:
                cowarp_kwargs = cowarp_kwargs or {}
                warped_curve, correlation = warp(reference_curve, target_curve, **cowarp_kwargs)
                aligned_curve = _pad_or_crop_curve(warped_curve, target_len)
                shift = estimate_rt_shift_xcorr(reference_curve, aligned_curve, max_lag=max_lag)
                if shift is None:
                    apex_ref = int(np.argmax(reference_curve)) if reference_curve.max() > 0 else target_len // 2
                    apex_aligned = int(np.argmax(aligned_curve)) if aligned_curve.max() > 0 else target_len // 2
                    shift = int(apex_ref - apex_aligned)
                return aligned_curve, int(shift), float(correlation)
            except Exception as exc:
                cowarp_error = exc

        global _COWARP_FALLBACK_LOGGED
        if not _COWARP_FALLBACK_LOGGED:
            _COWARP_FALLBACK_LOGGED = True
            Logger.warning(
                "COWARP requested but unavailable or failed; falling back to xcorr. reason=%s",
                repr(cowarp_error) if cowarp_error is not None else "no warp function or empty curve",
            )

    shift = compute_rt_alignment_shift(
        reference_curve,
        target_curve,
        rt_exp_center_ref,
        rt_exp_center_target,
        max_lag=max_lag,
        alignment_method="xcorr" if alignment_method == "cowarp" else alignment_method,
        gaussian_sigma=gaussian_sigma,
    )
    aligned_curve = shift_curve_zero_padded(target_curve, shift, target_len=target_len)
    ref_centered = reference_curve - reference_curve.mean()
    ali_centered = aligned_curve - aligned_curve.mean()
    ref_norm = np.linalg.norm(ref_centered)
    ali_norm = np.linalg.norm(ali_centered)
    score = float(np.dot(ref_centered, ali_centered) / (ref_norm * ali_norm)) if ref_norm > 1e-9 and ali_norm > 1e-9 else 0.0
    return aligned_curve, int(shift), score


def build_consensus_1d_curve(
    per_run_curves: dict[str, np.ndarray],
    per_run_shifts: dict[str, int],
    reference_run: str,
) -> np.ndarray:
    """Align curves by integer shifts and average into one consensus curve."""
    if not per_run_curves:
        return np.array([], dtype=np.float32)

    shifts_array = np.array(list(per_run_shifts.values()), dtype=np.int32)
    ref_len = len(per_run_curves[reference_run])
    min_shift = int(shifts_array.min())
    max_shift = int(shifts_array.max())
    consensus_len = max(1, int(ref_len + max_shift - min_shift))

    consensus = np.zeros(consensus_len, dtype=np.float32)
    count = np.zeros(consensus_len, dtype=np.float32)
    
    # Vectorized shifting and accumulation
    for run_name, curve in per_run_curves.items():
        curve = np.asarray(curve, dtype=np.float32)
        offset = int(per_run_shifts[run_name] - min_shift)
        end_idx = min(offset + len(curve), consensus_len)
        in_curve_end = end_idx - offset
        if in_curve_end > 0:
            consensus[offset:end_idx] += curve[:in_curve_end]
            count[offset:end_idx] += 1.0

    # Vectorized division
    return np.divide(consensus, count, where=count > 0, out=np.zeros_like(consensus))


def _preaggregate_run_batch(
    df: pd.DataFrame,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Pre-group activation by (mz_rank, frame_idx) once for the whole batch.

    Returns mz_rank -> (frame_idx_arr, activation_arr) for O(1) per-mz lookup,
    eliminating per-mz per-run pandas groupby calls in the hot loop.
    """
    if df.empty:
        return {}
    agg = (
        df.groupby(["mz_rank", "frame_idx"], sort=True)["activation"]
        .sum()
        .reset_index()
    )
    mz_col = agg["mz_rank"].values
    frame_col = agg["frame_idx"].values.astype(np.int32)
    act_col = agg["activation"].values.astype(np.float32)

    unique_mzs, counts = np.unique(mz_col, return_counts=True)
    split_pts = np.concatenate([[0], np.cumsum(counts)])
    return {
        int(mz_val): (
            frame_col[split_pts[i] : split_pts[i + 1]],
            act_col[split_pts[i] : split_pts[i + 1]],
        )
        for i, mz_val in enumerate(unique_mzs)
    }


def _build_1d_curve_from_arrays(
    frame_idx_arr: np.ndarray,
    activation_arr: np.ndarray,
    rt_start_idx: int,
    rt_end_idx: int,
    rt_exp_center: int,
) -> tuple[np.ndarray, int, float]:
    """Build 1D activation array from pre-aggregated frame arrays."""
    rt_range_len = rt_end_idx - rt_start_idx + 1
    activation_1d = np.zeros(rt_range_len, dtype=np.float32)
    mask = (frame_idx_arr >= rt_start_idx) & (frame_idx_arr <= rt_end_idx)
    if mask.any():
        indices = frame_idx_arr[mask] - rt_start_idx
        activation_1d[indices] = activation_arr[mask]
    return activation_1d, rt_exp_center, float(activation_1d.sum())


def _init_match_features_worker_non_ims(dict_ref,raw_file_list,result_dir,processing_kwargs):
    """Store immutable batch context once per worker process."""

    _WORKER_CONTEXT["dict_ref"] = dict_ref
    _WORKER_CONTEXT["raw_file_list"] = raw_file_list
    _WORKER_CONTEXT["result_dir"] = result_dir
    _WORKER_CONTEXT["processing_kwargs"] = processing_kwargs
    _WORKER_CONTEXT["dict_ref_by_mz"] = (
        dict_ref.set_index("mz_rank")
        if dict_ref["mz_rank"].is_unique
        else dict_ref.drop_duplicates("mz_rank").set_index("mz_rank")
    )


def _match_features_batch_worker_non_ims(batch):
    return match_features_batch_non_ims(
        dict_ref=_WORKER_CONTEXT["dict_ref"],
        raw_file_list=_WORKER_CONTEXT["raw_file_list"],
        result_dir=_WORKER_CONTEXT["result_dir"],
        batch=batch,
        processing_kwargs=_WORKER_CONTEXT["processing_kwargs"],
    )


def match_features_batch_non_ims(
    dict_ref,
    raw_file_list,
    result_dir,
    batch,
    processing_kwargs: dict | None = None,
    visualize_dir: str | None = None,
    match_decoy: bool = True,
):
    """Process one peptide batch using the non-IMS RT alignment path."""

    dict_ref_by_mz = _WORKER_CONTEXT.get("dict_ref_by_mz")
    if dict_ref_by_mz is None:
        dict_ref_by_mz = (
            dict_ref.set_index("mz_rank")
            if dict_ref["mz_rank"].is_unique
            else dict_ref.drop_duplicates("mz_rank").set_index("mz_rank")
        )

    debug_cfg = processing_kwargs or {}
    log_timing = bool(debug_cfg.get("debug_timing", False))
    log_consensus = bool(debug_cfg.get("log_consensus", False))
    alignment_method = str(debug_cfg.get("alignment_method", "xcorr") or "xcorr")
    xcorr_max_lag = int(debug_cfg.get("xcorr_max_lag", 20) or 20)
    gaussian_sigma = float(debug_cfg.get("rt_gaussian_sigma", 0.0) or 0.0)
    cowarp_kwargs = {
        "auto_segment": bool(debug_cfg.get("cowarp_auto_segment", False)),
        "num_intervals": debug_cfg.get("cowarp_num_intervals", 10),
        "segment_length": debug_cfg.get("cowarp_segment_length", None),
        "slack": debug_cfg.get("cowarp_slack", 10),
        "segmentation_type": str(debug_cfg.get("cowarp_segmentation_type", "stationary_points") or "stationary_points"),
        "deformation_coeff": debug_cfg.get("cowarp_deformation_coeff", None),
        "filter_func_code": str(debug_cfg.get("cowarp_filter_func_code", "gaussian") or "gaussian"),
        "filter_func_params": debug_cfg.get("cowarp_filter_func_params", 3),
        "process_filtered_signals": bool(debug_cfg.get("cowarp_process_filtered_signals", False)),
        "min_interval_length": debug_cfg.get("cowarp_min_interval_length", None),
        "verbose": bool(debug_cfg.get("cowarp_verbose", False)),
    }
    _decoy_kwargs = dict((debug_cfg.get("consensus_decoy_kwargs") or {}))
    _decoy_strategies = {
        str(s)
        for s in _decoy_kwargs.get("strategies", ["peptide_swap", "off_target_shift"])
    }
    _n_peptide_swap_decoys = max(int(_decoy_kwargs.get("n_peptide_swap_decoys", 1)), 0)
    _n_off_target_decoys = max(int(_decoy_kwargs.get("n_off_target_shift_decoys", 1)), 0)
    _off_target_min_frac = float(_decoy_kwargs.get("off_target_min_offset_frac", 0.35))
    _off_target_max_overlap = float(_decoy_kwargs.get("off_target_max_overlap_fraction", 0.05))

    results_target: list[dict] = []
    results_decoy: list[dict] = []
    pp_reference_list, pp_match_target_list, pp_match_decoy_list = [], [], []
    no_quant_log: list[dict] = []
    no_match_log: list[dict] = []
    snap_log_collection: dict = {}
    batch_np = np.asarray(batch)

    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    per_run_lookup: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    for run_name in raw_file_list:
        df = load_peptide_batch_df_from_partquet(
            os.path.join(result_dir, run_name, "activation"),
            batch_np,
            con=con,
        )
        per_run_lookup[run_name] = _preaggregate_run_batch(df)
    con.close()

    batch_dict_ref: dict[int, pd.Series] = {}
    for _mz in batch_np:
        try:
            batch_dict_ref[int(_mz)] = dict_ref_by_mz.loc[int(_mz)]
        except KeyError:
            pass

    for mz in batch_np:
        mz_start = time.perf_counter() if log_timing else None
        mz = int(mz)
        per_run_curves: dict[str, np.ndarray] = {}
        per_run_aligned_curves: dict[str, np.ndarray] = {}
        per_run_rt_exp: dict[str, int] = {}
        per_run_intensity: dict[str, float] = {}
        per_run_rt_range: dict[str, tuple[int, int]] = {}
        per_run_alignment_score: dict[str, float] = {}

        dict_ref_row = batch_dict_ref.get(mz)
        if dict_ref_row is None:
            continue

        for run_name in raw_file_list:
            lookup = per_run_lookup.get(run_name, {})
            if mz not in lookup:
                continue
            frame_idx_arr, activation_arr = lookup[mz]
            try:
                rt_start_idx, rt_end_idx, rt_exp_center = get_rt_window(dict_ref_row, run_name)
            except (KeyError, ValueError):
                continue
            activation_1d, rt_exp_center, intensity_sum = _build_1d_curve_from_arrays(
                frame_idx_arr, activation_arr, rt_start_idx, rt_end_idx, rt_exp_center,
            )
            if intensity_sum <= 0:
                continue
            per_run_curves[run_name] = activation_1d
            per_run_rt_exp[run_name] = rt_exp_center
            per_run_intensity[run_name] = intensity_sum
            per_run_rt_range[run_name] = (rt_start_idx, rt_end_idx)

        if not per_run_curves:
            no_match_log.append({"mz_rank": mz, "type": "no_curves_across_runs"})
            continue

        _str_vals = dict_ref_row[dict_ref_row.map(lambda x: isinstance(x, str))]
        _ref_candidates = _str_vals.index[_str_vals == "Reference"].tolist()
        if _ref_candidates and _ref_candidates[0] in per_run_curves:
            reference_run = _ref_candidates[0]
        else:
            reference_run = max(per_run_intensity, key=lambda r: per_run_intensity[r])

        if log_consensus:
            Logger.debug(
                "[mz=%d] curves extracted: %s | reference: %s",
                mz,
                {r: f"len={len(c)} intensity={per_run_intensity[r]:.1f}" for r, c in per_run_curves.items()},
                reference_run,
            )

        per_run_shifts: dict[str, int] = {}
        for run_name in per_run_curves:
            if run_name == reference_run:
                per_run_shifts[run_name] = 0
                per_run_aligned_curves[run_name] = per_run_curves[run_name]
                per_run_alignment_score[run_name] = 1.0
                continue

            aligned_curve, shift, score = align_1d_curve_to_reference(
                per_run_curves[reference_run],
                per_run_curves[run_name],
                per_run_rt_exp[reference_run],
                per_run_rt_exp[run_name],
                max_lag=xcorr_max_lag,
                alignment_method=alignment_method,
                cowarp_kwargs=cowarp_kwargs,
                gaussian_sigma=gaussian_sigma,
            )
            per_run_shifts[run_name] = int(shift)
            per_run_aligned_curves[run_name] = aligned_curve
            per_run_alignment_score[run_name] = float(score)

        if log_consensus:
            Logger.debug(
                "[mz=%d] alignment (%s): %s",
                mz,
                alignment_method,
                {r: f"shift={per_run_shifts[r]:+d} score={per_run_alignment_score[r]:.3f}" for r in per_run_curves},
            )

        consensus_inputs = per_run_aligned_curves
        consensus_shifts = {run_name: 0 for run_name in consensus_inputs}
        _runs_with_peak = [r for r in consensus_inputs if len(find_peaks(consensus_inputs[r], width=(1, None))[0]) > 0]
        _consensus_inputs = {r: consensus_inputs[r] for r in _runs_with_peak} if _runs_with_peak else consensus_inputs
        _consensus_shifts = {r: consensus_shifts[r] for r in _consensus_inputs}

        if log_consensus:
            _excluded = set(consensus_inputs) - set(_consensus_inputs)
            Logger.debug(
                "[mz=%d] consensus inputs: %d/%d runs (excluded no-peak runs: %s) | shifts: %s",
                mz,
                len(_consensus_inputs),
                len(consensus_inputs),
                _excluded or "none",
                {r: f"{_consensus_shifts[r]:+d}" for r in _consensus_inputs},
            )

        consensus_curve = build_consensus_1d_curve(_consensus_inputs, _consensus_shifts, reference_run if reference_run in _consensus_inputs else next(iter(_consensus_inputs)))
        if consensus_curve.size == 0:
            no_match_log.append({"mz_rank": mz, "type": "empty_consensus"})
            continue

        if log_consensus:
            _cs = list(_consensus_shifts.values())
            Logger.debug(
                "[mz=%d] consensus curve: len=%d min_shift=%d max_shift=%d | intensity range=[%.1f, %.1f]",
                mz,
                len(consensus_curve),
                min(_cs),
                max(_cs),
                float(consensus_curve.min()),
                float(consensus_curve.max()),
            )

        watershed_peaks, watershed_labels, _, _, _ = detect_1d_peak_with_watershed(consensus_curve)
        if watershed_peaks.size == 0:
            no_match_log.append({"mz_rank": mz, "type": "no_consensus_peak"})
            continue

        consensus_min_shift = min(_consensus_shifts.values())
        # consensus is in the reference frame; offset_in_consensus == per_run_shifts[r]
        _ref_exp_in_consensus = per_run_rt_exp[reference_run] - consensus_min_shift
        _nearest = int(np.argmin(np.abs(watershed_peaks - _ref_exp_in_consensus)))
        consensus_apex_idx = int(watershed_peaks[_nearest])

        if log_consensus:
            Logger.debug(
                "[mz=%d] watershed: %d peaks at %s | ref_exp_in_consensus=%d | elected apex=%d (label=%d)",
                mz,
                len(watershed_peaks),
                watershed_peaks.tolist(),
                _ref_exp_in_consensus,
                consensus_apex_idx,
                int(watershed_labels[consensus_apex_idx]),
            )

        _target_label = int(watershed_labels[consensus_apex_idx])
        _basin_consensus_idx = (
            np.where(watershed_labels == _target_label)[0]
            if _target_label > 0
            else np.empty(0, dtype=np.int64)
        )

        if log_consensus:
            Logger.debug(
                "[mz=%d] basin: label=%d size=%d consensus_idx=[%s..%s]",
                mz,
                _target_label,
                len(_basin_consensus_idx),
                int(_basin_consensus_idx[0]) if _basin_consensus_idx.size else "N/A",
                int(_basin_consensus_idx[-1]) if _basin_consensus_idx.size else "N/A",
            )

        per_run_pp: list[dict] = []
        for run_name in per_run_curves:
            shift = per_run_shifts[run_name]
            rt_start_idx, _rt_end_idx = per_run_rt_range[run_name]
            offset_in_consensus = shift - consensus_min_shift
            apex_in_run_coords = consensus_apex_idx - offset_in_consensus + rt_start_idx

            _curve = per_run_curves[run_name]
            _basin_local = _basin_consensus_idx - offset_in_consensus
            _valid = (_basin_local >= 0) & (_basin_local < len(_curve))
            intensity_sum = (
                float(_curve[_basin_local[_valid]].sum())
                if _valid.any()
                else float(per_run_intensity[run_name])
            )

            if log_consensus:
                Logger.debug(
                    "[mz=%d] quant %s: shift=%+d offset_in_consensus=%d apex_frame=%d basin_local=[%s..%s] valid=%d/%d intensity=%.1f",
                    mz,
                    run_name,
                    shift,
                    offset_in_consensus,
                    apex_in_run_coords,
                    int(_basin_local[0]) if _basin_local.size else "N/A",
                    int(_basin_local[-1]) if _basin_local.size else "N/A",
                    int(_valid.sum()),
                    len(_basin_local),
                    intensity_sum,
                )

            feat_id = f"{mz}_0"
            per_run_pp.append({
                "mz_rank": mz,
                "Run_name": run_name,
                "feature_instance_id": feat_id,
                "intensity_sum": intensity_sum,
                "apex_scan": int(apex_in_run_coords),
                "rt_shift": int(shift),
            })

        # Get reference run properties
        reference_props = per_run_pp[list(per_run_curves).index(reference_run)]
        pp_reference_list.append(reference_props)

        # Build jump anchor log more efficiently
        jump_anchor_log: dict[str, dict] = {
            run_name: {
                "anchor": (int(consensus_apex_idx), 0),
                "nearest_labeled_pixel": (int(consensus_apex_idx - shift), 0),
                "dist_to_label": float(abs(shift)),
            }
            for run_name, shift in per_run_shifts.items()
        }
        snap_log_collection[mz] = {
            "no_seg_log": None,
            "discard_record": False,
            "jump_anchor_log": jump_anchor_log,
        }

        # Add matches for non-reference runs
        for run_name, props in zip(per_run_curves.keys(), per_run_pp):
            if run_name == reference_run:
                continue

            results_target.append(
                {
                    "mz_rank": mz,
                    "feature_instance_id": props["feature_instance_id"],
                    "reference_run": reference_run,
                    "matched_run": run_name,
                    "rt_shift": props["rt_shift"],
                    "im_shift": 0,
                    "template_matching_score": 1.0,
                    "alignment_score": float(per_run_alignment_score.get(run_name, 1.0)),
                }
            )
            pp_match_target_list.append(props)

        if log_timing:
            Logger.info(
                "Non-IMS mz_rank=%d done in %.3fs | runs=%d | curves=%d | consensus_len=%d",
                mz,
                time.perf_counter() - mz_start,
                len(per_run_curves),
                len(per_run_pp),
                len(consensus_curve) if consensus_curve.size > 0 else 0,
            )

    matches_target = pd.DataFrame(results_target)
    matches_decoy = pd.DataFrame(results_decoy)
    pp_reference_target = pd.DataFrame(pp_reference_list) if pp_reference_list else pd.DataFrame()
    pp_match_target = pd.DataFrame(pp_match_target_list) if pp_match_target_list else pd.DataFrame()
    pp_match_decoy = pd.concat(pp_match_decoy_list, ignore_index=True) if pp_match_decoy_list else pd.DataFrame()
    df_no_quant = pd.DataFrame(no_quant_log)
    df_no_match = pd.DataFrame(no_match_log)

    return (
        matches_target,
        matches_decoy,
        pp_reference_target,
        pp_match_target,
        pp_match_decoy,
        df_no_quant,
        df_no_match,
        snap_log_collection,
    )


def match_features_batches_parallel_non_ims(
    dict_ref,
    raw_file_list,
    result_dir,
    peptide_indicies: np.ndarray | None = None,
    batch_size_max: int = 1500,
    max_workers: int = 4,
    processing_kwargs: dict | None = None,
):
    """Match non-IMS peptide features across runs using 1D RT alignment."""
    debug_cfg = processing_kwargs or {}
    debug_limit_mz = int(debug_cfg.get("debug_limit_mz", 0) or 0)
    debug_limit_runs = int(debug_cfg.get("debug_limit_runs", 0) or 0)

    if peptide_indicies is None:
        peptide_indicies = dict_ref["mz_rank"].values
        Logger.info("No peptide indices provided, using all mz_rank from dict_ref.")
    else:
        Logger.info("Using provided peptide indices. Total count: %d", len(peptide_indicies))

    # Sort mz_ranks so each batch is a contiguous range — this lets DuckDB skip
    # row groups in the mz-sorted parquet
    sorted_mz = np.sort(peptide_indicies)
    if debug_limit_mz > 0:
        sorted_mz = sorted_mz[:debug_limit_mz]
        Logger.info("Debug mode active: limiting to first %d mz_rank values", debug_limit_mz)

    active_runs = raw_file_list[:debug_limit_runs] if debug_limit_runs > 0 else raw_file_list
    if debug_limit_runs > 0:
        Logger.info("Debug mode active: limiting to first %d runs", debug_limit_runs)

    n_total = len(sorted_mz)

    n_batches = max(
        max_workers * 2, 
        int(np.ceil(n_total / batch_size_max))
    )
    peptide_batches = np.array_split(sorted_mz, n_batches)

    results_target, results_decoy = [], []
    pp_reference_list: list[pd.DataFrame] = []
    pp_match_target_list: list[pd.DataFrame] = []
    pp_match_decoy_list: list[pd.DataFrame] = []
    no_quant_log: list[dict] = []
    no_match_log: list[dict] = []
    snap_log_collection: dict = {}

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_match_features_worker_non_ims,
        initargs=(
            dict_ref,
            active_runs,
            result_dir,
            processing_kwargs,
        ),
    ) as executor:
        futures = [
            executor.submit(_match_features_batch_worker_non_ims, batch)
            for batch in peptide_batches
        ]

        for future in tqdm.tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Processing batches",
            unit="batch",
        ):
            (
                res_target,
                res_decoy,
                pp_reference_target,
                pp_match_target,
                pp_match_decoy,
                no_quant,
                no_match,
                batch_snap_log,
            ) = future.result()
            results_target.append(res_target)
            results_decoy.append(res_decoy)
            pp_reference_list.append(pp_reference_target)
            pp_match_target_list.append(pp_match_target)
            pp_match_decoy_list.append(pp_match_decoy)
            no_quant_log.extend(no_quant)
            no_match_log.extend(no_match)
            snap_log_collection.update(batch_snap_log)

    matches_target = (
        pd.concat(results_target, ignore_index=True)
        if results_target
        else pd.DataFrame()
    )
    matches_decoy = (
        pd.concat(results_decoy, ignore_index=True)
        if results_decoy
        else pd.DataFrame()
    )
    pp_reference_target = (
        pd.concat(pp_reference_list, ignore_index=True)
        if pp_reference_list
        else pd.DataFrame()
    )
    pp_match_target = (
        pd.concat(pp_match_target_list, ignore_index=True)
        if pp_match_target_list
        else pd.DataFrame()
    )
    pp_match_decoy = (
        pd.concat(pp_match_decoy_list, ignore_index=True)
        if pp_match_decoy_list
        else pd.DataFrame()
    )
    df_no_quant = pd.DataFrame(no_quant_log)
    df_no_match = pd.DataFrame(no_match_log)

    return (
        matches_target,
        matches_decoy,
        pp_reference_target,
        pp_match_target,
        pp_match_decoy,
        df_no_quant,
        df_no_match,
        snap_log_collection,
    )


match_features_batches_non_ims = match_features_batches_parallel_non_ims


def plot_rt_alignment(
    mz_rank: int,
    dict_ref: pd.DataFrame,
    raw_file_list: list[str],
    result_dir: str,
    processing_kwargs: dict | None = None,
    log_scale: bool = False,
    legend_outside: bool = False,
    show_basins: bool = False,
):
    """Plot before/after RT alignment curves for a single peptide.

    Returns a matplotlib Figure with two subplots.  Call from a notebook after
    running match_features_batches_non_ims to inspect any mz_rank.
    """
    import matplotlib.pyplot as plt

    debug_cfg = processing_kwargs or {}
    log_consensus = bool(debug_cfg.get("log_consensus", False))
    alignment_method = str(debug_cfg.get("alignment_method", "xcorr") or "xcorr")
    xcorr_max_lag = int(debug_cfg.get("xcorr_max_lag", 20) or 20)
    gaussian_sigma = float(debug_cfg.get("rt_gaussian_sigma", 0.0) or 0.0)
    cowarp_kwargs = {
        "auto_segment": bool(debug_cfg.get("cowarp_auto_segment", False)),
        "num_intervals": debug_cfg.get("cowarp_num_intervals", 10),
        "segment_length": debug_cfg.get("cowarp_segment_length", None),
        "slack": debug_cfg.get("cowarp_slack", 10),
        "segmentation_type": str(debug_cfg.get("cowarp_segmentation_type", "stationary_points") or "stationary_points"),
        "deformation_coeff": debug_cfg.get("cowarp_deformation_coeff", None),
        "filter_func_code": str(debug_cfg.get("cowarp_filter_func_code", "gaussian") or "gaussian"),
        "filter_func_params": debug_cfg.get("cowarp_filter_func_params", 3),
        "process_filtered_signals": bool(debug_cfg.get("cowarp_process_filtered_signals", False)),
        "min_interval_length": debug_cfg.get("cowarp_min_interval_length", None),
        "verbose": bool(debug_cfg.get("cowarp_verbose", False)),
    }

    dict_ref_by_mz = (
        dict_ref.set_index("mz_rank")
        if dict_ref["mz_rank"].is_unique
        else dict_ref.drop_duplicates("mz_rank").set_index("mz_rank")
    )
    try:
        dict_ref_row = dict_ref_by_mz.loc[int(mz_rank)]
    except KeyError:
        raise ValueError(f"mz_rank={mz_rank} not found in dict_ref")

    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    per_run_lookup: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    for run_name in raw_file_list:
        df = load_peptide_batch_df_from_partquet(
            os.path.join(result_dir, run_name, "activation"),
            np.array([mz_rank]),
            con=con,
        )
        per_run_lookup[run_name] = _preaggregate_run_batch(df)
    con.close()

    per_run_curves: dict[str, np.ndarray] = {}
    per_run_aligned_curves: dict[str, np.ndarray] = {}
    per_run_rt_exp: dict[str, int] = {}
    per_run_intensity: dict[str, float] = {}
    per_run_shifts: dict[str, int] = {}
    per_run_alignment_score: dict[str, float] = {}
    per_run_rt_start: dict[str, int] = {}

    for run_name in raw_file_list:
        lookup = per_run_lookup.get(run_name, {})
        if mz_rank not in lookup:
            continue
        frame_idx_arr, activation_arr = lookup[mz_rank]
        try:
            rt_start_idx, rt_end_idx, rt_exp_center = get_rt_window(dict_ref_row, run_name)
        except (KeyError, ValueError):
            continue
        activation_1d, rt_exp_center, intensity_sum = _build_1d_curve_from_arrays(
            frame_idx_arr, activation_arr, rt_start_idx, rt_end_idx, rt_exp_center,
        )
        if intensity_sum <= 0:
            continue
        per_run_curves[run_name] = activation_1d
        per_run_rt_exp[run_name] = rt_exp_center
        per_run_intensity[run_name] = intensity_sum
        per_run_rt_start[run_name] = rt_start_idx

    if not per_run_curves:
        raise ValueError(f"No activation data found for mz_rank={mz_rank}")

    _str_vals = dict_ref_row[dict_ref_row.map(lambda x: isinstance(x, str))]
    _ref_candidates = _str_vals.index[_str_vals == "Reference"].tolist()
    if _ref_candidates and _ref_candidates[0] in per_run_curves:
        reference_run = _ref_candidates[0]
    else:
        reference_run = max(per_run_intensity, key=lambda r: per_run_intensity[r])
    _override = debug_cfg.get("reference_run")
    if _override and _override in per_run_curves:
        reference_run = _override

    if log_consensus:
        print(f"[mz={mz_rank}] curves extracted:")
        for r, c in per_run_curves.items():
            print(f"  {r}: len={len(c)}  intensity={per_run_intensity[r]:.1f}  rt_exp_local={per_run_rt_exp[r]}")
        print(f"[mz={mz_rank}] reference run: {reference_run}")

    for run_name in per_run_curves:
        if run_name == reference_run:
            per_run_shifts[run_name] = 0
            per_run_aligned_curves[run_name] = per_run_curves[run_name]
            per_run_alignment_score[run_name] = 1.0
            continue
        aligned_curve, shift, score = align_1d_curve_to_reference(
            per_run_curves[reference_run],
            per_run_curves[run_name],
            per_run_rt_exp[reference_run],
            per_run_rt_exp[run_name],
            max_lag=xcorr_max_lag,
            alignment_method=alignment_method,
            cowarp_kwargs=cowarp_kwargs,
            gaussian_sigma=gaussian_sigma,
        )
        per_run_shifts[run_name] = int(shift)
        per_run_aligned_curves[run_name] = aligned_curve
        per_run_alignment_score[run_name] = float(score)

    if log_consensus:
        print(f"[mz={mz_rank}] alignment ({alignment_method}):")
        for r in per_run_curves:
            print(f"  {r}: shift={per_run_shifts[r]:+d}  score={per_run_alignment_score[r]:.3f}")

    colors = plt.cm.tab10.colors
    run_names = list(per_run_curves.keys())

    display_sigma = max(1.0, gaussian_sigma)

    def _smooth(curve: np.ndarray) -> np.ndarray:
        return smooth_curve_with_gaussian(curve, display_sigma)
        # return curve

    # consensus is always built from aligned curves (same as IMS 2D path).
    _proc_base = per_run_aligned_curves
    _proc_base_shifts = {r: 0 for r in _proc_base}
    _runs_with_peak = [r for r in _proc_base if len(find_peaks(_proc_base[r], width=(1, None))[0]) > 0]
    _proc_inputs = {r: _proc_base[r] for r in _runs_with_peak} if _runs_with_peak else _proc_base
    _proc_shifts = {r: _proc_base_shifts[r] for r in _proc_inputs}
    _proc_ref = reference_run if reference_run in _proc_inputs else next(iter(_proc_inputs))
    consensus_proc = build_consensus_1d_curve(_proc_inputs, _proc_shifts, _proc_ref)
    _ws_peaks, _ws_labels, _, _, _ = detect_1d_peak_with_watershed(consensus_proc)
    # Consensus index 0 maps to absolute frame: ref_start + min_shift
    _proc_x0 = per_run_rt_start[reference_run] + min(_proc_shifts.values())

    if log_consensus:
        _excluded = set(_proc_base) - set(_proc_inputs)
        _min_shift = min(_proc_shifts.values())
        _ref_exp_in_cons = per_run_rt_exp[reference_run] - _min_shift
        print(f"[mz={mz_rank}] consensus inputs: {len(_proc_inputs)}/{len(_proc_base)} runs")
        if _excluded:
            print(f"  excluded (no scipy peak): {_excluded}")
        print(f"  shifts into consensus: { {r: f'{_proc_shifts[r]:+d}' for r in _proc_inputs} }")
        print(f"  min_shift={_min_shift}  consensus_len={len(consensus_proc)}  proc_x0={_proc_x0}")
        print(f"[mz={mz_rank}] consensus curve: min={consensus_proc.min():.2f}  max={consensus_proc.max():.2f}")
        print(f"[mz={mz_rank}] watershed peaks (consensus idx): {_ws_peaks.tolist()}")
        print(f"  ref_exp_in_consensus={_ref_exp_in_cons}  → elected apex at consensus idx closest to that")
        if _ws_peaks.size:
            _nearest = int(np.argmin(np.abs(_ws_peaks - _ref_exp_in_cons)))
            _apex = int(_ws_peaks[_nearest])
            _label = int(_ws_labels[_apex])
            _basin = np.where(_ws_labels == _label)[0]
            print(f"  elected apex={_apex}  label={_label}  basin=[{_basin[0]}..{_basin[-1]}]  size={len(_basin)}")
            print(f"  absolute frame of apex: {_proc_x0 + _apex}")
            print(f"[mz={mz_rank}] per-run apex (absolute frame):")
            for r in per_run_curves:
                _s = per_run_shifts[r]
                _off = _s - _min_shift
                _apex_abs = _apex - _off + per_run_rt_start[r]
                _apex_back = _apex_abs + _off - per_run_rt_start[r]
                print(f"  {r}: offset_in_consensus={_off:+d}  apex_frame={_apex_abs}  → consensus_apex_idx={_apex_back}")

    apex_before: dict[str, int] = {}
    apex_after: dict[str, int] = {}

    fig, axes = plt.subplots(1, 3, figsize=(21, 4), sharey=False)

    # --- Before alignment ---
    ax = axes[0]
    ref_apex_abs = per_run_rt_exp[reference_run] + per_run_rt_start[reference_run]
    _ref_raw = per_run_curves[reference_run]
    real_ref_apex_abs = int(np.argmax(_ref_raw)) + per_run_rt_start[reference_run] if _ref_raw.max() > 0 else ref_apex_abs
    all_x_starts = [per_run_rt_start[r] for r in run_names]
    all_x_ends = [per_run_rt_start[r] + len(per_run_curves[r]) - 1 for r in run_names]
    for i, run_name in enumerate(run_names):
        curve = _smooth(per_run_curves[run_name])
        x = np.arange(len(curve)) + per_run_rt_start[run_name]
        label = f"{run_name}" + (" [ref]" if run_name == reference_run else "")
        color = colors[i % len(colors)]
        ax.plot(x, curve, color=color, label=label, lw=1.5)
        ax.fill_between(x, curve, alpha=0.08, color=color)
        run_apex_abs = per_run_rt_exp[run_name] + per_run_rt_start[run_name]
        ax.axvline(run_apex_abs, color=color, lw=0.7, ls=":", alpha=0.7)
        apex_before[run_name] = int(x[int(np.argmax(curve))])
    ax.axvline(real_ref_apex_abs, color="red", lw=1.2, ls=":", alpha=0.9, label="ref apex (actual)")
    ax.set_xlim(min(all_x_starts), max(all_x_ends))
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Activation")
    ax.set_title(f"mz_rank={mz_rank} — Before alignment")
    _legend_kw = dict(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0) if legend_outside else dict(fontsize=7, loc="upper right")
    ax.legend(**_legend_kw)

    # --- After alignment ---
    ax = axes[1]
    ref_start = per_run_rt_start[reference_run]
    ref_end = ref_start + len(per_run_curves[reference_run]) - 1
    for i, run_name in enumerate(run_names):
        curve = _smooth(per_run_aligned_curves[run_name])
        x = np.arange(len(curve)) + ref_start
        window_offset = per_run_rt_start[reference_run] - per_run_rt_start[run_name]
        label = f"{run_name} (shift={per_run_shifts[run_name]:+d}, win_off={window_offset:+d})" + (" [ref]" if run_name == reference_run else "")
        color = colors[i % len(colors)]
        ax.plot(x, curve, color=color, label=label, lw=1.5)
        ax.fill_between(x, curve, alpha=0.08, color=color)
        apex_after[run_name] = int(x[int(np.argmax(curve))])
    ax.axvline(ref_apex_abs, color="black", lw=0.7, ls="--", alpha=0.4, label="ref apex (exp)")
    ax.axvline(real_ref_apex_abs, color="red", lw=1.2, ls=":", alpha=0.9, label="ref apex (actual)")
    ax.set_xlim(ref_start, ref_end)
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Activation")
    ax.set_title(f"mz_rank={mz_rank} — After alignment ({alignment_method})")
    ax.legend(**_legend_kw)

   # --- Consensus ---
    ax = axes[2]
    basin_colors = plt.cm.Set2.colors

    x_proc = np.arange(len(consensus_proc)) + _proc_x0

    if show_basins:
        for li, label_id in enumerate(lbl for lbl in np.unique(_ws_labels) if lbl > 0):
            mask = _ws_labels == label_id
            idxs = np.where(mask)[0]
            regions = np.split(idxs, np.where(np.diff(idxs) > 1)[0] + 1)
            for ri, region in enumerate(regions):
                ax.axvspan(
                    _proc_x0 + region[0] - 0.5, _proc_x0 + region[-1] + 0.5,
                    alpha=0.2, color=basin_colors[li % len(basin_colors)],
                    label=f"basin {label_id}" if ri == 0 else "",
                )

    _proc_min_shift = min(_proc_shifts.values())
    for i, run_name in enumerate(run_names):
        curve = _smooth(_proc_base[run_name])
        norm = curve / (np.max(curve) or 1.0)
        offset = _proc_base_shifts[run_name] - _proc_min_shift
        x_run = np.arange(len(norm)) + _proc_x0 + offset
        ax.plot(x_run, norm, color=colors[i % len(colors)], lw=0.8, alpha=0.4)
        
    disp = _smooth(consensus_proc)
    disp = disp / (disp.max() or 1.0)
    ax.plot(x_proc, disp, color="black", lw=2.5, label="consensus (mean)")
    ax.fill_between(x_proc, disp, alpha=0.15, color="black")

    ax.axvline(ref_apex_abs, color="black", lw=0.7, ls="--", alpha=0.4)
    ax.axvline(real_ref_apex_abs, color="red", lw=1.2, ls=":", alpha=0.9, label="ref apex (actual)")
    ax.set_xlim(x_proc[0], x_proc[-1])
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Normalised activation")
    ax.set_title(f"mz_rank={mz_rank} — Consensus")
    ax.legend(**_legend_kw)

    if log_scale:
        for ax in axes:
            ax.set_yscale("log")

    fig.tight_layout()

    apex_table = pd.DataFrame(
        {
            "run": list(apex_before.keys()),
            "apex_frame_before": list(apex_before.values()),
            "apex_frame_after": [apex_after.get(r) for r in apex_before],
            "is_reference": [r == reference_run for r in apex_before],
        }
    )
    return fig, apex_table


def plot_consensus_watershed(
    mz_rank: int,
    dict_ref: pd.DataFrame,
    raw_file_list: list[str],
    result_dir: str,
    processing_kwargs: dict | None = None,
):
    """Visualize the consensus curve and 1D watershed segmentation for one peptide.

    Returns (fig, consensus_curve, ws_peaks, ws_labels) so the caller can do
    further analysis on the raw arrays.

    Two panels:
      Top  — consensus curve (raw intensity) with colored watershed basins,
              individual run curves as faint background lines, and markers for
              all detected peaks plus the elected apex.
      Bottom — log10-transformed curve (what the watershed algorithm actually
               sees) with the same basin coloring.
    """
    import matplotlib.pyplot as plt
    from .image_processing import detect_1d_peak_with_watershed

    debug_cfg = processing_kwargs or {}
    alignment_method = str(debug_cfg.get("alignment_method", "xcorr") or "xcorr")
    xcorr_max_lag = int(debug_cfg.get("xcorr_max_lag", 20) or 20)
    gaussian_sigma = float(debug_cfg.get("rt_gaussian_sigma", 0.0) or 0.0)
    cowarp_kwargs = {
        "auto_segment": bool(debug_cfg.get("cowarp_auto_segment", False)),
        "num_intervals": debug_cfg.get("cowarp_num_intervals", 10),
        "segment_length": debug_cfg.get("cowarp_segment_length", None),
        "slack": debug_cfg.get("cowarp_slack", 10),
        "segmentation_type": str(debug_cfg.get("cowarp_segmentation_type", "stationary_points") or "stationary_points"),
        "deformation_coeff": debug_cfg.get("cowarp_deformation_coeff", None),
        "filter_func_code": str(debug_cfg.get("cowarp_filter_func_code", "gaussian") or "gaussian"),
        "filter_func_params": debug_cfg.get("cowarp_filter_func_params", 3),
        "process_filtered_signals": bool(debug_cfg.get("cowarp_process_filtered_signals", False)),
        "min_interval_length": debug_cfg.get("cowarp_min_interval_length", None),
        "verbose": bool(debug_cfg.get("cowarp_verbose", False)),
    }

    dict_ref_by_mz = (
        dict_ref.set_index("mz_rank")
        if dict_ref["mz_rank"].is_unique
        else dict_ref.drop_duplicates("mz_rank").set_index("mz_rank")
    )
    try:
        dict_ref_row = dict_ref_by_mz.loc[int(mz_rank)]
    except KeyError:
        raise ValueError(f"mz_rank={mz_rank} not found in dict_ref")

    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    per_run_lookup: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    for run_name in raw_file_list:
        df = load_peptide_batch_df_from_partquet(
            os.path.join(result_dir, run_name, "activation"),
            np.array([mz_rank]),
            con=con,
        )
        per_run_lookup[run_name] = _preaggregate_run_batch(df)
    con.close()

    per_run_curves: dict[str, np.ndarray] = {}
    per_run_aligned_curves: dict[str, np.ndarray] = {}
    per_run_rt_exp: dict[str, int] = {}
    per_run_intensity: dict[str, float] = {}
    per_run_shifts: dict[str, int] = {}
    per_run_rt_start: dict[str, int] = {}
    per_run_alignment_score: dict[str, float] = {}

    for run_name in raw_file_list:
        lookup = per_run_lookup.get(run_name, {})
        if mz_rank not in lookup:
            continue
        frame_idx_arr, activation_arr = lookup[mz_rank]
        try:
            rt_start_idx, rt_end_idx, rt_exp_center = get_rt_window(dict_ref_row, run_name)
        except (KeyError, ValueError):
            continue
        activation_1d, rt_exp_center, intensity_sum = _build_1d_curve_from_arrays(
            frame_idx_arr, activation_arr, rt_start_idx, rt_end_idx, rt_exp_center,
        )
        if intensity_sum <= 0:
            continue
        per_run_curves[run_name] = activation_1d
        per_run_rt_exp[run_name] = rt_exp_center
        per_run_intensity[run_name] = intensity_sum
        per_run_rt_start[run_name] = rt_start_idx

    if not per_run_curves:
        raise ValueError(f"No activation data found for mz_rank={mz_rank}")

    _str_vals = dict_ref_row[dict_ref_row.map(lambda x: isinstance(x, str))]
    _ref_candidates = _str_vals.index[_str_vals == "Reference"].tolist()
    reference_run = (
        _ref_candidates[0]
        if _ref_candidates and _ref_candidates[0] in per_run_curves
        else max(per_run_intensity, key=lambda r: per_run_intensity[r])
    )
    _override = debug_cfg.get("reference_run")
    if _override and _override in per_run_curves:
        reference_run = _override

    for run_name in per_run_curves:
        if run_name == reference_run:
            per_run_shifts[run_name] = 0
            per_run_aligned_curves[run_name] = per_run_curves[run_name]
            per_run_alignment_score[run_name] = 1.0
            continue
        aligned_curve, shift, score = align_1d_curve_to_reference(
            per_run_curves[reference_run],
            per_run_curves[run_name],
            per_run_rt_exp[reference_run],
            per_run_rt_exp[run_name],
            max_lag=xcorr_max_lag,
            alignment_method=alignment_method,
            cowarp_kwargs=cowarp_kwargs,
            gaussian_sigma=gaussian_sigma,
        )
        per_run_shifts[run_name] = int(shift)
        per_run_aligned_curves[run_name] = aligned_curve
        per_run_alignment_score[run_name] = float(score)

    _proc_base = per_run_aligned_curves
    _proc_shifts = {r: 0 for r in _proc_base}
    _runs_with_peak = [r for r in _proc_base if len(find_peaks(_proc_base[r], width=(1, None))[0]) > 0]
    _proc_inputs = {r: _proc_base[r] for r in _runs_with_peak} if _runs_with_peak else _proc_base
    _proc_shifts = {r: _proc_shifts[r] for r in _proc_inputs}
    _proc_ref = reference_run if reference_run in _proc_inputs else next(iter(_proc_inputs))

    consensus_curve = build_consensus_1d_curve(_proc_inputs, _proc_shifts, _proc_ref)
    ws_peaks, ws_labels, curve_log, ws_labels_multi, _ = detect_1d_peak_with_watershed(consensus_curve)

    _proc_x0 = per_run_rt_start[reference_run]
    x_cons = np.arange(len(consensus_curve)) + _proc_x0

    elected_apex_idx: int | None = None
    elected_label: int | None = None
    if ws_peaks.size > 0:
        _ref_exp_in_cons = per_run_rt_exp[reference_run]
        _nearest = int(np.argmin(np.abs(ws_peaks - _ref_exp_in_cons)))
        elected_apex_idx = int(ws_peaks[_nearest])
        elected_label = int(ws_labels[elected_apex_idx])

    basin_colors = plt.cm.Set2.colors
    run_colors = plt.cm.tab10.colors
    run_names = list(per_run_curves.keys())

    def _fill_basins(ax: "plt.Axes", y: np.ndarray) -> None:
        for li, label_id in enumerate(lbl for lbl in np.unique(ws_labels) if lbl > 0):
            mask = ws_labels == label_id
            idxs = np.where(mask)[0]
            is_elected = label_id == elected_label
            regions = np.split(idxs, np.where(np.diff(idxs) > 1)[0] + 1)
            for ri, region in enumerate(regions):
                ax.axvspan(
                    x_cons[region[0]] - 0.5,
                    x_cons[region[-1]] + 0.5,
                    alpha=0.35 if is_elected else 0.18,
                    color=basin_colors[li % len(basin_colors)],
                    label=f"basin {label_id}" + (" (elected)" if is_elected else "") if ri == 0 else "",
                )

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # --- Top: raw consensus + individual runs ---
    ax = axes[0]
    cons_max = float(consensus_curve.max()) or 1.0
    _fill_basins(ax, consensus_curve)
    for i, run_name in enumerate(run_names):
        _off = 0
        x_run = np.arange(len(_proc_base[run_name])) + _proc_x0 + _off
        ax.plot(
            x_run,
            _proc_base[run_name],
            color=run_colors[i % len(run_colors)],
            lw=0.9,
            alpha=0.45,
            label=run_name + (" [ref]" if run_name == reference_run else ""),
        )
    ax.plot(x_cons, consensus_curve, color="black", lw=2.0, label="consensus (mean)")
    for pk in ws_peaks:
        ax.axvline(x_cons[pk], color="steelblue", lw=1.2, ls="--", alpha=0.8)
        ax.plot(x_cons[pk], consensus_curve[pk], "v", color="steelblue", ms=7, zorder=5)
    if elected_apex_idx is not None:
        ax.plot(
            x_cons[elected_apex_idx], consensus_curve[elected_apex_idx],
            "*", color="red", ms=14, zorder=6, label=f"elected apex (frame {x_cons[elected_apex_idx]:.0f})",
        )
    ax.set_ylabel("Activation (raw)")
    ax.set_title(f"mz_rank={mz_rank} — Consensus curve + watershed basins")
    ax.legend(fontsize=7, loc="upper right")

    # --- Bottom: log10 curve (what watershed sees) ---
    ax = axes[1]
    _fill_basins(ax, curve_log)
    ax.plot(x_cons, curve_log, color="darkorange", lw=1.8, label="log10(activation)")
    for pk in ws_peaks:
        ax.axvline(x_cons[pk], color="steelblue", lw=1.2, ls="--", alpha=0.8)
        ax.plot(x_cons[pk], curve_log[pk], "v", color="steelblue", ms=7, zorder=5,
                label="watershed peak" if pk == ws_peaks[0] else "")
    if elected_apex_idx is not None:
        ax.plot(
            x_cons[elected_apex_idx], curve_log[elected_apex_idx],
            "*", color="red", ms=14, zorder=6, label="elected apex",
        )
    ax.set_xlabel("Frame index")
    ax.set_ylabel("log10(activation)")
    ax.set_title(f"mz_rank={mz_rank} — Log10 curve (watershed input, {ws_peaks.size} peak(s) detected)")
    ax.legend(fontsize=7, loc="upper right")

    fig.tight_layout()
    return fig, consensus_curve, ws_peaks, ws_labels


def sweep_gaussian_sigma(
    mz_rank: int,
    dict_ref: pd.DataFrame,
    raw_file_list: list[str],
    result_dir: str,
    sigma_values: list[float] | None = None,
    xcorr_max_lag: int = 20,
):
    """Plot peak count, apex stability, and xcorr shift as a function of gaussian_sigma.

    Loads raw curves once, then for each sigma:
      - counts scipy peaks on the smoothed reference curve (noise suppression check)
      - records argmax of smoothed reference (apex stability check)
      - computes xcorr shift for every non-reference run (alignment consistency check)

    Use from a notebook:
        fig = sweep_gaussian_sigma(1230, dict_ref, raw_file_list, result_dir)
        fig.savefig("sigma_sweep.png", bbox_inches="tight")
    """
    import matplotlib.pyplot as plt

    if sigma_values is None:
        sigma_values = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

    dict_ref_by_mz = (
        dict_ref.set_index("mz_rank")
        if dict_ref["mz_rank"].is_unique
        else dict_ref.drop_duplicates("mz_rank").set_index("mz_rank")
    )
    try:
        dict_ref_row = dict_ref_by_mz.loc[int(mz_rank)]
    except KeyError:
        raise ValueError(f"mz_rank={mz_rank} not found in dict_ref")

    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    per_run_lookup: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    for run_name in raw_file_list:
        df = load_peptide_batch_df_from_partquet(
            os.path.join(result_dir, run_name, "activation"),
            np.array([mz_rank]),
            con=con,
        )
        per_run_lookup[run_name] = _preaggregate_run_batch(df)
    con.close()

    per_run_curves: dict[str, np.ndarray] = {}
    per_run_rt_exp: dict[str, int] = {}
    per_run_intensity: dict[str, float] = {}
    for run_name in raw_file_list:
        lookup = per_run_lookup.get(run_name, {})
        if mz_rank not in lookup:
            continue
        frame_idx_arr, activation_arr = lookup[mz_rank]
        try:
            rt_start_idx, rt_end_idx, rt_exp_center = get_rt_window(dict_ref_row, run_name)
        except (KeyError, ValueError):
            continue
        activation_1d, rt_exp_center, intensity_sum = _build_1d_curve_from_arrays(
            frame_idx_arr, activation_arr, rt_start_idx, rt_end_idx, rt_exp_center,
        )
        if intensity_sum <= 0:
            continue
        per_run_curves[run_name] = activation_1d
        per_run_rt_exp[run_name] = rt_exp_center
        per_run_intensity[run_name] = intensity_sum

    if not per_run_curves:
        raise ValueError(f"No activation data found for mz_rank={mz_rank}")

    _str_vals = dict_ref_row[dict_ref_row.map(lambda x: isinstance(x, str))]
    _ref_candidates = _str_vals.index[_str_vals == "Reference"].tolist()
    if _ref_candidates and _ref_candidates[0] in per_run_curves:
        reference_run = _ref_candidates[0]
    else:
        reference_run = max(per_run_intensity, key=lambda r: per_run_intensity[r])

    ref_curve = per_run_curves[reference_run]
    non_ref_runs = [r for r in per_run_curves if r != reference_run]

    peak_counts: list[int] = []
    apex_frames: list[int] = []
    shifts_per_run: dict[str, list[int | None]] = {r: [] for r in non_ref_runs}

    for sigma in sigma_values:
        smoothed_ref = smooth_curve_with_gaussian(ref_curve, sigma)
        peaks, _ = find_peaks(smoothed_ref, width=(1, None))
        peak_counts.append(len(peaks))
        apex_frames.append(int(np.argmax(smoothed_ref)) if smoothed_ref.max() > 0 else 0)
        for run_name in non_ref_runs:
            smoothed_tgt = smooth_curve_with_gaussian(per_run_curves[run_name], sigma)
            shifts_per_run[run_name].append(
                estimate_rt_shift_xcorr(smoothed_ref, smoothed_tgt, max_lag=xcorr_max_lag)
            )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sigmas = list(sigma_values)
    colors = plt.cm.tab10.colors

    ax = axes[0]
    ax.plot(sigmas, peak_counts, "o-", color="steelblue", lw=2)
    ax.axhline(1, color="red", ls="--", lw=1, alpha=0.6, label="target = 1")
    ax.set_xlabel("Gaussian sigma (frames)")
    ax.set_ylabel("N peaks on smoothed ref")
    ax.set_title("Peak count vs sigma")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(sigmas, apex_frames, "s-", color="darkorange", lw=2)
    ax.set_xlabel("Gaussian sigma (frames)")
    ax.set_ylabel("Ref apex (local frame idx)")
    ax.set_title("Apex stability vs sigma")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    for i, run_name in enumerate(non_ref_runs):
        ax.plot(sigmas, shifts_per_run[run_name], "^-", color=colors[i % len(colors)], lw=1.5, label=run_name)
    ax.axhline(0, color="black", ls="--", lw=0.8, alpha=0.4)
    ax.set_xlabel("Gaussian sigma (frames)")
    ax.set_ylabel("xcorr shift (frames)")
    ax.set_title("xcorr shift vs sigma")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"mz_rank={mz_rank} — Gaussian sigma sweep  |  reference: {reference_run}",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    return fig


def select_peak_from_activation(
    maxquant_result_ref: pd.DataFrame,
    ms1scans_no_array: pd.DataFrame,
    activation: np.ndarray,
    n_peaks: int = 1,
    return_peak_result: bool = False,
    **kwargs,
):
    """Select peak from activation"""

    search_rt_array = maxquant_result_ref[["id", "RT_search_center"]].values.reshape(
        [activation.shape[0], 2]
    )

    Logger.debug("shape of activation %s", activation.shape)
    Logger.debug("shape of reference RT %s", search_rt_array.shape)
    act_ref_rt = np.concatenate((activation, search_rt_array), axis=1)
    Logger.debug("shape of act_ref_RT %s", act_ref_rt.shape)
    if return_peak_result:
        Logger.warning("Returning peak result is significantly slower!")
        results = [
            extract_elution_peak_from_act_row(
                activation_row=act_ref_RT_row[:-3],
                pcm_id=act_ref_RT_row[-2],
                rt_search_center=act_ref_RT_row[-1],
                ms1scans_no_array=ms1scans_no_array,
                n_peaks=n_peaks,
                return_peak_result=True,
                **kwargs,
            )
            for act_ref_RT_row in act_ref_rt
        ]
        # logging.debug(results)
        peak_results, peak_sum_activation = zip(*results)
        sum_peak = pd.DataFrame({"AUCActivationPeak": peak_sum_activation})
        peak_results = pd.concat(peak_results, axis=0)
        return sum_peak, peak_results

    peak_sum_activation = np.apply_along_axis(
        lambda act_ref_RT_row: extract_elution_peak_from_act_row(
            activation_row=act_ref_RT_row[:-2],
            rt_search_center=act_ref_RT_row[-2],
            pcm_id=act_ref_RT_row[-1],
            ms1scans_no_array=ms1scans_no_array,
            n_peaks=n_peaks,
            return_peak_result=False,
            **kwargs,
        ),
        axis=1,
        arr=act_ref_rt,
    )
    sum_peak = pd.DataFrame({"AUCActivationPeak": peak_sum_activation})
    return sum_peak


# Define the Gaussian function
def gauss(x, A, B):
    """Gaussian function"""
    y = A * np.exp(-1 * B * x**2)
    return y