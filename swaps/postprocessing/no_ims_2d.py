import logging
from typing import Literal, Union

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.signal import find_peaks, peak_widths
from sklearn.metrics import auc

# import keras
# from postprocessing.peak_selection import prepare_peak_input, PeakClsModel

Logger = logging.getLogger(__name__)

_SmoothMethods = Literal["GaussianKernel", "LocalMinima", "Raw"]

# TODO: update once peak model is ready
# def activation_peak_cls_and_sum(
#     activation_df: pd.DataFrame,
#     peak_results: pd.DataFrame,
#     model_path: str,
#     return_peak_result: bool = False,
# ):

#     # filter out precursor activation with only one peak
#     peak_results_onepeak = (
#         peak_results.groupby("id").filter(lambda x: len(x) == 1).reset_index()
#     )
#     # filter out precursor activation with more than one peak
#     peak_results_filtered = (
#         peak_results.groupby("id").filter(lambda x: len(x) > 1).reset_index()
#     )
#     # prepare peak input
#     peak_arrays = prepare_peak_input(
#         peak_results_filtered,
#         activation_df,
#         log_intensity=True,
#         standardize="minmax",
#         method="mask",
#     )
#     # pad peak input
#     mdl_class = PeakClsModel(
#         peak_input=peak_arrays,
#         peak_results=peak_results_filtered,
#         initial_bias="auto",
#         use_class_weight=True,
#     )
#     mdl_class.pad_peak_input(maxlen=361, padding="two_side", value=-1.0)
#     peak_arrays = mdl_class.padded_sequences.reshape(
#         mdl_class.padded_sequences.shape[0],
#         mdl_class.padded_sequences.shape[1],
#         1,
#     )

#     # load model
#     model = keras.models.load_model(model_path)
#     Logger.info("Loaded model from %s", model_path)
#     # predict
#     peak_results_filtered["peak_prob"] = model.predict(peak_arrays)
#     # mark true peak of each precursor activation
#     peak_results_filtered["true_peak"] = (
#         peak_results_filtered.groupby("id")["peak_prob"].transform(max)
#         == peak_results_filtered["peak_prob"]
#     ) & (
#         peak_results_filtered["peak_prob"] > 0.5
#     )
#   # TODO: candidate peak with same prob: take the small or big one?
#     # merge back
#     peak_results_onepeak["peak_prob"] = 1
#     peak_results_onepeak["true_peak"] = True

#     peak_results_updated = pd.concat(
#         [peak_results_onepeak, peak_results_filtered], axis=0
#     )
#     peak_results_sum = peak_results_updated.loc[
#         peak_results_updated["true_peak"], ["id", "peak_intensity_auc"]
#     ]
#     peak_results_sum.index = peak_results_sum["id"]
#     sum_peak = pd.DataFrame(
#         {
#             "id": peak_results_sum["id"].values,
#             "AUCActivationPeak": peak_results_sum["peak_intensity_auc"].values,
#         }
#     )
#     # calculate sum of activation
#     if return_peak_result:
#         return sum_peak, peak_results_updated
#     else:
#         return sum_peak


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


# Define the Gaussian function
def gauss(x, A, B):
    """Gaussian function"""
    y = A * np.exp(-1 * B * x**2)
    return y


# Smooth results
def smooth_act_curve(
    precursor_activation,
    ms1scans_no_array: pd.DataFrame,
    method: _SmoothMethods = "GaussianKernel",
    b=0.02,
    minima_width: int = 3,
    minima_prominence: int = 10000,
):
    """Smooth activation curve"""
    replace_values = []
    smoothed_act = precursor_activation.copy()
    if precursor_activation.sum() > 0:
        non0idx = np.nonzero(precursor_activation)[0]
        if len(non0idx) > 2:
            match method:
                case "GaussianKernel":
                    gaussian_kernel_value = np.zeros(shape=non0idx.shape)
                    for time in ms1scans_no_array.loc[non0idx, "starttime"]:
                        gaussian_kernel_value = np.exp(
                            -((ms1scans_no_array.loc[non0idx, "starttime"] - time) ** 2)
                            / (2 * (b**2))
                        )
                        gaussian_kernel_value /= gaussian_kernel_value.sum()
                        replace_values.append(
                            round(
                                precursor_activation[non0idx] * gaussian_kernel_value
                            ).sum()
                        )
                    smoothed_act[non0idx] = replace_values
                    # smoothedActivation = np.array(smoothedActivation).reshape(-1)
                case "LocalMinima":
                    peaks, _ = find_peaks(
                        -precursor_activation, prominence=minima_prominence
                    )
                    if len(peaks) > 0:
                        (peak_width, _peak_height, left, right) = peak_widths(
                            -precursor_activation, peaks, rel_height=1
                        )
                        # print(peakWidth, left, right)
                        for idx, width in enumerate(peak_width):
                            # print(idx, width, peaks[idx], smoothedActivation[peaks[idx]] )
                            if width <= minima_width:
                                distance = [
                                    1 / abs(left[idx] - peaks[idx].item()),
                                    1 / abs(right[idx] - peaks[idx].item()),
                                ]
                                # print(distance)
                                smoothed_act[peaks[idx]] = np.average(
                                    smoothed_act[
                                        [
                                            int(np.floor(left[idx])),
                                            int(np.ceil(right[idx])),
                                        ]
                                    ],
                                    axis=0,
                                    weights=distance,
                                )
    return smoothed_act


def smooth_act_mat(
    activation: np.ndarray,  # TODO: doing more than 1 thing!
    ms1scans_no_array: pd.DataFrame,
    method: _SmoothMethods = "GaussianKernel",
    b=0.02,
    minima_width: int = 3,
):
    """smooth activation matrix"""

    # refit
    def smooth_activation(
        row,
        ms1scans_no_array: pd.DataFrame = ms1scans_no_array,
        method: _SmoothMethods = method,
        b: float = b,
        minima_width: int = minima_width,
    ):
        return smooth_act_curve(
            row,
            ms1scans_no_array=ms1scans_no_array,
            method=method,
            b=b,
            minima_width=minima_width,
        )

    if method != "Raw":
        refit_activation = np.apply_along_axis(
            func1d=smooth_activation, axis=1, arr=activation
        )
    else:
        refit_activation = activation

    # Sum activation
    sum_activation = pd.DataFrame(
        {
            "SumActivation" + method: refit_activation.sum(axis=1),
            "AUCActivation"
            + method: np.apply_along_axis(
                lambda x: auc(ms1scans_no_array["starttime"], x),
                1,
                refit_activation,
            ),
            "TRPZActivation"
            + method: np.apply_along_axis(
                lambda x: trapezoid(x=ms1scans_no_array["starttime"], y=x),
                1,
                refit_activation,
            ),
        }
    )
    return refit_activation, sum_activation
