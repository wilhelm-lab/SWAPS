import logging
import os
from multiprocessing import cpu_count
from typing import Callable, List, Union, Literal
import itertools
from operator import itemgetter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import spatial
from scipy.sparse import coo_matrix
from scipy.signal import find_peaks
from sklearn.decomposition import sparse_encode
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_squared_error,
)
from math import floor
from optimization.dictionary import Dict
from optimization.custom_models import CustomLinearModel, mean_square_root_error
from postprocessing import peak_selection
from utils.plot import plot_comparison, plot_isopattern_and_obs
from utils.config_json import (
    _algo,
    _alpha_criteria,
    _alpha_opt_metric,
    _loss,
    _pp_method,
)
import sparse

Logger = logging.getLogger(__name__)


class Quant:
    """
    Joint identification and quantification of candidates for given MS1 scan

    :preprocessing: if CandidateDict and obs_data are preprocessed, if 'sqrt' \
        then act should be squared, and cos_dist calculation should also be \
            squared
    """

    def __init__(
        self,
        candidate_dict: pd.DataFrame,
        obs_data: pd.DataFrame,
        filtered_precursor_idx: Union[np.ndarray, list],
        preprocessing_method: _pp_method,
    ) -> None:
        """
        Initialize the Inference class.

        :param candidate_dict: DataFrame containing candidate dictionary.
        :type candidate_dict: pd.DataFrame
        :param obs_data: DataFrame containing observed data. contains mzarray_obs and intensity.
        :type obs_data: pd.DataFrame
        :param filtered_precursor_idx: Filtered precursor indices.
        :type filtered_precursor_idx: Union[np.ndarray, list]
        :param preprocessing_method: Preprocessing method to be used.
        :type preprocessing_method: _pp_method

        :return: None
        """
        self.filter_precursor_idx = filtered_precursor_idx
        self.obs_data_raw = self.obs_data = obs_data[["intensity"]].values
        self.obs_mz = obs_data[["mzarray_obs"]].values[:, 0]
        Logger.debug("obs mz (index) dimension: %s", self.obs_mz.shape)
        if preprocessing_method == "raw":
            self.dictionary = candidate_dict[filtered_precursor_idx].values
            self.obs_data = obs_data[["intensity"]].values  # shape (n_mzvalues, 1)
        elif preprocessing_method == "sqrt":
            self.dictionary = np.sqrt(candidate_dict[filtered_precursor_idx].values)
            self.obs_data = np.sqrt(
                obs_data[["intensity"]].values
            )  # shape (n_mzvalues, 1)
        self.preprocessing = preprocessing_method
        self.alphas = []
        self.acts = []
        self.inferences = []
        self.nonzeros = []
        self.metric = []
        self.infer = None
        self.best_alpha = None
        self.act = None
        self.confusion_matrix = None
        self.metric_used = None
        self.id_result = None
        self.cls_report = None
        self.reconstruc_cos_dist = None

    def optimize(
        self,
        alpha: float,
        loss: _loss = "lasso",
        algorithm: _algo = "lasso_lars",
        max_iter: int = 1000,
        metric: _alpha_opt_metric = "RMSE",
    ):
        self.alphas.append(alpha)
        match loss:
            case "lasso":
                act = sparse_encode(
                    X=self.obs_data.T,
                    dictionary=self.dictionary.T,
                    algorithm=algorithm,
                    positive=True,
                    alpha=alpha,
                    max_iter=max_iter,
                    verbose=50,
                    # n_jobs=-1
                )
                Logger.debug("dimension of act: %s", act.shape)
            case "sqrt_lasso":
                sol = CustomLinearModel(
                    residue_loss=mean_square_root_error,
                    X=self.dictionary,
                    Y=self.obs_data,
                    reg_norm="l1",
                    reg_param=alpha,
                )
                sol.fit(method=algorithm)  # maxiter=max_iter,
                Logger.debug("dimension of sol.beta: %s", sol.beta.shape)
                act = sol.beta.reshape(1, -1)
                Logger.debug("dimension of act: %s", act.shape)
        if self.preprocessing == "raw":
            self.acts.append(act[0])
            self.infer = np.matmul(act, self.dictionary.T)
        elif self.preprocessing == "sqrt":
            self.acts.append(np.square(act[0]))
            self.infer = np.square(np.matmul(act[0], self.dictionary.T))

        self.inferences.append(self.infer)
        self.nonzeros.append(np.count_nonzero(act[0] > 1))

        self.metric_used = metric
        if metric == "cos_dist":
            self.metric.append(self.CalcCosDist())
        elif metric == "RMSE":
            self.metric.append(self.CalcRMSE())

    def optimizeAlphas(
        self,
        alphas: Union[List, np.ndarray],
        loss: _loss = "lasso",
        algorithm: _algo = "lasso_lars",
        criteria: _alpha_criteria = "min",
        metric: _alpha_opt_metric = "cos_dist",
        eps: Union[None, float] = None,
        max_iter: int = 1000,
        PlotTrace: bool = False,
        save_dir: Union[str, None] = None,
    ):
        """
        iterate through different alphas and algorithm,
        for ols, set algo = 'threshold' and alpha = 0.

        :alphas: if
        :algorithms:
        """

        if criteria == "convergence" and eps is None:
            eps = 0.0001
        if loss == "lasso":
            self.optimize(alpha=0, algorithm="threshold", metric=metric)
        else:
            self.optimize(
                alpha=0,
                loss=loss,
                algorithm=algorithm,
                metric=metric,
                max_iter=max_iter,
            )
        if len(alphas) > 0:
            match criteria:
                case "min":
                    for a in alphas:
                        self.optimize(
                            alpha=a,
                            loss=loss,
                            algorithm=algorithm,
                            metric=metric,
                            max_iter=max_iter,
                        )
                    BestAlphaIdx = int(np.array(self.metric).argmin())
                    self.best_alpha = self.alphas[int(BestAlphaIdx)]
                    Logger.info(
                        "Minimal distance reached at alpha = %s", self.best_alpha
                    )
                case "convergence":
                    BestAlphaIdx = None
                    for a in alphas:
                        Logger.debug("Current alpha = %s", a)
                        self.optimize(
                            alpha=a,
                            loss=loss,
                            metric=metric,
                            algorithm=algorithm,
                            max_iter=max_iter,
                        )
                        Logger.debug("Alpha list %s", self.alphas)
                        diff = self.metric[-2] - self.metric[-1]
                        Logger.debug("Alpha = %s, tol = %s", a, diff)
                        if diff <= eps and diff > 0:
                            BestAlphaIdx = self.alphas.index(a)
                            self.best_alpha = self.alphas[BestAlphaIdx]
                            Logger.info(
                                "Reached convergence criteria at alpha = %s",
                                self.best_alpha,
                            )
                            break
                        if diff < 0:
                            BestAlphaIdx = alphas.index(a) - 1
                            self.best_alpha = self.alphas[BestAlphaIdx]
                            Logger.warning(
                                "Increasing %s! Using previous alpha = %s as best"
                                " candidate!",
                                self.metric_used,
                                self.best_alpha,
                            )
                            break
                    if BestAlphaIdx is None:
                        BestAlphaIdx = int(np.array(self.metric).argmin())
                        self.best_alpha = self.alphas[int(BestAlphaIdx)]
                        Logger.warning(
                            "Convergence not reached! Using alpha = %s with minimal"
                            " distance as candidate!",
                            self.best_alpha,
                        )
        else:
            BestAlphaIdx = 0
            self.best_alpha = self.alphas[BestAlphaIdx]
            Logger.info("Alpha not specified, using alpha = %s", self.best_alpha)
        self.infer = self.inferences[BestAlphaIdx]
        self.act = self.acts[BestAlphaIdx]

        if PlotTrace:  # plot trace with 2 y-axis: metric and count_nonzeors
            fig, ax1 = plt.subplots()
            ax1.set_xlabel("log10(alpha+1)")

            color = "tab:red"
            ax1.set_ylabel(self.metric_used, color=color)
            ax1.plot(np.log10(np.array(self.alphas) + 1), self.metric, color=color)
            ax1.tick_params(axis="y", labelcolor=color)
            ax1.plot(
                np.log10(self.best_alpha + 1), self.metric[BestAlphaIdx], "x", color="r"
            )
            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

            color = "tab:blue"
            ax2.set_ylabel(
                "Num of Non-zero Act.", color=color
            )  # we already handled the x-label with ax1
            ax2.plot(np.log10(np.array(self.alphas) + 1), self.nonzeros, color=color)
            ax2.tick_params(axis="y", labelcolor=color)

            ax1.annotate(
                "N0 = " + str(self.dictionary.shape[1]),
                xy=(0.1, 0.9),
                xycoords="axes fraction",
            )

            fig.tight_layout()  # otherwise the right y-label is slightly clipped
            plt.show()

            if save_dir is not None:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                plt.savefig(fname=os.path.join(save_dir, "alphaTrace.png"), dpi=300)
                plt.close()
            else:
                plt.show()

    def analyzeID(self, trueIDidx: List, alpha: Union[float, None] = None):
        if alpha is None:
            Logger.warning(
                "Alpha not specified, using best alpha value = %s", self.best_alpha
            )
            alpha = self.best_alpha

        result_idx = self.alphas.index(alpha)
        inferIDidx = self.filter_precursor_idx[self.acts[result_idx] > 1]
        Logger.info("Number of non-zero actiavation = %s", len(inferIDidx))
        y_pred = [element in set(inferIDidx) for element in self.filter_precursor_idx]
        y_true = [element in set(trueIDidx) for element in self.filter_precursor_idx]
        self.confusion_matrix = confusion_matrix(
            y_true=y_true, y_pred=y_pred, normalize=None, labels=[True, False]
        )
        self.id_result = pd.DataFrame(
            {
                "Candidate": self.filter_precursor_idx,
                "Activation": self.acts[result_idx] > 1,
                "y_pred": y_pred,
                "y_true": y_true,
            }
        )
        self.cls_report = classification_report(
            y_true=y_true, y_pred=y_pred, output_dict=True
        )

    def PlotObsAndInfer(self, log_intensity: bool = False):
        plot_comparison(
            y_true=self.obs_data_raw.flatten(),
            y_pred=self.infer.flatten(),
            log_y=log_intensity,
        )

    def calc_precursor_reconstruct_cos_dist(self):
        self.reconstruc_cos_dist = []
        for idx, val in enumerate(self.act):
            if val > 0:
                mask_act = self.act.copy()
                mask_act[idx] = 0
                y_minus_i = np.matmul(self.dictionary, mask_act)
                y_i = self.obs_data_raw.reshape(-1) - y_minus_i
                iso_mz_mask = (
                    self.dictionary[:, idx] != 0
                )  # only consider nonzero input in dict
                self.reconstruc_cos_dist.append(
                    spatial.distance.cosine(
                        self.dictionary[:, idx][iso_mz_mask], y_i.flatten()[iso_mz_mask]
                    )
                )
            else:
                self.reconstruc_cos_dist.append(0)
        return self.reconstruc_cos_dist

    def CalcRMSE(self):
        return mean_squared_error(
            y_true=self.obs_data_raw.flatten(),
            y_pred=self.infer.flatten(),
            squared=False,
        )

    def CalcCosDist(self):
        return spatial.distance.cosine(
            self.obs_data_raw.flatten(), self.infer.flatten()
        )

    # def CalcExplainedValues(self): TODO: specify which alpha?
    #     return len(self.infer_nonzero)/self.obs_data.shape[0]
    def plot_iso_pattern_and_infer(
        self,
        Maxquant_result: pd.DataFrame,
        precursor_id: List[int] | None = None,
        precursor_idx: List[int] | None = None,
        log_intensity: bool = False,
    ):
        plot_isopattern_and_obs(
            maxquant_result=Maxquant_result,
            infer_intensity=pd.Series(data=self.infer[0], index=self.obs_mz),
            lower_plot="infer",
            precursor_id=precursor_id,
            precursor_idx=precursor_idx,
            log_intensity=log_intensity,
        )

    def CalcExplainedInt(
        self,
    ):  # TODO: does not consider the correctness of explained peaks
        return self.infer.sum() / self.obs_data_raw.sum()


def process_one_scan(
    scan_idx: int,
    OneScan: pd.core.series.Series,
    Maxquant_result: pd.DataFrame,
    scan_time: Union[float, None] = None,
    AbundanceMissingThres: float = 0.4,
    metric: _alpha_opt_metric = "cos_dist",
    alpha_criteria: _alpha_criteria = "convergence",
    alphas: Union[List, np.ndarray] = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
    loss: _loss = "lasso",
    opt_algo: _algo = "lasso_cd",
    preprocessing_method: _pp_method = "raw",
    corr_thres: float = 0.9,
    max_iter: int = 1000,
    return_interim_results: bool = False,
    return_precursor_scan_cos_dist: bool = False,
    return_collinear_candidates: bool = False,
    plot_alpha_trace: bool = False,
    plot_obs_and_infer: bool = False,
):
    """
    process one scan using lasso regression, return  alignment, \
        activation and scan summary

    :scan_idx:
    :OneScan: a row in dataframe MS1Scans
    :
    """
    Logger.debug("Start.")
    if scan_time is None:
        scan_time = OneScan["starttime"]
    CandidatePrecursorsByRT = Maxquant_result.loc[
        (Maxquant_result["RT_search_left"] <= scan_time)
        & (Maxquant_result["RT_search_right"] >= scan_time)
    ]
    Logger.debug("Filter by RT.")

    if CandidatePrecursorsByRT.shape[0] > 0:
        ScanDict = Dict(
            candidate_by_rt=CandidatePrecursorsByRT,
            one_scan=OneScan,
            abundance_missing_thres=AbundanceMissingThres,
        )
        CandidateDict = ScanDict.dict
        filteredPrecursorIdx = ScanDict.filtered_candidate_idx
        if CandidateDict is not None:
            y_true = ScanDict.obs_peak_int

            ScanDict.get_feature_corr(
                cos_sim_thres=corr_thres,
                calc_jaccard_sim=False,
                # plot_collinear_hist=False,
                # plot_hmap=False,
            )
            num_corr_dict_candidate = ScanDict.high_corr_sol.shape[0]
            Logger.debug("Construct dictionary")

            PrecursorQuant = Quant(
                candidate_dict=CandidateDict,
                obs_data=y_true,
                filtered_precursor_idx=filteredPrecursorIdx,
                preprocessing_method=preprocessing_method,
            )

            PrecursorQuant.optimizeAlphas(
                alphas=alphas,
                loss=loss,
                metric=metric,
                algorithm=opt_algo,
                criteria=alpha_criteria,
                PlotTrace=plot_alpha_trace,
                max_iter=max_iter,
            )
            Logger.debug("Quant - optimize alphas")
            if plot_obs_and_infer:
                PrecursorQuant.PlotObsAndInfer(log_intensity=False)
                Logger.debug("Quant - Plot")

            activation = {
                "precursor": filteredPrecursorIdx,
                "activation": PrecursorQuant.act,
            }
            collinear_candidates = {
                "precursor": ScanDict.collinear_precursors.index.values,
                "collinear_candidates": ScanDict.collinear_precursors.values,
            }

            if return_precursor_scan_cos_dist:
                cos_dist = PrecursorQuant.calc_precursor_reconstruct_cos_dist()
                precursor_cos_dist = {
                    "precursor": filteredPrecursorIdx,
                    "cos_dist": cos_dist,
                }
            cos_dist = PrecursorQuant.CalcCosDist()
            rmse = PrecursorQuant.CalcRMSE()
            scan_sum = (
                scan_idx,
                scan_time,
                CandidatePrecursorsByRT.index,
                filteredPrecursorIdx,
                num_corr_dict_candidate,
                PrecursorQuant.best_alpha,
                cos_dist,
                PrecursorQuant.CalcExplainedInt(),
            )
            Logger.info(
                "scan index %s: activation successfully calculated with cosine distance"
                " %s, RMSE %s.",
                scan_idx,
                cos_dist,
                rmse,
            )
        else:
            activation = None
            precursor_cos_dist = None
            collinear_candidates = None
            scan_sum = (
                scan_idx,
                scan_time,
                CandidatePrecursorsByRT.index,
                filteredPrecursorIdx,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            )
            Logger.info(
                "scan index %s: less than 2 valid candidates after isotope pattern"
                " matching.",
                scan_idx,
            )

    else:
        CandidateDict = None
        activation = None
        precursor_cos_dist = None
        collinear_candidates = None
        scan_sum = (
            scan_idx,
            scan_time,
            [],
            [],
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )
        Logger.info("scan index %s: no valid candidate by RT.", scan_idx)

    result_dict_onescan = {}
    result_dict_onescan[scan_idx] = {
        "activation": activation,
        "CandidateDict": CandidateDict,
        "scans_record": scan_sum,
        "precursor_collinear_sets": collinear_candidates,
    }
    match (return_interim_results, return_precursor_scan_cos_dist):
        case (True, True):
            result_dict_onescan[scan_idx]["precursor_cos_dist"] = precursor_cos_dist
            return result_dict_onescan, ScanDict, PrecursorQuant
        case (True, False):
            result_dict_onescan[scan_idx]["precursor_cos_dist"] = None
            return result_dict_onescan, ScanDict, PrecursorQuant
        case (False, True):
            result_dict_onescan[scan_idx]["precursor_cos_dist"] = precursor_cos_dist
            return result_dict_onescan
        case (False, False):
            result_dict_onescan[scan_idx]["precursor_cos_dist"] = None
            return result_dict_onescan


def sparse_encode_divide_and_conquer(frame_array, candidate_array):
    candidate_coo_blocks, col_start, col_end = slice_candidate_blocks_by_pept(
        candidate_array
    )
    # frame_coo_blocks = slice_frame_data_blocks(frame_array, col_start, col_end)
    act = []
    for idx, candidate_block in enumerate(candidate_coo_blocks):
        # Logger.debug("frame and candiate shape %s %s", frame_array.shape, candidate.shape)
        im_pept_act = sparse_encode(
            frame_array[:, col_start[idx] : col_end[idx]],
            candidate_block,
            algorithm="threshold",
            alpha=0,
            positive=True,
            # n_jobs=4,
        )
        Logger.debug("act shape %s", im_pept_act.shape)
        act.append(im_pept_act)
    frame_act = np.concatenate(act, axis=1)
    assert frame_act.shape[1] == candidate_array.shape[0]
    return frame_act


def slice_candidate_blocks_by_pept(matrix):
    # slice candidate block by not splitting up isotope envlopes
    if matrix.shape[0] > 6000 and matrix.shape[0] < 8000:
        Logger.info("Divide matrix into 2 blocks.")
        row_cut_indices = [matrix.shape[0] // 2]
    elif matrix.shape[0] >= 8000 and matrix.shape[0] < 12000:
        Logger.info("Divide matrix into 4 blocks.")
        row_cut_indices = [
            floor(matrix.shape[0] * 0.35),
            floor(matrix.shape[0] * 0.6),
            floor(matrix.shape[0] * 0.85),
            # floor(matrix.shape[0] * 0.3),
        ]
    elif matrix.shape[0] >= 12000 and matrix.shape[0] < 16000:
        Logger.info("Divide matrix into 6 blocks.")
        row_cut_indices = [
            floor(matrix.shape[0] * 0.2),
            floor(matrix.shape[0] * 0.4),
            floor(matrix.shape[0] * 0.6),
            floor(matrix.shape[0] * 0.8),
            floor(matrix.shape[0] * 0.9),
        ]
    elif matrix.shape[0] >= 16000:
        Logger.info("Divide matrix into 8 blocks.")
        row_cut_indices = [
            floor(matrix.shape[0] * 0.2),
            floor(matrix.shape[0] * 0.4),
            floor(matrix.shape[0] * 0.5),
            floor(matrix.shape[0] * 0.6),
            floor(matrix.shape[0] * 0.7),
            floor(matrix.shape[0] * 0.8),
            floor(matrix.shape[0] * 0.9),
        ]

    if row_cut_indices[-1] != matrix.shape[0] + 1:
        row_cut_indices.append(matrix.shape[0] + 1)
    blocks = []
    prev_row_cut = 0
    col_cut_indices_start = []
    col_cut_indices_end = []
    Logger.debug(row_cut_indices)
    for row_cut_index in row_cut_indices:
        # Logger.debug("row indices start and end %s %s", prev_row_cut, row_cut_index)
        block_rows = matrix[prev_row_cut:row_cut_index, :]
        blocks_col_sum = block_rows.sum(axis=0)
        # get the first and last col with non-zero entries

        non_zero_indices = np.flatnonzero(blocks_col_sum)
        # Logger.info("Block col sum shape %s", blocks_col_sum.shape)
        # Find index of the first non-zero value
        col_cut_index_start = non_zero_indices[0] if non_zero_indices.size > 0 else None

        # Find index of the last non-zero value
        col_cut_index_end = non_zero_indices[-1] if non_zero_indices.size > 0 else None
        # Logger.info(
        #     "col cut index start and end %s %s", col_cut_index_start, col_cut_index_end
        # )
        block = matrix[
            prev_row_cut:row_cut_index, col_cut_index_start : col_cut_index_end + 1
        ]
        blocks.append(block)
        prev_row_cut = row_cut_index
        col_cut_indices_start.append(col_cut_index_start)
        col_cut_indices_end.append(col_cut_index_end + 1)
        Logger.info("block shape %s", block.shape)
    assert sum([block.shape[0] for block in blocks]) == matrix.shape[0]
    return blocks, col_cut_indices_start, col_cut_indices_end


def _find_first_nonzero(arr, axis, invalid_val=-1):
    mask = arr != 0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)


def _find_last_nonzero(arr, axis, invalid_val=-1):
    mask = arr != 0
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)


def slice_candidate_blocks_by_mz(matrix):
    # TODO: finding valid slice position not done
    assert matrix.shape[1] >= 6000
    Logger.debug("Candidate matrix shape %s", matrix.shape)
    # slice candidate block by not splitting up isotope envlopes
    n_blocks = matrix.shape[1] // 3000
    Logger.debug("Slice candidate blocks into %s blocks.", n_blocks)
    block_size = matrix.shape[1] // n_blocks
    ref_col_cut_indices = [block_size * (i + 1) for i in range(n_blocks)]

    col_last_nonzero = _find_last_nonzero(arr=matrix, axis=0, invalid_val=-1)
    col_first_nonzero = _find_first_nonzero(arr=matrix, axis=0, invalid_val=-1)

    # comparison only starts with index 1, for actual indexing needs to +1
    valid_slice_pos = np.where(col_last_nonzero[0:-1] < col_first_nonzero[1:])[0] + 1
    # Logger.debug("valid slice pos %s", valid_slice_pos)
    col_cut_indices = []
    for i in range(n_blocks - 1):
        Logger.debug("column slice idx %s for block %s", ref_col_cut_indices[i], i + 1)
        idx = (np.abs(valid_slice_pos - ref_col_cut_indices[i])).argmin()
        col_cut_indices.append(valid_slice_pos[idx])
        valid_slice_pos = np.delete(valid_slice_pos, list(range(0, idx + 1)))
    if col_cut_indices[-1] != matrix.shape[1] + 1:
        col_cut_indices.append(matrix.shape[1] + 1)
    blocks = []
    prev_col_cut = 0
    col_cut_indices_start = []
    col_cut_indices_end = []
    Logger.debug("col cut indices: %s", col_cut_indices)
    for block_idx, col_cut_index in enumerate(col_cut_indices):
        Logger.debug(
            "Block %s, col indices start and end %s %s",
            block_idx,
            prev_col_cut,
            col_cut_index,
        )
        block_cols = matrix[:, prev_col_cut:col_cut_index]
        blocks_row_sum = block_cols.sum(axis=1)
        # get the first and last col with non-zero entries
        non_zero_indices = np.flatnonzero(blocks_row_sum)
        Logger.debug("Block row sum shape %s", blocks_row_sum.shape)

        # Find index of the first non-zero value
        row_cut_index_start = non_zero_indices[0] if non_zero_indices.size > 0 else None

        # Find index of the last non-zero value
        row_cut_index_end = non_zero_indices[-1] if non_zero_indices.size > 0 else None
        Logger.info(
            "row cut index start and end %s %s", row_cut_index_start, row_cut_index_end
        )
        block = matrix[
            row_cut_index_start : row_cut_index_end + 1, prev_col_cut:col_cut_index
        ]
        blocks.append(block)
        col_cut_indices_start.append(prev_col_cut)
        col_cut_indices_end.append(col_cut_index)
        prev_col_cut = col_cut_index
        # row_cut_indices_start.append(row_cut_index_start)
        # row_cut_indices_end.append(row_cut_index_end + 1)
        Logger.info("block shape %s", block.shape)
    assert sum([block.shape[0] for block in blocks]) == matrix.shape[0]
    return blocks, col_cut_indices_start, col_cut_indices_end


def slice_frame_data_blocks(frame_data, col_cut_indices_start, col_cut_indices_end):
    frame_data_blocks = []
    for start, end in zip(col_cut_indices_start, col_cut_indices_end):
        frame_data_block = frame_data[:, start:end]
        frame_data_blocks.append(frame_data_block)
        # Logger.info(frame_data_block.shape)
    return frame_data_blocks


def process_one_frame_ims(
    data: pd.DataFrame,
    ms1scans: pd.DataFrame,
    ms1_frame_idx: int,
    maxquant_result_ref_with_im_index_sortmz: pd.DataFrame,
    mobility_values: np.ndarray,
    delta_mobility_thres: int = 100,
    return_im_pept_act: bool = False,
    mz_bin_digits: int = 3,
    process_in_blocks: bool = True,
    extract_im_peak: bool = True,
    **kwargs,
):
    Logger.debug("Start data preparation.")
    # prepare data
    frame_data = data[{"frame_indices": [ms1scans.loc[ms1_frame_idx, "Id"]]}]
    Logger.debug("Frame data shape: %s", frame_data.shape[0])
    peaks_df = pd.DataFrame()
    im_pept_act_coo = {
        "coord_frame_indices": [],
        "coord_im_indices": [],
        "coord_pept_indices": [],
        "data": [],
    }
    if frame_data.shape[0] > 0:
        scan_time = np.round(ms1scans.loc[ms1_frame_idx, "Time_minute"], decimals=4)
        candidate_precursor_by_rt = maxquant_result_ref_with_im_index_sortmz.loc[
            (maxquant_result_ref_with_im_index_sortmz["RT_search_left"] <= scan_time)
            & (maxquant_result_ref_with_im_index_sortmz["RT_search_right"] >= scan_time)
        ]
        Logger.info(
            "Number of candidates by RT in frame %s: %s",
            ms1_frame_idx,
            candidate_precursor_by_rt.shape[0],
        )
        if candidate_precursor_by_rt.shape[0] > 0:
            candidate_precursor_by_rt.sort_values(
                "mz_rank", ascending=True, inplace=True
            )
            all_frame_pept_idx = candidate_precursor_by_rt.mz_rank.values
            (
                frame_array,
                candidate_array,
            ) = _prepare_sparse_matrices(
                candidate_precursor_by_rt,
                frame_data,
                mobility_values,
                all_frame_pept_idx,
                mz_bin_digits=mz_bin_digits,
            )

            assert frame_array.shape[1] == candidate_array.shape[1]
            Logger.debug("Start optimization with sparse encoding.")
            if candidate_precursor_by_rt.shape[0] > 6000 and process_in_blocks:
                im_pept_act = sparse_encode_divide_and_conquer(
                    frame_array, candidate_array
                )
            else:
                # optimization with sparse encoding
                im_pept_act = sparse_encode(
                    frame_array,
                    candidate_array,
                    algorithm="threshold",
                    alpha=0,
                    positive=True,
                    # n_jobs=4,
                )
            Logger.debug("Start peak selection.")
            if extract_im_peak:
                peaks_df = _select_im_peak_from_frame_act(
                    im_pept_act=im_pept_act,
                    all_pept_mzrank=all_frame_pept_idx,
                    maxquant_result_dict_with_im_index=candidate_precursor_by_rt,
                    delta_mobility_thres=delta_mobility_thres,
                    **kwargs,
                )
                peaks_df["frame_indices"] = ms1_frame_idx
            Logger.debug("Scan-wise opimtization completed.")
            if return_im_pept_act:
                im_pept_act_coo = {}
                nonzero_indices = np.nonzero(im_pept_act)
                im_pept_act_coo["data"] = im_pept_act[nonzero_indices]
                im_pept_act_coo["coord_frame_indices"] = np.repeat(
                    ms1_frame_idx, len(im_pept_act_coo["data"])
                )
                im_pept_act_coo["coord_im_indices"] = nonzero_indices[0]
                im_pept_act_coo["coord_pept_indices"] = all_frame_pept_idx[
                    nonzero_indices[1]
                ]
        else:
            Logger.info("No candidate precursor by RT from frame %s", ms1_frame_idx)
    else:
        Logger.info("No data for frame index %s", ms1_frame_idx)

    return (
        peaks_df,
        im_pept_act_coo,
        # candidate_array,
        # frame_array,
    )  # TODO: remove candidate array


def make_coo_from_dict(data_dict, shape: tuple, cutoff: List[int]):
    Logger.info("Shape of COO matrix: %s", shape)
    if len(cutoff) > 1:
        coo_list = []
        # n_pept_in_blocks = shape[2] // n_blocks_by_pept
        # cutoff = [(n_pept_in_blocks * (i + 1)) for i in range(n_blocks_by_pept - 1)]
        # cutoff.append(shape[2] + 1)
        Logger.debug("cutoff list %s", cutoff)
        prev_cutoff = 0
        for cutoff_i in cutoff:
            block_idx = np.where(
                (prev_cutoff <= np.array(data_dict["coord_pept_indices"]))
                & (np.array(data_dict["coord_pept_indices"]) < cutoff_i)
            )[0].astype(int)
            Logger.debug("block index %s", block_idx)
            coo_list.append(
                sparse.COO(
                    coords=[
                        list(itemgetter(*block_idx)(data_dict["coord_frame_indices"])),
                        list(itemgetter(*block_idx)(data_dict["coord_im_indices"])),
                        list(itemgetter(*block_idx)(data_dict["coord_pept_indices"])),
                    ],
                    data=list(itemgetter(*block_idx)(data_dict["data"])),
                    shape=shape,
                )
            )
            prev_cutoff = cutoff_i
        return coo_list
    else:
        return sparse.COO(
            coords=[
                data_dict["coord_frame_indices"],
                data_dict["coord_im_indices"],
                data_dict["coord_pept_indices"],
            ],
            data=data_dict["data"],
            shape=shape,
        )


def process_batch_frame_ims(
    data: pd.DataFrame,
    ms1scans: pd.DataFrame,
    batch_scan_idx: list,
    maxquant_result_ref_with_im_index: pd.DataFrame,
    mobility_values: np.ndarray,
    cutoff: List[int],
    delta_mobility_thres: int = 100,
    mz_bin_digits: int = 3,
    process_in_blocks: bool = True,
    batch_num: int = 0,
    save_dir: str = "",
    return_im_pept_act: bool = False,
    extract_im_peak: bool = True,
    **kwargs,
):
    batch_peaks_df = []
    batch_im_rt_pept_act_coo_dict = {
        "coord_frame_indices": [],
        "coord_im_indices": [],
        "coord_pept_indices": [],
        "data": [],
    }
    for scan_idx in batch_scan_idx:
        Logger.debug("Start processing frame index %s", scan_idx)
        peaks_df, frame_im_pept_act_coo = process_one_frame_ims(
            data=data,
            ms1scans=ms1scans,
            ms1_frame_idx=scan_idx,
            maxquant_result_ref_with_im_index_sortmz=maxquant_result_ref_with_im_index,
            mobility_values=mobility_values,
            delta_mobility_thres=delta_mobility_thres,
            mz_bin_digits=mz_bin_digits,
            process_in_blocks=process_in_blocks,
            return_im_pept_act=return_im_pept_act,
            extract_im_peak=extract_im_peak,
            **kwargs,
        )
        if extract_im_peak:
            batch_peaks_df.append(peaks_df)
        if return_im_pept_act:
            for key in batch_im_rt_pept_act_coo_dict.keys():
                batch_im_rt_pept_act_coo_dict[key].extend(frame_im_pept_act_coo[key])

    if extract_im_peak:
        batch_peaks_df = pd.concat(batch_peaks_df).reset_index(drop=True)
        batch_peaks_df.to_csv(
            os.path.join(save_dir, f"batch_peaks_df_{batch_num}.csv"), index=False
        )
    if return_im_pept_act:
        batch_im_rt_pept_act_coo = make_coo_from_dict(
            batch_im_rt_pept_act_coo_dict,
            shape=(
                len(ms1scans.index.values)
                + 1,  # this index is rank, starting from 1, add 1 for the last frame
                len(mobility_values),
                len(maxquant_result_ref_with_im_index.mz_rank)
                + 1,  # this index is rank, starting from 1, add 1 for the last frame
            ),
            cutoff=cutoff,
        )

        if isinstance(batch_im_rt_pept_act_coo, list):
            for pept_batch_idx, pept_batch_dict in enumerate(batch_im_rt_pept_act_coo):
                sparse.save_npz(
                    os.path.join(
                        save_dir,
                        f"im_rt_pept_act_coo_batch{batch_num}_peptbatch{pept_batch_idx}.npz",
                    ),
                    pept_batch_dict,
                )
                Logger.info(
                    "Size of COO matrix in batch %s, peptide batch %s: %s Mb",
                    batch_num,
                    pept_batch_idx,
                    pept_batch_dict.nbytes / 1e6,
                )
        else:
            sparse.save_npz(
                os.path.join(
                    save_dir, f"im_rt_pept_act_coo_batch{batch_num}_peptbatch0.npz"
                ),
                batch_im_rt_pept_act_coo,
            )
            Logger.info(
                "Size of COO matrix in batch %s: %s Mb",
                batch_num,
                batch_im_rt_pept_act_coo.nbytes / 1e6,
            )


def _prepare_sparse_matrices(
    candidate_precursor_by_rt,
    frame_data,
    mobility_values,
    all_id,
    mz_bin_digits: int = 3,
):
    # prepare arrays from sparse matrices
    candidate_id = np.repeat(
        candidate_precursor_by_rt.mz_rank.values, candidate_precursor_by_rt.mz_length
    )
    candidate_mz = np.round(
        np.array(list(itertools.chain(*candidate_precursor_by_rt.IsoMZ.values))),
        decimals=mz_bin_digits,
    )
    candidate_abundance = np.array(
        list(itertools.chain(*candidate_precursor_by_rt.IsoAbundance.values))
    )

    candidate_id_index = np.searchsorted(all_id, candidate_id)

    frame_mz = np.round(frame_data["mz_values"], decimals=mz_bin_digits)
    all_mz = np.sort(np.array(list(set(frame_mz).union(set(candidate_mz)))))
    Logger.debug(
        "Number of mz values in candidate, frame and joint:%s, %s, %s",
        len(set(candidate_mz)),
        len(set(frame_mz)),
        len(all_mz),
    )
    candidate_mz_index = np.searchsorted(all_mz, candidate_mz)
    frame_mz_index = np.searchsorted(all_mz, frame_mz)
    # Logger.info("all_mz shape %s", all_mz.shape)
    # Logger.info("frame mz shape %s", frame_mz.shape)
    all_im = np.sort(mobility_values["mobility_values"])
    # Logger.info("mobility values columns %s", mobility_values.columns)
    # Logger.info("all_im shape %s", all_im.shape)
    # Logger.info("frame mobility shape %s", frame_data["mobility_values"].shape)
    frame_im_index = np.searchsorted(all_im, frame_data["mobility_values"])

    frame_coo = coo_matrix(
        (frame_data["intensity_values"], (frame_im_index, frame_mz_index)),
    )
    candidate_coo = coo_matrix(
        (candidate_abundance, (candidate_id_index, candidate_mz_index))
    )
    # prepare arrays from sparse matrices
    min_mz_index = max(candidate_mz_index.min(), frame_mz_index.min())
    max_mz_index = min(candidate_mz_index.max(), frame_mz_index.max())
    Logger.debug("min and max mz index: %s %s", min_mz_index, max_mz_index)

    # make sure candidate mz index is not out of range of observed mz in frame
    candidate_mz_index_filtered = list(
        set(
            candidate_mz_index[
                (min_mz_index <= candidate_mz_index)
                & (candidate_mz_index <= max_mz_index)
            ]
        )
    )
    Logger.debug(
        "Number of mz values in filtered candidate index: %s",
        len(candidate_mz_index_filtered),
    )
    # only candidate mz is considered
    frame_array = frame_coo.toarray()[:, candidate_mz_index_filtered]
    candidate_array = candidate_coo.toarray()[:, candidate_mz_index_filtered]

    return (
        frame_array,
        candidate_array,
    )  # , pd.DataFrame({"mz_index":frame_mz_index, "mz_value":frame_mz}), pd.DataFrame({"mz_index":candidate_mz_index, "mz_value":candidate_mz}) #TODO: remove extra returns


def _select_im_peak_from_frame_act(
    im_pept_act: pd.DataFrame,
    all_pept_mzrank: np.ndarray,
    maxquant_result_dict_with_im_index: pd.DataFrame,
    delta_mobility_thres: int = 100,
    **kwargs,
):
    peak_properties_list = _select_peaks_from_im_pept_act(
        im_pept_act, pept_mzrank=all_pept_mzrank, **kwargs
    )
    # print(len(peak_properties_list))
    if peak_properties_list is None or all(v is None for v in peak_properties_list):
        peaks_df = pd.DataFrame()
        Logger.info("No peaks extracted.")
    else:
        peaks_df = pd.concat(peak_properties_list).reset_index(drop=True)
        Logger.info(
            "Number of peaks before delta mobility filter: %s", peaks_df.shape[0]
        )
        peaks_df = pd.merge(
            left=peaks_df,
            right=maxquant_result_dict_with_im_index[
                ["mobility_values_index", "mz_rank"]
            ],
            left_on="pept_mzrank",
            right_on="mz_rank",
            how="left",
        )
        peaks_df["delta_mobility"] = abs(
            peaks_df["peak"] - peaks_df["mobility_values_index"]
        )
        peaks_df = peaks_df.loc[peaks_df["delta_mobility"] <= delta_mobility_thres]
        if peaks_df.shape[0] > 0:
            Logger.info(
                "Number of peaks and peptide after delta mobility filter: %s %s",
                peaks_df.shape[0],
                peaks_df["pept_mzrank"].nunique(),
            )
            peaks_df = peaks_df.loc[
                peaks_df.groupby("pept_mzrank")["delta_mobility"].idxmin()
            ]
            peaks_df["peak_sum"] = peaks_df.apply(
                lambda x: _sum_im_act_per_pept(x, im_pept_act), axis=1
            )
        else:
            Logger.info("No peaks found after delta mobility filter.")
            peaks_df = pd.DataFrame()

    return peaks_df


def _sum_im_act_per_pept(peak_row, im_pept_act):
    pept_id_index = int(peak_row["pept_mzrank_index"])
    peak_start = int(peak_row["left_ips"])
    peak_end = int(peak_row["right_ips"])
    peak_sum = im_pept_act[peak_start : peak_end + 1, pept_id_index].sum()
    # Logger.debug("shape of peak_sum: %s", peak_sum.shape)
    return peak_sum


def _select_peaks_from_im_pept_act(
    im_pept_act: np.ndarray, pept_mzrank: np.ndarray, **kwargs
):
    # filtered only the columns with nonzero value > = 3
    pept_nonzero_count = np.count_nonzero(im_pept_act, axis=0)
    pept_valid_idx = np.where(pept_nonzero_count >= 3)[0]
    if pept_valid_idx.size > 0:
        Logger.info(
            "Number of peptides with nonzero mobility value >= 3: %s",
            len(pept_valid_idx),
        )
        peak_properties = [
            _extract_peaks_in_im(
                im_pept_act_array=im_pept_act[:, idx],
                index=idx,
                pept_mzrank=pept_mzrank[idx],
                **kwargs,
            )
            for idx in pept_valid_idx
        ]
        if all(v is None for v in peak_properties):
            Logger.info("No peaks extracted.")
            peak_properties = None
    else:
        Logger.info("No peptides with nonzero mobility value >= 3.")
        peak_properties = None
    return peak_properties


def _extract_peaks_in_im(
    im_pept_act_array, index: int, pept_mzrank: int, height=0.1, width=4, rel_height=1
):
    # Logger.debug("Peak extration width %s", width)
    peaks, peak_properties = find_peaks(
        im_pept_act_array, height=height, width=width, rel_height=rel_height
    )
    if peaks.size > 0:
        peak_properties["pept_mzrank_index"] = np.repeat(index, len(peaks))
        peak_properties["pept_mzrank"] = np.repeat(pept_mzrank, len(peaks))
        peak_properties["peak"] = peaks
        peak_properties = pd.DataFrame(peak_properties)
        # Logger.debug(
        #     "Number of peaks extracted from pept_id %s: %s", pept_id, len(peaks)
        # )
    else:
        # Logger.debug("No peak extracted from pept_id %s in this frame.", pept_id)
        peak_properties = None
    return peak_properties


def parallel(func=None, args=(), merge_func=lambda x: x, parallelism=cpu_count()):
    def decorator(func: Callable):
        def inner(*args, **kwargs):
            results = Parallel(n_jobs=parallelism)(
                delayed(func)(*args, **kwargs) for i in range()
            )
            return merge_func(results)

        return inner

    if func is None:
        # decorator was used like @parallel(...)
        return decorator
    else:
        # decorator was used like @parallel, without parens
        return decorator(func)


def generate_id_partitions(
    id_array,
    n_batch,
    how: Literal["round_robin", "block"] = "block",
    n_edge_counts: int = 50,
):
    id_partitions = [[] for _ in range(n_batch)]
    if how == "round_robin":
        Logger.info("Generate id partitions by block.")
        for i in range(n_batch):
            batch_idx = np.arange(i, len(id_array), n_batch)
            id_partitions[i] = id_array[batch_idx]
    elif how == "block":
        Logger.info("Generate id partitions by block.")
        block_size = (len(id_array) - 2 * n_edge_counts) // n_batch
        for i in range(n_batch):
            if i == 0:
                mark = block_size + n_edge_counts
                id_partitions[i] = id_array[:mark]
            elif i == n_batch - 1:
                id_partitions[i] = id_array[mark:]
            else:
                id_partitions[i] = id_array[mark : mark + block_size]
                mark = block_size * (i + 1) + n_edge_counts
    return id_partitions


def process_ims_frames_parallel(
    n_jobs: int,
    batch_scan_indices: list,
    data,
    ms1scans: pd.DataFrame,
    maxquant_ref: pd.DataFrame,
    mobility_values: np.array,
    cutoff: List[int],
    delta_mobility_thres: int = 100,
    mz_bin_digits: int = 3,
    process_in_blocks: bool = True,
    width: int = 4,
    save_dir: str = "",
    return_im_pept_act: bool = False,
    extract_im_peak: bool = True,
    #n_blocks_by_pept: int = 0,
):
    list_batch_im_pept_act_coo_dict = Parallel(n_jobs=n_jobs)(
        delayed(process_batch_frame_ims)(
            data=data,
            maxquant_result_ref_with_im_index=maxquant_ref,
            ms1scans=ms1scans,
            batch_scan_idx=batch,
            mobility_values=mobility_values,
            delta_mobility_thres=delta_mobility_thres,
            mz_bin_digits=mz_bin_digits,
            process_in_blocks=process_in_blocks,
            width=width,
            batch_num=batch[0],
            save_dir=save_dir,
            return_im_pept_act=return_im_pept_act,
            extract_im_peak=extract_im_peak,
            cutoff=cutoff,
        )
        for batch in batch_scan_indices
    )
    # scan_result_dict = dict(pair for d in scan_results_list for pair in d.items())
    # if there is multiple values in return
    # frame_results_df = pd.concat(frame_results_list).reset_index(drop=True)
    # return frame_results_df


def process_scans_parallel(
    n_jobs: int,
    ms1scans: pd.DataFrame,
    maxquant_ref: pd.DataFrame,
    abundance_missing_threshold: float = 0.4,
    alpha_criteria: _alpha_criteria = "convergence",
    alphas: Union[List, np.ndarray] = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
    loss: _loss = "lasso",
    opt_algo: _algo = "lasso_cd",
    metric: _alpha_opt_metric = "cos_dist",
    preprocessing_method: _pp_method = "raw",
    corr_thres: float = 0.95,
    max_iter: int = 1000,
    return_precursor_scan_cos_dist: bool = False,
):
    scan_result_list = Parallel(n_jobs=n_jobs)(
        delayed(process_one_scan)(
            scan_idx=scan_idx,
            OneScan=OneScan,
            Maxquant_result=maxquant_ref,
            AbundanceMissingThres=abundance_missing_threshold,
            alpha_criteria=alpha_criteria,
            alphas=alphas,
            metric=metric,
            loss=loss,
            opt_algo=opt_algo,
            preprocessing_method=preprocessing_method,
            corr_thres=corr_thres,
            max_iter=max_iter,
            return_interim_results=False,
            return_precursor_scan_cos_dist=return_precursor_scan_cos_dist,
        )
        for scan_idx, OneScan in ms1scans.iterrows()
    )
    scan_result_dict = dict(pair for d in scan_result_list for pair in d.items())
    return scan_result_dict
