import logging
import os
from typing import List, Literal, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from .compare_maxquant import (
    merge_with_maxquant_exp,
    evaluate_rt_overlap,
    filter_merged_by_rt_overlap,
    sum_pcm_intensity_from_exp,
    add_sum_act_cols,
)

from utils.plot import plot_scatter, plot_venn2, plot_venn3, save_plot

Logger = logging.getLogger(__name__)


def _report_intensity(intensity: pd.Series, save_dir: Union[str, None]):
    """Report the distribution of a given intensity column"""
    print("--------------", intensity.name, "-----------------")
    n_MQ_nonzero = intensity[intensity > 1].count()  # always filter at 1
    print("Non zero intensity in", intensity.name, "=", n_MQ_nonzero)
    np.log10(intensity + 1).hist()  # type: ignore
    plt.title("Distribution of " + str(intensity.name) + " Intensity (log)")
    plt.xlabel("Intensity (log)")
    plt.ylabel("Count")
    if save_dir is not None:
        plt.savefig(
            os.path.join(save_dir, "distr_" + str(intensity.name) + ".png"), dpi=300
        )
        plt.close()


def _report_distr(
    data: pd.Series, suffix: Union[str, None], save_dir: Union[str, None]
):
    """report the distribution of a given column"""
    name = str(data.name)
    if suffix is not None:  # add suffix to name if desired
        name += "_" + suffix
    print("--------------", name, "-----------------")
    print(data.describe())
    data.hist()
    plt.title("Distribution of " + name)
    plt.xlabel(name)
    plt.ylabel("Count")
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, "distr_" + name + ".png"), dpi=300)
        plt.close()


def generate_result_report(
    scan_record: pd.DataFrame,
    intensity_cols: list,
    save_dir: Union[str, None] = None,
):
    """Generate a report for the result of SBS"""
    for col in intensity_cols:
        _report_intensity(intensity=col, save_dir=save_dir)

    scan_record["n_CandidateByRT"] = scan_record["CandidatePrecursorByRT"].apply(
        lambda x: len(x) if x is not None else 0
    )
    scan_record["n_filteredCandidate"] = scan_record["FilteredPrecursor"].apply(
        lambda x: len(x) if x is not None else 0
    )
    scan_record["preservedRatio_IE_filter"] = (
        scan_record["n_filteredCandidate"] / scan_record["n_CandidateByRT"]
    )

    empty_scans = scan_record.loc[scan_record["n_filteredCandidate"] <= 1]
    non_empty_scans = scan_record.loc[scan_record["n_filteredCandidate"] > 1]
    print("--------------Empty Scans-----------------")
    for i in [
        "Time",
        "n_CandidateByRT",
        "n_filteredCandidate",
        "preservedRatio_IE_filter",
    ]:
        try:
            _report_distr(data=empty_scans[i], suffix="EmptyScans", save_dir=save_dir)
        except:
            print("Column ", i, "does not exist!")

    print("--------------Non Empty Scans-----------------")
    for i in [
        "Loss",
        "Cosine Dist",
        "n_CandidateByRT",
        "n_filteredCandidate",
        "preservedRatio_IE_filter",
        "IntensityExplained",
        "PeaksExplained",
        "NumberHighlyCorrDictCandidate",
    ]:
        try:
            _report_distr(
                data=non_empty_scans[i], suffix="nonEmptyScans", save_dir=save_dir
            )
        except:
            print("Column ", i, "does not exist!")

    sns.lineplot(x=non_empty_scans["Time"], y=non_empty_scans["Cosine Dist"])
    plt.title("Cosine Distance By Scan")
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, "cos_dist_byscan.png"), dpi=300)
        plt.close()

    return scan_record


def plot_corr_int_ref_and_act(
    ref_int,
    sum_act,
    data=None,
    filter_thres: float = 1,
    contour: bool = False,
    interactive: bool = False,
    show_diag: bool = True,
    color: Union[None, pd.Series] = None,
    hover_data: Union[None, List] = None,
    save_dir: Union[None, str] = None,
    log_x: bool = True,
    log_y: bool = True,
):
    if hover_data is None:
        hover_data = [
            "Modified sequence",
            "Leading proteins",
            "Charge",
            "id",
        ]
    reg_int, abs_residue, valid_idx = plot_scatter(
        x=ref_int,
        y=sum_act,
        log_x=log_x,
        log_y=log_y,
        data=data,
        filter_thres=filter_thres,
        contour=contour,
        interactive=interactive,
        hover_data=hover_data,
        show_diag=show_diag,
        color=color,
        title="Corr. of Quant. Results ",
        save_dir=save_dir,
        x_label="Reference (log)",
        y_label="Infered (log)",
    )
    return reg_int, abs_residue, valid_idx


def FindStartAndEndScan(activation: np.ndarray, thres: float = 1.0):
    """
    Given the activation matrix, finds the first and last value for each row (precursor)
    that is larger than given threshold

    :actiavtion:
    :thres:
    """
    # Find the indices where the values are greater than 1
    row_indices, col_indices = np.where(activation > thres)

    # Group col_indices by row_indices
    grouped_indices = pd.Series(col_indices).groupby(row_indices)

    # Calculate the minimum and maximum values for each group
    min_indices = grouped_indices.min()
    max_indices = grouped_indices.max()

    # Create a DataFrame to store the results
    df = pd.DataFrame(
        {
            "id": min_indices.index,
            "Scan Index_start_SBS": min_indices.values,
            "Scan Index_end_SBS": max_indices.values,
        }
    )
    df["CountScan_SBS"] = df["Scan Index_start_SBS"] - df["Scan Index_end_SBS"] + 1

    return df


def plot_alphas_across_scan(
    scan_record: pd.DataFrame,
    x: Literal["Scan", "Time"] = "Scan",
    save_dir: str | None = None,
):
    NonEmptyScans = scan_record.loc[scan_record["BestAlpha"] != None]
    EmptyScans = scan_record.loc[scan_record["BestAlpha"] == None]
    minimal_alpha = min(filter(lambda x: x > 0, NonEmptyScans["BestAlpha"]))
    zero_replace = minimal_alpha / 10
    Logger.info(
        "Alpha range: %s, replacing all zeros with %s",
        NonEmptyScans["BestAlpha"].unique(),
        zero_replace,
    )
    NonEmptyScans["alpha_to_plot"] = np.log10(
        NonEmptyScans["BestAlpha"].replace(0, zero_replace, inplace=False)
    )

    plt.plot(NonEmptyScans[x], NonEmptyScans["alpha_to_plot"])  # alphas
    plt.scatter(
        x=EmptyScans[x],
        y=np.repeat(np.log10(zero_replace), EmptyScans.shape[0]),
        s=10,
        marker="x",
        c="black",
    )  # empty scans
    plt.xlabel(x)
    plt.ylabel("Alpha (log)")
    plt.title("Best Alpha over " + x)
    save_plot(save_dir=save_dir, fig_type_name="Alpha_along", fig_spec_name=x)


class SBSResult:
    """SBSResult class for result analysis"""

    def __init__(
        self,
        maxquant_ref_df: pd.DataFrame,
        maxquant_exp_df: pd.DataFrame | None = None,
        sum_raw: pd.DataFrame | None = None,
        sum_gaussian: pd.DataFrame | None = None,
        sum_minima: pd.DataFrame | None = None,
        sum_peak: pd.DataFrame | None = None,
        sum_cols: List[str] | None = None,
        ims: bool = False,
    ) -> None:
        """Initialize SBSResult object and intergrate all activation data."""

        assert any(
            item is not None for item in [sum_raw, sum_peak, sum_gaussian, sum_minima]
        )
        sum_cols = []
        for s in [sum_raw, sum_peak, sum_gaussian, sum_minima]:
            if s is not None:
                s.reset_index(drop=True, inplace=True)
                sum_cols += list(s.columns)
        if sum_cols is None:
            self.sum_cols = sum_cols
        else:
            self.sum_cols = sum_cols
        pp_sumactivation = pd.concat(
            [
                sum_raw,
                sum_gaussian,
                sum_minima,
                sum_peak,
            ],
            axis=1,
        )
        maxquant_ref_df["Reverse"] = np.where(maxquant_ref_df["Reverse"].isnull(), 0, 1)
        Logger.info("Reference shape: %s", maxquant_ref_df.shape)
        Logger.info("Experiment shape: %s", maxquant_exp_df.shape)
        if ims:
            self.ref_df = pd.merge(
                left=maxquant_ref_df,
                right=sum_raw,
                left_on="mz_rank",
                right_index=True,
                how="inner",
            )
            Logger.debug(
                "Reference shape after merging activation sum: %s", self.ref_df.shape
            )
        else:
            pp_sumactivation = pp_sumactivation.set_index(maxquant_ref_df.index)
            self.ref_df = pd.concat([maxquant_ref_df, pp_sumactivation], axis=1)

        self.exp_df = maxquant_exp_df.copy()
        self.exp_df["Reverse"] = np.where(maxquant_exp_df["Reverse"].isnull(), 0, 1)

        self.ref_exp_df_inner = None
        Logger.debug("sum cols: %s", self.sum_cols)
        self.ref_df_non_zero = self.ref_df.loc[self.ref_df[self.sum_cols[0]] > 0, :]
        Logger.debug("Reference non zero shape: %s", self.ref_df_non_zero.shape)

    def compare_with_maxquant_exp_int(
        self,
        filter_by_rt_overlap: List[
            Literal["full_overlap", "partial_overlap", "no_overlap"]
        ]  # TODO: refactor to the same place as compare_maxquant
        | None = None,
        handle_mul_exp_pcm: Literal["drop", "agg", "preserve"] = "agg",
        # agg_pcm_intensity: bool = True,
        save_dir: str | None = None,
    ):
        """
        Compare activation with the intensity of the precursors from the experiment file.

        :param filter_by_rt_overlap: A list of conditions to filter the data based on retention time overlap.
            Options are "full_overlap", "partial_overlap", or "no_overlap".
        :type filter_by_rt_overlap: List[Literal["full_overlap", "partial_overlap", "no_overlap"]] or None, optional
        :param agg_pcm_intensity: Whether to aggregate the precursor intensity from the experiment file.
            If True, the intensity will be aggregated by the precursor charge combination.
            If False, the intensity will not be aggregated.
        :type agg_pcm_intensity: bool, optional
        :param save_dir: The directory to save the plot. If None, the plot will not be saved.
        :type save_dir: str or None, optional
        """
        # print("Hello")
        match handle_mul_exp_pcm:
            case "drop":
                n_pre_drop = self.exp_df.shape[0]
                self.exp_df = self.exp_df.drop_duplicates(
                    subset=["Modified sequence", "Charge"], keep=False
                )
                n_post_drop = self.exp_df.shape[0]
                Logger.info(
                    "Drop all duplicated pcm. %s -> %s", n_pre_drop, n_post_drop
                )
            case "agg":
                self.exp_df = sum_pcm_intensity_from_exp(self.exp_df)
            case "preserve":
                self.exp_df = self.exp_df

        maxquant_ref_and_exp = merge_with_maxquant_exp(
            maxquant_exp_df=self.exp_df,
            maxquant_ref_df=self.ref_df_non_zero,
            ref_cols=self.sum_cols,
        )

        maxquant_ref_and_exp = evaluate_rt_overlap(
            maxquant_ref_and_exp, save_dir=save_dir
        )
        if filter_by_rt_overlap is not None:
            maxquant_ref_and_exp = filter_merged_by_rt_overlap(
                condition=filter_by_rt_overlap,
                maxquant_ref_and_exp=maxquant_ref_and_exp,
            )
        else:
            Logger.info(
                "No filter_by_rt_overlap is specified, use all entries for plotting."
            )

        self.ref_exp_df_inner = maxquant_ref_and_exp
        Logger.info("Ref exp inner join shape: %s", self.ref_exp_df_inner.shape)

    def plot_intensity_corr(
        self,
        ref_col: str = "Intensity",
        inf_col: str = "AUCActivationRaw",
        interactive: bool = False,
        save_dir: str | None = None,
        **kwargs,
    ):
        """Plot the correlation between the intensity from the experiment file and the activation columns"""
        reg_int, abs_residue, valid_idx = plot_corr_int_ref_and_act(
            self.ref_exp_df_inner[ref_col],
            self.ref_exp_df_inner[inf_col],
            data=self.ref_exp_df_inner,
            interactive=interactive,
            save_dir=save_dir,
            **kwargs,
        )

    def plot_overlap_with_MQ(
        self,
        save_dir: str | None = None,
        save_format: str = "png",
        show_ref: bool = False,
    ):
        Logger.debug("Experiment columns: %s", self.exp_df.columns)
        self.exp_df_unique_PCM = (
            self.exp_df.groupby(["Modified sequence", "Charge", "Reverse"])
            .agg(
                {
                    "Retention time": "median",
                    "Intensity": "sum",
                    "Calibrated retention time start": "min",
                    "Calibrated retention time finish": "max",
                }
            )
            .reset_index()
        )
        Logger.info("Unique PCM TD in experiment: %s", self.exp_df_unique_PCM.shape)
        self.exp_df_unique_PCM["precursor"] = self.exp_df_unique_PCM[
            "Modified sequence"
        ] + self.exp_df_unique_PCM["Charge"].astype(str)

        self.ref_df_non_zero["precursor"] = self.ref_df_non_zero[
            "Modified sequence"
        ] + self.ref_df_non_zero["Charge"].astype(str)
        Logger.info(
            "Unique PCM TD in non zero activations: %s", self.ref_df_non_zero.shape
        )
        if show_ref:
            self.ref_df["precursor"] = self.ref_df["Modified sequence"] + self.ref_df[
                "Charge"
            ].astype(str)
            plot_venn3(
                set1=set(
                    self.exp_df_unique_PCM.loc[
                        self.exp_df_unique_PCM["Reverse"] == 0, "precursor"
                    ].to_list()
                ),
                set2=set(
                    self.ref_df_non_zero.loc[
                        self.ref_df_non_zero["Reverse"] == 0, "precursor"
                    ].to_list()
                ),
                set3=set(
                    self.ref_df.loc[self.ref_df["Reverse"] == 0, "precursor"].to_list()
                ),
                label1="Maxquant",
                label2="SBS",
                label3="Reference",
                save_dir=save_dir,
                save_format=save_format,
                title="IdentificationOfTarget",
            )

            plot_venn3(
                set1=set(
                    self.exp_df_unique_PCM.loc[
                        (self.exp_df_unique_PCM["Reverse"] == 0)
                        & (self.exp_df_unique_PCM["Intensity"] > 0),
                        "precursor",
                    ].to_list()
                ),
                set2=set(
                    self.ref_df_non_zero.loc[
                        self.ref_df_non_zero["Reverse"] == 0, "precursor"
                    ].to_list()
                ),
                set3=set(
                    self.ref_df.loc[self.ref_df["Reverse"] == 0, "precursor"].to_list()
                ),
                label1="Maxquant",
                label2="SBS",
                label3="Reference",
                save_dir=save_dir,
                save_format=save_format,
                title="QuantificationOfTarget",
            )
        else:
            plot_venn2(
                set1=set(
                    self.exp_df_unique_PCM.loc[
                        self.exp_df_unique_PCM["Reverse"] == 0, "precursor"
                    ].to_list()
                ),
                set2=set(
                    self.ref_df_non_zero.loc[
                        self.ref_df_non_zero["Reverse"] == 0, "precursor"
                    ].to_list()
                ),
                label1="Maxquant",
                label2="SBS",
                save_dir=save_dir,
                save_format=save_format,
                title="IdentificationOfTarget",
            )

            plot_venn2(
                set1=set(
                    self.exp_df_unique_PCM.loc[
                        (self.exp_df_unique_PCM["Reverse"] == 0)
                        & (self.exp_df_unique_PCM["Intensity"] > 0),
                        "precursor",
                    ].to_list()
                ),
                set2=set(
                    self.ref_df_non_zero.loc[
                        self.ref_df_non_zero["Reverse"] == 0, "precursor"
                    ].to_list()
                ),
                label1="Maxquant",
                label2="SBS",
                save_dir=save_dir,
                save_format=save_format,
                title="QuantificationOfTarget",
            )

    def eval_target_decoy(
        self, ref_col: str = "AUCActivationRaw", save_dir: str | None = None
    ):
        self.TDC_table = self.ref_df_non_zero.groupby("Reverse").agg(
            {"id": "count", ref_col: "mean"}
        )
        _, axs = plt.subplots(ncols=3, width_ratios=[1, 1, 2], figsize=(12, 5))
        sns.countplot(data=self.ref_df_non_zero, x="Reverse", ax=axs[0])
        axs[0].bar_label(axs[0].containers[0])
        sns.barplot(
            data=self.ref_df_non_zero, x="Reverse", y="AUCActivationRaw", ax=axs[1]
        )
        axs[1].bar_label(axs[1].containers[0])
        axs[1].set_ylabel("Mean Activation")
        sns.kdeplot(
            data=self.ref_df_non_zero,
            x=np.log10(1 + self.ref_df_non_zero["AUCActivationRaw"]),
            hue="Reverse",
            ax=axs[2],
        )
        axs[2].set_xlabel("Activation (log)")
        save_plot(save_dir=save_dir, fig_type_name="TDC", fig_spec_name="")
        return self.TDC_table
