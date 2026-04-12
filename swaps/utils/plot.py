import logging
import os
from typing import List, Set, Union, Literal, Optional

import matplotlib.pyplot as plt
from matplotlib import colormaps, patches  # type: ignore
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from matplotlib_venn import venn2, venn3
from matplotlib.patches import Rectangle
from scipy import stats
from sparse import SparseArray
from utils.tools import ExtractPeak
from postprocessing.ims_3d import (
    get_ref_rt_im_range,
    slice_pept_act,
    prepare_slice_pept_act_df,
    get_bbox_from_mq_exp,
)

Logger = logging.getLogger(__name__)


def plot_pie(
    sizes: np.ndarray,
    labels: np.ndarray,
    title: str,
    save_dir: str | None = None,
    fig_spec_name: str | None = None,
    accumulative_threshold: float | None = None,
):
    """plot pie charts with accumulative threshold"""
    _, ax1 = plt.subplots()
    if accumulative_threshold is not None:
        # Sort sizes in descending order and get the sorted labels
        sorted_indices = np.argsort(sizes)[::-1]
        sorted_sizes = sizes[sorted_indices]
        sorted_labels = labels[sorted_indices]

        # Calculate cumulative sum
        cumulative_sizes = np.cumsum(sorted_sizes)

        # Find the index where cumulative sum exceeds 95% of the total
        idx = np.where(
            cumulative_sizes >= accumulative_threshold * cumulative_sizes[-1]
        )[0][0]

        # Include only the segments needed to reach 95% of the total
        sizes = sorted_sizes[: idx + 1]
        labels = sorted_labels[: idx + 1]

        # Add 'Others' segment if there are remaining sizes
        if idx < len(sorted_sizes) - 1:
            sizes = np.append(sizes, sum(sorted_sizes[idx + 1 :]))
            labels = np.append(labels, "Others")

    ax1.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
    ax1.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax1.set_title(title)
    save_plot(
        save_dir=save_dir,
        fig_type_name="PieChart",
        fig_spec_name=fig_spec_name,
        fig_format="png",
        bbox_inches="tight",
    )


def plot_scatter(
    x: pd.Series,
    y: pd.Series,
    log_x: bool = False,
    log_y: bool = False,
    data: Union[None, pd.DataFrame] = None,
    filter_thres: float = 0,
    contour: bool = False,
    interactive: bool = False,
    hover_data: Union[None, List] = None,  # only used if interactive is true
    color: Union[None, pd.Series] = None,
    show_diag: bool = True,
    show_conf: Union[None, tuple] = None,
    save_dir: Union[None, str] = None,
    x_label: Union[None, str] = None,
    y_label: Union[None, str] = None,
    title: Union[None, str] = "auto",
    fig_spec_name: str = "",
    line_width: int = 2,
    font_size: int = 22,
    fig_size: tuple = (8, 6),
):
    """
    Generate scatter plot with correlation coefficient and number of data points,
    with other annotations possible

    :color: a pandas series for color, default is color with density info

    return
    -> valid_idx: index kept after filter
    """
    valid_idx = np.where((x > filter_thres) & (y > filter_thres))
    x_name = str(x.name)
    y_name = str(y.name)

    if data is not None:
        data = data.iloc[valid_idx[0], :].copy()
    else:
        data = pd.DataFrame({x_name: x.values[valid_idx], y_name: y.values[valid_idx]})

    x_log = x.values[valid_idx]
    y_log = y.values[valid_idx]
    if log_x:
        x_name += "_log"
        x_log = np.log10(x.values[valid_idx])
        data[x_name] = x_log
    if log_y:
        y_name += "_log"
        y_log = np.log10(y.values[valid_idx])
        data[y_name] = y_log

    if x_label is None:
        x_label = x_name
    if y_label is None:
        y_label = y_name
    if title == "auto":
        title = "Corr. of " + x_name + " and " + y_name

    PearsonR = stats.pearsonr(x=x_log, y=y_log)  # w/ log and w/o outliers
    SpearmanR = stats.spearmanr(a=x_log, b=y_log)
    slope, intercept, _, _, _ = stats.linregress(x=x_log, y=y_log)
    Logger.info(
        "Data: %s, %s, slope = %s, intercept = %s, Pearson's R = %s, Spearman's R = %s",
        x_name,
        y_name,
        np.round(slope, 3).item(),
        np.round(intercept, 3).item(),
        np.round(PearsonR[0], 3).item(),
        np.round(SpearmanR[0], 3).item(),
    )

    # calculate the point density
    if color is None:
        xy = np.vstack([x_log, y_log])
        color = stats.gaussian_kde(xy)(xy)
    else:
        color = color.values[valid_idx]

    RegressionY = x_log * slope + intercept
    AbsResidue = abs(y_log - RegressionY)

    if interactive:
        fig = px.scatter(
            data,
            x=x_name,
            y=y_name,
            color=color,
            hover_data=hover_data,
            title=title,
            labels={x_name: x_label, y_name: y_label},
            trendline="ols",
        )
        fig.update_layout(
            autosize=False,
            width=800,
            height=600,
        )
        if show_diag:  # Add a diagonal line y = x
            fig.update_layout(
                shapes=[
                    dict(
                        type="line",
                        x0=min(x_log),
                        y0=min(x_log),
                        x1=max(x_log),
                        y1=max(x_log),
                        line=dict(color="red", width=2),
                    )
                ]
            )
        # TODO: change to smart/relative positioning
        fig.add_annotation(
            x=6.5, y=10, text="N = " + str(x_log.shape[0]), showarrow=False
        )
        fig.add_annotation(
            x=6.5,
            y=11,
            text="Prs.r = "
            + "{:.3f}".format(PearsonR[0])
            + ", Sprm.r = "
            + "{:.3f}".format(SpearmanR[0]),
            showarrow=False,
        )
        fig.show()

    else:
        plt.rc(
            "font", size=font_size
        )  # Set the default font size for all text elements
        plt.figure(figsize=fig_size)  # Set the size of the figure
        # Plot with correlation
        if contour:
            ax = sns.jointplot(x=x_log, y=y_log, kind="kde")
            ax.ax_marg_x.remove()
            ax.ax_marg_y.remove()
            fig_type_name = "CorrQuantificationDensity"
        else:
            ax = sns.regplot(x=x_log, y=y_log, scatter=False, fit_reg=True)
            sns.scatterplot(x=x_log, y=y_log, hue=color, linewidth=0, legend=False, ax=ax)  # type: ignore

            ax.annotate(
                "N = " + str(x_log.shape[0]),
                xy=(0.2, 0.85),
                xycoords="figure fraction",
                horizontalalignment="left",
                verticalalignment="top",
                fontsize=0.9 * font_size,
            )
            ax.annotate(
                "Prs.r = "
                + "{:.3f}".format(PearsonR[0])
                + ", Sprm.r = "
                + "{:.3f}".format(SpearmanR[0]),
                xy=(0.2, 0.8),
                xycoords="figure fraction",
                horizontalalignment="left",
                verticalalignment="top",
                fontsize=0.9 * font_size,
            )
            # ax.annotate(
            #     "slp. = "
            #     + "{:.3f}".format(slope)
            #     + ", intrcpt. = "
            #     + "{:.3f}".format(intercept),
            #     xy=(0.2, 0.75),
            #     xycoords="figure fraction",
            #     horizontalalignment="left",
            #     verticalalignment="top",
            #     fontsize=7,
            # )
            fig_type_name = "CorrQuantification"
        # Set the width of all spines (top, right, bottom, left)
        for spine in ax.spines.values():
            spine.set_linewidth(line_width)  # Increase this value for thicker lines
        # Optional: you can also make tick marks thicker
        ax.tick_params(width=line_width)
        min_val = min(x_log)
        max_val = max(x_log)
        if show_diag:
            plt.plot(
                [min_val, max_val],
                [min_val, max_val],
                linestyle="--",
                color="k",
                lw=1,
                label="y=x",
            )
        if show_conf is not None:
            plt.plot(
                [min_val, max_val],
                [min_val + show_conf[0], max_val + show_conf[0]],
                linestyle="--",
                color="green",
                lw=2,
                label="lower bound",
            )
            plt.plot(
                [min_val, max_val],
                [min_val + show_conf[1], max_val + show_conf[1]],
                linestyle="--",
                color="green",
                lw=2,
                label="upper bound",
            )
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        if title is not None:
            plt.title(title)
        save_plot(
            save_dir=save_dir,
            fig_type_name=fig_type_name,
            fig_spec_name=y_name + fig_spec_name,
        )

    return RegressionY.T, AbsResidue.T, valid_idx


def plot_venn2(
    set1: Set,
    set2: Set,
    label1: str,
    label2: str,
    save_dir: str | None = None,
    # save_format: str = "png",
    title: str | None = None,
    fig_spec_name: str | None = None,
):
    venn2([set1, set2], set_labels=(label1, label2))
    if title is not None:
        plt.title(title)
    save_plot(
        save_dir=save_dir,
        fig_type_name="VennDiag",
        fig_spec_name=fig_spec_name,
        # fig_format=save_format,
    )


def plot_venn3(
    set1: Set,
    set2: Set,
    set3: Set,
    label1: str,
    label2: str,
    label3: str,
    save_dir: str | None = None,
    # save_format: str = "png",
    title: str | None = None,
    fig_spec_name: str | None = None,
    **kwargs,
):
    venn3([set1, set2, set3], set_labels=(label1, label2, label3))
    if title is not None:
        plt.title(title)
    save_plot(
        save_dir=save_dir,
        fig_type_name="VennDiag",
        fig_spec_name=fig_spec_name,
        # fig_format=save_format,
        **kwargs,
    )


def plot_comparison(
    y_true: pd.Series,
    y_pred: pd.Series,
    x_true: Union[None, pd.Series] = None,
    x_pred: Union[None, pd.Series] = None,
    log_y: bool = False,
    align_y: bool = False,
):
    fig, axs = plt.subplots(2, 1, sharex=True, sharey=False)
    fig.subplots_adjust(hspace=0)
    if log_y:
        y_pred = np.log10(y_pred + 1)
        y_true = np.log10(y_true + 1)
    if x_true is None:
        x_true = np.arange(len(y_true))
    if x_pred is None:
        x_pred = np.arange(len(y_pred))
    axs[0].vlines(x=x_true, ymin=0, ymax=y_true)
    axs[1].vlines(x=x_pred, ymin=-y_pred, ymax=0)
    if align_y:
        # enforce same y axis limits
        axs[0].set_ylim([0, max(axs[0].get_ylim()[1], abs(axs[1].get_ylim()[0]))])
        axs[1].set_ylim([-axs[0].get_ylim()[1], 0])


def save_plot(
    save_dir, fig_type_name, fig_spec_name, fig_format=["png", "svg"], **kwargs
):
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for fmt in fig_format:
            plt.savefig(
                fname=os.path.join(
                    save_dir, fig_type_name + "_" + fig_spec_name + "." + fmt
                ),
                dpi=300,
                format=fmt,
                bbox_inches="tight",
                **kwargs,
            )
            Logger.info(
                "Save plot at %s",
                os.path.join(save_dir, fig_type_name + "_" + fig_spec_name + "." + fmt),
            )
            # plt.show()
        plt.close()
    else:
        plt.show()


def plot_true_and_predict(x, prediction, true, log: bool = False):
    if log:
        prediction = np.log10(prediction + 1)
        true = np.log10(true + 1)
    fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0)

    axs[0].vlines(x=x, ymin=0, ymax=true)
    axs[1].vlines(x=x, ymin=0, ymax=prediction)


def plot_isopattern_and_obs(
    maxquant_result,
    ms1_scans: pd.DataFrame | None = None,
    infer_intensity: Union[pd.Series, np.ndarray, None] = None,
    lower_plot: Literal["infer", "obs"] = "obs",
    scan_idx: Union[int, None] = None,
    precursor_idx: Union[List[int], None] = None,
    precursor_id: Union[List[int], None] = None,
    mzrange: Union[None, list] = None,
    log_intensity: bool = False,
    save_dir=None,
):
    # preprocess data
    # TODO: return within scan Pearson and Jaccard distance --> redundant
    # TODO: return atom composition --> IsoSpecPy incompatibility
    # TODO: clean code and if possible factorize part of it! --> scan by scan notebook

    # Find precursor index if only precursor id is provided:
    if precursor_id is not None:
        precursor_idx = maxquant_result.loc[
            maxquant_result["id"].isin(precursor_id)
        ].index

    match lower_plot:
        case "obs":
            # Find an appropriate scan if not provided
            if scan_idx is None:
                if precursor_idx is None:
                    raise ValueError("Please provide a precursor index.")
                rt = np.max(
                    maxquant_result.loc[precursor_idx, "Retention time"].values
                )  # take the later RT of the precursors
                scan_idx = np.abs(ms1_scans["starttime"] - rt).argmin()
                scan_time = ms1_scans.loc[scan_idx, "starttime"]
                Logger.info(
                    "Precursors %s retention time %s, \n show later RT %s with"
                    " corresponding scan index %s         with scan time %s",
                    precursor_idx,
                    maxquant_result.loc[precursor_idx, "Retention time"].values,
                    rt,
                    scan_idx,
                    scan_time,
                )
            one_scan = ms1_scans.iloc[scan_idx, :]
            one_scan_mz = np.array(one_scan["mzarray"])
            iso_mz = None

            # Find the range of mz in MS1 scan to plot
            if precursor_idx is not None:
                iso_mz = maxquant_result.loc[precursor_idx, "IsoMZ"]
                iso_mz_flatten = np.concatenate(iso_mz.values).ravel()
                iso_mz_range = [
                    np.min(iso_mz_flatten) - 1,
                    np.max(iso_mz_flatten) + 1,
                ]
                one_scan_mz_in_range = one_scan_mz[
                    (one_scan_mz > iso_mz_range[0]) & (one_scan_mz < iso_mz_range[1])
                ]
                one_scan_mz_in_range_idx = np.where(
                    (one_scan_mz > iso_mz_range[0]) & (one_scan_mz < iso_mz_range[1])
                )[0]

            else:
                if not isinstance(mzrange, list):
                    raise TypeError(
                        "mzrange should be a list, or provide an int for precursor"
                        " index."
                    )
                one_scan_mz_in_range = one_scan_mz[
                    (one_scan_mz > mzrange[0]) & (one_scan_mz < mzrange[1])
                ]
                one_scan_mz_in_range_idx = np.where(
                    (one_scan_mz > mzrange[0]) & (one_scan_mz < mzrange[1])
                )[0]

            # Calculating values for visualization
            intensity = np.array(one_scan["intarray"])[one_scan_mz_in_range_idx]
            if log_intensity:  # +1 to avoid divide by zero error
                intensity = np.log10(intensity + 1)
            peak_results = ExtractPeak(x=one_scan_mz_in_range, y=intensity)
            peaks_idx = peak_results["apex_mzidx"]
            print("Peak results:")
            print(peak_results)
        case "infer":
            if infer_intensity is None:
                raise ValueError("please provide infer_intensity.")
            intensity = infer_intensity.values
            if log_intensity:
                intensity = np.log10(infer_intensity.values + 1)
            if precursor_idx is not None:
                iso_mz = maxquant_result.loc[precursor_idx, "IsoMZ"]
                iso_mz_flatten = np.concatenate(iso_mz.values).ravel()
                iso_mz_range = [np.min(iso_mz_flatten) - 1, np.max(iso_mz_flatten) + 1]
                infer_in_range = intensity[
                    (infer_intensity.index > iso_mz_range[0])
                    & (infer_intensity.index < iso_mz_range[1])
                ]
                infer_in_range_idx = infer_intensity.index[
                    (infer_intensity.index > iso_mz_range[0])
                    & (infer_intensity.index < iso_mz_range[1])
                ]

    # Plot
    fig, axs = plt.subplots(2, 1, sharex=True, sharey=False)
    fig.subplots_adjust(hspace=0)
    if precursor_idx is not None:
        colormap = colormaps["bwr"]  # nipy_spectral, Set1,Paired
        colors = [colormap(i) for i in np.linspace(0, 1, len(precursor_idx))]
        for i, precursor in enumerate(precursor_idx):
            axs[0].vlines(
                x=maxquant_result.loc[precursor, "IsoMZ"],
                ymin=0,
                ymax=maxquant_result.loc[precursor, "IsoAbundance"],
                label=precursor,
                color=colors[i],
            )
            print(
                "Isotope Pattern:",
                precursor,
                maxquant_result.loc[precursor, "IsoMZ"],
                maxquant_result.loc[precursor, "IsoAbundance"],
            )
        axs[0].set_title("Up: Isotope Pattern, Down: MS1 Scan " + str(scan_idx))
        if mzrange is not None:
            axs[0].set_xlim(mzrange)
    match lower_plot:
        case "obs":
            axs[1].vlines(
                x=one_scan_mz_in_range, ymin=-intensity, ymax=0, label="MS peaks"
            )
            axs[1].hlines(
                y=-peak_results["peak_height"],
                xmin=peak_results["start_mz"],
                xmax=peak_results["end_mz"],
                linewidth=2,
                color="black",
            )
            axs[1].vlines(
                x=one_scan_mz_in_range[peaks_idx],
                ymin=-intensity[peaks_idx],
                ymax=0,
                color="orange",
                label="inferred apex",
            )
            axs[1].plot(
                one_scan_mz_in_range[peaks_idx],
                -intensity[peaks_idx],
                "x",
                color="orange",
                label="inferred apex",
            )
            if mzrange is not None:
                axs[1].set_xlim(mzrange)
        case "infer":
            # Logger.debug(
            #     "infer m/z and intensities: %s, %s", infer_in_range_idx, infer_in_range
            # )
            axs[1].vlines(  # x = infer_intensity.index,
                x=infer_in_range_idx,
                ymin=-infer_in_range,
                ymax=0,
                label="inferred intensity",
            )
            axs[1].set_xlim(iso_mz_range)
    fig.legend(loc="center right", bbox_to_anchor=(1.15, 0.5))

    # save result
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        figname = (
            "SpecAndIsoPatterns_scan"
            + str(scan_idx)
            + "_precursor"
            + str(precursor_idx)
            + ".png"
        )
        plt.savefig(fname=os.path.join(save_dir, figname), dpi=300)
        plt.close()
    else:
        plt.show()

    return None


def plot_activation(
    maxquant_ref_row: pd.Series,
    maxquant_exp_df: pd.DataFrame,
    precursor_activations: list,
    ms1scan_no_array: pd.DataFrame,
    activation_labels: list,
    precursor_cos_dists: list | None = None,
    cos_dist_labels: list | None = None,
    log_intensity: bool = False,
    x_ticks: Literal["time", "scan index"] = "time",
    save_dir=None,
    save_format: str = "png",
    ms1scan_time_col="starttime",
):
    """Plot the activation profile of a precursor"""
    # get the RT search range
    rt_search_range = [
        maxquant_ref_row["RT_search_left"].values[0],
        maxquant_ref_row["RT_search_right"].values[0],
    ]
    rt_search_center = maxquant_ref_row["RT_search_center"].values[0]
    Logger.debug("RT search range: %s", rt_search_range)

    # find the corresponding experiment match
    mod_seq = maxquant_ref_row["Modified sequence"].values[0]
    charge = maxquant_ref_row["Charge"].values[0]
    Logger.debug("mod_seq: %s, charge: %s", mod_seq, charge)
    exp_match = maxquant_exp_df.loc[
        (maxquant_exp_df["Modified sequence"] == mod_seq)
        & (maxquant_exp_df["Charge"] == charge),
        :,
    ]
    # find the smallest and largest RT in the experiment and RT search range and filter activation
    rt_min = min(
        [
            exp_match["Calibrated retention time start"].min(),
            maxquant_ref_row["RT_search_left"].min(),
        ]
    )
    rt_max = max(
        [
            exp_match["Calibrated retention time finish"].max(),
            maxquant_ref_row["RT_search_right"].max(),
        ]
    )
    Logger.debug(
        "RT exp range %s",
        [
            exp_match["Calibrated retention time start"].values,
            exp_match["Calibrated retention time finish"].values,
        ],
    )
    scan_index = ms1scan_no_array.loc[
        (ms1scan_no_array[ms1scan_time_col] >= rt_min)
        & (ms1scan_no_array[ms1scan_time_col] <= rt_max),
        ms1scan_time_col,
    ]
    # Logger.debug("ScanIdx: %s", scan_index)

    time_profiles = pd.DataFrame(dict(zip(activation_labels, precursor_activations)))
    time_profiles = time_profiles.set_index(ms1scan_no_array[ms1scan_time_col])
    activation_in_range = time_profiles.loc[scan_index, :]

    if log_intensity:
        activation_in_range = np.log10(activation_in_range + 1)

    fig, ax1 = plt.subplots()
    sns.scatterplot(data=activation_in_range, legend=False, ax=ax1)  # type: ignore
    sns.lineplot(
        data=activation_in_range,
        legend=False,
        ax=ax1,
    )
    y_min, y_max = ax1.get_ylim()

    # experimental RT range
    if exp_match.shape[0] == 0:
        Logger.warning("No experiment match is found. RT elution will not be plotted.")
    else:
        for _, row in exp_match.iterrows():
            ax1.fill_between(
                [
                    row["Calibrated retention time start"],
                    row["Calibrated retention time finish"],
                ],
                [y_min, y_min],
                [y_max, y_max],
                color="grey",
                alpha=0.5,
                label="Elution Range",
            )

    # reference RT
    ax1.fill_between(
        rt_search_range,
        [y_min, y_min],
        [y_max, y_max],
        color="pink",
        alpha=0.3,
        label="SBS Search Range",
    )
    ax1.axvline(
        x=rt_search_center,
        linewidth=2,
        color="black",
        label="SBS_RT_ref",
    )

    ax1.set_xlabel("Time (Minutes)")
    ax1.set_ylabel("Activation")

    # Get the legend handles and labels for both axes
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles = handles1
    labels = labels1

    if precursor_cos_dists is not None:
        cos_dist = pd.DataFrame(dict(zip(cos_dist_labels, precursor_cos_dists)))
        cos_dist = cos_dist.set_index(ms1scan_no_array["starttime"])
        cos_dist_in_range = cos_dist.iloc[scan_index, :]
        cos_dist = cos_dist.replace(0, np.nan)
        ax2 = ax1.twinx()
        sns.lineplot(
            data=cos_dist_in_range,
            ax=ax2,
            legend=False,
            palette=["pink", "yellow", "brown"],
            label="cos dist",
        )
        ax2.set_ylabel("cosine distance")
        ax2.set_ylim(0, 1)
        handles2, labels2 = ax2.get_legend_handles_labels()
        # Merge the legend handles and labels
        handles = handles1 + handles2
        labels = labels1 + labels2
    plt.legend(handles, labels, bbox_to_anchor=(1.5, 1), loc="upper right")

    title = mod_seq + ", " + charge.astype(str)
    ax1.set_title(title)

    if save_dir is not None:
        save_plot(
            save_dir=save_dir,
            fig_type_name="Activation",
            fig_spec_name=title.replace("|", "_"),
            fig_format=save_format,
            bbox_inches="tight",
        )

    activation_in_range_df = activation_in_range
    activation_in_range_df["Scan index"] = scan_index
    activation_in_range_df = activation_in_range_df.reset_index()
    if x_ticks == "scan index":
        x, loc = plt.xticks()
        print(loc)
        plt.xticks(
            x,
            activation_in_range_df.loc[
                activation_in_range_df["starttime"].isin(x), "Scan index"
            ].values,
        )
        plt.xlabel("Scan index")
    return activation_in_range_df


def plot_im_mz(
    sliced_frame: pd.DataFrame,
    labels: np.ndarray = None,
    label_name: str = "Label",
    mark_mz: List[float] = None,
    mark_im: List[float] = None,
    distr_mz: List[float] = None,
    title: str = "MS1 Frame",
    save_dir: str | None = None,
    fig_spec_name: str = "",
):
    fig, ax = plt.subplots()
    if labels is None:
        labels = sliced_frame["intensity_values"]
        label_name = "Intensity"
    sliced_frame["mz_values"] = np.round(sliced_frame["mz_values"], 2)
    scatter = ax.scatter(
        sliced_frame["mobility_values"],
        sliced_frame["mz_values"],
        c=labels,
        cmap="cividis_r",
        s=0.1,
    )
    if mark_im is not None:
        for im in mark_im:
            ax.axvline(im, color="r", alpha=0.5)
    if mark_mz is not None:
        for mz, abu in zip(mark_mz, distr_mz):
            ax.axhline(mz, color="r", linewidth=abu * 10, alpha=0.5)
    cbar = plt.colorbar(scatter)
    cbar.set_label(label_name, rotation=270, labelpad=15)
    ax.set_ylabel("m/z")
    ax.set_xlabel("1/K0")
    if title == "MS1 Frame":
        title = title + " " + str(sliced_frame["frame_indices"].unique())
    plt.title(title)
    save_plot(
        save_dir, fig_type_name="precursor_IM_MZ_raw_int", fig_spec_name=fig_spec_name
    )


def plot_im_mz_int(sliced_frame: pd.DataFrame):
    fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
    ax.stem(
        sliced_frame["mobility_values"].values,
        sliced_frame["mz_values"].values,
        sliced_frame["intensity_values"].values,
    )
    # ax.bar3d(
    #     sliced_frame["mobility_values"].index,
    #     sliced_frame["mz_values"].index,
    #     sliced_frame["intensity_values"].index,
    #     sliced_frame["mobility_values"].values,
    #     sliced_frame["mz_values"].values,
    #     sliced_frame["intensity_values"].values,
    # )
    ax.set_ylabel("m/z")
    ax.set_xlabel("mobility")
    ax.set_zlabel("intensity")


def plot_im_or_mz_int_reduced(
    sliced_frame: pd.DataFrame,
    group_by: str,
    labels: np.ndarray = None,
    label_name="Label",
):
    assert group_by in [
        "mz_values",
        "mobility_values",
    ], "Please provide either mz_values or mobility_values for group_by"
    sliced_frame["mz_vales"] = np.round(sliced_frame["mz_values"], 2)
    sliced_frame_grouped = sliced_frame.groupby(group_by).sum().reset_index()
    fig, ax = plt.subplots()
    # if labels is not None:
    #     scatter = plt.plot(
    #         sliced_frame_grouped[group_by],
    #         sliced_frame_grouped["intensity_values"],
    #         #c=labels,
    #         #s=0.5,
    #     )
    #     cbar = plt.colorbar(scatter)
    #     cbar.set_label(label_name, rotation=270)
    # else:
    #     scatter = plt.plot(
    #         sliced_frame_grouped[group_by],
    #         sliced_frame_grouped["intensity_values"],
    #         #s=10,
    #     )

    plt.vlines(
        x=sliced_frame_grouped[group_by],
        ymin=0,
        ymax=sliced_frame_grouped["intensity_values"],
    )
    ax.set_ylabel("intensity")
    ax.set_xlabel(group_by)
    plt.title("Ion mobility spectrum frame " + str(sliced_frame["frame_indices"].min()))
    # if mz_value is not None:
    #     sliced_frame_filtered = sliced_frame.loc[sliced_frame["mz_values"] == mz_value]
    #     x_value = "mobility_values"
    #     title = (
    #         "Ion mobility spectrum frame "
    #         + str(sliced_frame["frame_indices"].min())
    #         + "m/z "
    #         + str(mz_value)
    #     )
    # else:
    #     x_value = "mz_values"
    #     sliced_frame_filtered = sliced_frame.loc[
    #         sliced_frame["mobility_values"] == im_value
    #     ]
    #     title = (
    #         "Ion mobility spectrum frame "
    #         + str(sliced_frame["frame_indices"].min())
    #         + " im "
    #         + str(im_value)
    #     )
    # fig, ax = plt.subplots()
    # if labels is not None:
    #     scatter = ax.scatter(
    #         sliced_frame_filtered[x_value],
    #         sliced_frame_filtered["intensity_values"],
    #         c=labels,
    #         s=0.5,
    #     )
    #     cbar = plt.colorbar(scatter)
    #     cbar.set_label(label_name, rotation=270)
    # else:
    #     scatter = ax.scatter(
    #         sliced_frame_filtered[x_value],
    #         sliced_frame_filtered["intensity_values"],
    #         s=10,
    #     )
    # ax.set_ylabel("intensity")
    # ax.set_xlabel(x_value)
    # plt.title(title)

    # plt.show()


def plot_im_or_mz_int(
    sliced_frame: pd.DataFrame,
    mz_value: float = None,
    im_value: float = None,
    labels: np.ndarray = None,
    label_name="Label",
):
    assert (
        mz_value is not None or im_value is not None
    ), "Please provide either mz_value or im_value"
    if mz_value is not None:
        sliced_frame_filtered = sliced_frame.loc[sliced_frame["mz_values"] == mz_value]
        x_value = "mobility_values"
        title = (
            "Ion mobility spectrum frame "
            + str(sliced_frame["frame_indices"].min())
            + "m/z "
            + str(mz_value)
        )
    else:
        x_value = "mz_values"
        sliced_frame_filtered = sliced_frame.loc[
            sliced_frame["mobility_values"] == im_value
        ]
        title = (
            "Ion mobility spectrum frame "
            + str(sliced_frame["frame_indices"].min())
            + " im "
            + str(im_value)
        )
    fig, ax = plt.subplots()
    if labels is not None:
        scatter = ax.scatter(
            sliced_frame_filtered[x_value],
            sliced_frame_filtered["intensity_values"],
            c=labels,
            s=0.5,
        )
        cbar = plt.colorbar(scatter)
        cbar.set_label(label_name, rotation=270)
    else:
        scatter = ax.scatter(
            sliced_frame_filtered[x_value],
            sliced_frame_filtered["intensity_values"],
            s=10,
        )
    ax.set_ylabel("intensity")
    ax.set_xlabel(x_value)
    plt.title(title)

    plt.show()


def plot_pept_im_rt_heatmap(
    pept_mz_rank: int,
    act_3d: SparseArray,
    maxquant_result_dict: pd.DataFrame,
    maxquant_result_exp: pd.DataFrame | None,
    mobility_values_df: pd.DataFrame,
    ms1scans: pd.DataFrame,
    plot_range: Literal["nonzero", "custom"] = "nonzero",
    rt_range: tuple | None = None,
    im_range: tuple | None = None,
    pept_batch_idx: int = 0,
    pept_batch_size: int = 50000,
    log_intensity: bool = False,
):
    """plot the heatmap of peptide ion mobility and retention time"""
    batch_corrected_pept_mz_rank = pept_mz_rank - pept_batch_idx * pept_batch_size
    (modseq, charge) = maxquant_result_dict.loc[
        maxquant_result_dict["mz_rank"] == pept_mz_rank, ["Modified sequence", "Charge"]
    ].values[0]
    Logger.info(
        "Dictionary entry %s",
        maxquant_result_dict.loc[
            maxquant_result_dict["mz_rank"] == pept_mz_rank,
            [
                "Modified sequence",
                "Charge",
                "1/K0",
                "RT_search_left",
                "RT_search_right",
                "RT_search_center",
                "Retention time",
                "Number of data points",
                "Number of scans",
            ],
        ],
    )
    if maxquant_result_exp is not None:
        maxquant_result_exp_row = maxquant_result_exp.loc[
            (maxquant_result_exp["Modified sequence"] == modseq)
            & (maxquant_result_exp["Charge"] == charge)
        ]
        if maxquant_result_exp_row.shape[0] == 0:
            Logger.warning("No experiment match is found.")
            bbox = None
        else:
            bbox = get_bbox_from_mq_exp(maxquant_result_exp_row)
            Logger.info(
                "Experiment result: %s, bounding box available: %s",
                maxquant_result_exp_row[
                    [
                        "Modified sequence",
                        "Charge",
                        "Calibrated retention time start",
                        "Calibrated retention time finish",
                        "1/K0",
                        "1/K0 length",
                        "Number of data points",
                        "Retention length",
                    ]
                ],
                bbox,
            )

    else:
        bbox = None

    (
        rt_idx_range,
        im_idx_range,
        reference_entry,
        reference_entry_idx,
    ) = get_ref_rt_im_range(
        pept_mz_rank=pept_mz_rank,
        maxquant_result_dict=maxquant_result_dict,
        mobility_values_df=mobility_values_df,
        ms1scans=ms1scans,
        ref_rt_range=rt_range,
        ref_im_range=im_range,
    )
    Logger.info("Reference entry: %s, im_idx_range: %s", reference_entry, im_idx_range)
    slice_pept_act_sparse, rt_idx_range, im_idx_range = slice_pept_act(
        pept_act_sparse=act_3d[:, :, batch_corrected_pept_mz_rank],  # type: ignore
        plot_range=plot_range,
        rt_idx_range=rt_idx_range,
        im_idx_range=im_idx_range,
    )
    data_3d_heatmap = prepare_slice_pept_act_df(
        slice_pept_act_sparse, rt_idx_range, im_idx_range, mobility_values_df, ms1scans
    )
    Logger.info("Sum of data 3D heatmap: %s", data_3d_heatmap.sum().sum())
    if log_intensity:
        data_3d_heatmap = np.log10(data_3d_heatmap + 1)
    Logger.info("Data 3D heatmap shape: %s", data_3d_heatmap.shape)
    # sns.color_palette("icefire", as_cmap=True)
    ax = sns.heatmap(data_3d_heatmap, cmap="icefire")
    if bbox is not None:
        bbox_rt_min_idx = max(
            np.searchsorted(data_3d_heatmap.index, bbox[0], side="left") - 1, 0
        )
        bbox_rt_max_idx = np.searchsorted(data_3d_heatmap.index, bbox[1], side="right")
        bbox_im_min_idx = max(
            np.searchsorted(data_3d_heatmap.columns, bbox[3], side="left") - 1, 0
        )
        bbox_im_max_idx = np.searchsorted(
            data_3d_heatmap.columns, bbox[4], side="right"
        )
        Logger.info(
            "Bounding box: %s",
            [
                bbox_rt_min_idx,
                bbox_rt_max_idx,
                bbox_im_min_idx,
                bbox_im_max_idx,
            ],
        )
        ax.add_patch(
            patches.Rectangle(
                xy=(bbox_im_min_idx, bbox_rt_min_idx),
                width=bbox_im_max_idx - bbox_im_min_idx,
                height=bbox_rt_max_idx - bbox_rt_min_idx,
                edgecolor="red",
                fill=False,
                lw=3,
            )
        )
        plt.show()
    else:
        ref_rt_idx = np.searchsorted(
            data_3d_heatmap.index, reference_entry[0], side="left"
        )
        ref_im_idx = np.searchsorted(
            data_3d_heatmap.columns, reference_entry[1], side="left"
        )
        ax.scatter(
            ref_im_idx,
            ref_rt_idx,
            marker="o",
            color="red",
            s=50,
            label="Reference Point",
        )
        ax.legend()
        plt.show()


def plot_pept_act_heatmap(
    pept_act,
    max_rt_index,
    max_im_index,
    fwhm_rt_left_index=None,
    fwhm_rt_right_index=None,
    rt_left_index=None,
    rt_right_index=None,
    rt_offset=75,
    im_offset=30,
    title=None,
    log_scale=False,
):
    heatmap_data = pept_act.todense() + 1
    if log_scale:
        heatmap_data = np.log10(heatmap_data)
    ax = sns.heatmap(heatmap_data, cmap="viridis")
    ax.set_xlabel("Ion mobility index")
    ax.set_ylabel("MS1 frame index")
    ax.set_ylim(
        max(0, max_rt_index - rt_offset),
        min(max_rt_index + rt_offset, pept_act.shape[0] - 1),
    )
    ax.set_xlim(
        max(0, max_im_index - im_offset),
        min(max_im_index + im_offset, pept_act.shape[1]),
    )
    plt.vlines(
        x=max_im_index,
        ymin=max(0, max_rt_index - rt_offset),
        ymax=min(max_rt_index + rt_offset, pept_act.shape[0] - 1),
        color="red",
        linestyle="-",
    )
    plt.hlines(
        y=max_rt_index,
        xmin=max(0, max_im_index - im_offset),
        xmax=min(max_im_index + im_offset, pept_act.shape[1]),
        color="red",
        linestyle="-",
    )
    if fwhm_rt_left_index is not None and fwhm_rt_right_index is not None:
        plt.hlines(
            y=[fwhm_rt_left_index, fwhm_rt_right_index],
            xmin=max(0, max_im_index - im_offset),
            xmax=min(max_im_index + im_offset, pept_act.shape[1]),
            color="white",
            linestyle="--",
        )
    if rt_left_index is not None and rt_right_index is not None:
        plt.hlines(
            y=[rt_left_index, rt_right_index],
            xmin=max(0, max_im_index - im_offset),
            xmax=min(max_im_index + im_offset, pept_act.shape[1]),
            color="red",
            linestyle="--",
        )
    if title is not None:
        plt.title(title)


def plot_pept_act_align_channels(
    act_raw,
    act_log,
    act_transformed_log,
    grad_mag,
    grad_mag_smoothed,
    rt_exp_start,
    rt_exp_end,
    rt_exp_center,
    im_exp_start,
    im_exp_center,
    im_exp_end,
    coordinates=None,
    labels=None,
    peak_properties=None,
    sub_fig3=None,
    figsize=(15, 5),
    annotate=True,
    title: Optional[str] = None,
    save_dir: Optional[str] = None,
    fig_spec_name: str = "",
):
    """
    Plot peptide activity, gradient magnitude and watershed labels with annotations.

    Parameters:
    - pept_act_log: 2D array of log-transformed intensities
    - grad_mag: 2D array of gradient magnitudes
    - coordinates: Nx2 array of detected peak coordinates (row, col)
    - labels: 2D int array of watershed labels
    - peak_properties: DataFrame containing detected peak properties (must include 'label',
      'centroid-0', 'centroid-1' and 'row_smoothness' to annotate)
    - rt_exp_start, rt_exp_end: horizontal lines (RT window) in pixel/index units
    - figsize: figure size tuple
    - annotate: whether to annotate centroids with label and smoothness
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    # left: log intensity with detected peak locations
    ax = axes[0]
    im0 = ax.imshow(act_raw, cmap="viridis", aspect="auto", interpolation="nearest")
    ax.set_title("Raw")
    fig.colorbar(im0, ax=ax)

    ax = axes[1]
    im1 = ax.imshow(act_log, cmap="viridis", aspect="auto", interpolation="nearest")
    ax.set_title("Raw, Log")
    fig.colorbar(im1, ax=ax)

    ax = axes[2]
    im2 = ax.imshow(
        act_transformed_log, cmap="viridis", aspect="auto", interpolation="nearest"
    )
    if coordinates is not None and len(coordinates) > 0:
        ax.scatter(coordinates[:, 1], coordinates[:, 0], color="r", s=20)
    ax.set_title("Transformed(smoothed, cleaned up), Log + Local Max")
    fig.colorbar(im2, ax=ax)

    if sub_fig3 is None:
        # right: watershed labels (only plot labels present in peak_properties)
        assert isinstance(peak_properties, pd.DataFrame)
        filtered_labels = peak_properties["label"].values
        labels_mask = np.isin(labels, filtered_labels)  # type: ignore
        sub_fig3 = labels * labels_mask
    ax = axes[3]
    im5 = ax.imshow(
        sub_fig3,
        cmap=plt.cm.nipy_spectral,
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_title("Watershed Segmentation")
    fig.colorbar(im5, ax=ax)

    # annotate each detected coordinate with its watershed label and row_smoothness
    if annotate and not peak_properties.empty:
        # use centroid coords and row_smoothness from peak_properties
        cols = ["centroid-0", "centroid-1", "intensity_sum"]
        # guard if some columns missing
        if all(c in peak_properties.columns for c in cols):
            for r, c, s in peak_properties[cols].values:
                rr, cc = int(round(r)), int(round(c))
                # ensure indices are inside bounds
                if 0 <= rr < labels.shape[0] and 0 <= cc < labels.shape[1]:
                    lbl = int(labels[rr, cc])
                else:
                    lbl = 0
                ax.text(
                    c,
                    r,
                    str(lbl),
                    color="white",
                    fontsize=12,
                    ha="center",
                    va="center",
                    bbox=dict(facecolor="black", alpha=0.6, pad=0.2),
                )
                ax.text(
                    c,
                    r + 10,
                    f"{np.log10(s+1):.2f}",
                    color="yellow",
                    fontsize=12,
                    ha="center",
                    va="center",
                    bbox=dict(facecolor="black", alpha=0.6, pad=0.2),
                )
        # middle: gradient magnitude
    ax = axes[4]
    im3 = ax.imshow(grad_mag, cmap="gray", aspect="auto", interpolation="nearest")
    ax.set_title("Gradient Magnitude")
    fig.colorbar(im3, ax=ax)

    # right: smoothed gradient magnitude
    ax = axes[5]
    im4 = ax.imshow(
        grad_mag_smoothed, cmap="gray", aspect="auto", interpolation="nearest"
    )
    ax.set_title("Smoothed Gradient Magnitude")
    fig.colorbar(im4, ax=ax)

    for ax in axes:
        xlim = ax.get_xlim()
        ax.hlines(
            [rt_exp_start, rt_exp_end],
            xmin=xlim[0],
            xmax=xlim[1],
            colors="white",
            linestyles="dashed",
        )
        ax.vlines(
            [im_exp_start, im_exp_end],
            ymin=ax.get_ylim()[0],
            ymax=ax.get_ylim()[1],
            colors="white",
            linestyles="dashed",
        )
        ax.hlines(
            [rt_exp_center],
            xmin=xlim[0],
            xmax=xlim[1],
            colors="white",
            linestyles="solid",
        )
        ax.vlines(
            [im_exp_center],
            ymin=ax.get_ylim()[0],
            ymax=ax.get_ylim()[1],
            colors="white",
            linestyles="solid",
        )
    if title is not None:
        fig.suptitle(title)
    plt.tight_layout()
    if save_dir is not None:
        save_plot(
            save_dir,
            fig_type_name="pept_act_align_channels",
            fig_spec_name=fig_spec_name,
            fig_format=["png"],
        )
    else:
        plt.show()


def plot_heat(
    ax,
    heat_df,
    title,
    filtered_precursor,
    rows,
    match_col="mz_rank",
    mark_col="mobility_values_index_center_exp",
):
    sns.heatmap(heat_df, cmap="viridis", ax=ax)
    ax.set_title(title)
    ax.tick_params(axis="y", labelrotation=0)
    for lbl in ax.get_yticklabels():
        lbl.set_ha("right")
    cols = heat_df.columns.values
    for y_pos, mz_rank in enumerate(rows):
        sel = filtered_precursor.loc[filtered_precursor[match_col] == mz_rank, mark_col]

        if sel.empty:
            continue
        mobility_idx = sel.values[0]
        # print(y_pos, mz_rank, mobility_idx)
        if pd.isna(mobility_idx):
            continue
        # find nearest mobility column in the heatmap
        col_pos = np.argmin(np.abs(cols - mobility_idx))
        ax.scatter(col_pos + 0.5, y_pos + 0.5, marker="x", color="red", s=20)


def plot_frame_act_and_mono_mz(
    frame_data,
    filtered_precursor,
    act_3d_dict,
    ppm_tol=20,
    log_int=False,
    log_act=False,
    raw_file="",
):
    """
    Plot side-by-side heatmaps:
      1. Experimental mobility–intensity map
      2. Activation (3D) map
    with red markers indicating precursor positions.

    Parameters
    ----------
    frame_data : pd.DataFrame
        Columns: "mz_values", "intensity_values", "mobility_values"
    filtered_precursor : pd.DataFrame
        Must contain columns: "IsoMZ", "1/K0", "mz_rank", "mobility_values_index_center_exp"
    act_3d_dict : dict
        Keys: "coord_pept_indices", "coord_im_indices", "data"
    ppm_tol : float, optional
        PPM tolerance for selecting mz window around each precursor center
    """

    # --- Extract basic arrays ---
    mz_vals = frame_data["mz_values"].values
    intensities = frame_data["intensity_values"].values
    scans = frame_data["mobility_values"].values

    # --- Precursor centers and tolerances ---
    centers = np.array([mz[0] for mz in filtered_precursor["IsoMZ"].values])
    tolerances = centers * ppm_tol / 1e6

    # --- Aggregate intensity per mobility for each precursor ---
    records = []
    for c, tol in zip(centers, tolerances):
        mask = (mz_vals >= (c - tol)) & (mz_vals <= (c + tol))
        if not np.any(mask):
            continue
        df_sub = (
            pd.DataFrame(
                {"mobility_values": scans[mask], "intensity_values": intensities[mask]}
            )
            .groupby("mobility_values", as_index=False)["intensity_values"]
            .sum()
        )
        df_sub["center_mz"] = c
        records.append(df_sub)

    result_df = pd.concat(records, ignore_index=True)
    heat_exp = result_df.pivot_table(
        index="center_mz",
        columns="mobility_values",
        values="intensity_values",
        aggfunc="sum",
    )
    left_title = f"Experimental MonoM/Z (ppm={ppm_tol}) - 1/K0 - Intensity"
    if log_int:
        heat_exp = np.log1p(heat_exp)
        left_title += " (log-scaled)"
    # --- Activation data ---
    act_3d_df = pd.DataFrame(act_3d_dict)
    act_3d_df = act_3d_df[
        act_3d_df["coord_pept_indices"].isin(filtered_precursor["mz_rank"].values)
    ]
    right_title = "Activation Map"
    if log_act:
        act_3d_df["data"] = np.log1p(act_3d_df["data"])
        right_title += " (log-scaled)"
    heat_act = act_3d_df.pivot_table(
        index="coord_pept_indices",
        columns="coord_im_indices",
        values="data",
        aggfunc="sum",
    ).fillna(0)

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    filtered_precursor["mono_mz"] = np.array(
        [mz[0] for mz in filtered_precursor["IsoMZ"].values]
    )
    # Left: experimental
    plot_heat(
        axes[0],
        heat_exp,
        left_title,
        filtered_precursor,
        heat_exp.index.values,
        "mono_mz",
        f"{raw_file}_1K0",
    )
    # Right: activation
    plot_heat(
        axes[1],
        heat_act,
        right_title,
        filtered_precursor,
        heat_act.index.values,
        "mz_rank",
        f"{raw_file}_mobility_values_index_exp",
    )

    plt.tight_layout()
    plt.show()


def plot_image_comparison(
    fig_a,
    fig_b,
    rt_exp_start=None,
    rt_exp_end=None,
    im_exp_start=None,
    im_exp_end=None,
    fig_dir=None,
    fig_name=None,
):
    """
    Compare two figures side by side with highlighted experimental region.
    Parameters:
    fig_a: 2D array-like
        First figure to display.
    fig_b: 2D array-like
        Second figure to display.
    rt_exp_start: int, optional
        Start index of the experimental region in the RT dimension.
    rt_exp_end: int, optional
        End index of the experimental region in the RT dimension.
    im_exp_start: int, optional
        Start index of the experimental region in the IM dimension.
    im_exp_end: int, optional
        End index of the experimental region in the IM dimension.
    fig_dir: str, optional
        Directory to save the figure.
    fig_name: str, optional
        Name of the figure file.

    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    im0 = axes[0].imshow(fig_a, aspect="auto", cmap="viridis")
    axes[0].set_title(fig_name)
    fig.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(fig_b, aspect="auto", cmap="viridis")
    axes[1].set_title(fig_name)
    fig.colorbar(im1, ax=axes[1])
    if None not in (im_exp_start, rt_exp_start, im_exp_end, rt_exp_end):
        axes[0].add_patch(
            Rectangle(
                (im_exp_start, rt_exp_start),
                im_exp_end - im_exp_start,
                rt_exp_end - rt_exp_start,
                fill=False,
                edgecolor="red",
                linewidth=2,
            )
        )
        axes[1].add_patch(
            Rectangle(
                (im_exp_start, rt_exp_start),
                im_exp_end - im_exp_start,
                rt_exp_end - rt_exp_start,
                fill=False,
                edgecolor="red",
                linewidth=2,
            )
        )

    plt.tight_layout()
    if fig_dir and fig_name:
        os.makedirs(fig_dir, exist_ok=True)
        plt.savefig(os.path.join(fig_dir, fig_name), dpi=300)
        plt.close()
