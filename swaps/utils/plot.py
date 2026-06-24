import logging
import os
from typing import List, Set, Union, Literal, Optional

import matplotlib.pyplot as plt
from matplotlib import colormaps, patches  # type: ignore
from matplotlib.figure import Figure
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


def calc_quant_corr(pp_reference, pp_match_target, quant_dir):
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(quant_dir, exist_ok=True)
    if "source_type" in pp_match_target.columns:
        pp_quant_only = pp_match_target.loc[
            pp_match_target["source_type"] == "Quant_Only"
        ].copy()
        pp_match_target_only = pp_match_target.loc[
            pp_match_target["source_type"] != "Quant_Only"
        ].copy()
    else:
        pp_quant_only = pd.DataFrame(columns=pp_match_target.columns)
        pp_match_target_only = pp_match_target
    pp_quant_only_pivoted = pp_quant_only.pivot_table(
        index="mz_rank",
        columns="Run_name",
        values="intensity_sum",
        aggfunc="max",
    ).reset_index()
    pp_reference_pivoted = pp_reference.pivot_table(
        index="mz_rank",
        columns="Run_name",
        values="intensity_sum",
        aggfunc="max",
    ).reset_index()
    # Log-transform numeric columns and compute pairwise Pearson correlations (pairwise complete cases)
    pp_match_target_pivoted = pp_match_target_only.pivot_table(
        index="mz_rank",
        columns="Run_name",
        values="intensity_sum",
        aggfunc="max",
    ).reset_index()
    # Log-transform numeric columns and compute pairwise Pearson correlations (pairwise complete cases)
    num_cols = pp_match_target_pivoted.select_dtypes(
        include=[np.number]
    ).columns.difference(["mz_rank"])
    pp_log = pp_match_target_pivoted.copy()
    pp_log[num_cols] = np.log2(pp_log[num_cols] + 1)

    corr_matrix = pp_log[num_cols].corr(method="pearson", min_periods=1)
    corr_matrix.to_csv(
        os.path.join(
            quant_dir, "pp_match_target_filtered_log_intensity_correlation_matrix.csv"
        )
    )

    # 1. Concatenate with MultiIndex
    pp_all_pivoted = pd.concat(
        [
            pp_reference_pivoted.set_index("mz_rank"),
            pp_quant_only_pivoted.set_index("mz_rank"),
            pp_match_target_pivoted.set_index("mz_rank"),
        ],
        axis=1,
        keys=["reference", "quant_only", "match_target"],
    )

    # 2. Identify numeric columns (excluding the index 'mz_rank')
    # Since mz_rank is the index now, we just take all columns
    num_cols = pp_all_pivoted.select_dtypes(include=[np.number]).columns

    # 3. Log transformation (using log2(x+1) to handle zeros)
    pp_log = np.log2(pp_all_pivoted[num_cols] + 1)

    # 4. Correlation Matrix
    # min_periods=1 ensures you get a value even if there's only one overlapping point
    corr_matrix = pp_log.corr(method="pearson", min_periods=1)
    corr_matrix.to_csv(
        os.path.join(quant_dir, "pp_all_log_intensity_correlation_matrix.csv")
    )
    # Optional: Flatten the MultiIndex for easier viewing if it's too cluttered

    count_matrix = pp_log.notna().astype(int).T.dot(pp_log.notna().astype(int))
    sns.heatmap(corr_matrix)
    ax = plt.gca()
    for i in range(count_matrix.shape[0]):
        for j in range(count_matrix.shape[1]):
            _ = ax.text(
                j + 0.5,
                i + 0.5,
                str(int(count_matrix.iloc[i, j])),
                ha="center",
                va="center",
                fontsize=3,
                color="white",
            )

    plt.xticks(
        ticks=np.arange(len(corr_matrix.columns)),
        labels=[c[0] + c[1][-5:] for c in corr_matrix.columns.values],
        fontsize=5,
        # rotation=45,
    )
    plt.yticks(
        ticks=np.arange(len(corr_matrix.index)),
        labels=[c[0] + c[1][-5:] for c in corr_matrix.index.values],
        fontsize=5,
    )
    plt.savefig(
        os.path.join(
            quant_dir,
            "correlation_matrix_of_log_intensity_with_counts.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_run_summary(
    df,
    mode: str = "match_type",
    # --- match_type-specific ---
    colors=None,
    # --- nonzero_quant-specific ---
    id_cols=None,
    int_col_keyword=None,
    zero_color: str = "#d73027",
    nonzero_color: str = "#1a9850",
    # --- shared ---
    labels=None,
    stack_order=None,
    sort_columns: bool = False,
    label_char_range=None,
    dataset_name: str = "",
    fig_name_suffix: str = "",
    ax=None,
    fig_dir=None,
):
    """
    Plot per-run stacked bar counts in one of two modes.

    Parameters
    ----------
    df
        Input DataFrame.
    mode
        ``"match_type"`` – stacked bars broken down by MS/MS / MBR / unmatched
        category, one bar per run.
        ``"nonzero_quant"`` – stacked bars of quantified vs non-quantified
        entries, one bar per intensity column / run.
    colors
        [match_type] Dict mapping category label → hex color.
    id_cols
        [nonzero_quant] Columns to exclude from plotting.
    int_col_keyword
        [match_type] Keyword identifying the intensity column per run (e.g. ``"Intensity"``).
        Derived by replacing ``"Match Type"`` in each match-type column name. When found,
        rows with zero intensity are counted as ``"Zero Quant"`` and match types are only
        counted for non-zero rows.
        [nonzero_quant] Only columns whose name contains this substring are plotted.
    zero_color
        [nonzero_quant] Color for zero / missing counts.
    nonzero_color
        [nonzero_quant] Color for quantified counts.
    labels
        X-tick labels for each bar; if None, derived from the data.
    stack_order
        Ordered list controlling which categories are shown and in what stack
        order. For ``"match_type"`` these are category names; for
        ``"nonzero_quant"`` this sets the column display order.
    sort_columns
        Sort bars by total count descending (applied before ``stack_order``).
    label_char_range
        ``(m, n)`` – slice each derived x-axis label as ``label[m:n]``.
        Ignored when ``labels`` is provided explicitly.
    dataset_name
        Used in the plot title and saved filename.
    fig_name_suffix
        Appended to the saved filename.
    ax
        Optional existing Axes to draw on.
    fig_dir
        Directory to save PNG + SVG; if None the figure is shown interactively.
    """
    if mode == "match_type":
        if colors is None:
            colors = {
                "MS/MS": "#55A868",
                "MS/MS Quant": "#55A868",
                "MS/MS Ref": "#4C72B0",
                "MBR": "#C44E52",
                "unmatched": "#BBBBBB",
                "Zero Quant": "#DD8452",
            }
        if stack_order is None:
            stack_order = [
                "MS/MS",
                "MS/MS Quant",
                "MS/MS Ref",
                "MBR",
                "unmatched",
                "Zero Quant",
            ]

        match_type_cols = [col for col in df.columns if "Match Type" in col]
        counts_dict = {}
        for col in match_type_cols:
            int_col = (
                col.replace("Match Type", int_col_keyword)
                if int_col_keyword is not None
                else None
            )
            if int_col is not None and int_col in df.columns and int_col != col:
                mask_zero_quant = (df[int_col] == 0) & (df[col] != "unmatched")
                mt_counts = df.loc[~mask_zero_quant, col].value_counts(dropna=True)
                mt_counts["Zero Quant"] = mask_zero_quant.sum()
            else:
                mt_counts = df[col].value_counts(dropna=True)
            counts_dict[col] = mt_counts
        counts = pd.DataFrame(counts_dict).T.fillna(0)
        ordered_cols = [c for c in stack_order if c in counts.columns]
        counts = counts[ordered_cols]

        if sort_columns:
            counts = counts.loc[counts.sum(axis=1).sort_values(ascending=False).index]

        derived_labels = list(counts.index.astype(str))
        if label_char_range is not None and labels is None:
            m, n = label_char_range
            derived_labels = [lbl[m:n] for lbl in derived_labels]
        x_labels = labels if labels is not None else derived_labels

        if ax is None:
            fig, ax = plt.subplots(
                figsize=(max(8, 0.5 * len(counts) + 2), 6),
                constrained_layout=True,
            )
        else:
            fig = ax.figure

        counts.plot(kind="bar", stacked=True, color=colors, ax=ax)
        ax.set_xlabel("Run")
        ax.set_ylabel("Count")
        ax.set_title(
            f"Entry Counts per Match Type{' (' + dataset_name + ')' if dataset_name else ''}"
        )
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha="right")

        for container in ax.containers:
            for bar in container:
                height = bar.get_height()
                if height > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + height / 2,
                        f"{int(height)}",
                        ha="center",
                        va="center",
                        fontsize=8,
                    )

        ax.legend(title="Entry", bbox_to_anchor=(1.02, 1), loc="upper left")
        fig_name = f"match_type_counts{fig_name_suffix}"

    elif mode == "nonzero_quant":
        assert (
            id_cols or int_col_keyword
        ), "Must provide either id_cols or int_col_keyword to identify columns to plot."
        if id_cols is not None:
            cols_to_plot = [
                col
                for col in df.columns
                if col not in id_cols
                and (not int_col_keyword or int_col_keyword in col)
            ]
        else:
            cols_to_plot = [
                col for col in df.columns if int_col_keyword and int_col_keyword in col
            ]
        if not cols_to_plot:
            raise ValueError(
                f"No columns available for plotting after excluding '{id_cols}'."
            )

        if stack_order is not None:
            cols_to_plot = [c for c in stack_order if c in cols_to_plot] + [
                c for c in cols_to_plot if c not in stack_order
            ]

        values = df[cols_to_plot]
        zero_counts = values.eq(0).sum(axis=0)
        na_counts = values.isna().sum(axis=0)
        nonquant_counts = zero_counts + na_counts
        nonzero_counts = values.gt(0).sum(axis=0)

        counts_df = pd.DataFrame(
            {
                "column": cols_to_plot,
                "non_quantified_counts": nonquant_counts.values,
                "quantified_counts": nonzero_counts.values,
            }
        )
        counts_df["total_count"] = (
            counts_df["non_quantified_counts"] + counts_df["quantified_counts"]
        )
        if sort_columns:
            counts_df = counts_df.sort_values("total_count", ascending=False)

        if ax is None:
            fig, ax = plt.subplots(
                figsize=(max(8, 0.5 * len(counts_df) + 2), 5),
                constrained_layout=True,
            )
        else:
            fig = ax.figure

        x = np.arange(len(counts_df))
        ax.bar(
            x,
            counts_df["quantified_counts"],
            color=nonzero_color,
            label="Quantified count",
        )
        ax.bar(
            x,
            counts_df["non_quantified_counts"],
            bottom=counts_df["quantified_counts"],
            color=zero_color,
            label="Non-quantified count",
        )

        for i, (qc, non_qc) in enumerate(
            zip(counts_df["quantified_counts"], counts_df["non_quantified_counts"])
        ):
            ax.text(
                i,
                qc / 2,
                str(int(qc)),
                ha="center",
                va="center",
                color="black",
                fontsize=10,
                fontweight="bold",
                rotation=45,
            )
            ax.text(
                i,
                qc + non_qc / 2,
                str(int(non_qc)),
                ha="center",
                va="center",
                color="black",
                fontsize=10,
                fontweight="bold",
                rotation=45,
            )

        derived_labels = counts_df["column"].astype(str).tolist()
        if label_char_range is not None and labels is None:
            m, n = label_char_range
            derived_labels = [lbl[m:n] for lbl in derived_labels]
        x_labels = labels if labels is not None else derived_labels

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.set_ylabel("Count")
        ax.set_xlabel("Columns")
        ax.set_title(
            f"Quantification per Run{' (' + dataset_name + ')' if dataset_name else ''}"
        )
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        fig_name = f"quantification_count_{dataset_name}{fig_name_suffix}"

    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'match_type' or 'nonzero_quant'.")

    if fig_dir is not None:
        os.makedirs(fig_dir, exist_ok=True)
        fig.savefig(
            os.path.join(fig_dir, f"{fig_name}.png"), dpi=300, bbox_inches="tight"
        )
        fig.savefig(
            os.path.join(fig_dir, f"{fig_name}.svg"), dpi=300, bbox_inches="tight"
        )
        plt.close(fig)
    else:
        plt.show()

    if mode == "nonzero_quant":
        return fig, ax, counts_df.reset_index(drop=True)
    return fig, ax


def plot_match_type_from_combined(
    df,
    int_col_keyword=None,
    colors=None,
    labels=None,
    stack_order=None,
    fig_dir=None,
    fig_name_suffix="",
):
    """Backward-compatible wrapper around plot_run_summary(mode='match_type')."""
    return plot_run_summary(
        df,
        mode="match_type",
        int_col_keyword=int_col_keyword,
        colors=colors,
        labels=labels,
        stack_order=stack_order,
        fig_name_suffix=fig_name_suffix,
        fig_dir=fig_dir,
    )


def plot_intensity_coverage_by_species(
    df: pd.DataFrame,
    *,
    int_col_keyword: str = "Intensity",
    species_col: str = "species",
    threshold: float = 10,
    species_order: Optional[list[str]] = None,
    species_colors: Optional[dict[str, str]] = None,
    missing_color: str = "#D3D3D3",
    label_shorten_fn=None,
    match_filters: Optional[dict[str, list]] = None,
    sort_columns: bool = False,
    dataset_name: str = "",
    fig_name_suffix: str = "",
    ax: Optional[plt.Axes] = None,
    fig_dir: Optional[str] = None,
    separate_by_match_type: bool = False,
    match_type_col_keyword: str = "Match Type",
    match_types: Optional[list[str]] = None,
) -> tuple[Figure, plt.Axes, pd.DataFrame]:
    """Stacked bar plot of intensity coverage broken down by species and missing values.

    Each bar corresponds to one intensity column. Values are counted as present
    when they are non-NaN and >= *threshold*; all others are counted as missing.

    Parameters
    ----------
    df:
        DataFrame containing intensity columns and a species column.
    int_col_keyword:
        Only columns whose name contains this substring are plotted.
    species_col:
        Column that holds the species label for each row.
    threshold:
        Minimum value to count as present (exclusive lower bound; NaN always missing).
    species_order:
        Ordered list of species to show; inferred from the data when None.
    species_colors:
        Dict mapping species label → hex color. Unmapped species get automatic colors.
    missing_color:
        Color for missing / below-threshold bars.
    label_shorten_fn:
        Optional callable ``(col_name: str) -> str`` applied to each column name
        to produce x-tick labels, e.g. ``lambda x: "_".join([x.split("_")[0], x.split("_")[-2]])``.
    match_filters:
        Optional dict ``{match_col_keyword: allowed_values}``.  For each intensity
        column whose name is ``"<stem> <int_col_keyword>"``, the corresponding
        column ``"<stem> <match_col_keyword>"`` is used to pre-filter rows: only
        rows whose value is in *allowed_values* are counted.
        Example: ``{"Match Type": ["unmatched"]}`` restricts every bar to rows
        where the paired Match Type column contains ``"unmatched"``.
    sort_columns:
        Sort bars by total present count descending.
    dataset_name:
        Used in the plot title and saved filename.
    fig_name_suffix:
        Appended to the saved filename.
    ax:
        Optional existing Axes to draw on.
    fig_dir:
        Directory to save PNG + SVG; if None the figure is shown interactively.
    separate_by_match_type:
        When True, each run gets two side-by-side grouped bars — one for "MS/MS"
        identifications (solid) and one for "MBR" match-between-runs (hatched).
        The match type is read from ``"<stem> <match_type_col_keyword>"`` for each
        intensity column. Returns a ``dict[str, pd.DataFrame]`` keyed by match type
        instead of a single DataFrame.
    match_type_col_keyword:
        Column keyword used to locate the Match Type columns (default ``"Match Type"``).
        Only used when ``separate_by_match_type=True``.
    match_types:
        Ordered list of match type values to plot as grouped bars (default
        ``["MS/MS", "MBR", "unmatched"]``). Offsets and hatch patterns are
        assigned automatically based on the list length. Only used when
        ``separate_by_match_type=True``.
    """
    int_cols = [c for c in df.columns if int_col_keyword in c]
    if not int_cols:
        raise ValueError(f"No columns containing '{int_col_keyword}' found.")

    if species_order is None:
        species_order = sorted(df[species_col].dropna().unique().tolist())

    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map = {
        sp: default_colors[i % len(default_colors)]
        for i, sp in enumerate(species_order)
    }
    if species_colors:
        color_map.update(species_colors)

    if separate_by_match_type:
        # Discover match types from data (like value_counts in plot_match_type_comparison)
        if match_types is not None:
            found_types = list(match_types)
        else:
            all_mt: set[str] = set()
            for col in int_cols:
                stem = col[: col.rfind(int_col_keyword)].rstrip()
                mt_col = f"{stem} {match_type_col_keyword}"
                if mt_col in df.columns:
                    all_mt.update(df[mt_col].dropna().unique())
            found_types = sorted(all_mt)

        counts_by_type: dict[str, pd.DataFrame] = {}
        for mt in found_types:
            mt_records = []
            for col in int_cols:
                stem = col[: col.rfind(int_col_keyword)].rstrip()
                row_mask = pd.Series(True, index=df.index)
                if match_filters:
                    for match_kw, allowed_values in match_filters.items():
                        match_col = f"{stem} {match_kw}"
                        if match_col in df.columns:
                            row_mask = row_mask & df[match_col].isin(allowed_values)
                mt_col = f"{stem} {match_type_col_keyword}"
                if mt_col in df.columns:
                    row_mask = row_mask & (df[mt_col] == mt)
                sub = df[row_mask]
                # unmatched rows have no intensity by definition — count all of them
                if mt == "unmatched":
                    mask = pd.Series(True, index=sub.index)
                else:
                    mask = sub[col].notna() & (sub[col] >= threshold)
                row: dict = {"col": col}
                for sp in species_order:
                    row[sp] = int((mask & (sub[species_col] == sp)).sum())
                mt_records.append(row)
            counts_by_type[mt] = pd.DataFrame(mt_records).set_index("col")

        # Add "Zero Quant" only when matched (MS/MS or MBR) entries fall below threshold
        # (mirrors Zero Quant logic in plot_match_type_comparison)
        nq_records = []
        has_zero_quant = False
        for col in int_cols:
            stem = col[: col.rfind(int_col_keyword)].rstrip()
            row_mask = pd.Series(True, index=df.index)
            if match_filters:
                for match_kw, allowed_values in match_filters.items():
                    match_col = f"{stem} {match_kw}"
                    if match_col in df.columns:
                        row_mask = row_mask & df[match_col].isin(allowed_values)
            sub = df[row_mask]
            mt_col = f"{stem} {match_type_col_keyword}"
            not_unmatched = (
                sub[mt_col] != "unmatched"
                if mt_col in sub.columns
                else pd.Series(True, index=sub.index)
            )
            below_threshold = sub[col].isna() | (sub[col] < threshold)
            zero_quant_mask = not_unmatched & below_threshold
            row: dict = {"col": col}
            for sp in species_order:
                row[sp] = int((zero_quant_mask & (sub[species_col] == sp)).sum())
            if not has_zero_quant and any(row[sp] > 0 for sp in species_order):
                has_zero_quant = True
            nq_records.append(row)

        if has_zero_quant:
            counts_by_type["Zero Quant"] = pd.DataFrame(nq_records).set_index("col")
        all_bar_types = found_types + (["Zero Quant"] if has_zero_quant else [])

        counts = counts_by_type[found_types[0]].copy()
        for mt in found_types[1:]:
            counts = counts.add(counts_by_type[mt], fill_value=0)
        counts = counts.astype(int)
        if sort_columns:
            total_present = counts[species_order].sum(axis=1)
            counts = counts.loc[total_present.sort_values(ascending=False).index]
            for bt in all_bar_types:
                counts_by_type[bt] = counts_by_type[bt].loc[counts.index]

        x_labels = (
            [label_shorten_fn(c) for c in counts.index]
            if label_shorten_fn is not None
            else list(counts.index)
        )
        n = len(counts)
        bar_width = 0.7 / len(all_bar_types)
        _default_hatches = [None, "///", "xxx", "...", "---", "|||"]
        offsets = {
            bt: (i - (len(all_bar_types) - 1) / 2) * bar_width
            for i, bt in enumerate(all_bar_types)
        }
        hatch_map = {
            bt: _default_hatches[i % len(_default_hatches)]
            for i, bt in enumerate(all_bar_types)
        }
        x = np.arange(n)

        if ax is None:
            fig, ax = plt.subplots(
                figsize=(max(10, 1.8 * n + 2), 6),
                constrained_layout=True,
            )
        else:
            fig = ax.figure

        for bt in all_bar_types:
            ct = counts_by_type[bt].loc[counts.index]
            bottom = np.zeros(n)
            hatch = hatch_map[bt]
            for sp in species_order:
                vals = ct[sp].values.astype(float)
                ax.bar(
                    x + offsets[bt],
                    vals,
                    bottom=bottom,
                    color=color_map[sp],
                    width=bar_width,
                    hatch=hatch,
                    edgecolor="white",
                )
                bottom += vals

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=90)
        ax.set_xlabel("Run")
        ax.set_ylabel("Ion count")
        ax.set_title(
            f"Intensity coverage by species — {' vs '.join(found_types)} (threshold={threshold})"
            + (f" — {dataset_name}" if dataset_name else "")
        )
        legend_handles = [
            patches.Patch(color=color_map[sp], label=sp) for sp in species_order
        ]
        for bt in all_bar_types:
            legend_handles.append(
                patches.Patch(
                    facecolor="white",
                    edgecolor="black",
                    hatch=hatch_map[bt] or "",
                    label=bt,
                )
            )
        ax.legend(
            handles=legend_handles,
            title="Species / Match Type",
            bbox_to_anchor=(1.01, 1),
            loc="upper left",
        )

        fig_name = f"intensity_coverage_by_species{fig_name_suffix}"
        if fig_dir is not None:
            os.makedirs(fig_dir, exist_ok=True)
            fig.savefig(
                os.path.join(fig_dir, f"{fig_name}_{dataset_name}.png"),
                dpi=300,
                bbox_inches="tight",
            )
            fig.savefig(
                os.path.join(fig_dir, f"{fig_name}_{dataset_name}.svg"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)
        else:
            plt.show()

        return fig, ax, pd.concat(counts_by_type, names=["match_type"])

    records = []
    for col in int_cols:
        stem = col[: col.rfind(int_col_keyword)].rstrip()
        row_mask = pd.Series(True, index=df.index)
        if match_filters:
            for match_kw, allowed_values in match_filters.items():
                match_col = f"{stem} {match_kw}"
                if match_col in df.columns:
                    row_mask = row_mask & df[match_col].isin(allowed_values)
        sub = df[row_mask]
        present = sub[col].notna() & (sub[col] >= threshold)
        row: dict = {"col": col, "Not Quantified": int((~present).sum())}
        for sp in species_order:
            row[sp] = int((present & (sub[species_col] == sp)).sum())
        records.append(row)

    counts = pd.DataFrame(records).set_index("col")

    if sort_columns:
        total_present = counts[species_order].sum(axis=1)
        counts = counts.loc[total_present.sort_values(ascending=False).index]

    x_labels = (
        [label_shorten_fn(c) for c in counts.index]
        if label_shorten_fn is not None
        else list(counts.index)
    )

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(max(10, 0.6 * len(counts) + 2), 6),
            constrained_layout=True,
        )
    else:
        fig = ax.figure

    bottom = np.zeros(len(counts))
    for sp in species_order:
        vals = counts[sp].values
        ax.bar(x_labels, vals, bottom=bottom, color=color_map[sp], label=sp, width=0.7)
        bottom += vals
    ax.bar(
        x_labels,
        counts["Not Quantified"].values,
        bottom=bottom,
        color=missing_color,
        label="Not Quantified",
        width=0.7,
    )

    ax.set_xlabel("Run")
    ax.set_ylabel("Ion count")
    ax.set_title(
        f"Intensity coverage by species (threshold={threshold})"
        + (f" — {dataset_name}" if dataset_name else "")
    )
    ax.tick_params(axis="x", labelsize=8, rotation=90)
    ax.legend(title="Species", bbox_to_anchor=(1.01, 1), loc="upper left")

    fig_name = f"intensity_coverage_by_species{fig_name_suffix}"
    if fig_dir is not None:
        os.makedirs(fig_dir, exist_ok=True)
        fig.savefig(
            os.path.join(fig_dir, f"{fig_name}_{dataset_name}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        fig.savefig(
            os.path.join(fig_dir, f"{fig_name}_{dataset_name}.svg"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)
    else:
        plt.show()

    return fig, ax, counts


def plot_dict_ref_search_windows(
    dict_ref: pd.DataFrame,
    mz_bin: float,
    tolerance: float,
    figsize_windows: tuple = (8, 6),
    iso_row_height: float = 2.0,
    save_dir: Optional[str] = None,
) -> tuple[Figure, Figure]:
    """
    Two-panel diagnostic for dict_ref entries near a given mz_bin.

    Plot 1 – RT/IM search windows: one rectangle per row, colored by mz_bin,
    annotated with mz_rank.  X-axis = 1/K0, Y-axis = RT (min).

    Plot 2 – Isotope spike plots: one row of stacked subplots (shared x-axis)
    per dict_ref entry; X = IsoMZ, Y = IsoAbundance.
    """
    filtered = dict_ref[np.abs(dict_ref["mz_bin"] - mz_bin) <= tolerance].copy()
    if filtered.empty:
        raise ValueError(f"No entries within tolerance {tolerance} of mz_bin {mz_bin}")
    filtered = filtered.sort_values("mz_rank")
    Logger.info(
        "Filtered dict_ref: %s",
        filtered[["mz_rank", "mz_bin", "confounders", "count_confounders"]],
    )
    unique_bins = sorted(filtered["mz_bin"].unique())
    cmap = colormaps["tab10"] if len(unique_bins) <= 10 else colormaps["viridis"]
    n_bins = max(len(unique_bins) - 1, 1)
    bin_color = {b: cmap(i / n_bins) for i, b in enumerate(unique_bins)}

    # ── Plot 1: RT/IM rectangles ──────────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=figsize_windows)
    for _, row in filtered.iterrows():
        im_l, im_r = row["IM_search_left"], row["IM_search_right"]
        rt_l, rt_r = row["RT_search_left"], row["RT_search_right"]
        c = bin_color[row["mz_bin"]]
        ax1.add_patch(
            Rectangle(
                (im_l, rt_l),
                im_r - im_l,
                rt_r - rt_l,
                linewidth=1.5,
                edgecolor=c,
                facecolor=(*c[:3], 0.15),
            )
        )
        ax1.text(
            (im_l + im_r) / 2,
            (rt_l + rt_r) / 2,
            str(row["mz_rank"]),
            ha="center",
            va="center",
            fontsize=7,
            color=c[:3],
        )

    from matplotlib.patches import Patch

    ax1.legend(
        handles=[
            Patch(facecolor=bin_color[b], edgecolor=bin_color[b], label=f"{b:.4f}")
            for b in unique_bins
        ],
        title="mz_bin",
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        fontsize=8,
    )
    ax1.autoscale_view()
    ax1.set_xlabel("1/K0 (ion mobility)")
    ax1.set_ylabel("RT (min)")
    ax1.set_title(f"RT/IM Search Windows  mz_bin≈{mz_bin}  tol={tolerance}")
    fig1.tight_layout()

    # ── Plot 2: Isotope spike subplots ────────────────────────────────────────
    n = len(filtered)
    fig2, axes = plt.subplots(
        n,
        1,
        figsize=(8, max(n * iso_row_height, 3)),
        sharex=True,
        squeeze=False,
    )
    axes = axes.flatten()

    all_mz = np.concatenate([np.asarray(r["IsoMZ"]) for _, r in filtered.iterrows()])
    x_min, x_max = all_mz.min() - 0.5, all_mz.max() + 0.5

    for i, (_, row) in enumerate(filtered.iterrows()):
        ax = axes[i]
        iso_mz = np.asarray(row["IsoMZ"])
        iso_abu = np.asarray(row["IsoAbundance"])
        c = bin_color[row["mz_bin"]]
        ax.vlines(iso_mz, ymin=0, ymax=iso_abu, color=c, linewidth=1)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(bottom=0)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_ylabel(
            f"rank {row['mz_rank']}", fontsize=7, rotation=0, labelpad=50, va="center"
        )

    axes[-1].set_xlabel("m/z")
    fig2.suptitle(f"Isotope Patterns  mz_bin≈{mz_bin}  tol={tolerance}")
    fig2.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        for fmt in ["png", "svg"]:
            fig1.savefig(
                os.path.join(
                    save_dir,
                    f"DictRefWindows_{mz_bin}_{tolerance}.{fmt}",
                ),
                dpi=300,
                bbox_inches="tight",
            )
            fig2.savefig(
                os.path.join(
                    save_dir,
                    f"DictRefIsoPatterns_{mz_bin}_{tolerance}.{fmt}",
                ),
                dpi=300,
                bbox_inches="tight",
            )
        plt.close(fig1)
        plt.close(fig2)
    else:
        plt.show()

    return fig1, fig2


def plot_quantification_by_run(
    df: pd.DataFrame,
    *,
    id_cols: list[str] | None = None,
    int_col_keyword: Optional[str] = None,
    zero_color: str = "#d73027",
    nonzero_color: str = "#1a9850",
    sort_columns: bool = False,
    label_char_range: Optional[tuple[int, int]] = None,
    dataset_name: str = "",
    fig_dir: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
):
    """Wrapper around plot_run_summary(mode='nonzero_quant')."""
    return plot_run_summary(
        df,
        mode="nonzero_quant",
        id_cols=id_cols,
        int_col_keyword=int_col_keyword,
        zero_color=zero_color,
        nonzero_color=nonzero_color,
        sort_columns=sort_columns,
        label_char_range=label_char_range,
        dataset_name=dataset_name,
        fig_dir=fig_dir,
        ax=ax,
    )
