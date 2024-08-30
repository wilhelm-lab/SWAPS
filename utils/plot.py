import logging
import os
from typing import List, Set, Union, Literal

import matplotlib.pyplot as plt
from matplotlib import colormaps, patches
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from matplotlib_venn import venn2, venn3
from scipy import stats
from sparse import SparseArray
from .tools import ExtractPeak
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
    title: Union[None, str] = None,
    fig_spec_name: str = "",
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
    if title is None:
        title = "Corr. of" + x_name + " and " + y_name

    PearsonR = stats.pearsonr(x=x_log, y=y_log)  # w/ log and w/o outliers
    SpearmanR = stats.spearmanr(a=x_log, b=y_log)
    slope, intercept, _, _, _ = stats.linregress(x=x_log, y=y_log)
    print(
        "Data: ",
        x_name,
        y_name,
        ", slope = ",
        np.round(slope, 3).item(),
        ", intercept = ",
        np.round(intercept, 3).item(),
        ", PearsonR = ",
        np.round(PearsonR[0], 3).item(),
        ", SpearmanR = ",
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
                fontsize=7,
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
                fontsize=7,
            )
            ax.annotate(
                "slp. = "
                + "{:.3f}".format(slope)
                + ", intrcpt. = "
                + "{:.3f}".format(intercept),
                xy=(0.2, 0.75),
                xycoords="figure fraction",
                horizontalalignment="left",
                verticalalignment="top",
                fontsize=7,
            )
            fig_type_name = "CorrQuantification"

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
        plt.suptitle(title + y_name)
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
    save_format: str = "png",
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
        fig_format=save_format,
    )


def plot_venn3(
    set1: Set,
    set2: Set,
    set3: Set,
    label1: str,
    label2: str,
    label3: str,
    save_dir: str | None = None,
    save_format: str = "png",
    title: str | None = None,
    fig_spec_name: str | None = None,
):
    venn3([set1, set2, set3], set_labels=(label1, label2, label3))
    if title is not None:
        plt.title(title)
    save_plot(
        save_dir=save_dir,
        fig_type_name="VennDiag",
        fig_spec_name=fig_spec_name,
        fig_format=save_format,
    )


def plot_comparison(
    y_true: pd.Series,
    y_pred: pd.Series,
    x_true: Union[None, pd.Series] = None,
    x_pred: Union[None, pd.Series] = None,
    log_y: bool = False,
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

    # enforce same y axis limits
    axs[0].set_ylim([0, max(axs[0].get_ylim()[1], abs(axs[1].get_ylim()[0]))])
    axs[1].set_ylim([-axs[0].get_ylim()[1], 0])


def save_plot(save_dir, fig_type_name, fig_spec_name, fig_format="png", **kwargs):
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(
            fname=os.path.join(
                save_dir, fig_type_name + "_" + fig_spec_name + "." + fig_format
            ),
            dpi=300,
            format=fig_format,
            **kwargs,
        )
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
            Logger.debug(
                "infer m/z and intensities: %s, %s", infer_in_range_idx, infer_in_range
            )
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
    sliced_frame: pd.DataFrame, labels: np.ndarray = None, label_name: str = "Label"
):
    fig, ax = plt.subplots()
    if labels is None:
        labels = sliced_frame["intensity_values"]
        label_name = "Intensity"
    scatter = ax.scatter(
        sliced_frame["mobility_values"], sliced_frame["mz_values"], c=labels, s=0.5
    )
    cbar = plt.colorbar(scatter)
    cbar.set_label(label_name, rotation=270)
    ax.set_ylabel("m/z")
    ax.set_xlabel("mobility")

    plt.title("Ion mobility spectrum frame " + str(sliced_frame["frame_indices"].min()))

    plt.show()


def plot_im_mz_int(sliced_frame: pd.DataFrame):
    fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
    ax.stem(
        sliced_frame["mobility_values"].values,
        sliced_frame["mz_values"].values,
        sliced_frame["intensity_values"].values,
    )
    ax.set_ylabel("m/z")
    ax.set_xlabel("mobility")
    ax.set_zlabel("intensity")


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
    Logger.info("Reference entry: %s", reference_entry)
    slice_pept_act_sparse, rt_idx_range, im_idx_range = slice_pept_act(
        pept_act_sparse=act_3d[:, :, batch_corrected_pept_mz_rank],
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
