import logging
import os
from typing import List, Set, Union, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from matplotlib_venn import venn2, venn3
from scipy import stats
from .tools import ExtractPeak


Logger = logging.getLogger(__name__)


def plot_scatter(
    x: pd.Series,
    y: pd.Series,
    log_x: bool = False,
    log_y: bool = False,
    data: Union[None, pd.DataFrame] = None,
    filter_thres: float = 0,
    interactive: bool = False,
    hover_data: Union[None, List] = None,  # only used if interactive is true
    color: Union[None, pd.Series] = None,
    show_diag: bool = True,
    show_conf: Union[None, tuple] = None,
    save_dir: Union[None, str] = None,
    x_label: Union[None, str] = None,
    y_label: Union[None, str] = None,
    title: Union[None, str] = None,
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
    slope, intercept, r_value, p_value, std_err = stats.linregress(x=x_log, y=y_log)
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
        ax = sns.regplot(x=x_log, y=y_log, scatter=False, fit_reg=True)
        sns.scatterplot(x=x_log, y=y_log, hue=color, linewidth=0, legend=False)  # type: ignore
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
            save_dir=save_dir, fig_type_name="CorrQuantification", fig_spec_name=y_name
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
):
    venn2([set1, set2], set_labels=(label1, label2))
    if title is not None:
        plt.title(title)
    save_plot(
        save_dir=save_dir,
        fig_type_name="VennDiag",
        fig_spec_name=title,
        format=save_format,
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
):
    venn3([set1, set2, set3], set_labels=(label1, label2, label3))
    if title is not None:
        plt.title(title)
    save_plot(
        save_dir=save_dir,
        fig_type_name="VennDiag",
        fig_spec_name=title,
        format=save_format,
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


def save_plot(save_dir, fig_type_name, fig_spec_name, format="png", **kwargs):
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(
            fname=os.path.join(
                save_dir, fig_type_name + "_" + fig_spec_name + "." + format
            ),
            dpi=300,
            format=format,
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
    Maxquant_result,
    MS1Scans: pd.DataFrame | None = None,
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
        precursor_idx = Maxquant_result.loc[
            Maxquant_result["id"].isin(precursor_id)
        ].index

    match lower_plot:
        case "obs":
            # Find an appropriate scan if not provided
            if scan_idx is None:
                if precursor_idx is None:
                    raise ValueError("Please provide a precursor index.")
                RT = np.max(
                    Maxquant_result.loc[precursor_idx, "Retention time"].values
                )  # take the later RT of the precursors
                scan_idx = np.abs(MS1Scans["starttime"] - RT).argmin()
                scan_time = MS1Scans.loc[scan_idx, "starttime"]
                Logger.info(
                    "Precursors %s retention time %s, \n show later RT %s with"
                    " corresponding scan index %s         with scan time %s",
                    precursor_idx,
                    Maxquant_result.loc[precursor_idx, "Retention time"].values,
                    RT,
                    scan_idx,
                    scan_time,
                )
            OneScan = MS1Scans.iloc[scan_idx, :]
            OneScanMZ = np.array(OneScan["mzarray"])
            IsoMZ = None

            # Find the range of mz in MS1 scan to plot
            if precursor_idx is not None:
                IsoMZ = Maxquant_result.loc[precursor_idx, "IsoMZ"]
                IsoMZ_flatten = np.concatenate(IsoMZ.values).ravel()
                IsoMZ_range = [np.min(IsoMZ_flatten) - 1, np.max(IsoMZ_flatten) + 1]
                OneScanMZinRange = OneScanMZ[
                    (OneScanMZ > IsoMZ_range[0]) & (OneScanMZ < IsoMZ_range[1])
                ]
                OneScanMZinRangeIdx = np.where(
                    (OneScanMZ > IsoMZ_range[0]) & (OneScanMZ < IsoMZ_range[1])
                )[0]
            else:
                if not isinstance(mzrange, list):
                    raise TypeError(
                        "mzrange should be a list, or provide an int for precursor"
                        " index."
                    )
                OneScanMZinRange = OneScanMZ[
                    (OneScanMZ > mzrange[0]) & (OneScanMZ < mzrange[1])
                ]
                OneScanMZinRangeIdx = np.where(
                    (OneScanMZ > mzrange[0]) & (OneScanMZ < mzrange[1])
                )[0]

            # Calculating values for visualization
            Intensity = np.array(OneScan["intarray"])[OneScanMZinRangeIdx]
            if log_intensity:  # +1 to avoid divide by zero error
                Intensity = np.log10(Intensity + 1)
            peak_results = ExtractPeak(x=OneScanMZinRange, y=Intensity)
            peaks_idx = peak_results["apex_mzidx"]
            print("Peak results:")
            print(peak_results)
        case "infer":
            if infer_intensity is None:
                raise ValueError("please provide infer_intensity.")
            Intensity = infer_intensity.values
            if log_intensity:
                Intensity = np.log10(infer_intensity.values + 1)
            if precursor_idx is not None:
                IsoMZ = Maxquant_result.loc[precursor_idx, "IsoMZ"]
                IsoMZ_flatten = np.concatenate(IsoMZ.values).ravel()
                IsoMZ_range = [np.min(IsoMZ_flatten) - 1, np.max(IsoMZ_flatten) + 1]
                InferinRange = Intensity[
                    (infer_intensity.index > IsoMZ_range[0])
                    & (infer_intensity.index < IsoMZ_range[1])
                ]
                InferinRangeIdx = infer_intensity.index[
                    (infer_intensity.index > IsoMZ_range[0])
                    & (infer_intensity.index < IsoMZ_range[1])
                ]

    # Plot
    fig, axs = plt.subplots(2, 1, sharex=True, sharey=False)
    fig.subplots_adjust(hspace=0)
    if precursor_idx is not None:
        colormap = plt.cm.bwr  # nipy_spectral, Set1,Paired
        colors = [colormap(i) for i in np.linspace(0, 1, len(precursor_idx))]
        for i, precursor in enumerate(precursor_idx):
            axs[0].vlines(
                x=Maxquant_result.loc[precursor, "IsoMZ"],
                ymin=0,
                ymax=Maxquant_result.loc[precursor, "IsoAbundance"],
                label=precursor,
                color=colors[i],
            )
            print(
                "Isotope Pattern:",
                precursor,
                Maxquant_result.loc[precursor, "IsoMZ"],
                Maxquant_result.loc[precursor, "IsoAbundance"],
            )
        axs[0].set_title("Up: Isotope Pattern, Down: MS1 Scan " + str(scan_idx))
    match lower_plot:
        case "obs":
            axs[1].vlines(x=OneScanMZinRange, ymin=-Intensity, ymax=0, label="MS peaks")
            axs[1].hlines(
                y=-peak_results["peak_height"],
                xmin=peak_results["start_mz"],
                xmax=peak_results["end_mz"],
                linewidth=2,
                color="black",
            )
            axs[1].vlines(
                x=OneScanMZinRange[peaks_idx],
                ymin=-Intensity[peaks_idx],
                ymax=0,
                color="orange",
                label="inferred apex",
            )
            axs[1].plot(
                OneScanMZinRange[peaks_idx],
                -Intensity[peaks_idx],
                "x",
                color="orange",
                label="inferred apex",
            )
        case "infer":
            Logger.debug(
                "infer m/z and intensities: %s, %s", InferinRangeIdx, InferinRange
            )
            axs[1].vlines(  # x = infer_intensity.index,
                x=InferinRangeIdx,
                ymin=-InferinRange,
                ymax=0,
                label="inferred intensity",
            )
            axs[1].set_xlim(IsoMZ_range)
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
    precursor_cos_dists: list | None,
    activation_labels: list,
    cos_dist_labels: list | None,
    ms1scan_no_array: pd.DataFrame,
    log_intensity: bool = False,
    x_ticks: Literal["time", "scan index"] = "time",
    save_dir=None,
    save_format: str = "png",
):
    """Plot the activation profile of a precursor"""
    # get the RT search range
    rt_search_range = [
        maxquant_ref_row["RT_search_left"],
        maxquant_ref_row["RT_search_right"],
    ]
    rt_search_center = maxquant_ref_row["RT_search_center"]
    Logger.debug("RT search range: %s", rt_search_range)

    # find the corresponding experiment match
    mod_seq = maxquant_ref_row["Modified sequence"]
    charge = maxquant_ref_row["Charge"]
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
    scan_index = ms1scan_no_array[
        (ms1scan_no_array["starttime"] >= rt_min)
        & (ms1scan_no_array["starttime"] <= rt_max)
    ].index
    Logger.debug("ScanIdx: %s", scan_index)

    time_profiles = pd.DataFrame(dict(zip(activation_labels, precursor_activations)))
    time_profiles = time_profiles.set_index(ms1scan_no_array["starttime"])
    activation_in_range = time_profiles.iloc[scan_index, :]

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
            format=save_format,
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
