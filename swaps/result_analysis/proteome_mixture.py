import logging
import os
from numpy import save
import pandas as pd
from peak_detection_2d.utils import (
    plot_target_decoy_distr,
    calc_fdr_and_thres,
    plot_per_image_metric_distr,
)

Logger = logging.getLogger(__name__)


def species_fdr_analysis(
    pept_act_sum_ps: pd.DataFrame, speices_name: str, save_dir: str
):
    if save_dir is None:
        os.makedirs(save_dir, exist_ok=True)
    df = pept_act_sum_ps.loc[pept_act_sum_ps["Taxonomy names"] == speices_name]
    Logger.info(f"Species: {speices_name}")
    Logger.info("Number of targets and decoys: %s", df["Decoy"].value_counts())
    # Logger.info("Number of precursors per species: %s", df["Taxonomy names"].nunique())
    plot_target_decoy_distr(df, save_dir, dataset_name=speices_name)
    calc_fdr_and_thres(
        df,
        return_plot=True,
        mark_x=[0.1, 0.2, 0.4],
        xlim=(0, 1),
        save_dir=save_dir,
        dataset_name=speices_name,
    )


def all_species_fdr_analysis(
    pept_act_sum_ps: pd.DataFrame, save_dir: str = None, species: list = None
):
    if save_dir is None:
        os.makedirs(save_dir, exist_ok=True)
    Logger.info(
        "Number of precursors per species: %s",
        pept_act_sum_ps["Taxonomy names"].value_counts(),
    )
    if species is None:
        Logger.info("Species not given, analyzing all species")
        species = pept_act_sum_ps["Taxonomy names"].unique()
    for speices_name in species:
        species_fdr_analysis(pept_act_sum_ps, speices_name, save_dir)


def species_wiou_analysis(
    pept_act_sum_ps: pd.DataFrame, speices_name: str, save_dir: str = None
):
    if save_dir is None:
        os.makedirs(save_dir, exist_ok=True)
    Logger.info(f"Species: {speices_name}")

    df = pept_act_sum_ps.loc[
        (pept_act_sum_ps["Taxonomy names"] == speices_name)
        & (pept_act_sum_ps["Decoy"] == 0)
    ]
    # Logger.info(
    #     "weighted IoU by species: %s", df["per_image_weighted_iou_metric"].describe()
    # )
    plot_per_image_metric_distr(
        loss_array=df["per_image_weighted_iou_metric"],
        metric_name="weighted IoU",
        save_dir=save_dir,
        dataset_name=speices_name,
    )


def all_species_wiou_analysis(
    pept_act_sum_ps: pd.DataFrame, species: list = None, save_dir: str = None
):
    if save_dir is None:
        os.makedirs(save_dir, exist_ok=True)
    Logger.info(
        "Number of precursors per species: %s",
        pept_act_sum_ps["Taxonomy names"].value_counts(),
    )
    if species is None:
        Logger.info("Species not given, analyzing all species")
        species = pept_act_sum_ps["Taxonomy names"].unique()
    for speices_name in species:
        species_wiou_analysis(pept_act_sum_ps, speices_name, save_dir)
