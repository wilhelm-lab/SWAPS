import logging
from typing import Optional, Literal
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

import alphatims.bruker

Logger = logging.getLogger(__name__)


def load_dotd_data(dotd_file_path: str, swaps_result_dir: str = ""):
    """
    Load .d file data and save hdf5 if not already exists.
    :param: dotd_file_path: str, path to the .d file
    :param: swaps_result_dir: str, path to the directory to save the hdf5 file, optional, default .d directory
    :return: data: alphatims.bruker.TimsTOF, data object
    :return: hdf_file_name: str, path to the saved hdf5 file
    """
    data = alphatims.bruker.TimsTOF(dotd_file_path)
    if swaps_result_dir == "":
        Logger.info("No output directory provided, using the directory of the .d file")
        swaps_result_dir = os.path.dirname(dotd_file_path)
    os.makedirs(os.path.join(swaps_result_dir), exist_ok=True)
    hdf_path = os.path.join(swaps_result_dir, f"{data.sample_name}.hdf")
    if not os.path.isfile(hdf_path):
        hdf_file_name = data.save_as_hdf(
            directory=swaps_result_dir,
            file_name=f"{data.sample_name}.hdf",
            overwrite=False,
        )
        Logger.info("HDF file saved as %s", hdf_file_name)
    else:
        hdf_file_name = hdf_path
        Logger.info("HDF file %s already exists", hdf_file_name)
    return data, hdf_file_name


def export_im_and_ms1scans(
    data: alphatims.bruker.TimsTOF, swaps_result_dir: Optional[str] = None
):
    """
    Export IM and MS1 scans to csv files.
    :param: data: alphatims.bruker.TimsTOF, data object
    :param: swaps_result_dir: str, path to the directory to save the csv files, optional, default None then no export
    :return: ms1scans: pd.DataFrame, MS1 scans
    :return: mobility_values_df: pd.DataFrame, mobility values
    """
    # ms1scans
    ms1scans = data.frames.loc[data.frames.MsMsType == 0].copy()
    ms1scans["Time_minute"] = ms1scans["Time"] / 60
    ms1scans["MS1_frame_idx"] = (
        ms1scans["Time"].rank(axis=0, method="first", ascending=True).astype(int) - 1
    )  # 0-based index
    ms1scans.set_index("MS1_frame_idx", inplace=True, drop=False)
    Logger.info(
        "Double check MS1 frame index range: %s - %s",
        ms1scans["MS1_frame_idx"].min(),
        ms1scans["MS1_frame_idx"].max(),
    )

    # mobilty values
    mobility_values = np.sort(data.mobility_values)
    mobility_values_df = pd.DataFrame(
        mobility_values, columns=["mobility_values"]
    ).reset_index()
    mobility_values_df = mobility_values_df.rename(
        columns={"index": "mobility_values_index"}
    )
    Logger.info(
        "Double check mobility values index range: %s - %s",
        mobility_values_df["mobility_values_index"].min(),
        mobility_values_df["mobility_values_index"].max(),
    )
    ms1scans = pd.DataFrame(ms1scans)
    # export if swaps_result_dir is not None
    if swaps_result_dir is not None:
        os.makedirs(os.path.join(swaps_result_dir), exist_ok=True)
        ms1scans.to_csv(os.path.join(swaps_result_dir, "ms1scans.csv"))
        mobility_values_df.to_csv(os.path.join(swaps_result_dir, "mobility_values.csv"))
    return ms1scans, mobility_values_df
