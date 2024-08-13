""" config class for ScanByScan """
import json
import logging
import os
from pathlib import Path
import pickle
from typing import Literal

Logger = logging.getLogger(__name__)

_alpha_opt_metric = Literal["cos_dist", "RMSE"]
_loss = Literal["lasso", "sqrt_lasso"]
_algo = Literal["lasso_lars", "lasso_cd", "lars", "omp", "threshold"]
_alpha_criteria = Literal["min", "convergence"]
_pp_method = Literal["raw", "sqrt"]
_AlignMethods = Literal["2stepNN", "peakRange"]


class Config:
    """Read config file and get information from it."""

    @property
    def n_cpu(self) -> int:
        """Get number of CPUs from config file."""
        return self.data.get("n_cpu", 1)

    @property
    def n_batch(self) -> int:
        """Get number of batches from config file."""
        return self.data.get("n_batch", 1)

    @property
    def n_blocks_by_pept(self) -> int:
        """Ger number of blocks seperated by peptide when saving output"""
        return self.data.get("n_blocks_by_pept", 0)

    @property
    def mzml_path(self) -> str:
        """Get mzml path from config file."""
        return self.data.get("mzml_path")

    @property
    def d_path(self) -> str:
        """Get d path from config file."""
        return self.data.get("d_path", None)

    @property
    def use_ims(self) -> bool:
        """Get use ims from config file."""
        return self.data.get("use_ims", False)

    @property
    def mq_exp_path(self) -> str:
        """Get MQ experiment path from config file."""
        return self.data.get("MQ_exp_path")

    @property
    def mq_ref_path(self) -> str:
        """Get MQ reference path from config file."""
        return self.data.get("MQ_ref_path")

    @property
    def notes(self) -> str:
        """Get notes from config file."""
        return self.data.get("notes")

    @property
    def how_batch(self) -> Literal["robin_round", "block"]:
        """Get method for generating batch from config file."""
        return self.data.get("how_batch")

    @property
    def rt_tol(self) -> float:
        """Get RT tolerance from config file."""
        return self.data.get("RT_tol")

    @property
    def iso_ab_mis_thres(self) -> float:
        """Get isotope abundance missing threshold from config file."""
        return self.data.get("iso_ab_mis_thres", 0.5)

    @property
    def alpha_criteria(self) -> str:
        """Get alpha criteria from config file."""
        return self.data.get("alpha_criteria", "convergence")

    @property
    def rt_ref(self) -> str:
        """Get RT reference from config file."""
        return self.data.get("RT_ref")

    @property
    def peak_sel_cos_dist(self) -> bool:
        """Get peak selection cosine distance from config file."""
        return self.data.get("PS_cos_dist", False)

    @property
    def opt_algo(self) -> str:
        """Get optimization algorithm from config file."""
        return self.data.get("opt_algo", "threshold")

    @property
    def alphas(self) -> list:
        """Get alphas from config file."""
        if self.data["opt_algo"] == "threshold":
            self.data["alphas"] = []
            Logger.info("Threshold method selected, set alphas to empty list.")
        return self.data.get("alphas", None)

    @property
    def mz_bin_digits(self) -> int:
        """Get mz bin digits from config file."""
        return self.data.get("mz_bin_digits")

    @property
    def im_peak_extraction_width(self) -> int:
        """Get peak extraction width from config file."""
        return self.data.get("im_peak_width")

    @property
    def start_frame(self) -> int:
        """Get start frame from config file."""
        return self.data.get("start_frame")

    @property
    def end_frame(self) -> int:
        """Get end frame from config file."""
        return self.data.get("end_frame")

    @property
    def filename(self) -> str:
        """Get filename from config file."""
        if self.start_frame is not None and self.end_frame is not None:
            notes = (
                "frame"
                + str(self.start_frame)
                + "_"
                + str(self.end_frame)
                + "_"
                + self.notes
            )
        else:
            notes = self.notes
        self.data["filename"] = (
            notes
            + self.basename
            + "_ScanByScan"
            + "_RTtol"
            + str(self.rt_tol)
            + "_"
            + self.opt_algo
            + "_missabthres"
            + str(self.iso_ab_mis_thres)
            + "_"
            + self.alpha_criteria
            + "_NoIntercept"
            + "_"
            + self.rt_ref
            + "_mzBinDigits"
            + str(self.mz_bin_digits)
            + "_imPeakWidth"
            + str(self.im_peak_extraction_width)
        )
        if self.peak_sel_cos_dist:
            self.data["filename"] += "_PScosDist"
        if self.delta_mobility_thres:
            self.data["filename"] += "_deltaMobilityThres" + str(
                self.delta_mobility_thres
            )
        return self.data.get("filename")

    @property
    def result_dir(self) -> str:
        """Get result directory from config file."""
        self.data["result_dir"] = os.path.join(self.dirname, self.filename)
        return self.data.get("result_dir")

    @property
    def output_file(self) -> str:
        """Get output file from config file."""
        self.data["output_file"] = os.path.join(self.data["result_dir"], "output")
        return self.data.get("output_file")

    @property
    def dirname(self) -> str:
        """Get dirname from config file."""
        try:
            self.data["dirname"] = os.path.dirname(self.data["mzml_path"])
        except KeyError as e:
            self.data["dirname"] = os.path.dirname(self.data["d_path"])
        return self.data.get("dirname")

    @property
    def basename(self) -> str:
        """Get basename from config file."""
        try:
            self.data["basename"] = Path(os.path.basename(self.data["mzml_path"])).stem
        except KeyError as e:
            self.data["basename"] = Path(os.path.basename(self.data["d_path"])).stem
        return self.data.get("basename")

    @property
    def ms1scans_no_array_name(self) -> str:
        """Get ms1scans_no_array_name from config file."""
        self.data["ms1scans_no_array_name"] = self.basename + "_MS1Scans_NoArray.csv"
        return self.data.get("ms1scans_no_array_name")

    @property
    def report_dir(self) -> str:
        """Get report directory from config file."""
        self.data["report_dir"] = os.path.join(self.data["result_dir"], "report")
        return self.data.get("report_dir")

    @property
    def rt_ref_act_peak(self) -> str:
        """Get RT reference elution peak from RT reference method"""
        match self.rt_ref:
            case "mix":
                self.data["RT_ref_act_peak"] = "Retention time new"
            case "pred":
                self.data["RT_ref_act_peak"] = "predicted_RT"
            case "exp":
                self.data["RT_ref_act_peak"] = "Calibrated retention time"
        return self.data.get("RT_ref_act_peak")

    @property
    def delta_mobility_thres(self) -> int:
        """Get delta mobility threshold from config file."""
        return self.data.get("delta_mobility_thres")

    ########################
    # functions start here #
    ########################

    def __init__(self, config_path):
        self.path = config_path
        self.data = self.read_config()
        self.check_config()

    def read_config(self):
        """Read config file."""
        with open(self.path, mode="r", encoding="utf-8") as file:
            config_data = json.load(file)

        return config_data

    def check_config(self):
        """Check config file validation."""
        pass

    def make_result_dirs(self):
        """Make result directory from config."""
        try:
            Logger.info(
                "Use RT reference method %s and RT elution peak %s",
                self.rt_ref,
                self.rt_ref_act_peak,
            )
            Logger.info("Result directory: %s", self.result_dir)
            if not os.path.exists(self.result_dir):
                os.makedirs(self.result_dir)

            if not os.path.exists(self.report_dir):
                os.makedirs(self.report_dir)
                os.makedirs(os.path.join(self.report_dir, "activation"))

            json_saved_args = json.dumps(self.data)
            with open(
                os.path.join(self.result_dir, "param.json"),
                mode="w",
                encoding="utf-8",
            ) as f:
                f.write(json_saved_args)

            with open(os.path.join(self.result_dir, "param.pkl"), "wb") as f:
                pickle.dump(self.data, f)
        except AttributeError as e:
            raise AttributeError("Please set mzml_path first.") from e
