"""
Unit tests for utils.config and the YACS configuration system.

Tests cover:
- get_cfg_defaults returns a frozen CfgNode with expected keys
- Merging a YAML override file applies values correctly
- ADD_TIMESTAMP_TO_RESULT_PATH behaviour
- N_CPU / N_BATCH auto-detection defaults
"""

import os
import tempfile

import pytest
import yaml

from utils.config import get_cfg_defaults
from utils.singleton_swaps_optimization import swaps_optimization_cfg


@pytest.fixture
def cfg():
    """Fresh default config for each test."""
    return get_cfg_defaults(swaps_optimization_cfg)


@pytest.fixture
def minimal_yaml(tmp_path):
    """Write a minimal override YAML and return its path."""
    data = {
        "RESULT_PATH": str(tmp_path / "results"),
        "DATA_PATH": [str(tmp_path / "data")],
        "N_CPU": 4,
        "PREPARE_DICT": {
            "SEARCH_ENGINE": "fragpipe",
            "PPM_TOL": 15,
        },
    }
    p = tmp_path / "test_override.yaml"
    p.write_text(yaml.dump(data))
    return str(p)


@pytest.fixture
def nested_yaml(tmp_path):
    """Write an override YAML exercising all three new sub-nodes."""
    data = {
        "PREPARE_DICT": {
            "PRED": {
                "TRAIN_FRAC": 0.8,
                "RT_TRAIN_EPOCHS": 20,
                "UPDATED_RT_MODEL_PATH": "/tmp/rt_model.pt",
            },
            "REF": {
                "RT_TOL": 0.5,
                "IM_LENGTH": 10,
                "DELTA_IM_95": 0.05,
                "SUMMARIZE_WITHOUT_MATCH": True,
            },
            "SAGE": {
                "RT_WINDOW": 1.5,
                "IM_WINDOW": 0.2,
            },
        },
    }
    p = tmp_path / "test_nested_override.yaml"
    p.write_text(yaml.dump(data))
    return str(p)


# ---------------------------------------------------------------------------
# Default config shape
# ---------------------------------------------------------------------------


class TestGetCfgDefaults:
    def test_returns_cfg_node(self, cfg):
        from yacs.config import CfgNode
        assert isinstance(cfg, CfgNode)

    def test_top_level_keys_present(self, cfg):
        for key in ["DATA_PATH", "RESULT_PATH", "N_CPU", "SWA", "USE_IMS",
                    "PREPARE_DICT", "OPTIMIZATION"]:
            assert hasattr(cfg, key), f"Missing top-level key: {key}"

    def test_prepare_dict_keys_present(self, cfg):
        for key in ["SEARCH_ENGINE", "PPM_TOL", "BIN_WIDTH",
                    "PRED", "REF", "SAGE", "OK"]:
            assert hasattr(cfg.PREPARE_DICT, key), f"Missing PREPARE_DICT.{key}"

    def test_prepare_dict_pred_keys_present(self, cfg):
        for key in ["UPDATED_RT_MODEL_PATH", "UPDATED_IM_MODEL_PATH",
                    "TRAIN_FRAC", "RT_TRAIN_EPOCHS", "IM_TRAIN_EPOCHS"]:
            assert hasattr(cfg.PREPARE_DICT.PRED, key), f"Missing PREPARE_DICT.PRED.{key}"

    def test_prepare_dict_ref_keys_present(self, cfg):
        for key in ["RT_TOL", "IM_LENGTH", "DELTA_IM_95", "SUMMARIZE_WITHOUT_MATCH"]:
            assert hasattr(cfg.PREPARE_DICT.REF, key), f"Missing PREPARE_DICT.REF.{key}"

    def test_prepare_dict_sage_keys_present(self, cfg):
        for key in ["RT_WINDOW", "IM_WINDOW"]:
            assert hasattr(cfg.PREPARE_DICT.SAGE, key), f"Missing PREPARE_DICT.SAGE.{key}"

    def test_prepare_dict_ok_keys_present(self, cfg):
        for key in ["DIR", "OUTPUT", "FDR"]:
            assert hasattr(cfg.PREPARE_DICT.OK, key), f"Missing PREPARE_DICT.OK.{key}"

    def test_optimization_keys_present(self, cfg):
        assert hasattr(cfg.OPTIMIZATION, "N_BATCH")

    def test_default_search_engine_is_maxquant(self, cfg):
        assert cfg.PREPARE_DICT.SEARCH_ENGINE == "maxquant"

    def test_default_swa_is_true(self, cfg):
        assert cfg.SWA is True

    def test_default_use_ims_is_true(self, cfg):
        assert cfg.USE_IMS is True

    def test_default_n_cpu_is_negative(self, cfg):
        assert cfg.N_CPU < 0

    def test_default_n_batch_is_negative(self, cfg):
        assert cfg.OPTIMIZATION.N_BATCH < 0

    def test_default_data_path_is_empty_list(self, cfg):
        assert cfg.DATA_PATH == []

    def test_default_ok_dir_is_empty(self, cfg):
        assert cfg.PREPARE_DICT.OK.DIR == ""

    def test_default_ok_output_is_psms(self, cfg):
        assert cfg.PREPARE_DICT.OK.OUTPUT == "psms"

    def test_default_pred_model_paths_empty(self, cfg):
        assert cfg.PREPARE_DICT.PRED.UPDATED_RT_MODEL_PATH == ""
        assert cfg.PREPARE_DICT.PRED.UPDATED_IM_MODEL_PATH == ""

    def test_default_ref_tols_negative(self, cfg):
        """Negative values trigger auto-detection from data."""
        assert cfg.PREPARE_DICT.REF.RT_TOL < 0
        assert cfg.PREPARE_DICT.REF.IM_LENGTH < 0
        assert cfg.PREPARE_DICT.REF.DELTA_IM_95 < 0

    def test_default_sage_windows_zero(self, cfg):
        """Zero values trigger auto-detection."""
        assert cfg.PREPARE_DICT.SAGE.RT_WINDOW == 0.0
        assert cfg.PREPARE_DICT.SAGE.IM_WINDOW == 0.0

    def test_cfg_is_mutable_after_get_defaults(self, cfg):
        """get_cfg_defaults should return an unfrozen copy, ready to merge."""
        cfg.N_CPU = 8  # should not raise
        assert cfg.N_CPU == 8


# ---------------------------------------------------------------------------
# YAML merge
# ---------------------------------------------------------------------------


class TestCfgMergeFromFile:
    def test_merge_applies_n_cpu(self, cfg, minimal_yaml):
        cfg.merge_from_file(minimal_yaml)
        assert cfg.N_CPU == 4

    def test_merge_applies_nested_key(self, cfg, minimal_yaml):
        cfg.merge_from_file(minimal_yaml)
        assert cfg.PREPARE_DICT.SEARCH_ENGINE == "fragpipe"
        assert cfg.PREPARE_DICT.PPM_TOL == 15

    def test_merge_applies_result_path(self, cfg, minimal_yaml, tmp_path):
        cfg.merge_from_file(minimal_yaml)
        assert cfg.RESULT_PATH == str(tmp_path / "results")

    def test_unset_keys_keep_defaults(self, cfg, minimal_yaml):
        original_swa = cfg.SWA
        cfg.merge_from_file(minimal_yaml)
        assert cfg.SWA == original_swa


class TestNestedSubNodeMerge:
    def test_merge_pred_keys(self, cfg, nested_yaml):
        cfg.merge_from_file(nested_yaml)
        assert cfg.PREPARE_DICT.PRED.TRAIN_FRAC == 0.8
        assert cfg.PREPARE_DICT.PRED.RT_TRAIN_EPOCHS == 20
        assert cfg.PREPARE_DICT.PRED.UPDATED_RT_MODEL_PATH == "/tmp/rt_model.pt"

    def test_merge_ref_keys(self, cfg, nested_yaml):
        cfg.merge_from_file(nested_yaml)
        assert cfg.PREPARE_DICT.REF.RT_TOL == 0.5
        assert cfg.PREPARE_DICT.REF.IM_LENGTH == 10
        assert cfg.PREPARE_DICT.REF.DELTA_IM_95 == 0.05
        assert cfg.PREPARE_DICT.REF.SUMMARIZE_WITHOUT_MATCH is True

    def test_merge_sage_keys(self, cfg, nested_yaml):
        cfg.merge_from_file(nested_yaml)
        assert cfg.PREPARE_DICT.SAGE.RT_WINDOW == 1.5
        assert cfg.PREPARE_DICT.SAGE.IM_WINDOW == 0.2

    def test_unrelated_defaults_unchanged(self, cfg, nested_yaml):
        cfg.merge_from_file(nested_yaml)
        assert cfg.PREPARE_DICT.PRED.IM_TRAIN_EPOCHS == 8  # default untouched
        assert cfg.PREPARE_DICT.REF.IM_LENGTH == 10  # overridden
        assert cfg.N_CPU < 0  # top-level default untouched


# ---------------------------------------------------------------------------
# DATA_PATH normalization (string → list)
# ---------------------------------------------------------------------------


class TestDataPathNormalization:
    def test_scalar_string_is_wrapped_in_list(self, cfg, tmp_path):
        """A YAML with DATA_PATH as a plain string must be normalized to a 1-element list."""
        from utils.config import merge_cfg_from_file

        p = tmp_path / "scalar_path.yaml"
        p.write_text(f"DATA_PATH: {tmp_path}/data\n")
        merge_cfg_from_file(cfg, str(p))
        assert cfg.DATA_PATH == [str(tmp_path / "data")]

    def test_list_of_paths_is_preserved(self, cfg, tmp_path):
        """A YAML with DATA_PATH as a list stays a list with all entries."""
        import yaml as _yaml
        from utils.config import merge_cfg_from_file

        paths = [str(tmp_path / "run1"), str(tmp_path / "run2")]
        p = tmp_path / "list_path.yaml"
        p.write_text(_yaml.dump({"DATA_PATH": paths}))
        merge_cfg_from_file(cfg, str(p))
        assert list(cfg.DATA_PATH) == paths

    def test_default_data_path_not_set_stays_empty(self, cfg, tmp_path):
        """When DATA_PATH is absent from the YAML, it stays as the default []."""
        from utils.config import merge_cfg_from_file

        p = tmp_path / "no_data_path.yaml"
        p.write_text("N_CPU: 2\n")
        merge_cfg_from_file(cfg, str(p))
        assert cfg.DATA_PATH == []


# ---------------------------------------------------------------------------
# ADD_TIMESTAMP_TO_RESULT_PATH logic (mirrors sbs_runner_ims.py behaviour)
# ---------------------------------------------------------------------------


class TestTimestampLogic:
    def test_timestamp_appended_when_flag_true(self, cfg, tmp_path):
        cfg.RESULT_PATH = str(tmp_path / "my_result")
        cfg.ADD_TIMESTAMP_TO_RESULT_PATH = True
        # Mimic the runner logic
        from datetime import datetime
        name_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        if cfg.ADD_TIMESTAMP_TO_RESULT_PATH:
            cfg.RESULT_PATH = cfg.RESULT_PATH + "_" + name_timestamp
            cfg.ADD_TIMESTAMP_TO_RESULT_PATH = False
        assert name_timestamp in cfg.RESULT_PATH
        assert cfg.ADD_TIMESTAMP_TO_RESULT_PATH is False

    def test_no_timestamp_when_flag_false(self, cfg, tmp_path):
        base = str(tmp_path / "my_result")
        cfg.RESULT_PATH = base
        cfg.ADD_TIMESTAMP_TO_RESULT_PATH = False
        if cfg.ADD_TIMESTAMP_TO_RESULT_PATH:
            cfg.RESULT_PATH = cfg.RESULT_PATH + "_TIMESTAMP"
        assert cfg.RESULT_PATH == base


# ---------------------------------------------------------------------------
# N_CPU / N_BATCH auto-detection (mirrors sbs_runner_ims.py logic)
# ---------------------------------------------------------------------------


class TestCpuAutoDetection:
    def test_n_cpu_reads_slurm_env(self, cfg, monkeypatch):
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "16")
        if cfg.N_CPU < 0:
            cfg.N_CPU = int(os.getenv("SLURM_CPUS_PER_TASK", 1))
        assert cfg.N_CPU == 16

    def test_n_cpu_defaults_to_1_without_slurm(self, cfg, monkeypatch):
        monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
        if cfg.N_CPU < 0:
            cfg.N_CPU = int(os.getenv("SLURM_CPUS_PER_TASK", 1))
        assert cfg.N_CPU == 1

    def test_n_batch_set_to_n_cpu_when_negative(self, cfg):
        cfg.N_CPU = 8
        if cfg.OPTIMIZATION.N_BATCH < 0:
            cfg.OPTIMIZATION.N_BATCH = cfg.N_CPU
        assert cfg.OPTIMIZATION.N_BATCH == 8
