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
        "DATA_PATH": str(tmp_path / "data"),
        "N_CPU": 4,
        "PREPARE_DICT": {
            "SEARCH_ENGINE": "fragpipe",
            "PPM_TOL": 15,
        },
    }
    p = tmp_path / "test_override.yaml"
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
        for key in ["SEARCH_ENGINE", "PPM_TOL", "BIN_WIDTH", "GENERATE_DECOY"]:
            assert hasattr(cfg.PREPARE_DICT, key), f"Missing PREPARE_DICT.{key}"

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

    def test_default_generate_decoy_is_false(self, cfg):
        assert cfg.PREPARE_DICT.GENERATE_DECOY is False

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
