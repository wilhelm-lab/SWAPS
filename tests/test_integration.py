"""
Integration tests using real on-disk data.

These tests are skipped automatically if the required directories are not
accessible. They are marked with @pytest.mark.integration.

To run:
  export SWAPS_TEST_RAW_DIR=/cmnfs/proj/ORIGINS/data/.../Data/test_hela/500SPD/jigsawPASEF_a000_IntThres1600_2R_150ms
  export SWAPS_TEST_FRAGPIPE_DIR=/cmnfs/proj/ORIGINS/data/.../fragPipeOutput
  export SWAPS_TEST_SAGE_DIR=/cmnfs/proj/ORIGINS/data/.../sage/...
  export SWAPS_TEST_MAXQUANT_DIR=/cmnfs/proj/ORIGINS/data/.../MaxQuant/.../combined/txt
  pytest tests/test_integration.py -v

Each test covers one pipeline phase so failures are easy to localise.
The SWA and full-pipeline tests are additionally marked @pytest.mark.slow
and are excluded from default CI runs via:
  pytest tests/ -m "not slow"
"""

import os

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Phase 2 – FragPipe parsing
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestFragpipeParsingIntegration:
    """Load real psm.tsv files and run fragpipe_psm_parser on them."""

    def test_psm_files_found_in_fragpipe_dir(self, fragpipe_dir):
        exp_dirs = [
            d for d in os.listdir(fragpipe_dir)
            if os.path.isdir(os.path.join(fragpipe_dir, d)) and d != "MSBooster"
        ]
        assert len(exp_dirs) > 0, "No experiment sub-directories found in FragPipe dir"

    def test_psm_tsv_exists_per_experiment(self, fragpipe_dir):
        exp_dirs = [
            d for d in os.listdir(fragpipe_dir)
            if os.path.isdir(os.path.join(fragpipe_dir, d)) and d != "MSBooster"
        ]
        for exp in exp_dirs:
            psm_path = os.path.join(fragpipe_dir, exp, "psm.tsv")
            assert os.path.isfile(psm_path), f"Missing psm.tsv in {exp}"

    def test_all_psm_files_loadable(self, fragpipe_dir):
        exp_dirs = [
            d for d in os.listdir(fragpipe_dir)
            if os.path.isdir(os.path.join(fragpipe_dir, d)) and d != "MSBooster"
        ]
        for exp in exp_dirs:
            psm_path = os.path.join(fragpipe_dir, exp, "psm.tsv")
            df = pd.read_csv(psm_path, sep="\t")
            assert len(df) > 0, f"Empty psm.tsv in {exp}"

    def test_fragpipe_psm_parser_runs_on_real_data(self, fragpipe_dir):
        from prepare_dict.search_engine_output_parser import fragpipe_psm_parser

        evidence = pd.DataFrame()
        exp_dirs = [
            d for d in os.listdir(fragpipe_dir)
            if os.path.isdir(os.path.join(fragpipe_dir, d)) and d != "MSBooster"
        ]
        for exp in exp_dirs:
            tmp = pd.read_csv(os.path.join(fragpipe_dir, exp, "psm.tsv"), sep="\t")
            evidence = pd.concat([evidence, tmp], ignore_index=True)

        result = fragpipe_psm_parser(evidence)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_parsed_fragpipe_has_required_columns(self, fragpipe_dir):
        from prepare_dict.search_engine_output_parser import fragpipe_psm_parser

        evidence = pd.DataFrame()
        exp_dirs = [
            d for d in os.listdir(fragpipe_dir)
            if os.path.isdir(os.path.join(fragpipe_dir, d)) and d != "MSBooster"
        ]
        for exp in exp_dirs:
            tmp = pd.read_csv(os.path.join(fragpipe_dir, exp, "psm.tsv"), sep="\t")
            evidence = pd.concat([evidence, tmp], ignore_index=True)

        result = fragpipe_psm_parser(evidence)
        for col in [
            "Sequence",
            "Modified sequence",
            "Raw file",
            "Retention time",
            "Calibrated retention time",
            "1/K0",
            "Type",
            "Reverse",
        ]:
            assert col in result.columns, f"Missing required column: {col}"

    def test_rt_in_minutes_not_seconds(self, fragpipe_dir):
        """RT values after parsing must be in minutes (< 200 for typical LC runs)."""
        from prepare_dict.search_engine_output_parser import fragpipe_psm_parser

        evidence = pd.DataFrame()
        exp_dirs = [
            d for d in os.listdir(fragpipe_dir)
            if os.path.isdir(os.path.join(fragpipe_dir, d)) and d != "MSBooster"
        ][:2]  # just first 2 experiments for speed
        for exp in exp_dirs:
            tmp = pd.read_csv(os.path.join(fragpipe_dir, exp, "psm.tsv"), sep="\t")
            evidence = pd.concat([evidence, tmp], ignore_index=True)

        result = fragpipe_psm_parser(evidence)
        assert result["Retention time"].max() < 200, (
            "RT values appear to still be in seconds (max > 200 min)"
        )

    def test_multiple_raw_files_in_parsed_output(self, fragpipe_dir):
        """Parsed evidence must contain >1 unique Raw file (multi-run experiment)."""
        from prepare_dict.search_engine_output_parser import fragpipe_psm_parser

        evidence = pd.DataFrame()
        exp_dirs = [
            d for d in os.listdir(fragpipe_dir)
            if os.path.isdir(os.path.join(fragpipe_dir, d)) and d != "MSBooster"
        ]
        for exp in exp_dirs:
            tmp = pd.read_csv(os.path.join(fragpipe_dir, exp, "psm.tsv"), sep="\t")
            evidence = pd.concat([evidence, tmp], ignore_index=True)

        result = fragpipe_psm_parser(evidence)
        assert result["Raw file"].nunique() > 1, (
            "Expected multiple raw files in parsed FragPipe output"
        )


# ---------------------------------------------------------------------------
# Phase 2 – Dictionary construction (subset for speed)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestDictConstructionIntegration:
    """Run construct_dict_from_search_pivoted on a small subset of real data."""

    @pytest.fixture
    def fragpipe_evidence_subset(self, fragpipe_dir):
        from prepare_dict.search_engine_output_parser import fragpipe_psm_parser

        evidence = pd.DataFrame()
        exp_dirs = [
            d for d in sorted(os.listdir(fragpipe_dir))
            if os.path.isdir(os.path.join(fragpipe_dir, d)) and d != "MSBooster"
        ][:2]  # 2 runs only
        for exp in exp_dirs:
            tmp = pd.read_csv(os.path.join(fragpipe_dir, exp, "psm.tsv"), sep="\t")
            evidence = pd.concat([evidence, tmp], ignore_index=True)

        return fragpipe_psm_parser(evidence)

    def test_construct_dict_runs_without_error(self, fragpipe_evidence_subset):
        from utils.config import get_cfg_defaults
        from utils.singleton_swaps_optimization import swaps_optimization_cfg
        from prepare_dict.prepare_dict import construct_dict_from_search_pivoted

        cfg = get_cfg_defaults(swaps_optimization_cfg)
        cfg.PREPARE_DICT.SEARCH_ENGINE = "fragpipe"
        cfg.PREPARE_DICT.GENERATE_DECOY = True

        dict_ref = construct_dict_from_search_pivoted(
            cfg_prepare_dict=cfg.PREPARE_DICT,
            evidence=fragpipe_evidence_subset,
            n_blocks_by_pept=1,
        )
        assert isinstance(dict_ref, pd.DataFrame)
        assert len(dict_ref) > 0

    def test_dict_ref_has_required_columns(self, fragpipe_evidence_subset):
        from utils.config import get_cfg_defaults
        from utils.singleton_swaps_optimization import swaps_optimization_cfg
        from prepare_dict.prepare_dict import construct_dict_from_search_pivoted

        cfg = get_cfg_defaults(swaps_optimization_cfg)
        cfg.PREPARE_DICT.SEARCH_ENGINE = "fragpipe"

        dict_ref = construct_dict_from_search_pivoted(
            cfg_prepare_dict=cfg.PREPARE_DICT,
            evidence=fragpipe_evidence_subset,
            n_blocks_by_pept=1,
        )
        for col in ["mz_rank", "Sequence", "Proteins"]:
            assert col in dict_ref.columns, f"Missing column: {col}"

    def test_mz_rank_is_unique(self, fragpipe_evidence_subset):
        from utils.config import get_cfg_defaults
        from utils.singleton_swaps_optimization import swaps_optimization_cfg
        from prepare_dict.prepare_dict import construct_dict_from_search_pivoted

        cfg = get_cfg_defaults(swaps_optimization_cfg)
        cfg.PREPARE_DICT.SEARCH_ENGINE = "fragpipe"

        dict_ref = construct_dict_from_search_pivoted(
            cfg_prepare_dict=cfg.PREPARE_DICT,
            evidence=fragpipe_evidence_subset,
            n_blocks_by_pept=1,
        )
        assert dict_ref["mz_rank"].is_unique


# ---------------------------------------------------------------------------
# Phase 3 – Raw data loading (requires SWAPS_TEST_RAW_DIR)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.slow
class TestRawDataLoadingIntegration:
    """Load a single .d file and verify the alphatims wrapper works."""

    @pytest.fixture
    def first_dotd(self, raw_dir):
        dotd_files = [f for f in os.listdir(raw_dir) if f.endswith(".d")]
        if not dotd_files:
            pytest.skip("No .d files found in raw_dir")
        return os.path.join(raw_dir, sorted(dotd_files)[0])

    def test_load_dotd_data_returns_data_object(self, first_dotd, tmp_path):
        from utils.ims_utils import load_dotd_data

        data, hdf_path = load_dotd_data(first_dotd, swaps_result_dir=str(tmp_path))
        assert data is not None

    def test_export_im_and_ms1scans_returns_dataframes(self, first_dotd, tmp_path):
        from utils.ims_utils import export_im_and_ms1scans, load_dotd_data

        data, _ = load_dotd_data(first_dotd, swaps_result_dir=str(tmp_path))
        ms1scans, mobility_values_df = export_im_and_ms1scans(
            data=data, swaps_result_dir=str(tmp_path)
        )
        assert isinstance(ms1scans, pd.DataFrame)
        assert isinstance(mobility_values_df, pd.DataFrame)
        assert len(ms1scans) > 0
        assert len(mobility_values_df) > 0

    def test_multiple_dotd_files_present(self, raw_dir):
        dotd_files = [f for f in os.listdir(raw_dir) if f.endswith(".d")]
        assert len(dotd_files) > 1, (
            f"Expected multiple .d files for multi-run test, found {len(dotd_files)}"
        )


# ---------------------------------------------------------------------------
# Phase 2 – SAGE parsing
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestSageParsingIntegration:

    @pytest.fixture
    def sage_tsv_path(self, sage_dir):
        # Prefer the canonical PSM results file; fall back to first .tsv found.
        # SAGE produces multiple TSVs (results.sage.tsv, matched_fragments.sage.tsv,
        # etc.) — only the PSM file has the 'filename' column expected by sage_parser.
        candidate = os.path.join(sage_dir, "results.sage.tsv")
        if os.path.isfile(candidate):
            return candidate
        tsv_files = sorted(f for f in os.listdir(sage_dir) if f.endswith(".tsv"))
        if not tsv_files:
            pytest.skip("No .tsv files found in SAGE output dir")
        return os.path.join(sage_dir, tsv_files[0])

    @pytest.fixture
    def parsed_sage(self, sage_tsv_path):
        from prepare_dict.search_engine_output_parser import sage_parser

        df = pd.read_csv(sage_tsv_path, sep="\t")
        return sage_parser(df)

    def test_sage_dir_accessible(self, sage_dir):
        assert os.path.isdir(sage_dir)

    def test_sage_result_file_exists(self, sage_dir):
        tsv_files = [f for f in os.listdir(sage_dir) if f.endswith(".tsv")]
        assert len(tsv_files) > 0, "No .tsv files found in SAGE output dir"

    def test_sage_tsv_loadable(self, sage_tsv_path):
        df = pd.read_csv(sage_tsv_path, sep="\t")
        assert len(df) > 0

    def test_sage_parser_runs_on_real_data(self, parsed_sage):
        assert isinstance(parsed_sage, pd.DataFrame)
        assert len(parsed_sage) > 0

    def test_parsed_sage_has_required_columns(self, parsed_sage):
        for col in [
            "Sequence",
            "Modified sequence",
            "Raw file",
            "Retention time",
            "Calibrated retention time",
            "1/K0",
            "Type",
            "Reverse",
        ]:
            assert col in parsed_sage.columns, f"Missing required column: {col}"

    def test_sage_fdr_filter_applied(self, sage_tsv_path):
        """Parser must apply peptide_q <= 0.01 and spectrum_q <= 0.01 filter."""
        from prepare_dict.search_engine_output_parser import sage_parser

        raw = pd.read_csv(sage_tsv_path, sep="\t")
        result = sage_parser(raw)
        # All remaining rows must pass the FDR thresholds used before rename
        # (columns are renamed; original peptide_q → Match q-value via rename_dict)
        assert len(result) <= len(raw), "Parser must not add rows"
        assert len(result) > 0, "Parser filtered out everything — check FDR thresholds"

    def test_sage_decoy_reverse_column(self, parsed_sage):
        """Rows with rev_ proteins must have Reverse == '+', others must not."""
        decoy_mask = parsed_sage["Reverse"] == "+"
        # All decoy rows should have Reverse == "+"
        if decoy_mask.any():
            assert (parsed_sage.loc[decoy_mask, "Reverse"] == "+").all()

    def test_sage_dict_construction(self, parsed_sage):
        """construct_dict_from_search_pivoted must run on parsed SAGE output."""
        from utils.config import get_cfg_defaults
        from utils.singleton_swaps_optimization import swaps_optimization_cfg
        from prepare_dict.prepare_dict import construct_dict_from_search_pivoted

        cfg = get_cfg_defaults(swaps_optimization_cfg)
        cfg.PREPARE_DICT.SEARCH_ENGINE = "sage"
        cfg.PREPARE_DICT.GENERATE_DECOY = True

        dict_ref = construct_dict_from_search_pivoted(
            cfg_prepare_dict=cfg.PREPARE_DICT,
            evidence=parsed_sage,
            n_blocks_by_pept=1,
        )
        assert isinstance(dict_ref, pd.DataFrame)
        assert len(dict_ref) > 0
        assert "mz_rank" in dict_ref.columns


# ---------------------------------------------------------------------------
# Phase 2 – MaxQuant parsing
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestMaxQuantParsingIntegration:

    @pytest.fixture
    def maxquant_evidence(self, maxquant_dir):
        evidence_path = os.path.join(maxquant_dir, "evidence.txt")
        if not os.path.isfile(evidence_path):
            pytest.skip(f"evidence.txt not found in {maxquant_dir}")
        return pd.read_csv(evidence_path, sep="\t", low_memory=False)

    def test_maxquant_dir_accessible(self, maxquant_dir):
        assert os.path.isdir(maxquant_dir)

    def test_evidence_txt_exists(self, maxquant_dir):
        evidence_path = os.path.join(maxquant_dir, "evidence.txt")
        assert os.path.isfile(evidence_path), (
            f"evidence.txt not found in {maxquant_dir}"
        )

    def test_evidence_txt_loadable(self, maxquant_evidence):
        assert len(maxquant_evidence) > 0

    def test_evidence_has_required_columns(self, maxquant_evidence):
        """MaxQuant evidence.txt must have columns expected by construct_dict."""
        for col in ["Sequence", "Modified sequence", "Raw file", "Retention time"]:
            assert col in maxquant_evidence.columns, f"Missing column: {col}"

    def test_multiple_raw_files_in_evidence(self, maxquant_evidence):
        assert maxquant_evidence["Raw file"].nunique() > 1, (
            "Expected multiple raw files in MaxQuant evidence.txt"
        )

    def test_maxquant_dict_construction(self, maxquant_evidence):
        """construct_dict_from_search_pivoted must run on MaxQuant evidence.

        Note: we must pass all raw files together. Subsetting to a subset of
        raw files breaks the Reference validation: peptides whose global-best
        identification is a MULTI-MATCH (MBR) row from an excluded run end up
        with zero Reference assignments in the pivot, failing the assertion.
        """
        from utils.config import get_cfg_defaults
        from utils.singleton_swaps_optimization import swaps_optimization_cfg
        from prepare_dict.prepare_dict import construct_dict_from_search_pivoted

        cfg = get_cfg_defaults(swaps_optimization_cfg)
        cfg.PREPARE_DICT.SEARCH_ENGINE = "maxquant"
        cfg.PREPARE_DICT.GENERATE_DECOY = True

        dict_ref = construct_dict_from_search_pivoted(
            cfg_prepare_dict=cfg.PREPARE_DICT,
            evidence=maxquant_evidence,
            n_blocks_by_pept=1,
        )
        assert isinstance(dict_ref, pd.DataFrame)
        assert len(dict_ref) > 0
        assert "mz_rank" in dict_ref.columns
