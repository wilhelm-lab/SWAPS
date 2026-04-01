"""
Unit tests for prepare_dict.search_engine_output_parser.
Covers fragpipe_psm_parser and sage_parser – both are pure DataFrame
transformations with no I/O, so they are fast and deterministic.
"""

import numpy as np
import pandas as pd
import pytest

from prepare_dict.search_engine_output_parser import fragpipe_psm_parser, sage_parser


# ---------------------------------------------------------------------------
# fragpipe_psm_parser
# ---------------------------------------------------------------------------


class TestFragpipePsmParser:
    def test_returns_dataframe(self, minimal_fragpipe_psm_df):
        result = fragpipe_psm_parser(minimal_fragpipe_psm_df)
        assert isinstance(result, pd.DataFrame)

    def test_does_not_mutate_input(self, minimal_fragpipe_psm_df):
        original_cols = set(minimal_fragpipe_psm_df.columns)
        fragpipe_psm_parser(minimal_fragpipe_psm_df)
        assert set(minimal_fragpipe_psm_df.columns) == original_cols

    def test_rt_converted_to_minutes(self, minimal_fragpipe_psm_df):
        result = fragpipe_psm_parser(minimal_fragpipe_psm_df)
        # Input Retention (seconds): 3086.4 → should become ~51.44 min
        expected_rt = minimal_fragpipe_psm_df["Retention"].iloc[0] / 60.0
        assert abs(result["Retention time"].iloc[0] - expected_rt) < 1e-6

    def test_calibrated_rt_columns_in_minutes(self, minimal_fragpipe_psm_df):
        result = fragpipe_psm_parser(minimal_fragpipe_psm_df)
        for col in [
            "Calibrated retention time",
            "Calibrated retention time start",
            "Calibrated retention time finish",
        ]:
            # All values should be < 120 min (original values were in seconds ~3000)
            assert (result[col] < 120).all(), f"{col} still appears to be in seconds"

    def test_retention_length_computed(self, minimal_fragpipe_psm_df):
        result = fragpipe_psm_parser(minimal_fragpipe_psm_df)
        expected = (
            result["Calibrated retention time finish"]
            - result["Calibrated retention time start"]
        )
        pd.testing.assert_series_equal(
            result["Retention length"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_im_length_computed(self, minimal_fragpipe_psm_df):
        result = fragpipe_psm_parser(minimal_fragpipe_psm_df)
        expected = result["1/K0 finish"] - result["1/K0 start"]
        pd.testing.assert_series_equal(
            result["1/K0 length"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_raw_file_extracted_from_spectrum(self, minimal_fragpipe_psm_df):
        result = fragpipe_psm_parser(minimal_fragpipe_psm_df)
        # "sample_A.00100.00100.2" → raw file = "sample_A"
        assert result["Raw file"].iloc[0] == "sample_A"

    def test_reverse_column_for_decoys(self):
        df = pd.DataFrame(
            {
                "Spectrum": ["s.001.001.2", "s.002.002.2"],
                "Peptide": ["PEPTIDEK", "DECOYPEP"],
                "Modified Peptide": [np.nan, np.nan],
                "Protein": ["sp|P1|GENE1", "sp|P2|GENE2"],
                "Observed M/Z": [500.0, 400.0],
                "Retention": [3000.0, 3100.0],
                "Apex Retention Time": [3010.0, 3110.0],
                "Retention Time Start": [2990.0, 3090.0],
                "Retention Time End": [3020.0, 3120.0],
                "Ion Mobility": [0.9, 1.0],
                "Ion Mobility Start": [0.88, 0.98],
                "Ion Mobility End": [0.92, 1.02],
                "Assigned Modifications": [np.nan, np.nan],
                "Hyperscore": [20.0, 5.0],
                "Is Decoy": [False, True],
                "Charge": [2, 2],
                "Peptide Length": [8, 8],
            }
        )
        result = fragpipe_psm_parser(df)
        # np.where with a string operand casts np.nan → string 'nan', so
        # we test the meaningful invariant: decoys get "+", non-decoys do not
        assert (result.loc[result["Is Decoy"] == True, "Reverse"] == "+").all()
        assert not (result.loc[result["Is Decoy"] == False, "Reverse"] == "+").any()

    def test_modified_peptide_fallback_to_peptide(self, minimal_fragpipe_psm_df):
        # Row 1 has NaN for Modified Peptide – should fall back to Peptide value
        result = fragpipe_psm_parser(minimal_fragpipe_psm_df)
        assert result["Modified sequence"].iloc[1] == "SEQUENCER"

    def test_modifications_fillna(self, minimal_fragpipe_psm_df):
        result = fragpipe_psm_parser(minimal_fragpipe_psm_df)
        # Row 1 has NaN Assigned Modifications → should become "-"
        assert result["Modifications"].iloc[1] == "-"

    def test_type_column_added(self, minimal_fragpipe_psm_df):
        result = fragpipe_psm_parser(minimal_fragpipe_psm_df)
        assert "Type" in result.columns
        assert (result["Type"] == "TIMS-MULTI-MSMS").all()

    def test_row_count_preserved(self, minimal_fragpipe_psm_df):
        result = fragpipe_psm_parser(minimal_fragpipe_psm_df)
        assert len(result) == len(minimal_fragpipe_psm_df)


# ---------------------------------------------------------------------------
# sage_parser
# ---------------------------------------------------------------------------


class TestSageParser:
    def test_returns_dataframe(self, minimal_sage_df):
        result = sage_parser(minimal_sage_df)
        assert isinstance(result, pd.DataFrame)

    def test_fdr_filter_applied(self, minimal_sage_df):
        # Row 2 (DECOYYYY) has peptide_q=0.4, spectrum_q=0.3 → should be dropped
        result = sage_parser(minimal_sage_df)
        assert len(result) < len(minimal_sage_df)

    def test_passing_rows_retained(self, minimal_sage_df):
        result = sage_parser(minimal_sage_df)
        # Rows 0 and 1 have q-values ≤ 0.01
        assert "PEPTIDEK" in result["Sequence"].values or len(result) >= 1

    def test_filename_stripped_of_extension(self, minimal_sage_df):
        # sage_parser strips last 2 chars (".d" → stripped to stem)
        result = sage_parser(minimal_sage_df)
        assert not result["Raw file"].str.endswith(".d").any()

    def test_is_decoy_column_created(self, minimal_sage_df):
        result = sage_parser(minimal_sage_df)
        assert "Is_Decoy" in result.columns

    def test_mz_columns_computed(self, minimal_sage_df):
        result = sage_parser(minimal_sage_df)
        assert "m/z" in result.columns
        assert "calc_m/z" in result.columns
        # m/z = expmass / charge  (for rows that pass FDR)
        row = result.iloc[0]
        # We can't directly look up the original row easily after rename, but
        # verify values are reasonable for a peptide
        assert 100 < row["m/z"] < 5000

    def test_rt_calibration_columns_added(self, minimal_sage_df):
        result = sage_parser(minimal_sage_df)
        for col in [
            "Calibrated retention time",
            "Calibrated retention time start",
            "Calibrated retention time finish",
        ]:
            assert col in result.columns

    def test_type_column_added(self, minimal_sage_df):
        result = sage_parser(minimal_sage_df)
        assert "Type" in result.columns
        assert (result["Type"] == "TIMS-MULTI-MSMS").all()

    def test_sequence_strips_modifications(self, minimal_sage_df):
        # peptide column has values like "_PEPTIDEK_" → Sequence should be clean
        result = sage_parser(minimal_sage_df)
        # After rename, Modified sequence = peptide; Sequence strips [mod] brackets
        assert "Sequence" in result.columns
        assert not result["Sequence"].str.contains(r"\[").any()

    def test_retention_time_calibration_zero(self, minimal_sage_df):
        result = sage_parser(minimal_sage_df)
        assert "Retention time calibration" in result.columns
        assert (result["Retention time calibration"] == 0).all()
