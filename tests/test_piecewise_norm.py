"""
Unit tests for postprocessing.piecewise_norm.piecewise_normalize.

The algorithm (IonQuant-style piecewise normalization):
  1. Reference run  = column with the most non-NaN / non-zero ions.
  2. For every other run, compute log2-ratios of overlapping ions.
  3. Bin the overlapping log2-intensities into N equal-width ranges.
  4. Apply the median log2-ratio of each bin as a correction to all
     ions falling in that bin.
"""

import numpy as np
import pandas as pd
import pytest

from postprocessing.piecewise_norm import _fill_nan_bins, piecewise_normalize

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
RNG = np.random.default_rng(42)


def _make_df(**runs):
    """Build a DataFrame with one Intensity column per keyword argument."""
    return pd.DataFrame({f"{name} Intensity": vals for name, vals in runs.items()})


# ---------------------------------------------------------------------------
# _fill_nan_bins
# ---------------------------------------------------------------------------


class TestFillNanBins:
    def test_no_nans_unchanged(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = _fill_nan_bins(arr)
        np.testing.assert_array_equal(result, arr)

    def test_all_nans_returns_zeros(self):
        arr = np.full(4, np.nan)
        result = _fill_nan_bins(arr)
        assert np.all(result == 0.0)

    def test_interior_nan_interpolated(self):
        arr = np.array([0.0, np.nan, 2.0])
        result = _fill_nan_bins(arr)
        assert result[1] == pytest.approx(1.0)

    def test_leading_nan_clipped_to_first_valid(self):
        arr = np.array([np.nan, np.nan, 3.0, 6.0])
        result = _fill_nan_bins(arr)
        # np.interp clips to leftmost valid value for indices before it
        assert result[0] == pytest.approx(3.0)
        assert result[1] == pytest.approx(3.0)

    def test_trailing_nan_clipped_to_last_valid(self):
        arr = np.array([1.0, 2.0, np.nan])
        result = _fill_nan_bins(arr)
        assert result[2] == pytest.approx(2.0)

    def test_does_not_modify_input(self):
        arr = np.array([1.0, np.nan, 3.0])
        original = arr.copy()
        _fill_nan_bins(arr)
        np.testing.assert_array_equal(arr, original)


# ---------------------------------------------------------------------------
# piecewise_normalize – basic contract
# ---------------------------------------------------------------------------


class TestPiecewiseNormalizeContract:
    def test_raises_if_no_intensity_columns(self):
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        with pytest.raises(ValueError, match="Intensity"):
            piecewise_normalize(df)

    def test_returns_dataframe(self):
        df = _make_df(A=[100.0, 200.0, 300.0, 400.0, 500.0] * 3)
        result = piecewise_normalize(df)
        assert isinstance(result, pd.DataFrame)

    def test_does_not_modify_input(self):
        df = _make_df(
            A=[100.0, 200.0, 300.0, 400.0, 500.0] * 4,
            B=[80.0, 160.0, 240.0, 320.0, 400.0] * 4,
        )
        original_A = df["A Intensity"].copy()
        piecewise_normalize(df)
        pd.testing.assert_series_equal(df["A Intensity"], original_A)

    def test_output_same_columns_and_index(self):
        df = _make_df(
            A=list(range(10, 110, 10)),
            B=list(range(5, 105, 10)),
        )
        result = piecewise_normalize(df)
        assert list(result.columns) == list(df.columns)
        pd.testing.assert_index_equal(result.index, df.index)

    def test_non_intensity_columns_preserved(self):
        df = _make_df(A=list(range(10, 110, 10)))
        df["some_other_col"] = "kept"
        result = piecewise_normalize(df)
        assert "some_other_col" in result.columns
        assert (result["some_other_col"] == "kept").all()


# ---------------------------------------------------------------------------
# Reference run selection
# ---------------------------------------------------------------------------


class TestReferenceRunSelection:
    def test_reference_run_is_unchanged(self):
        """The run with the most ions must be returned untouched."""
        n = 50
        ref_vals = RNG.uniform(100, 1000, n).tolist()
        # run_B has fewer ions (10 NaNs)
        other_vals = ref_vals.copy()
        for i in range(10):
            other_vals[i] = np.nan

        df = _make_df(A=ref_vals, B=other_vals)
        result = piecewise_normalize(df)
        # A has more ions → reference; must be bit-for-bit identical
        pd.testing.assert_series_equal(result["A Intensity"], df["A Intensity"])

    def test_run_with_fewer_ions_is_the_nonreference(self):
        """If B has fewer ions, A is reference; B should be corrected."""
        n = 40
        ref_vals = RNG.uniform(100, 1000, n)
        other_vals = ref_vals * 2.0  # constant 2× offset

        df = _make_df(A=ref_vals.tolist(), B=other_vals.tolist())
        # Give A more ions by adding extra NaN to B's last entry
        df.loc[n - 1, "B Intensity"] = np.nan

        result = piecewise_normalize(df)
        # A unchanged
        pd.testing.assert_series_equal(result["A Intensity"], df["A Intensity"])
        # B changed
        assert not np.allclose(
            result["B Intensity"].dropna(), df["B Intensity"].dropna(), rtol=1e-3
        )


# ---------------------------------------------------------------------------
# Correction behaviour
# ---------------------------------------------------------------------------


class TestCorrectionBehaviour:
    def test_constant_offset_is_removed(self):
        """
        If run B = run A * constant_factor for all ions, after normalisation
        B should be approximately equal to A (the constant log-ratio is fully
        corrected in every bin).
        """
        n = 100
        ref_vals = RNG.uniform(100, 10_000, n)
        factor = 3.5
        other_vals = ref_vals * factor

        df = _make_df(A=ref_vals.tolist(), B=other_vals.tolist())
        result = piecewise_normalize(df)

        # After correction, B ≈ A for ions present in both
        np.testing.assert_allclose(
            result["B Intensity"].values, result["A Intensity"].values, rtol=0.05
        )

    def test_intensity_dependent_bias_is_reduced(self):
        """
        A bias that grows with intensity (e.g. multiplicative per-range factor)
        should be substantially reduced after piecewise normalization compared
        with simple global median normalization.
        """
        n = 200
        ref_vals = np.sort(RNG.uniform(100, 100_000, n))

        # Bias: low intensities are 2× high, high intensities are 0.5×
        bias = np.where(np.log2(ref_vals) < np.median(np.log2(ref_vals)), 2.0, 0.5)
        other_vals = ref_vals * bias

        df = _make_df(A=ref_vals.tolist(), B=other_vals.tolist())
        result = piecewise_normalize(df)

        # Residual log2-ratio after piecewise correction
        log_ratio_after = np.log2(result["B Intensity"].values) - np.log2(
            result["A Intensity"].values
        )
        # Global median correction residual (naive baseline)
        log_ratio_global = np.log2(other_vals) - np.log2(ref_vals)
        global_median = np.median(log_ratio_global)
        log_ratio_global_corrected = log_ratio_global - global_median

        # Piecewise should yield a smaller MAD than global median correction
        mad_piecewise = np.median(np.abs(log_ratio_after))
        mad_global = np.median(np.abs(log_ratio_global_corrected))
        assert mad_piecewise < mad_global

    def test_zeros_treated_as_missing(self):
        """Zeros in the input must be treated as unquantified (→ NaN output)."""
        df = _make_df(
            A=[0.0, 200.0, 300.0, 400.0, 500.0] * 10,
            B=[100.0, 200.0, 300.0, 400.0, 500.0] * 10,
        )
        result = piecewise_normalize(df)
        # Original zeros in A must remain NaN / zero after normalization
        zero_positions = df["A Intensity"] == 0.0
        assert result.loc[zero_positions, "A Intensity"].isna().all()

    def test_nan_values_remain_nan(self):
        """NaN ions in the input must be NaN in the output."""
        vals_A = [float("nan")] + list(range(10, 510, 10))
        vals_B = [100.0] + list(range(10, 510, 10))

        df = _make_df(A=vals_A, B=vals_B)
        result = piecewise_normalize(df)
        assert np.isnan(result["A Intensity"].iloc[0])

    def test_normalized_values_positive(self):
        """All non-NaN normalized intensities must be strictly positive."""
        n = 60
        ref_vals = RNG.uniform(100, 5000, n).tolist()
        other_vals = (RNG.uniform(100, 5000, n)).tolist()

        df = _make_df(A=ref_vals, B=other_vals)
        result = piecewise_normalize(df)
        for col in [c for c in result.columns if "Intensity" in c]:
            valid = result[col].dropna()
            assert (valid > 0).all(), f"{col} contains non-positive values"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_run_returns_unchanged(self):
        """With only one intensity column there is nothing to normalize."""
        df = _make_df(A=list(range(10, 110, 10)))
        result = piecewise_normalize(df)
        pd.testing.assert_series_equal(result["A Intensity"], df["A Intensity"])

    def test_too_few_overlapping_ions_skips_run(self):
        """
        If a run shares fewer than n_bins ions with the reference, it should
        be left unchanged (no correction applied).
        """
        n_bins = 10
        n = 50
        ref_vals = RNG.uniform(100, 1000, n).tolist()
        # Only 5 overlapping ions (below n_bins=10)
        sparse_vals = [np.nan] * (n - 5) + ref_vals[:5]

        df = _make_df(A=ref_vals, B=sparse_vals)
        result = piecewise_normalize(df, n_bins=n_bins)
        # B should be unchanged (no correction applied)
        pd.testing.assert_series_equal(
            result["B Intensity"], df["B Intensity"].replace(0.0, np.nan)
        )

    def test_three_runs_all_corrected_relative_to_reference(self):
        """Reference is unchanged; both other runs are normalised."""
        n = 80
        ref_vals = RNG.uniform(100, 5000, n)
        run_b = ref_vals * 2.0
        run_c = ref_vals * 0.5

        df = _make_df(A=ref_vals.tolist(), B=run_b.tolist(), C=run_c.tolist())
        result = piecewise_normalize(df)

        pd.testing.assert_series_equal(result["A Intensity"], df["A Intensity"])
        # B and C should be closer to A after normalization
        assert not np.allclose(result["B Intensity"].values, df["B Intensity"].values)
        assert not np.allclose(result["C Intensity"].values, df["C Intensity"].values)

    def test_custom_n_bins(self):
        """The function must accept an arbitrary n_bins value."""
        n = 100
        ref_vals = RNG.uniform(100, 5000, n).tolist()
        other_vals = (RNG.uniform(100, 5000, n)).tolist()

        df = _make_df(A=ref_vals, B=other_vals)
        # Should not raise for n_bins in [2, 20]
        for nb in [2, 5, 20]:
            result = piecewise_normalize(df.copy(), n_bins=nb)
            assert isinstance(result, pd.DataFrame)
