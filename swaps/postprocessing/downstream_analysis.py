"""
Differential-expression analysis of directLFQ protein intensities across
brain regions.

Re-implements the normalization -> imputation -> ANOVA/pairwise-t-test ->
fold-change-threshold-simulation -> region-enrichment-classification
(Sjostedt et al, 2020 scheme) protocol described for a published 13-region
brain proteome atlas, applied to directLFQ protein quantification output
(from either the SWAPS or FragPipe search pipelines).

Reference
---------
Methods section of the TUM brain-region-atlas paper (13 FFPE brain regions +
fresh tissue, MassIVE accession MSV000091835).
"""

import itertools
import logging
import os
import warnings
from dataclasses import dataclass

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.integrate import cumulative_trapezoid
from scipy.special import digamma, polygamma
from statsmodels.stats.multitest import multipletests

from swaps.postprocessing.helper import (
    SAMPLE_TAG_PATTERN as _SAMPLE_TAG_PATTERN,
    REGION_REPLICATE_PATTERN as _REGION_REPLICATE_PATTERN,
)

Logger = logging.getLogger(__name__)


def load_protein_intensities(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load a directLFQ ``*.protein_intensities.tsv`` table and parse region/
    replicate sample metadata from its column names.

    Sample columns are expected to encode the sample tag between ``P{6 digits}_``
    and ``_5ug`` (e.g. ``..._P064051_Fresh2_5ug_..._Intensity``), with the tag
    itself starting with letters (the region) followed by a replicate number.
    Any column that doesn't match this pattern (other than the protein ID
    column itself) is dropped with a warning -- some directLFQ exports (e.g.
    from FragPipe input) carry a stray duplicate ID column.

    Parameters
    ----------
    path : str
        Path to a ``*.protein_intensities.tsv`` file.

    Returns
    -------
    intensities : pd.DataFrame
        Proteins (index) x samples (columns, renamed to ``<region><replicate>``
        tags). Zero values are treated as missing (NaN).
    sample_info : pd.DataFrame
        Indexed by the same sample tags, with ``region`` and ``replicate`` columns.
    """
    df = pd.read_csv(path, sep="\t")

    protein_id_cols = [c for c in df.columns if c.lower() == "protein"]
    if not protein_id_cols:
        raise ValueError(f"No 'protein' ID column found in {path!r}")
    df = df.set_index(protein_id_cols[0])

    rename_map = {}
    sample_info_rows = []
    dropped_cols = []
    for col in df.columns:
        if col in protein_id_cols:
            dropped_cols.append(col)
            continue
        tag_match = _SAMPLE_TAG_PATTERN.search(col)
        if tag_match is None:
            dropped_cols.append(col)
            continue
        region_match = _REGION_REPLICATE_PATTERN.match(tag_match.group("tag"))
        if region_match is None:
            raise ValueError(
                f"Could not parse region/replicate from tag: {tag_match.group('tag')!r}"
            )

        region = region_match.group("region")
        replicate = int(region_match.group("replicate"))
        sample = f"{region}{replicate}"
        rename_map[col] = sample
        sample_info_rows.append(
            {"sample": sample, "region": region, "replicate": replicate}
        )

    if dropped_cols:
        Logger.warning(
            "Dropping columns that aren't sample-tagged intensities: %s", dropped_cols
        )

    intensities = (
        df.drop(columns=dropped_cols).rename(columns=rename_map).replace(0.0, np.nan)
    )
    sample_info = pd.DataFrame(sample_info_rows).set_index("sample")
    Logger.info(
        "Loaded %d proteins x %d samples across %d regions from %s",
        intensities.shape[0],
        intensities.shape[1],
        sample_info["region"].nunique(),
        path,
    )
    return intensities, sample_info


def normalize_median_ratio(
    intensities: pd.DataFrame, min_presence_frac: float = 0.7
) -> pd.DataFrame:
    """
    Sample-wise median-ratio normalization, anchored on proteins detected in
    at least ``min_presence_frac`` of all samples.

    Each sample's median is computed over that stable protein subset only;
    every protein in the sample is then rescaled by
    ``mean(all sample medians) / that sample's median``.

    Parameters
    ----------
    intensities : pd.DataFrame
        Proteins x samples, linear-scale intensities (missing values as NaN).
    min_presence_frac : float
        Minimum fraction of samples a protein must be detected in to be used
        for computing the per-sample medians.

    Returns
    -------
    pd.DataFrame
        Normalized intensities, same shape as input.
    """
    presence_frac = intensities.notna().mean(axis=1)
    stable_proteins = intensities.loc[presence_frac >= min_presence_frac]
    sample_medians = stable_proteins.median(axis=0, skipna=True)
    correction_factor = sample_medians.mean() / sample_medians
    return intensities.multiply(correction_factor, axis=1)


def log2_transform(intensities: pd.DataFrame) -> pd.DataFrame:
    """Log2-transform normalized intensities (NaN stays NaN)."""
    return intensities.apply(np.log2)


def impute_downshifted_normal(
    log_intensities: pd.DataFrame,
    width: float = 0.3,
    downshift: float = 1.8,
    random_state: int | None = None,
) -> pd.DataFrame:
    """
    Perseus-style left-censored imputation of missing log-intensities.

    For each sample (column), missing values are drawn from
    ``N(obs_mean - downshift * obs_std, width * obs_std)``, where ``obs_mean``
    and ``obs_std`` are computed from that sample's observed values.

    Parameters
    ----------
    log_intensities : pd.DataFrame
        Proteins x samples, log2-scale intensities with NaN for missing values.
    width : float
        Standard deviation of the imputed distribution, as a fraction of the
        observed standard deviation.
    downshift : float
        Downshift of the imputed mean from the observed mean, in units of the
        observed standard deviation.
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Fully imputed log-intensities, same shape as input.
    """
    rng = np.random.default_rng(random_state)
    imputed = log_intensities.copy()
    for sample in imputed.columns:
        missing = imputed[sample].isna()
        if not missing.any():
            continue
        obs_mean = imputed[sample].mean(skipna=True)
        obs_std = imputed[sample].std(skipna=True)
        imputed.loc[missing, sample] = rng.normal(
            loc=obs_mean - downshift * obs_std,
            scale=width * obs_std,
            size=int(missing.sum()),
        )
    return imputed


def anova_across_groups(
    values: pd.DataFrame,
    sample_info: pd.DataFrame,
    group_col: str = "region",
    presence_mask: pd.DataFrame | None = None,
    min_valid_per_group: int = 2,
) -> pd.DataFrame:
    """
    Per-protein one-way ANOVA across groups (e.g. brain regions).

    Parameters
    ----------
    values : pd.DataFrame
        Proteins x samples. Fully imputed, or raw with missing values left as
        NaN (group stats are computed NaN-aware either way).
    sample_info : pd.DataFrame
        Indexed by sample, with a ``group_col`` column assigning each sample
        to a group.
    group_col : str
        Column in ``sample_info`` to group samples by.
    presence_mask : pd.DataFrame, optional
        Proteins x samples boolean mask of real (non-imputed) detections. If
        given, a group is only included in a protein's ANOVA if it has at
        least ``min_valid_per_group`` real values -- this is the
        presence/validity filter, applied regardless of whether ``values``
        was imputed, so imputed-but-mostly-fake groups don't contribute a
        spuriously tight variance estimate. Pass ``None`` to disable.
    min_valid_per_group : int
        Minimum real-value count per group required for that group to be
        included (only used when ``presence_mask`` is given).

    Returns
    -------
    pd.DataFrame
        Indexed by protein, with ``f_stat``, ``anova_p``, and BH-adjusted
        ``anova_padj``. Proteins with fewer than 2 valid groups (or fewer
        than 1 within-group degree of freedom) get NaN -- they have
        insufficient real data to test.
    """
    group_stats = _group_stats(
        values,
        sample_info,
        group_col,
        presence_mask=presence_mask,
        min_valid_per_group=min_valid_per_group,
    )
    groups = list(group_stats)
    n_arr = np.stack([group_stats[g]["n"] for g in groups]).astype(float)
    mean_arr = np.stack([group_stats[g]["mean"] for g in groups])
    var_arr = np.stack([group_stats[g]["var"] for g in groups])

    # A group needs >= 2 real/effective values to contribute a variance estimate.
    valid_group = n_arr >= 2
    n_eff = np.where(valid_group, n_arr, 0.0)
    mean_eff = np.where(valid_group, mean_arr, 0.0)
    var_eff = np.where(valid_group, var_arr, 0.0)

    n_total = n_eff.sum(axis=0)
    k_valid = valid_group.sum(axis=0)

    with np.errstate(invalid="ignore", divide="ignore"):
        grand_mean = (mean_eff * n_eff).sum(axis=0) / n_total
        ss_between = (n_eff * (mean_eff - grand_mean) ** 2).sum(axis=0)
        ss_within = (var_eff * (n_eff - 1)).sum(axis=0)

        df_between, df_within = k_valid - 1, n_total - k_valid
        f_stat = (ss_between / df_between) / (ss_within / df_within)
        p_value = stats.f.sf(f_stat, df_between, df_within)

    insufficient = (df_between < 1) | (df_within < 1)
    f_stat = np.where(insufficient, np.nan, f_stat)
    p_value = np.where(insufficient, np.nan, p_value)

    result = pd.DataFrame({"f_stat": f_stat, "anova_p": p_value}, index=values.index)
    result["anova_padj"] = _bh_correct_with_nan(result["anova_p"].to_numpy())
    return result


def pairwise_ttests_across_groups(
    values: pd.DataFrame,
    sample_info: pd.DataFrame,
    group_col: str = "region",
    presence_mask: pd.DataFrame | None = None,
    min_valid_per_group: int = 2,
    moderate: bool = True,
) -> pd.DataFrame:
    """
    Per-protein Welch's pairwise t-test between every pair of groups.

    Parameters
    ----------
    values : pd.DataFrame
        Proteins x samples. Fully imputed, or raw with missing values left as
        NaN (group stats are computed NaN-aware either way).
    sample_info : pd.DataFrame
        Indexed by sample, with a ``group_col`` column assigning each sample
        to a group.
    group_col : str
        Column in ``sample_info`` to group samples by.
    presence_mask : pd.DataFrame, optional
        Proteins x samples boolean mask of real (non-imputed) detections. If
        given, a (protein, group) combination is only used in a pairwise test
        if it has at least ``min_valid_per_group`` real values -- the
        presence/validity filter. A protein failing this for one group pair
        is simply skipped (NaN row) *for that pair*; it remains eligible for
        any other pair where both groups still have enough real data. Pass
        ``None`` to disable.
    min_valid_per_group : int
        Minimum real-value count per group required for a pair to be tested
        (only used when ``presence_mask`` is given).
    moderate : bool
        If True, apply limma-style empirical-Bayes moderation (Smyth 2004) to
        the per-pair variance estimates before computing t-statistics: each
        protein's Welch variance is shrunk towards a common prior fitted
        across all proteins for that same group pair, and its degrees of
        freedom increased accordingly. This stabilizes the test at low
        per-group replicate counts (n=2-3), where a per-pair-only variance
        estimate is otherwise extremely noisy and prone to false positives
        from chance variance underestimation -- exactly the effect ANOVA is
        protected from by pooling variance across all groups. No external
        limma/inmoose package is required; the shrinkage is implemented
        directly (``_squeeze_variance``/``_trigamma_inverse``).

    Returns
    -------
    pd.DataFrame
        Long-format, one row per protein x group pair, with ``log2fc``
        (group_a mean - group_b mean), ``t_stat``, ``p_value``, and BH-adjusted
        ``padj``. ``padj`` is computed *within* each group pair (i.e. across
        proteins, separately per pairwise comparison) -- the standard
        per-contrast FDR, matching "filter the proteins for each brain region"
        in the brain-atlas protocol rather than one global correction across
        all pairs (which would be needlessly conservative).
    """
    group_stats = _group_stats(
        values,
        sample_info,
        group_col,
        presence_mask=presence_mask,
        min_valid_per_group=min_valid_per_group,
    )
    groups = sorted(group_stats)

    records = []
    for group_a, group_b in itertools.combinations(groups, 2):
        a, b = group_stats[group_a], group_stats[group_b]
        valid = (a["n"] >= 2) & (b["n"] >= 2)
        se_sq, welch_df = _welch_se_and_df(a, b)
        se_sq = np.where(valid, se_sq, np.nan)
        welch_df = np.where(valid, welch_df, np.nan)

        if moderate:
            se_sq_used, df_used, _, _ = _squeeze_variance(se_sq, welch_df)
        else:
            se_sq_used, df_used = se_sq, welch_df

        with np.errstate(invalid="ignore", divide="ignore"):
            t_stat = (a["mean"] - b["mean"]) / np.sqrt(se_sq_used)
            p_value = stats.t.sf(np.abs(t_stat), df_used) * 2

        log2fc = np.where(valid, a["mean"] - b["mean"], np.nan)
        t_stat = np.where(valid, t_stat, np.nan)
        p_value = np.where(valid, p_value, np.nan)

        records.append(
            pd.DataFrame(
                {
                    "protein": values.index,
                    "group_a": group_a,
                    "group_b": group_b,
                    "log2fc": log2fc,
                    "t_stat": t_stat,
                    "p_value": p_value,
                    "padj": _bh_correct_with_nan(p_value),
                }
            )
        )

    return pd.concat(records, ignore_index=True)


def estimate_worst_case_cv(
    values: pd.DataFrame,
    sample_info: pd.DataFrame,
    group_col: str = "region",
    percentile: float = 99.0,
) -> float:
    """
    Estimate the worst-case coefficient of variation (CV) across groups.

    For every group, the CV (``std / mean``, ddof=1) is computed per protein
    on complete (non-missing) replicate sets, on linear-scale intensities.
    The ``percentile``-th value across all groups/proteins is returned. The
    brain-atlas protocol used the literal max; in practice that is usually
    dominated by a single near-noise-floor protein and destabilizes the
    downstream simulation (large fraction of negative replicate-mean draws),
    so the default here is the 99th percentile -- a robust stand-in for
    "worst realistic case". Pass ``percentile=100`` to reproduce the literal
    max.

    Parameters
    ----------
    values : pd.DataFrame
        Proteins x samples, linear-scale (normalized, non-imputed) intensities.
    sample_info : pd.DataFrame
        Indexed by sample, with a ``group_col`` column assigning each sample
        to a group.
    group_col : str
        Column in ``sample_info`` to group samples by.
    percentile : float
        Percentile of the observed CV distribution to use as "worst case".

    Returns
    -------
    float
        The estimated worst-case CV.
    """
    cvs = []
    for _, group_samples in sample_info.groupby(group_col):
        cols = values.columns.intersection(group_samples.index)
        complete = values[cols].dropna(how="any")
        if complete.empty:
            continue
        cvs.append(complete.std(axis=1, ddof=1) / complete.mean(axis=1))
    return float(np.percentile(pd.concat(cvs), percentile))


@dataclass
class FoldChangeThreshold:
    """Simulated null fold-change distribution and derived CI threshold."""

    n_reps: int
    cv: float
    log2fc_samples: np.ndarray
    pdf_x: np.ndarray
    pdf: np.ndarray
    cdf: np.ndarray
    ci: float
    log2fc_ci: tuple[float, float]
    fold_change_ci: tuple[float, float]


def simulate_fold_change_threshold(
    cv: float,
    n_reps: int = 3,
    mu_mean: float = 1000.0,
    mu_std: float = 200.0,
    n_simulations: int = 10_000,
    ci: float = 0.95,
    n_bins: int = 200,
    random_state: int | None = None,
) -> FoldChangeThreshold:
    """
    Simulate the null (no true difference) fold-change distribution between two
    groups of ``n_reps`` replicates at a given CV, and derive a two-tailed CI
    fold-change threshold.

    Mirrors the brain-atlas protocol: per simulated protein, a true mean is
    drawn from ``N(mu_mean, mu_std)``, replicate intensities for each of the
    two groups are drawn from ``N(mu, mu * cv)``, and the fold change is the
    ratio of the two groups' replicate means. The resulting distribution
    (an uncorrelated noncentral normal ratio, Hinkley 1969) is histogrammed
    and numerically integrated (trapezoidal rule) to obtain a CDF, from which
    the two-tailed ``ci`` confidence interval is read off.

    Parameters
    ----------
    cv : float
        Worst-case coefficient of variation to simulate under (see
        ``estimate_worst_case_cv``).
    n_reps : int
        Number of replicates per group.
    mu_mean, mu_std : float
        Mean/std of the simulated "true" protein intensity distribution.
    n_simulations : int
        Number of simulated proteins.
    ci : float
        Two-tailed confidence level for the derived threshold.
    n_bins : int
        Number of histogram bins used to approximate the density.
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    FoldChangeThreshold
    """
    rng = np.random.default_rng(random_state)
    mu = rng.normal(mu_mean, mu_std, size=n_simulations)
    sigma = mu * cv
    group_a = rng.normal(
        mu[:, None], sigma[:, None], size=(n_simulations, n_reps)
    ).mean(axis=1)
    group_b = rng.normal(
        mu[:, None], sigma[:, None], size=(n_simulations, n_reps)
    ).mean(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        log2fc_samples = np.log2(group_a / group_b)

    valid = np.isfinite(log2fc_samples)
    dropped_frac = 1 - valid.mean()
    if dropped_frac > 0.01:
        Logger.warning(
            "Dropped %.1f%% of simulated proteins with non-positive replicate means "
            "(cv=%.2f, n_reps=%d); consider a lower cv_percentile.",
            dropped_frac * 100,
            cv,
            n_reps,
        )
    log2fc_samples = log2fc_samples[valid]

    pdf, edges = np.histogram(log2fc_samples, bins=n_bins, density=True)
    pdf_x = (edges[:-1] + edges[1:]) / 2
    cdf = cumulative_trapezoid(pdf, pdf_x, initial=0)
    cdf /= cdf[-1]

    tail = (1 - ci) / 2
    log2fc_lo = float(np.interp(tail, cdf, pdf_x))
    log2fc_hi = float(np.interp(1 - tail, cdf, pdf_x))

    return FoldChangeThreshold(
        n_reps=n_reps,
        cv=cv,
        log2fc_samples=log2fc_samples,
        pdf_x=pdf_x,
        pdf=pdf,
        cdf=cdf,
        ci=ci,
        log2fc_ci=(log2fc_lo, log2fc_hi),
        fold_change_ci=(2**log2fc_lo, 2**log2fc_hi),
    )


def classify_region_enrichment(
    log_intensities: pd.DataFrame,
    sample_info: pd.DataFrame,
    presence_mask: pd.DataFrame,
    group_col: str = "region",
    fold_change_log2_threshold: float = 2.0,
    exclusive_min_own_replicates: int = 2,
    exclusive_max_other_replicates: int = 1,
    group_size_range: tuple[int, int] = (2, 7),
) -> pd.DataFrame:
    """
    Classify per-protein region specificity (Sjostedt et al, 2020 / Human
    Protein Atlas scheme, as adopted by the brain-region-atlas protocol).

    - Class 1 "region-exclusive": detected in >= ``exclusive_min_own_replicates``
      replicates of one region, and in <= ``exclusive_max_other_replicates``
      replicates summed across all other regions.
    - Class 2 "region-enriched": >= ``fold_change_log2_threshold`` log2-fold
      higher than *every* other region (pairwise).
    - Class 3 "group-enriched": >= ``fold_change_log2_threshold`` log2-fold
      higher in a group of ``group_size_range`` regions vs. the rest (the
      largest such group, if any).
    - Class 4 "region-enhanced": >= ``fold_change_log2_threshold`` log2-fold
      higher than the *average* of all other regions.

    Each protein is assigned the most specific class it qualifies for
    (1 > 2 > 3 > 4); proteins qualifying for none are left unclassified.

    Parameters
    ----------
    log_intensities : pd.DataFrame
        Proteins x samples, log2-scale intensities (used for the
        fold-change-based Classes 2-4). May be fully imputed or raw with
        missing values left as NaN -- group means are computed NaN-aware.
    sample_info : pd.DataFrame
        Indexed by sample, with a ``group_col`` column.
    presence_mask : pd.DataFrame
        Proteins x samples boolean mask of detection *before* imputation
        (used for the presence-based Class 1).
    group_col : str
        Column in ``sample_info`` to group samples by.
    fold_change_log2_threshold : float
        log2 fold-change cutoff for Classes 2-4 (default 2.0 = 4-fold, the
        brain-atlas protocol's adopted criterion).
    exclusive_min_own_replicates, exclusive_max_other_replicates : int
        Class 1 presence thresholds.
    group_size_range : tuple[int, int]
        Inclusive (min, max) group size considered for Class 3.

    Returns
    -------
    pd.DataFrame
        Indexed by protein, with ``enrichment_class`` (1-4, or NaN if
        unclassified) and ``regions`` (comma-joined region name(s); a single
        region for Classes 1/2/4, multiple for Class 3).
    """
    mean_stats = _group_stats(log_intensities, sample_info, group_col)
    groups = sorted(mean_stats)
    means = np.column_stack([mean_stats[g]["mean"] for g in groups])

    detected_count = {
        group: presence_mask[presence_mask.columns.intersection(group_samples.index)]
        .sum(axis=1)
        .to_numpy()
        for group, group_samples in sample_info.groupby(group_col)
    }

    n_proteins = means.shape[0]
    enrichment_class = np.full(n_proteins, np.nan)
    regions_out: list[list | None] = [None] * n_proteins

    def assign(mask: np.ndarray, class_id: int, region_names: list[str]) -> None:
        for protein_idx in np.where(mask & np.isnan(enrichment_class))[0]:
            enrichment_class[protein_idx] = class_id
            regions_out[protein_idx] = region_names

    # Class 1: region-exclusive (presence/absence based)
    total_detected = sum(detected_count.values())
    for region in groups:
        own, other = detected_count[region], total_detected - detected_count[region]
        mask = (own >= exclusive_min_own_replicates) & (
            other <= exclusive_max_other_replicates
        )
        assign(mask, 1, [region])

    # Class 2: region-enriched (beats every other region, pairwise)
    for i, region in enumerate(groups):
        margin = means[:, i : i + 1] - np.delete(means, i, axis=1)
        assign(margin.min(axis=1) >= fold_change_log2_threshold, 2, [region])

    # Class 3: group-enriched (largest group of group_size_range regions that
    # clears every region outside the group by the threshold)
    order = np.argsort(-means, axis=1)
    sorted_means = np.take_along_axis(means, order, axis=1)
    lo, hi = group_size_range
    for k in range(hi, lo - 1, -1):
        gap = sorted_means[:, k - 1] - sorted_means[:, k]
        for protein_idx in np.where(
            (gap >= fold_change_log2_threshold) & np.isnan(enrichment_class)
        )[0]:
            enrichment_class[protein_idx] = 3
            regions_out[protein_idx] = [groups[g] for g in order[protein_idx, :k]]

    # Class 4: region-enhanced (beats the average of all other regions)
    for i, region in enumerate(groups):
        other_mean = (means.sum(axis=1) - means[:, i]) / (len(groups) - 1)
        assign(means[:, i] - other_mean >= fold_change_log2_threshold, 4, [region])

    return pd.DataFrame(
        {
            "enrichment_class": enrichment_class,
            "regions": [",".join(r) if r else None for r in regions_out],
        },
        index=log_intensities.index,
    )


@dataclass
class RegionDEAResult:
    """Outputs of :func:`run_region_differential_expression`."""

    raw_intensities: pd.DataFrame
    normalized_intensities: pd.DataFrame
    log_intensities: pd.DataFrame
    imputed_intensities: pd.DataFrame
    sample_info: pd.DataFrame
    anova_results: pd.DataFrame
    pairwise_results: pd.DataFrame
    worst_case_cv: float
    fold_change_thresholds: dict[int, FoldChangeThreshold]
    enrichment_results: pd.DataFrame
    impute: bool


def run_region_differential_expression(
    protein_intensities_path: str,
    group_col: str = "region",
    min_presence_frac: float = 0.7,
    impute: bool = True,
    impute_width: float = 0.3,
    impute_downshift: float = 1.8,
    min_valid_per_group: int = 2,
    moderate_variance: bool = True,
    cv_percentile: float = 99.0,
    enrichment_fold_change_log2_threshold: float = 2.0,
    random_state: int | None = 0,
) -> RegionDEAResult:
    """
    Run the brain-atlas DEA protocol on a directLFQ protein-intensity table:
    normalization -> log2 -> (optional) imputation -> ANOVA + pairwise
    t-tests -> fold-change-threshold simulation -> region-enrichment
    classification.

    Parameters
    ----------
    protein_intensities_path : str
        Path to a directLFQ ``*.protein_intensities.tsv`` file.
    group_col : str
        Column in the parsed sample metadata to group samples by (default:
        ``"region"``, parsed from the sample tag).
    min_presence_frac : float
        Minimum per-protein presence fraction for the normalization anchor set.
    impute : bool
        If True (default, matches the brain-atlas protocol), missing values
        are filled via downshifted-normal imputation before testing. If
        False, no imputation is performed; ANOVA/pairwise tests are computed
        directly on the real values only (NaN-aware), and any (protein,
        group) combination with fewer than ``min_valid_per_group`` real
        values is excluded from tests involving that group (see
        ``min_valid_per_group``) rather than synthesized.
    impute_width, impute_downshift : float
        Downshifted-normal imputation parameters (only used if ``impute``).
    min_valid_per_group : int
        Presence/validity filter: a group is only used for a given protein's
        ANOVA/pairwise test if it has at least this many *real* (non-imputed)
        values. Applied regardless of ``impute`` -- with imputation on, this
        still excludes groups whose real-value count is too low to trust,
        even though imputed values exist to fill them; with imputation off,
        it determines which (protein, group-pair) tests are skipped.
    moderate_variance : bool
        If True (default), apply limma-style empirical-Bayes variance
        moderation (Smyth 2004) to the pairwise t-tests, shrinking each
        protein's per-pair variance toward a common prior fitted across all
        proteins for that pair. Stabilizes pairwise testing at low (n=2-3)
        replicate counts, where per-pair-only variance estimates are
        otherwise highly unstable (unlike ANOVA, which pools variance across
        all groups). No moderation is applied to the ANOVA, which is already
        comparatively robust due to that pooling.
    cv_percentile : float
        Percentile used to derive the "worst case" CV for the fold-change
        threshold simulation (100 = true max).
    enrichment_fold_change_log2_threshold : float
        log2 fold-change cutoff used by the region-enrichment classification
        (default 2.0 = 4-fold).
    random_state : int, optional
        Seed for imputation and fold-change simulation.

    Returns
    -------
    RegionDEAResult
    """
    raw_intensities, sample_info = load_protein_intensities(protein_intensities_path)
    normalized = normalize_median_ratio(
        raw_intensities, min_presence_frac=min_presence_frac
    )
    log_intensities = log2_transform(normalized)
    presence_mask = normalized.notna()

    if impute:
        test_values = impute_downshifted_normal(
            log_intensities,
            width=impute_width,
            downshift=impute_downshift,
            random_state=random_state,
        )
    else:
        test_values = log_intensities

    anova_results = anova_across_groups(
        test_values,
        sample_info,
        group_col=group_col,
        presence_mask=presence_mask,
        min_valid_per_group=min_valid_per_group,
    )
    pairwise_results = pairwise_ttests_across_groups(
        test_values,
        sample_info,
        group_col=group_col,
        presence_mask=presence_mask,
        min_valid_per_group=min_valid_per_group,
        moderate=moderate_variance,
    )

    worst_case_cv = estimate_worst_case_cv(
        normalized, sample_info, group_col=group_col, percentile=cv_percentile
    )
    fold_change_thresholds = {
        int(n_reps): simulate_fold_change_threshold(
            cv=worst_case_cv, n_reps=int(n_reps), random_state=random_state
        )
        for n_reps in sorted(sample_info[group_col].value_counts().unique())
    }

    enrichment_results = classify_region_enrichment(
        test_values,
        sample_info,
        presence_mask=presence_mask,
        group_col=group_col,
        fold_change_log2_threshold=enrichment_fold_change_log2_threshold,
    )

    return RegionDEAResult(
        raw_intensities=raw_intensities,
        normalized_intensities=normalized,
        log_intensities=log_intensities,
        imputed_intensities=test_values,
        sample_info=sample_info,
        anova_results=anova_results,
        pairwise_results=pairwise_results,
        worst_case_cv=worst_case_cv,
        fold_change_thresholds=fold_change_thresholds,
        enrichment_results=enrichment_results,
        impute=impute,
    )


def plot_fold_change_null_distribution(threshold: FoldChangeThreshold, ax=None):
    """
    Plot the simulated null log2-fold-change density and CDF with the derived
    CI bounds marked (analogous to the brain-atlas Appendix Fig S1D-F).

    Parameters
    ----------
    threshold : FoldChangeThreshold
        Result of :func:`simulate_fold_change_threshold`.
    ax : matplotlib.axes.Axes, optional
        Axes to plot the density on; a twin axis is added for the CDF.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    ax.plot(threshold.pdf_x, threshold.pdf, color="black", label="simulated density")
    ax_cdf = ax.twinx()
    ax_cdf.plot(
        threshold.pdf_x, threshold.cdf, color="tab:blue", alpha=0.6, label="CDF"
    )

    for bound in threshold.log2fc_ci:
        ax.axvline(bound, color="tab:red", linestyle="--", linewidth=1)

    ax.set_xlabel("log2 fold change (null)")
    ax.set_ylabel("density")
    ax_cdf.set_ylabel("CDF")
    ax.set_title(
        f"n_reps={threshold.n_reps}, CV={threshold.cv:.2f}, "
        f"{int(threshold.ci * 100)}% CI log2FC=({threshold.log2fc_ci[0]:.2f}, {threshold.log2fc_ci[1]:.2f})"
    )
    return ax


def paired_pearson_correlation(
    df_a: pd.DataFrame, df_b: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pairwise Pearson correlation between every shared-name column of ``df_a``
    and every shared-name column of ``df_b`` (e.g. two protein-intensity
    tables with matching sample tags), using only the proteins non-missing
    in both columns of each pair.

    Parameters
    ----------
    df_a, df_b : pd.DataFrame
        Proteins (index) x samples (columns). Columns are matched by name;
        columns unique to either frame are ignored.

    Returns
    -------
    corr : pd.DataFrame
        Shared columns x shared columns Pearson r.
    n_used : pd.DataFrame
        Same shape, number of proteins used for each pairwise correlation.
    """
    joint_df = pd.merge(
        df_a,
        df_b,
        left_index=True,
        right_index=True,
        how="inner",
        suffixes=("_swaps", "_ionquant"),
    )
    shared_cols = df_a.columns.intersection(df_b.columns)
    if shared_cols.empty:
        raise ValueError("df_a and df_b share no column names")
    corr = {}
    n_used = {}
    for col in shared_cols:
        pair = joint_df[[f"{col}_swaps", f"{col}_ionquant"]].dropna()
        if pair.empty:
            Logger.warning(
                "No shared non-missing proteins for column %s; correlation will be NaN",
                col,
            )
        else:
            r = pair.iloc[:, 0].corr(pair.iloc[:, 1])
            Logger.info(
                "Pearson r between %s and %s: %.3f (n=%d)",
                col,
                col,
                r,
                len(pair),
            )
            corr[col] = r
            n_used[col] = len(pair)
    result_df = pd.DataFrame(
        {"pearson_r": pd.Series(corr), "n_used": pd.Series(n_used)}
    )
    return result_df


def plot_paired_correlation_heatmap(
    corr: pd.DataFrame, n_used: pd.DataFrame, ax=None, title: str | None = None
):
    """
    Heatmap of a :func:`paired_pearson_correlation` result, cells annotated
    with the protein count behind each correlation.
    """
    if ax is None:
        _, ax = plt.subplots(
            figsize=(0.5 * len(corr.columns) + 3, 0.5 * len(corr.index) + 3)
        )
    sns.heatmap(
        corr,
        annot=n_used.astype(int),
        fmt="d",
        cmap="viridis",
        vmin=-1,
        vmax=1,
        ax=ax,
        cbar_kws={"label": "Pearson r"},
    )
    ax.set_title(title or "Pairwise Pearson correlation (annotated with protein count)")
    return ax


def compute_pairwise_de_overlap(
    pairwise_a: pd.DataFrame,
    pairwise_b: pd.DataFrame,
    padj_threshold: float = 0.05,
) -> pd.DataFrame:
    """
    Per region-pair overlap of significant (``padj < padj_threshold``)
    proteins between two :func:`pairwise_ttests_across_groups` results (e.g.
    SWAPS vs IonQuant pairwise DE on the same regions).

    Parameters
    ----------
    pairwise_a, pairwise_b : pd.DataFrame
        ``RegionDEAResult.pairwise_results`` from two pipelines.

    Returns
    -------
    pd.DataFrame
        Indexed by ``(group_a, group_b)``, columns ``n_shared``,
        ``n_a_only``, ``n_b_only``.
    """

    def _sig_sets(df: pd.DataFrame) -> pd.Series:
        sig = df.loc[df["padj"] < padj_threshold]
        return sig.groupby(["group_a", "group_b"])["protein"].apply(set)

    sets_a = _sig_sets(pairwise_a)
    sets_b = _sig_sets(pairwise_b)
    pairs = sorted(set(sets_a.index) | set(sets_b.index))

    records = []
    for group_a, group_b in pairs:
        set_a = sets_a.get((group_a, group_b), set())
        set_b = sets_b.get((group_a, group_b), set())
        records.append(
            {
                "group_a": group_a,
                "group_b": group_b,
                "n_shared": len(set_a & set_b),
                "n_a_only": len(set_a - set_b),
                "n_b_only": len(set_b - set_a),
            }
        )
    return pd.DataFrame(records).set_index(["group_a", "group_b"])


def plot_pairwise_de_overlap(
    overlap: pd.DataFrame,
    ax=None,
    label_a: str = "SWAPS",
    label_b: str = "IonQuant",
    color_a: str = "tomato",
    color_b: str = "steelblue",
    min_unique_de: int = 10,
    title: str | None = None,
    fig_dir: str | None = None,
    fig_name_suffix: str = "",
):
    """
    Diverging stacked bar of the per-region-pair significant-protein overlap
    from :func:`compute_pairwise_de_overlap`.

    Positive stack: shared proteins (blend of ``color_a``/``color_b``) plus
    ``label_a``-only proteins (``color_a``). Negative bar: ``label_b``-only
    proteins (``color_b``), kept separate rather than also stacked positively
    so the shared count is only shown once.

    Region pairs are sorted by total unique DE proteins (``n_shared +
    n_a_only + n_b_only``, i.e. the union of either pipeline's significant
    calls) descending, and pairs below ``min_unique_de`` are dropped.

    Parameters
    ----------
    fig_dir : str, optional
        If given, saves PNG + SVG to this directory instead of leaving the
        figure for the caller to show/save.
    fig_name_suffix : str
        Appended to the saved filename (only used when ``fig_dir`` is given).
    """
    total_unique = overlap["n_shared"] + overlap["n_a_only"] + overlap["n_b_only"]
    overlap = overlap.loc[total_unique >= min_unique_de]
    overlap = overlap.loc[
        total_unique.loc[overlap.index].sort_values(ascending=False).index
    ]

    if ax is None:
        _, ax = plt.subplots(figsize=(max(8, 0.6 * len(overlap) + 2), 5))

    shared_color = mcolors.to_hex(
        np.mean([mcolors.to_rgb(color_a), mcolors.to_rgb(color_b)], axis=0)
    )
    labels = [f"{a} vs {b}" for a, b in overlap.index]
    x = np.arange(len(overlap))

    ax.bar(
        x, overlap["n_shared"], color=shared_color, edgecolor="black", label="Shared"
    )
    ax.bar(
        x,
        overlap["n_a_only"],
        bottom=overlap["n_shared"],
        color=color_a,
        edgecolor="black",
        label=f"{label_a} only",
    )
    ax.bar(
        x,
        -overlap["n_b_only"],
        color=color_b,
        edgecolor="black",
        label=f"{label_b} only",
    )

    top = overlap["n_shared"] + overlap["n_a_only"]
    more_a = overlap["n_a_only"] > overlap["n_b_only"]
    for xi, yi, flag in zip(x, top, more_a):
        if flag:
            ax.text(xi, yi, "+", ha="center", va="bottom", fontsize=14, fontweight="bold")

    n_plus = int(more_a.sum())
    n_total = len(overlap)
    pct_plus = 100 * n_plus / n_total if n_total else 0.0

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_xlim(-0.6, len(x) - 0.4)
    ax.set_ylabel("# significant proteins")
    ax.set_title(title or f"Region-pair DE overlap: {label_a} vs {label_b}")
    ax.text(
        1.0,
        1.0,
        f"+ ({label_a} > {label_b}): {n_plus}/{n_total} ({pct_plus:.0f}%)",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")

    if fig_dir is not None:
        fig = ax.figure
        os.makedirs(fig_dir, exist_ok=True)
        fig_name = f"pairwise_de_overlap{fig_name_suffix}"
        fig.savefig(
            os.path.join(fig_dir, f"{fig_name}.png"), dpi=300, bbox_inches="tight"
        )
        fig.savefig(
            os.path.join(fig_dir, f"{fig_name}.svg"), dpi=300, bbox_inches="tight"
        )
    return ax


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _group_stats(
    values: pd.DataFrame,
    sample_info: pd.DataFrame,
    group_col: str,
    presence_mask: pd.DataFrame | None = None,
    min_valid_per_group: int = 0,
) -> dict[str, dict[str, object]]:
    """
    Per-group mean/variance/n for every protein (row), vectorized across rows.

    NaN-aware (``np.nanmean``/``np.nanvar``), so ``values`` may contain
    missing entries (e.g. when imputation is skipped). If ``presence_mask``
    is given, any group with fewer than ``min_valid_per_group`` *real*
    (non-imputed) values for a given protein is masked out for that protein
    (mean/var become NaN, n becomes 0) -- the presence/validity filter,
    applied independently of whether ``values`` itself was imputed.
    """
    if presence_mask is not None:
        values = _mask_below_min_valid(
            values, sample_info, group_col, presence_mask, min_valid_per_group
        )

    group_stats = {}
    for group, group_samples in sample_info.groupby(group_col):
        cols = values.columns.intersection(group_samples.index)
        arr = values[cols].to_numpy(dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean = np.nanmean(arr, axis=1)
            var = np.nanvar(arr, axis=1, ddof=1)
        n = np.sum(~np.isnan(arr), axis=1)
        group_stats[group] = {"mean": mean, "var": var, "n": n}
    return group_stats


def _mask_below_min_valid(
    values: pd.DataFrame,
    sample_info: pd.DataFrame,
    group_col: str,
    presence_mask: pd.DataFrame,
    min_valid_per_group: int,
    groups: list[str] | None = None,
) -> pd.DataFrame:
    """
    Return a copy of ``values`` with (protein, group) cells set to NaN for any
    group where the protein has fewer than ``min_valid_per_group`` real
    (non-imputed) values, per ``presence_mask``. Restricts to ``groups`` if
    given (default: all groups in ``group_col``). This is the presence/
    validity filter, materialized as an actual masked matrix (rather than
    applied implicitly inside ``_group_stats``) so it can also be exported
    for external validation (e.g. feeding the exact same masked subset into
    real limma).
    """
    masked = values.copy()
    for group, group_samples in sample_info.groupby(group_col):
        if groups is not None and group not in groups:
            continue
        cols = values.columns.intersection(group_samples.index)
        real_count = presence_mask[cols].to_numpy().sum(axis=1)
        masked.loc[real_count < min_valid_per_group, cols] = np.nan
    return masked


def _welch_se_and_df(
    a: dict[str, object], b: dict[str, object]
) -> tuple[np.ndarray, np.ndarray]:
    """Welch's pooled SE^2 and Satterthwaite df for two ``_group_stats`` groups."""
    with np.errstate(invalid="ignore", divide="ignore"):
        se_sq = a["var"] / a["n"] + b["var"] / b["n"]
        welch_df = se_sq**2 / (
            (a["var"] / a["n"]) ** 2 / (a["n"] - 1)
            + (b["var"] / b["n"]) ** 2 / (b["n"] - 1)
        )
    return se_sq, welch_df


def _bh_correct_with_nan(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction that passes NaN p-values through as NaN."""
    padj = np.full(p_values.shape, np.nan)
    valid = ~np.isnan(p_values)
    if valid.any():
        padj[valid] = multipletests(p_values[valid], method="fdr_bh")[1]
    return padj


def _trigamma_inverse(x: np.ndarray) -> np.ndarray:
    """
    Numerically invert the trigamma function via Newton's method.

    Ports limma's ``trigammaInverse`` (Smyth 2004), needed to fit the prior
    degrees of freedom in :func:`_squeeze_variance`.
    """
    x = np.asarray(x, dtype=float)
    y = np.where(x > 1e7, 1.0 / np.sqrt(x), np.where(x < 1e-6, 1.0 / x, 0.5 + 1.0 / x))
    for _ in range(50):
        trigamma_y = polygamma(1, y)
        delta = trigamma_y * (1.0 - trigamma_y / x) / polygamma(2, y)
        y = y + delta
        if np.all(np.abs(delta / y) < 1e-8):
            break
    return y


def _squeeze_variance(
    s2: np.ndarray, df: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Empirical-Bayes moderation of per-protein variance estimates towards a
    common prior (Smyth 2004; the core of limma's ``squeezeVar``/``eBayes``).

    Fits a scaled-inverse-chi-squared prior (prior df ``d0``, prior variance
    ``s0_sq``) to the observed ``(s2, df)`` pairs by the method of moments
    (matching the mean/variance of ``log(s2)`` against its expectation under
    the prior), then shrinks each ``s2`` towards ``s0_sq`` in proportion to
    its own degrees of freedom -- proteins with few/unstable replicates (low
    ``df``) get pulled hardest towards the common estimate; proteins with
    more replicates are trusted more.

    Parameters
    ----------
    s2 : np.ndarray
        Per-protein variance estimates (NaN for proteins to exclude from
        both fitting and shrinkage).
    df : np.ndarray
        Per-protein degrees of freedom associated with each ``s2`` (may be
        non-integer, e.g. Welch-Satterthwaite df).

    Returns
    -------
    s2_post : np.ndarray
        Moderated variances, same shape as ``s2``.
    df_post : np.ndarray
        Moderated degrees of freedom (``df + d0``).
    d0, s0_sq : float
        Fitted prior degrees of freedom and prior variance.
    """
    valid = np.isfinite(s2) & np.isfinite(df) & (s2 > 0) & (df > 0)
    if valid.sum() < 2:
        return s2, df, 0.0, np.nan

    z = np.log(s2[valid])
    d = df[valid]
    e = z - digamma(d / 2) + np.log(d / 2)
    emean = e.mean()
    evar = e.var(ddof=1) - np.mean(polygamma(1, d / 2))

    if evar > 0:
        d0 = float(2 * _trigamma_inverse(np.array([evar]))[0])
        s0_sq = float(np.exp(emean + digamma(d0 / 2) - np.log(d0 / 2)))
    else:
        d0 = np.inf
        s0_sq = float(np.exp(emean))

    with np.errstate(invalid="ignore", divide="ignore"):
        if np.isinf(d0):
            s2_post = np.full_like(s2, s0_sq)
            df_post = np.full_like(df, np.inf)
        else:
            s2_post = (d0 * s0_sq + df * s2) / (d0 + df)
            df_post = df + d0

    s2_post = np.where(valid, s2_post, np.nan)
    df_post = np.where(valid, df_post, np.nan)
    return s2_post, df_post, d0, s0_sq
