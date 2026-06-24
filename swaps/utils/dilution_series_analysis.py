"""
Evaluate per-feature AUC (target vs. decoy) across dilution-series datasets and
correlate feature discriminability with total identifications at 1% FDR.

Usage:
    python swaps/utils/feature_strength.py [--out_dir <path>]
"""

import argparse
import re
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_auc_score

BENCHMARK_DIR = Path("/cmnfs/proj/ORIGINS/data/SWAPS_FFM_timsTOF_benchmark")
PERCOLATOR_SUBDIR = "quantification/tmp_percolator_ONLY_SCORE_MATCH"
FEATURES = [
    "im_shift_abs_scaled",
    "rt_shift_abs_scaled",
    "rt_shift",
    "im_shift",
    "sift_similarities",
    "zernike_similarities",
    "sift_distance",
    "zernike_distance",
    "template_matching_score",
    "count_confounders",
]

# Dilution amount sort order (ascending = more challenging first)
_AMOUNT_ORDER = {"125pg": 0, "250pg": 1, "1ng": 2, "5ng": 3}


def _extract_amount(dir_name: str) -> str:
    m = re.search(r"(\d+(?:pg|ng))", dir_name)
    return m.group(1) if m else dir_name


def _sort_key(label: str) -> int:
    return _AMOUNT_ORDER.get(label, 99)


def collect_data(benchmark_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Return auc_df (features × datasets) and ids_series (datasets)."""
    dirs = sorted(
        [d for d in benchmark_dir.iterdir() if d.name.startswith("Ultra_nanoflow")],
        key=lambda d: _sort_key(_extract_amount(d.name)),
    )

    auc_records: dict[str, dict[str, float]] = {}
    ids_records: dict[str, int] = {}

    for d in dirs:
        label = _extract_amount(d.name)
        perc_dir = d / PERCOLATOR_SUBDIR

        pin = pd.read_csv(perc_dir / "percolator_input.tsv", sep="\t")
        # percolator uses +1 / -1 labels; roc_auc_score needs 0/1
        y = (pin["label"] == 1).astype(int)
        auc_records[label] = {
            feat: max(auc := roc_auc_score(y, pin[feat]), 1 - auc) for feat in FEATURES
        }

        psms = pd.read_csv(perc_dir / "percolator_psms.tsv", sep="\t")
        ids_records[label] = (psms["q-value"] <= 0.01).sum() / len(psms) * 100

    auc_df = pd.DataFrame(auc_records)  # rows=features, cols=datasets
    ids_series = pd.Series(ids_records, name="% PSMs at 1% FDR")
    return auc_df, ids_series


def plot_line(auc_df: pd.DataFrame, out_dir: Path) -> None:
    """Normalized AUC across datasets — one line per feature."""
    auc_norm = auc_df.div(auc_df.max(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(7, 4))
    palette = sns.color_palette("tab10", n_colors=len(auc_norm))
    for (feat, row), color in zip(auc_norm.iterrows(), palette):
        ax.plot(auc_norm.columns, row.values, marker="o", label=feat, color=color)

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_ylabel("Relative AUC (normalized to best dataset)")
    ax.set_xlabel("Dataset (increasing input amount →)")
    ax.set_ylim(0, 1.08)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.set_title("Feature discriminability across dilution series")
    fig.tight_layout()
    fig.savefig(out_dir / "feature_auc_lines.pdf", dpi=150)
    fig.savefig(out_dir / "feature_auc_lines.png", dpi=150)
    plt.close(fig)


def plot_heatmap(auc_df: pd.DataFrame, out_dir: Path) -> None:
    """Raw AUC heatmap, features sorted by AUC range (robust → sensitive)."""
    auc_range = auc_df.max(axis=1) - auc_df.min(axis=1)
    auc_sorted = auc_df.loc[auc_range.sort_values().index]

    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(
        auc_sorted,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0.5,
        vmax=1.0,
        linewidths=0.4,
        ax=ax,
    )
    ax.set_title("AUC per feature × dataset\n(sorted by robustness ↑)")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(out_dir / "feature_auc_heatmap.pdf", dpi=150)
    fig.savefig(out_dir / "feature_auc_heatmap.png", dpi=150)
    plt.close(fig)


def plot_auc_vs_ids(auc_df: pd.DataFrame, ids_series: pd.Series, out_dir: Path) -> None:
    """Scatter: per-dataset mean AUC vs IDs at 1% FDR."""
    mean_auc = auc_df.mean(axis=0)
    datasets = auc_df.columns.tolist()

    fig, ax = plt.subplots(figsize=(4, 4))
    for ds in datasets:
        ax.scatter(mean_auc[ds], ids_series[ds], zorder=3)
        ax.annotate(ds, (mean_auc[ds], ids_series[ds]), textcoords="offset points",
                    xytext=(5, 3), fontsize=8)

    ax.set_xlabel("Mean feature AUC across all features")
    ax.set_ylabel("PSMs at 1% FDR (%)")
    ax.set_title("Feature quality vs. identifications")
    fig.tight_layout()
    fig.savefig(out_dir / "feature_auc_vs_ids.pdf", dpi=150)
    fig.savefig(out_dir / "feature_auc_vs_ids.png", dpi=150)
    plt.close(fig)


def _mbr_fill_rates_per_run(df: pd.DataFrame, min_count: int = 10) -> np.ndarray:
    """Return per-run MBR fill rate (%) across all Match Type columns.

    For each run column, entries whose match-type category (MS/MS or MBR) has
    fewer than *min_count* occurrences in that column are reclassified as
    'unmatched' before computing the rate.

    Rate = MBR / (MBR + unmatched) per column, after reclassification.
    """
    mt_cols = [c for c in df.columns if c.endswith("Match Type")]
    rates = []
    for col in mt_cols:
        s = df[col].copy()
        for label in ("MS/MS", "MBR"):
            if (s == label).sum() < min_count:
                s = s.replace(label, "unmatched")
        mbr = (s == "MBR").sum()
        non_msms = (s != "MS/MS").sum()
        rates.append(mbr / non_msms * 100 if non_msms > 0 else 0.0)
    return np.array(rates)


def plot_mbr_fill_rate(
    benchmark_dir: Path = BENCHMARK_DIR,
    out_dir: Path | None = None,
    min_count: int = 10,
) -> plt.Figure:
    """Line plot of MBR fill rate (MBR / non-MS/MS) for SWAPS vs FragPipe.

    Per-run rates are computed and the mean is plotted with ±1 std shadow.
    Runs where MS/MS or MBR count < *min_count* have those entries reclassified
    as unmatched before computing the rate.

    X-axis: 5ng → 1ng → 250pg → 125pg (decreasing input amount).
    Y-axis: % of non-MS/MS slots successfully filled by MBR.

    Returns the Figure for inline display in Jupyter.
    """
    x_order = ["5ng", "1ng", "250pg", "125pg"]

    dirs = {
        _extract_amount(d.name): d
        for d in benchmark_dir.iterdir()
        if d.name.startswith("Ultra_nanoflow")
    }

    # mean and std per dataset for each tool
    swaps_mean, swaps_std = [], []
    fp_mean, fp_std = [], []

    for amount in x_order:
        d = dirs[amount]
        perc_dir = d / PERCOLATOR_SUBDIR

        swaps_df = pd.read_parquet(perc_dir / "swaps_combined_ions.parquet")
        sr = _mbr_fill_rates_per_run(swaps_df, min_count)
        swaps_mean.append(sr.mean())
        swaps_std.append(sr.std())

        cfg = (d / "quantification" / "effective_config.yaml").read_text()
        search_path = re.search(r"SEARCH_OUTPUT_PATH:\s*(\S+)", cfg).group(1)
        fp_df = pd.read_csv(Path(search_path) / "combined_ion.tsv", sep="\t")
        fr = _mbr_fill_rates_per_run(fp_df, min_count)
        fp_mean.append(fr.mean())
        fp_std.append(fr.std())

    swaps_mean, swaps_std = np.array(swaps_mean), np.array(swaps_std)
    fp_mean, fp_std = np.array(fp_mean), np.array(fp_std)
    xs = np.arange(len(x_order))

    fig, ax = plt.subplots(figsize=(5, 4))
    for mean, std, marker, label, color in (
        (swaps_mean, swaps_std, "o", "SWAPS", "steelblue"),
        (fp_mean, fp_std, "s", "FragPipe", "tomato"),
    ):
        ax.plot(xs, mean, marker=marker, label=label, color=color)
        ax.fill_between(xs, mean - std, mean + std, alpha=0.2, color=color)

    ax.set_xticks(xs)
    ax.set_xticklabels(x_order)
    ax.set_xlabel("Dataset (decreasing input amount →)")
    ax.set_ylabel("MBR fill rate (MBR / non-MS/MS, %)")
    ax.set_title(f"MBR fill rate across dilution series\n(min_count={min_count})")
    legend = ax.legend()
    for handle in legend.legend_handles:
        handle.set_marker("")
    fig.tight_layout()

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "mbr_fill_rate.pdf", dpi=150)
        fig.savefig(out_dir / "mbr_fill_rate.png", dpi=150)

    return fig


_CORR_CONDITIONS = ["MS/MS – MS/MS", "MS/MS – MBR", "MBR – MBR", "All Quantified"]
_CONDITION_LINESTYLES = {
    "MS/MS – MS/MS": "-",
    "MS/MS – MBR": "--",
    "MBR – MBR": ":",
    "All Quantified": "-.",
}


def _pairwise_corrs_by_condition(
    df: pd.DataFrame,
    min_count: int = 10,
    mt_keyword: str = "Match Type",
    int_keyword: str = "Intensity",
    mbr_label: str = "MBR",
    unmatched_label: str = "unmatched",
) -> dict[str, list[float]]:
    """Return {condition: [Pearson r per valid run pair]} for a combined-ions table."""
    mt_cols = [c for c in df.columns if mt_keyword in c]
    runs = []
    for mt_col in mt_cols:
        int_col = mt_col.replace(mt_keyword, int_keyword)
        if int_col in df.columns:
            runs.append((mt_col, int_col))

    corrs: dict[str, list[float]] = {c: [] for c in _CORR_CONDITIONS}
    for (mt_i, int_i), (mt_j, int_j) in combinations(runs, 2):
        mti, mtj = df[mt_i], df[mt_j]
        vi, vj = df[int_i], df[int_j]
        pos = (vi > 0) & (vj > 0)

        ms2_i = mti.str.contains("MS/MS", na=False)
        ms2_j = mtj.str.contains("MS/MS", na=False)
        mbr_i = mti == mbr_label
        mbr_j = mtj == mbr_label

        masks = {
            "MS/MS – MS/MS": ms2_i & ms2_j & pos,
            "MS/MS – MBR": ((ms2_i & mbr_j) | (mbr_i & ms2_j)) & pos,
            "MBR – MBR": mbr_i & mbr_j & pos,
            "All Quantified": (mti != unmatched_label) & (mtj != unmatched_label) & pos,
        }
        for cond, mask in masks.items():
            n = mask.sum()
            if n >= min_count:
                r, _ = stats.pearsonr(np.log2(vi[mask]), np.log2(vj[mask]))
                corrs[cond].append(r)
    return corrs


def plot_intensity_correlation_by_dilution(
    benchmark_dir: Path = BENCHMARK_DIR,
    out_dir: Path | None = None,
    min_count: int = 10,
) -> plt.Figure:
    """Line plot of pairwise Pearson r (log2 intensity) across dilution datasets.

    X-axis: 5ng → 1ng → 250pg → 125pg.
    Y-axis: mean Pearson r across all valid run pairs; ±1 std shadow.
    Color: SWAPS (blue) vs FragPipe (red).
    Line style: match-type pair condition.

    Returns the Figure for inline display in Jupyter.
    """
    x_order = ["5ng", "1ng", "250pg", "125pg"]

    dirs = {
        _extract_amount(d.name): d
        for d in benchmark_dir.iterdir()
        if d.name.startswith("Ultra_nanoflow")
    }

    # {tool: {condition: [mean_r per dataset]}}
    tool_means: dict[str, dict[str, list[float]]] = {
        "SWAPS": {c: [] for c in _CORR_CONDITIONS},
        "FragPipe": {c: [] for c in _CORR_CONDITIONS},
    }
    tool_stds: dict[str, dict[str, list[float]]] = {
        "SWAPS": {c: [] for c in _CORR_CONDITIONS},
        "FragPipe": {c: [] for c in _CORR_CONDITIONS},
    }

    for amount in x_order:
        d = dirs[amount]
        perc_dir = d / PERCOLATOR_SUBDIR

        swaps_df = pd.read_parquet(perc_dir / "swaps_combined_ions.parquet")
        swaps_corrs = _pairwise_corrs_by_condition(swaps_df, min_count)

        cfg = (d / "quantification" / "effective_config.yaml").read_text()
        search_path = re.search(r"SEARCH_OUTPUT_PATH:\s*(\S+)", cfg).group(1)
        fp_df = pd.read_csv(Path(search_path) / "combined_ion.tsv", sep="\t")
        fp_corrs = _pairwise_corrs_by_condition(fp_df, min_count)

        for cond in _CORR_CONDITIONS:
            for tool, corrs in (("SWAPS", swaps_corrs), ("FragPipe", fp_corrs)):
                vals = np.array(corrs[cond])
                tool_means[tool][cond].append(vals.mean() if len(vals) else np.nan)
                tool_stds[tool][cond].append(
                    vals.std(ddof=1) if len(vals) > 1 else 0.0
                )

    xs = np.arange(len(x_order))
    tool_colors = {"SWAPS": "steelblue", "FragPipe": "tomato"}

    fig, ax = plt.subplots(figsize=(6, 4))
    for tool, color in tool_colors.items():
        for cond in _CORR_CONDITIONS:
            mean = np.array(tool_means[tool][cond])
            std = np.array(tool_stds[tool][cond])
            ls = _CONDITION_LINESTYLES[cond]
            ax.plot(xs, mean, color=color, linestyle=ls, marker="o",
                    label=f"{tool} · {cond}")
            ax.fill_between(xs, mean - std, mean + std, color=color, alpha=0.1)

    ax.set_xticks(xs)
    ax.set_xticklabels(x_order)
    ax.set_xlabel("Dataset (decreasing input amount →)")
    ax.set_ylabel("Pearson r (log₂ intensity)")
    ax.set_title(f"Intensity correlation by match-type pair\n(min_count={min_count})")
    legend = ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    for handle in legend.legend_handles:
        handle.set_marker("")
    fig.tight_layout()

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "intensity_corr_by_dilution.pdf", dpi=150)
        fig.savefig(out_dir / "intensity_corr_by_dilution.png", dpi=150)

    return fig


def print_summary(auc_df: pd.DataFrame, ids_series: pd.Series) -> None:
    auc_range = (auc_df.max(axis=1) - auc_df.min(axis=1)).sort_values()
    print("\n=== Feature AUC ===")
    print(auc_df.to_string(float_format="{:.3f}".format))
    print("\n=== AUC range across datasets (low = robust) ===")
    print(auc_range.to_string(float_format="{:.3f}".format))
    print("\n=== IDs at 1% FDR ===")
    print(ids_series.to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("/cmnfs/proj/ORIGINS/data/SWAPS_FFM_timsTOF_benchmark/feature_strength"),
        help="Directory for output figures",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    auc_df, ids_series = collect_data(BENCHMARK_DIR)
    print_summary(auc_df, ids_series)

    plot_line(auc_df, args.out_dir)
    plot_heatmap(auc_df, args.out_dir)
    plot_auc_vs_ids(auc_df, ids_series, args.out_dir)
    fig = plot_mbr_fill_rate(out_dir=args.out_dir)
    plt.close(fig)
    fig = plot_intensity_correlation_by_dilution(out_dir=args.out_dir)
    plt.close(fig)

    print(f"\nFigures saved to {args.out_dir}")


if __name__ == "__main__":
    main()
