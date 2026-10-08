import logging
from typing import Optional, Sequence
 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os

Logger = logging.getLogger(__name__)

def validate_pipecho_cfg(cfg):
    if not cfg.PIPECHO.ENABLED:
        return

    select_by = cfg.PIPECHO.SELECT_BY
    if not isinstance(select_by, str) or not select_by:
        raise ValueError(f"PIPECHO.SELECT_BY must be a non-empty string, got {select_by!r}")
    if cfg.PIPECHO.N_PER_RUN < 1:
        raise ValueError(f"PIPECHO.N_PER_RUN must be >= 1, got {cfg.PIPECHO.N_PER_RUN}")

    if not cfg.SWA:
        raise ValueError(
            "PIPECHO.ENABLED requires SWA: true. "
            "dict_ref_with_activation.pkl and discards the censored dictionary."
        )
    dict_ref_path = os.path.join(cfg.RESULT_PATH, "dict_ref.pkl")
    if os.path.exists(dict_ref_path):
        raise FileExistsError(
            f"PIPECHO.ENABLED but {dict_ref_path} exists"
        )

def resolve_select_by(evidence, select_by):
    if select_by == "random": #random sampling
        return None
    if select_by not in evidence.columns:
        raise KeyError(
            f"PIPECHO.SELECT_BY={select_by!r} is not valid "
        )
    return select_by

def select_censored_combination( #combination of peptide and run to be censored
        evidence, 
        select_by, 
        n_per_run, 
        seed, 
        descending=True,
        id_cols=("Sequence", "Modifications", "Charge"),
        run_col="Raw file"):
    """Draw n_per_run (peptide, run) pairs per run from the eligible pool."""
    id_cols = list(id_cols)
    pool = eligible_combination(evidence, select_by=select_by, descending=descending,)          
    rng = np.random.default_rng(seed)

    combination = []
    for run, grp in pool.groupby(run_col, sort=True):
        if len(grp) < n_per_run:
            logging.warning(
                "PIPECHO: run %s has only %d eligible combination, taking all (wanted %d)",
                run, len(grp), n_per_run)
        k = min(n_per_run, len(grp))
        if select_by is None:
            take = grp.loc[rng.choice(grp.index.to_numpy(), size=k, replace=False)]
        else:
            take = grp.sort_values(select_by, ascending=not descending, kind="mergesort").head(k)
        combination.append(take)

    keep_cols = id_cols + [run_col]
    
    for extra in ("ms2_rt", "ms2_intensity", "Score"):
        if extra in pool.columns:
            keep_cols.append(extra)
    censored_combination = pd.concat(combination, ignore_index=True)[keep_cols]
    logging.info("PIPECHO: selected %d combinations across %d runs",
                 len(censored_combination), censored_combination[run_col].nunique())
    return censored_combination

def eligible_combination(evidence,
                        select_by=None,
                        descending=True,
                        id_cols=("Sequence", "Modifications", "Charge"),
                        run_col="Raw file",
                        score_col="Score",
                        rt_col="Retention time",
                        intensity_col="Intensity"):
    """(peptide, run) combinations to be censored."""
    id_cols = list(id_cols)

    agg = {"n_spectra": ("Spectrum", "nunique"),
           score_col: (score_col, "max")}
    if rt_col in evidence.columns:
        agg["ms2_rt"] = (rt_col, "max")
    else:
        logging.warning(
            "PIPECHO: evidence has no %r column -- censored combinations will "
            "lack ms2_rt and the RT-error evaluation will not be possible.", rt_col)
    if intensity_col in evidence.columns:
        agg["ms2_intensity"] = (intensity_col, "max")
    if select_by is not None and select_by != score_col:
        agg[select_by] = (select_by, "max" if descending else "min")


    pairs = evidence.groupby(id_cols + [run_col], as_index=False).agg(**agg)

    pairs["best_score"] = pairs.groupby(id_cols)[score_col].transform("max")

    pool = pairs[(pairs["n_spectra"] == 1) & (pairs[score_col] < pairs["best_score"])]

    logging.info("PIPECHO: %d eligible combinations across %d runs",
                len(pool), pool[run_col].nunique())
    
    return pool.reset_index(drop=True)

def censoring(evidence, censored_combination,
                    id_cols=("Sequence", "Modifications", "Charge"),
                    run_col="Raw file"):
    key = list(id_cols) + [run_col]
    marked = evidence.merge(censored_combination[key].assign(_c=True), on=key, how="left")
    mask = marked["_c"].fillna(False).astype(bool).to_numpy()
    out = marked.loc[~mask].drop(columns=["_c"])
    logging.info("PIPECHO: removed %d of %d evidence rows (%d combinations)",
                 int(mask.sum()), len(evidence), len(censored_combination))
    return out.reset_index(drop=True)


CLASS_ORDER = ["accepted_correct", "accepted_wrong", "rejected", "not_found"]
CLASS_COLORS = {
    "accepted_correct": "#2a9d8f",
    "accepted_wrong": "#e76f51",
    "rejected": "#e9c46a",
    "not_found": "#bdbdbd",
}

_RUN_NAME_EXTENSIONS = (".d", ".mzML", ".mzml", ".raw", ".RAW")


def _norm_run(name):
    base = os.path.basename(str(name))
    for ext in _RUN_NAME_EXTENSIONS:
        if base.endswith(ext):
            return base[: -len(ext)]
    return base


def _run_rt_lookup(result_path, run_name):
    """MS1_frame_idx -> Time_minute for one run, or None if ms1scans.csv is absent."""
    path = os.path.join(result_path, run_name, "ms1scans.csv")
    if not os.path.exists(path):
        return None
    ms1 = pd.read_csv(path)
    return ms1.set_index("MS1_frame_idx")["Time_minute"]


def map_combination_to_mz_rank(censored_combination, dict_ref,
                               id_cols=("Sequence", "Modifications", "Charge")):
    """Attach the censored dictionary's mz_rank to each censored combination."""
    id_cols = list(id_cols)
    missing = [c for c in id_cols if c not in dict_ref.columns]
    if missing:
        raise KeyError(f"dict_ref is missing id column(s) {missing}")
    lookup = dict_ref[id_cols + ["mz_rank"]].drop_duplicates(subset=id_cols)
    mapped = censored_combination.merge(lookup, on=id_cols, how="left")
    n_unmapped = int(mapped["mz_rank"].isna().sum())
    if n_unmapped:
        logging.warning(
            "PIPECHO: %d of %d censored combinations have no mz_rank in the "
            "censored dict_ref (peptide dropped from dictionary entirely?)",
            n_unmapped, len(mapped))
    return mapped


def _load_percolator_qvalues(work_dir):
    """(mz_rank, run) -> q-value from a percolator work_dir, or None."""
    path = os.path.join(work_dir, "percolator_psms.tsv")
    if not os.path.exists(path):
        return None
    psms = pd.read_csv(path, sep="\t")
    psms["mz_rank"] = psms["PSMId"].astype(str).str.split("_").str[0].astype(int)
    if "filename" in psms.columns:
        psms["run"] = psms["filename"].map(_norm_run)
    else:
        # PSMId = "{mz_rank}_{run}_{decoy}" 
        psms["run"] = (
            psms["PSMId"].astype(str)
            .str.split("_", n=1).str[1]
            .str.rsplit("_", n=1).str[0]
            .map(_norm_run)
        )
    return psms[["mz_rank", "run", "q-value"]].drop_duplicates(["mz_rank", "run"])


def assemble_censoring_detail(
    result_path,
    quant_dir,
    fdr_dir,
    rt_tolerance_pct=0.01,
    gradient_length_min=None,
    run_col="Raw file",
    id_cols=("Sequence", "Modifications", "Charge"),
):
    """One row per censored combination with candidate/acceptance/RT info.

    Reads pipecho_censored_combination.csv + dict_ref_with_activation.pkl from
    result_path, the pre-FDR candidates (pp_match_target.parquet) from
    quant_dir, the accepted set from quant_dir/fdr_dir, and percolator
    q-values from the work_dir one level above fdr_dir (if present).
    gradient_length_min=None derives the RT tolerance per run from the run's
    max Time_minute in ms1scans.csv.
    """
    combos = pd.read_csv(os.path.join(result_path, "pipecho_censored_combination.csv"))
    dict_ref = pd.read_pickle(os.path.join(result_path, "dict_ref_with_activation.pkl"))
    detail = map_combination_to_mz_rank(combos, dict_ref, id_cols=id_cols)
    detail["run"] = detail[run_col].map(_norm_run)

    # Pre-FDR MBR candidates: did the matching stage produce a peak at all?
    candidate_cols = [
        "mz_rank", "Run_name", "apex_scan", "intensity_sum",
        "template_matching_score",
    ]
    candidates = pd.read_parquet(
        os.path.join(quant_dir, "pp_match_target.parquet"))
    candidate_cols = [c for c in candidate_cols if c in candidates.columns]
    candidates = candidates[candidate_cols].assign(
        run=lambda df: df["Run_name"].map(_norm_run)).drop(columns=["Run_name"])
    detail = detail.merge(
        candidates.drop_duplicates(["mz_rank", "run"]),
        on=["mz_rank", "run"], how="left")
    detail["candidate_found"] = detail["apex_scan"].notna()

    # Accepted set: what actually survived this fdr_dir's filtering.
    accepted = pd.read_parquet(
        os.path.join(quant_dir, fdr_dir, "pp_match_target_filtered.parquet"),
        columns=["mz_rank", "Run_name"],
    )
    accepted = accepted.assign(run=accepted["Run_name"].map(_norm_run))
    accepted_keys = set(zip(accepted["mz_rank"], accepted["run"]))
    detail["accepted"] = [
        (mz, run) in accepted_keys
        for mz, run in zip(detail["mz_rank"], detail["run"])
    ]

    # Censoring check: a censored combo must not retain MS/MS status
    # (Reference/Quant_Only) in the censored dictionary.
    from swaps.postprocessing.rescore import get_msms_run_keys

    msms_keys = get_msms_run_keys(dict_ref)
    msms_keys = set(zip(
        msms_keys["mz_rank"], msms_keys["Run_name"].map(_norm_run)))
    n_leaks = sum(
        (mz, run) in msms_keys
        for mz, run in zip(detail["mz_rank"], detail["run"])
    )
    if n_leaks:
        logging.warning(
            "PIPECHO: %d censored combinations still have MS/MS status in the "
            "censored dict_ref -- their evidence survived censoring.", n_leaks)

    qvals = _load_percolator_qvalues(
        os.path.dirname(os.path.join(quant_dir, fdr_dir)))
    if qvals is not None:
        detail = detail.merge(qvals, on=["mz_rank", "run"], how="left")
    else:
        detail["q-value"] = np.nan
        logging.warning(
            "PIPECHO: no percolator_psms.tsv next to %s -- q-value calibration "
            "plot will not be available.", fdr_dir)

    detail["pip_rt"] = np.nan
    detail["rt_tolerance_min"] = np.nan
    for run, grp in detail.groupby("run"):
        rt_lookup = _run_rt_lookup(result_path, run)
        if rt_lookup is None:
            logging.warning(
                "PIPECHO: no ms1scans.csv for run %s under %s -- cannot convert "
                "apex_scan to RT for this run.", run, result_path)
            continue
        found = grp.loc[grp["candidate_found"]]
        apex = found["apex_scan"].astype(int)
        detail.loc[found.index, "pip_rt"] = rt_lookup.reindex(
            apex.to_numpy()).to_numpy()
        gradient = gradient_length_min or float(rt_lookup.max())
        detail.loc[grp.index, "rt_tolerance_min"] = rt_tolerance_pct * gradient

    if "ms2_rt" not in detail.columns:
        logging.warning(
            "PIPECHO: censored combination CSV has no ms2_rt column -- recovery "
            "is reported but RT errors cannot be computed.")
        detail["ms2_rt"] = np.nan
    detail["rt_diff_min"] = (detail["ms2_rt"] - detail["pip_rt"]).abs()
    detail["rt_correct"] = detail["rt_diff_min"] <= detail["rt_tolerance_min"]

    detail["class"] = np.select(
        [
            ~detail["candidate_found"],
            ~detail["accepted"],
            detail["rt_correct"],
        ],
        ["not_found", "rejected", "accepted_correct"],
        default="accepted_wrong",
    )
    return detail


def summarize_censoring(detail):
    """Per-run + TOTAL recovery and native-peak-error summary."""

    def _summarize(grp):
        n_accepted = int(grp["accepted"].sum())
        # Native Peak Pairs: accepted with both RTs available
        pairs = grp.loc[grp["accepted"] & grp["rt_diff_min"].notna()]
        n_errors = int((~pairs["rt_correct"]).sum())
        return pd.Series({
            "n_censored": len(grp),
            "n_in_dict": int(grp["mz_rank"].notna().sum()),
            "n_candidate_found": int(grp["candidate_found"].sum()),
            "n_accepted": n_accepted,
            "recovery_rate": n_accepted / len(grp) if len(grp) else np.nan,
            "n_native_peak_pairs": len(pairs),
            "n_native_peak_errors": n_errors,
            "native_peak_error_rate": n_errors / len(pairs) if len(pairs) else np.nan,
        })

    summary = (
        detail.groupby("run").apply(_summarize, include_groups=False).reset_index()
    )
    total = _summarize(detail)
    total["run"] = "TOTAL"
    return pd.concat([summary, total.to_frame().T], ignore_index=True)


def evaluate_censoring(
    result_path,
    quant_dir,
    fdr_dir,
    rt_tolerance_pct=0.01,
    gradient_length_min=None,
    save=True,
):
    """PIP-ECHO data-censoring evaluation; returns (summary, detail).
    When save=True both tables are written as CSVs into quant_dir/fdr_dir.
    """
    detail = assemble_censoring_detail(
        result_path, quant_dir, fdr_dir,
        rt_tolerance_pct=rt_tolerance_pct,
        gradient_length_min=gradient_length_min,
    )
    summary = summarize_censoring(detail)
    total = summary.loc[summary["run"] == "TOTAL"].iloc[0]
    logging.info(
        "PIPECHO evaluation (%s): %d/%d censored accepted (%.1f%%), "
        "native peak error rate %.3f (%d/%d pairs)",
        fdr_dir, total["n_accepted"], total["n_censored"],
        100.0 * total["recovery_rate"], total["native_peak_error_rate"],
        total["n_native_peak_errors"], total["n_native_peak_pairs"],
    )
    if save:
        out_dir = os.path.join(quant_dir, fdr_dir)
        detail.to_csv(
            os.path.join(out_dir, "pipecho_evaluation_detail.csv"), index=False)
        summary.to_csv(
            os.path.join(out_dir, "pipecho_evaluation_summary.csv"), index=False)
    return summary, detail


def _savefig(fig, fig_dir, name):
    if fig_dir is None:
        return
    os.makedirs(fig_dir, exist_ok=True)
    fig.savefig(os.path.join(fig_dir, name), dpi=300, bbox_inches="tight")


def plot_rt_diff_distribution(detail, fig_dir=None, bins=50):
    """Histogram + ECDF of |MS2-RT - PIP-RT| over accepted native peak pairs."""
    pairs = detail.loc[detail["accepted"] & detail["rt_diff_min"].notna()]
    if pairs.empty:
        logging.warning("PIPECHO: no accepted native peak pairs to plot.")
        return None
    tol = float(pairs["rt_tolerance_min"].median())
    diffs = pairs["rt_diff_min"].clip(lower=1e-4)

    fig, (ax_hist, ax_ecdf) = plt.subplots(1, 2, figsize=(10, 4))
    log_bins = np.logspace(
        np.log10(diffs.min()), np.log10(max(diffs.max(), tol * 2)), bins)
    ax_hist.hist(diffs, bins=log_bins, color="#457b9d")
    ax_hist.set_xscale("log")
    ax_hist.set_xlabel("|MS2-RT − PIP-RT| (min)")
    ax_hist.set_ylabel("Native peak pairs")

    sns.ecdfplot(x=diffs, ax=ax_ecdf, color="#457b9d")
    ax_ecdf.set_xscale("log")
    ax_ecdf.set_xlabel("|MS2-RT − PIP-RT| (min)")
    ax_ecdf.set_ylabel("ECDF")

    err_rate = float((pairs["rt_diff_min"] > pairs["rt_tolerance_min"]).mean())
    for ax in (ax_hist, ax_ecdf):
        ax.axvline(tol, color="#e76f51", linestyle="--",
                   label=f"1% gradient = {tol:.2f} min")
    ax_ecdf.axhline(1 - err_rate, color="#e76f51", linestyle=":",
                    label=f"error rate = {err_rate:.3f}")
    ax_hist.legend()
    ax_ecdf.legend()
    fig.tight_layout()
    _savefig(fig, fig_dir, "pipecho_rt_diff_distribution.png")
    return fig


def plot_recovery_by_run(detail, fig_dir=None):
    """Stacked per-run bars of the four censoring outcome classes."""
    counts = (
        detail.groupby(["run", "class"]).size().unstack(fill_value=0)
        .reindex(columns=CLASS_ORDER, fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(counts)), 4.5))
    bottom = np.zeros(len(counts))
    for cls in CLASS_ORDER:
        ax.bar(counts.index, counts[cls], bottom=bottom,
               color=CLASS_COLORS[cls], label=cls)
        bottom += counts[cls].to_numpy()
    for i, (run, row) in enumerate(counts.iterrows()):
        n = row.sum()
        acc = row["accepted_correct"] + row["accepted_wrong"]
        if n:
            ax.text(i, n, f"{100 * acc / n:.0f}%", ha="center", va="bottom",
                    fontsize=8)
    ax.set_ylabel("Censored peptides")
    ax.set_xlabel("Run")
    ax.tick_params(axis="x", rotation=60)
    ax.legend(title="Outcome", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    _savefig(fig, fig_dir, "pipecho_recovery_by_run.png")
    return fig


def _wilson_interval(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return center - half, center + half


def plot_fdr_calibration(detail, fig_dir=None, q_grid=None):
    """Observed native-peak error rate among transfers accepted at q <= cutoff.
    """
    cand = detail.loc[detail["q-value"].notna() & detail["rt_diff_min"].notna()]
    if cand.empty:
        logging.warning(
            "PIPECHO: no candidates with both q-value and RT info -- "
            "calibration plot skipped.")
        return None
    if q_grid is None:
        q_grid = np.logspace(-3, 0, 30)

    rows = []
    for q in q_grid:
        sel = cand.loc[cand["q-value"] <= q]
        n = len(sel)
        if n == 0:
            continue
        k = int((~sel["rt_correct"]).sum())
        lo, hi = _wilson_interval(k, n)
        rows.append({"q": q, "n": n, "error_rate": k / n, "lo": lo, "hi": hi})
    calib = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(calib["q"], calib["error_rate"], marker="o", ms=3,
            color="#2a9d8f", label="observed error rate")
    ax.fill_between(calib["q"], calib["lo"].clip(lower=0), calib["hi"],
                    color="#2a9d8f", alpha=0.2, label="95% Wilson CI")
    lims = (q_grid.min(), 1.0)
    ax.plot(lims, lims, color="grey", linestyle="--", lw=1, label="y = x")
    ax.set_xscale("log")
    ax.set_xlabel("q-value cutoff")
    ax.set_ylabel("Native peak error rate among accepted")
    ax.set_xlim(lims)

    ax_n = ax.twinx()
    ax_n.plot(calib["q"], calib["n"], color="grey", linestyle=":", lw=1)
    ax_n.set_ylabel("Native peak pairs (n)", color="grey")
    ax_n.tick_params(axis="y", colors="grey")
    ax.legend(loc="upper left")
    fig.tight_layout()
    _savefig(fig, fig_dir, "pipecho_fdr_calibration.png")
    return fig


def plot_error_diagnostics(detail, fig_dir=None):
    found = detail.loc[detail["candidate_found"]].copy()
    if found.empty:
        logging.warning("PIPECHO: no candidates found -- diagnostics skipped.")
        return None
    order = [c for c in CLASS_ORDER if c in found["class"].unique()]
    palette = {c: CLASS_COLORS[c] for c in order}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    if "template_matching_score" in found.columns:
        sns.boxplot(data=found, x="class", y="template_matching_score",
                    order=order, palette=palette, ax=axes[0])
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=30)

    if "intensity_sum" in found.columns:
        found["log10_intensity"] = np.log10(found["intensity_sum"].clip(lower=1))
        sns.boxplot(data=found, x="class", y="log10_intensity",
                    order=order, palette=palette, ax=axes[1])
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=30)

    pairs = found.loc[found["rt_diff_min"].notna()]
    for cls in order:
        grp = pairs.loc[pairs["class"] == cls]
        axes[2].scatter(grp["ms2_rt"], grp["rt_diff_min"].clip(lower=1e-4),
                        s=8, alpha=0.5, color=CLASS_COLORS[cls], label=cls)
    tol = pairs["rt_tolerance_min"].median()
    if pd.notna(tol):
        axes[2].axhline(tol, color="black", linestyle="--", lw=1)
    axes[2].set_yscale("log")
    axes[2].set_xlabel("MS2-RT (min, gradient position)")
    axes[2].set_ylabel("|MS2-RT − PIP-RT| (min)")
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    _savefig(fig, fig_dir, "pipecho_error_diagnostics.png")
    return fig


def plot_intensity_agreement(detail, fig_dir=None):
    """log2 transfer intensity vs log2 original MS2-feature intensity."""
    if "ms2_intensity" not in detail.columns:
        logging.warning(
            "PIPECHO: censored CSV has no ms2_intensity column -- intensity "
            "agreement plot skipped.")
        return None
    sub = detail.loc[
        detail["accepted"]
        & (detail["intensity_sum"] > 0)
        & (detail["ms2_intensity"] > 0)
        & detail["rt_diff_min"].notna()
    ].copy()
    if sub.empty:
        logging.warning("PIPECHO: no accepted transfers with intensities.")
        return None
    sub["log2_pip"] = np.log2(sub["intensity_sum"])
    sub["log2_ms2"] = np.log2(sub["ms2_intensity"])

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for correct, label, color in (
        (True, "accepted_correct", CLASS_COLORS["accepted_correct"]),
        (False, "accepted_wrong", CLASS_COLORS["accepted_wrong"]),
    ):
        grp = sub.loc[sub["rt_correct"] == correct]
        if grp.empty:
            continue
        r = grp["log2_pip"].corr(grp["log2_ms2"])
        ax.scatter(grp["log2_ms2"], grp["log2_pip"], s=8, alpha=0.5,
                   color=color, label=f"{label} (r={r:.2f}, n={len(grp)})")
    ax.set_xlabel("log2 original MS2-feature intensity")
    ax.set_ylabel("log2 censored PIP intensity")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _savefig(fig, fig_dir, "pipecho_intensity_agreement.png")
    return fig