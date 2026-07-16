# TODO: move it to helper instead
import logging
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from typing import Optional
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess

Logger = logging.getLogger(__name__)
ALLOWED_ORGANISMS = ("HUMAN", "YEAST", "ECOLI")
ORGANISM_NAME_MAP = {
    "HUMAN": "Homo sapiens",
    "YEAST": "Saccharomyces cerevisiae",
    "ECOLI": "Escherichia coli",
}


def undistinguishable_excl_output_name(output_name: str) -> str:
    """DirectLFQ input filename for the version that excludes ions flagged as
    belonging to a spatially undistinguishable coSWA confounder group (see
    reformat_swaps_combined_for_directlfq). Only actually written when at
    least one ion is flagged -- callers should check the path exists before
    using it (e.g. before running DirectLFQ on it)."""
    base_name, ext = os.path.splitext(output_name)
    return f"{base_name}_excl_undistinguishable{ext}"


def _log_and_write_undistinguishable_split(
    reformatted_df: pd.DataFrame,
    output_cols: list,
    output_dir: str,
    output_name: str,
) -> None:
    """Log the undistinguishable_group_id breakdown and, if any ion actually
    belongs to a flagged coSWA confounder group (a value other than -1 or
    NaN), additionally write a second DirectLFQ input with those ions
    dropped entirely -- for comparing LFQ results with/without them.
    """
    gid = reformatted_df["undistinguishable_group_id"]
    is_nan = gid.isna()
    is_neg1 = ~is_nan & gid.astype(str).isin(["-1", "-1.0"])
    is_other = ~is_nan & ~is_neg1
    if not is_other.any():
        return
    Logger.info(
        "undistinguishable_group_id breakdown among %d ions: -1=%d, NaN=%d, "
        "other=%d (%d unique group id(s))",
        len(gid),
        int(is_neg1.sum()),
        int(is_nan.sum()),
        int(is_other.sum()),
        int(gid[is_other].nunique()),
    )
    excl_name = undistinguishable_excl_output_name(output_name)
    reformatted_df.loc[~is_other, output_cols].to_csv(
        os.path.join(output_dir, excl_name), index=False, sep="\t"
    )
    Logger.info(
        "Wrote DirectLFQ input excluding %d undistinguishable ion(s) to %s",
        int(is_other.sum()),
        excl_name,
    )


def reformat_swaps_combined_for_directlfq(
    combined_ion,
    dict_ref,
    output_dir,
    keep_match_type_col: bool = False,
    ion_id_col="mz_rank",
    protein_id_col="Proteins",
    output_name="swaps.aq_reformat.tsv",
):
    intensity_cols = [col for col in combined_ion.columns if col.endswith("Intensity")]
    combined_ion = combined_ion.copy()
    for intensity_col in intensity_cols:
        match_type_col = intensity_col.replace("Intensity", "Match Type")
        if match_type_col in combined_ion.columns:
            # MBR hits inside a coSWA confounder group whose independently-
            # computed segments overlap cannot be attributed to one specific
            # member (see build_pivot) -- excluded from LFQ input entirely.
            combined_ion.loc[
                combined_ion[match_type_col] == "MBR_undistinguished", intensity_col
            ] = np.nan
    if keep_match_type_col:
        intensity_cols += [col for col in combined_ion.columns if col == "Match Type"]

    has_group_id = "undistinguishable_group_id" in combined_ion.columns
    merge_cols = intensity_cols + [ion_id_col]
    if has_group_id:
        merge_cols = merge_cols + ["undistinguishable_group_id"]
    reformatted_df = pd.merge(
        combined_ion[merge_cols],
        dict_ref[[ion_id_col, protein_id_col]],
        on=ion_id_col,
        how="left",
    )
    intensity_cols_rename_map = {col: "run_" + col for col in intensity_cols}
    intensity_cols_rename_map[protein_id_col] = "protein"
    intensity_cols_rename_map[ion_id_col] = "ion"
    reformatted_df = reformatted_df.rename(columns=intensity_cols_rename_map)
    output_cols = ["protein", "ion"] + list(intensity_cols_rename_map.values())[:-2]

    if has_group_id:
        _log_and_write_undistinguishable_split(
            reformatted_df, output_cols, output_dir, output_name
        )

    reformatted_df = reformatted_df[output_cols]
    reformatted_df.to_csv(os.path.join(output_dir, output_name), index=False, sep="\t")
    return reformatted_df


def extract_organism_from_protein_id(protein_id: object) -> str:
    """Extract organism code from the last '_' token in a protein id."""
    if not isinstance(protein_id, str) or not protein_id:
        return ""

    # Keep first accession when multiple ids are concatenated with ';'
    first_token = protein_id.split(";")[0]
    if "_" not in first_token:
        return ""
    return first_token.rsplit("_", 1)[-1].upper()


def add_filtered_organism_column(
    df: pd.DataFrame,
    *,
    protein_col: str = "Protein IDs",
    organism_col: str = "Species",
    allowed_species: tuple[str, ...] = ALLOWED_ORGANISMS,
) -> pd.DataFrame:
    """
    Create `organism_col` from `protein_col` and keep only allowed species.
    """
    if protein_col not in df.columns:
        raise ValueError(f"Column '{protein_col}' not found in input dataframe.")

    out = df.copy()
    out[organism_col] = out[protein_col].map(extract_organism_from_protein_id)
    out = out[out[organism_col].isin(allowed_species)].copy()
    return out


def resolve_organisms_to_plot(
    requested_organisms: list[str], available_organisms: set[str]
) -> list[str]:
    """
    Resolve configured organism labels to available labels (code <-> full-name).
    """
    inverse_map = {v: k for k, v in ORGANISM_NAME_MAP.items()}
    resolved = []
    for org in requested_organisms:
        if org in available_organisms:
            resolved.append(org)
            continue
        if org in ORGANISM_NAME_MAP and ORGANISM_NAME_MAP[org] in available_organisms:
            resolved.append(ORGANISM_NAME_MAP[org])
            continue
        if org in inverse_map and inverse_map[org] in available_organisms:
            resolved.append(inverse_map[org])
            continue
    return resolved


def get_log_fc_ratio_for_entry(
    df: pd.DataFrame,
    cond_A_keyword: str = "HYE124_A",
    cond_B_keyword: str = "HYE124_B",
) -> pd.DataFrame:
    """
    Compute per-row log2 fold-change ratio in the directLFQ results table.
    This is a helper function for plotting expected vs observed fold changes.

    Parameters
    ----------
    df
        DataFrame containing the directLFQ results with columns for each sample.
    cond_A_keyword
        Substring to identify condition A samples in column names.
    cond_B_keyword
        Substring to identify condition B samples in column names.

    Returns
    -------
    pd.DataFrame
        Copy of `df` with a new `log2_fc_A_B` column. Each row is calculated as
        log2(median(cond_B) / median(cond_A)) using matching columns.
    """
    cond_A_cols = [col for col in df.columns if cond_A_keyword in col]
    cond_B_cols = [col for col in df.columns if cond_B_keyword in col]
    if not cond_A_cols or not cond_B_cols:
        raise ValueError(
            "Could not find condition columns with the provided keywords: "
            f"cond_A_keyword='{cond_A_keyword}', cond_B_keyword='{cond_B_keyword}'."
        )

    out = df.copy()

    # Convert to numeric, replace zeros with NaN, then aggregate per row.
    cond_A_values = (
        out[cond_A_cols].apply(pd.to_numeric, errors="coerce").replace(0, np.nan)
    )
    cond_B_values = (
        out[cond_B_cols].apply(pd.to_numeric, errors="coerce").replace(0, np.nan)
    )
    out["median_A"] = cond_A_values.median(axis=1, skipna=True)
    out["median_B"] = cond_B_values.median(axis=1, skipna=True)
    median_A = cond_A_values.median(axis=1, skipna=True)
    median_B = cond_B_values.median(axis=1, skipna=True)

    ratio = median_B / median_A
    ratio = ratio.where((median_A > 0) & ratio.notna())
    out["log2_fc_A_B"] = np.log2(ratio)

    return out


def _save_fig(
    fig,
    fig_dir: str,
    fig_name: str,
    qdf: pd.DataFrame | None = None,
    save_svg: bool = True,
) -> None:
    """Save figure and underlying table for protein quant plots."""
    os.makedirs(fig_dir, exist_ok=True)
    png_path = os.path.join(fig_dir, f"{fig_name}.png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    Logger.info("Saved plot to %s", png_path)
    if save_svg:
        svg_path = os.path.join(fig_dir, f"{fig_name}.svg")
        fig.savefig(svg_path, dpi=300, bbox_inches="tight")
        Logger.info("Saved plot to %s", svg_path)

    if qdf is not None:
        csv_path = os.path.join(fig_dir, f"{fig_name}.csv")
        qdf.to_csv(csv_path, index=False)
        Logger.info("Saved plot table to %s", csv_path)


def _prepare_qdf_from_combined_protein(
    combined_protein: pd.DataFrame,
    int_col_keyword: str = "MaxLFQ Intensity",
    filtered_proteins: Optional[list] = None,
    id_cols: list[str] = ["Protein ID", "Organism"],
    cond_A_keyword: str = "HYE124_A",
    cond_B_keyword: str = "HYE124_B",
    min_valid_per_cond: int = 2,
) -> tuple[pd.DataFrame, int]:
    """
    Prepare a long-form quantification table with columns:
    `log2_Intensity_ref`, `ratio`, and `Organism`.
    """
    if combined_protein.empty:
        raise ValueError("combined_protein is empty.")

    n_total_proteins = int(combined_protein.shape[0])
    qdf = combined_protein.copy()

    missing_id_cols = [c for c in id_cols if c not in qdf.columns]
    if missing_id_cols:
        raise ValueError(f"Missing required id columns: {missing_id_cols}")

    protein_id_col = id_cols[0]
    organism_col = id_cols[1]
    if organism_col != "Organism":
        qdf = qdf.rename(columns={organism_col: "Organism"})

    if filtered_proteins:
        qdf = qdf[qdf[protein_id_col].isin(filtered_proteins)].copy()

    intensity_cols = [c for c in qdf.columns if int_col_keyword in c]
    if not intensity_cols:
        raise ValueError(f"No columns contain int_col_keyword='{int_col_keyword}'.")

    cond_A_cols = [c for c in intensity_cols if cond_A_keyword in c]
    cond_B_cols = [c for c in intensity_cols if cond_B_keyword in c]

    # Fallback heuristics for condition columns if explicit keywords are absent.
    if not cond_A_cols:
        cond_A_cols = [c for c in intensity_cols if ("_A" in c or "MixA" in c)]
    if not cond_B_cols:
        cond_B_cols = [c for c in intensity_cols if ("_B" in c or "MixB" in c)]

    if not cond_A_cols or not cond_B_cols:
        raise ValueError(
            "Could not infer condition columns for A/B from intensity columns. "
            f"Found A={len(cond_A_cols)}, B={len(cond_B_cols)}."
        )

    cond_A_values = (
        qdf[cond_A_cols].apply(pd.to_numeric, errors="coerce").replace(0, np.nan)
    )
    cond_B_values = (
        qdf[cond_B_cols].apply(pd.to_numeric, errors="coerce").replace(0, np.nan)
    )
    valid_A = cond_A_values.notna().sum(axis=1) >= min_valid_per_cond
    valid_B = cond_B_values.notna().sum(axis=1) >= min_valid_per_cond
    valid_mask = valid_A & valid_B

    median_A = cond_A_values.median(axis=1, skipna=True).where(valid_mask)
    median_B = cond_B_values.median(axis=1, skipna=True).where(valid_mask)

    qdf["log2_Intensity_ref"] = np.log2((median_A + median_B) / 2.0)
    qdf["ratio"] = np.log2(median_B / median_A)

    qdf = qdf.replace([np.inf, -np.inf], np.nan)
    qdf = qdf.dropna(subset=["log2_Intensity_ref", "ratio", "Organism"]).copy()
    return qdf, n_total_proteins


def plot_protein_quant(
    combined_protein: pd.DataFrame,
    color_map: Optional[dict] = None,
    fig_dir: Optional[str] = None,
    dataset_name: str = "",
    filtered_proteins: Optional[list] = None,
    annot_fontsize: int = 12,
    label_map: Optional[dict] = None,
    int_col_keyword: str = "MaxLFQ Intensity",
    bar_ylim: tuple[float, float] = (0, 4500),
    scatter_xlim: Optional[tuple[float, float]] = None,
    id_cols: list[str] = ["Protein ID", "Organism"],
    cond_A_keyword: str = "HYE124_A",
    cond_B_keyword: str = "HYE124_B",
    min_valid_per_cond: int = 2,
):
    plt.rcParams.update({"font.size": annot_fontsize})
    qdf, n_total_proteins = _prepare_qdf_from_combined_protein(
        combined_protein=combined_protein,
        int_col_keyword=int_col_keyword,
        filtered_proteins=filtered_proteins,
        id_cols=id_cols,
        cond_A_keyword=cond_A_keyword,
        cond_B_keyword=cond_B_keyword,
        min_valid_per_cond=min_valid_per_cond,
    )

    if label_map is None:
        label_map = {}
    if color_map is None:
        orgs = sorted(qdf["Organism"].dropna().unique().tolist())
        palette = sns.color_palette("tab10", n_colors=max(1, len(orgs)))
        color_map = {org: palette[i] for i, org in enumerate(orgs)}

    organisms_present = [org for org in color_map.keys() if org in set(qdf["Organism"])]
    if not organisms_present:
        organisms_present = sorted(qdf["Organism"].dropna().unique().tolist())

    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(1, 3, width_ratios=[1, 3, 1], wspace=0.25)
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_scatter = fig.add_subplot(gs[0, 1])
    ax_kde = fig.add_subplot(gs[0, 2], sharey=ax_scatter)

    # Left: bar plot of quantifiable protein counts per organism.
    count_df = qdf.groupby("Organism").size().reindex(organisms_present).fillna(0)
    x_pos = np.arange(len(count_df.index))
    ax_bar.bar(
        x_pos,
        count_df.values,  # type: ignore[union-attr]
        color=[color_map[o] for o in count_df.index],
    )
    ax_bar.set_ylabel("Quantified Proteins Count")
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels([label_map.get(x, x) for x in count_df.index], rotation=90)
    ax_bar.set_ylim(bar_ylim[0], bar_ylim[1])
    for i, v in enumerate(count_df.values):
        ax_bar.text(
            i, v, str(int(v)), ha="center", va="bottom", fontsize=annot_fontsize * 0.8
        )

    # Middle: scatter plot (rasterized points to keep vector export lightweight).
    sns.scatterplot(
        data=qdf,
        x="log2_Intensity_ref",
        y="ratio",
        hue="Organism",
        palette=color_map,
        hue_order=organisms_present,
        alpha=0.2,
        s=4,
        legend=False,
        ax=ax_scatter,
    )
    for artist in ax_scatter.collections:
        artist.set_rasterized(True)

    for org in organisms_present:
        sub = qdf[qdf["Organism"] == org].dropna(subset=["log2_Intensity_ref", "ratio"])
        if len(sub) < 10:
            continue
        smoothed = lowess(
            sub["ratio"].values, sub["log2_Intensity_ref"].values, frac=0.3
        )
        ax_scatter.plot(
            smoothed[:, 0],
            smoothed[:, 1],
            color=color_map[org],
            # color="black",
            linewidth=1.5,
            linestyle="-",
            zorder=3,
        )

    stats = qdf.groupby("Organism")["ratio"].agg(
        count="count",
        q1=lambda x: x.quantile(0.25),
        median="median",
        q3=lambda x: x.quantile(0.75),
    )
    Logger.info("Quantile stats:\n%s", stats)

    for i, (org, row) in enumerate(stats.iterrows()):
        c = color_map[org] if org in color_map else "black"
        display_org = label_map.get(org, org)
        # ax_scatter.axhline(row["median"], linestyle="-", linewidth=1.2, color=c)
        # ax_scatter.axhline(row["q1"], linestyle="--", linewidth=1, color=c)
        # ax_scatter.axhline(row["q3"], linestyle="--", linewidth=1, color=c)
        text = (
            f"{display_org}: n={int(row['count'])}, "
            f"Med={row['median']:.2f}, Q3-Q1={row['q3'] - row['q1']:.2f}"
        )
        ax_scatter.text(
            0.02,
            0.98 - (i * 0.055),
            text,
            transform=ax_scatter.transAxes,
            fontsize=annot_fontsize * 0.8,
            verticalalignment="top",
        )

    ax_scatter.set_xlabel("log2 Intensity, Mean of MixA and MixB")
    ax_scatter.set_ylabel("log2 Ratio, MixB/MixA")
    ax_scatter.set_ylim(-3, 4)
    ax_scatter.set_xlim(scatter_xlim if scatter_xlim is not None else (6, 22))
    ax_scatter.set_title(
        f"{dataset_name} | Total IDs: {n_total_proteins}, Quantifiable: {len(qdf)}"
    )
    ax_scatter.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax_scatter.axhline(-1, color="black", linewidth=0.8, linestyle=":")
    ax_scatter.axhline(2, color="black", linewidth=0.8, linestyle=":")

    # Right: marginal ratio density per organism.
    for org in organisms_present:
        sub = qdf[qdf["Organism"] == org]
        if sub.empty:
            continue
        if sub["ratio"].nunique(dropna=True) > 1:
            sns.kdeplot(
                y=sub["ratio"],
                ax=ax_kde,
                color=color_map[org],
                fill=True,
                alpha=0.4,
                linewidth=1,
            )
        if org in stats.index:
            ax_kde.axhline(
                stats.loc[org, "q1"], color=color_map[org], linewidth=1, linestyle="--"
            )
            ax_kde.axhline(stats.loc[org, "median"], color=color_map[org], linewidth=1)
            ax_kde.axhline(
                stats.loc[org, "q3"], color=color_map[org], linewidth=1, linestyle="--"
            )

    ax_kde.set_xlabel("Density")
    ax_kde.set_ylabel("")
    ax_kde.tick_params(labelleft=False)
    ax_kde.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax_kde.axhline(-1, color="black", linewidth=0.8, linestyle=":")
    ax_kde.axhline(2, color="black", linewidth=0.8, linestyle=":")
    plt.tight_layout()

    if fig_dir:
        fig_name = f"protein_quant_{dataset_name.replace(' ', '_')}"
        if filtered_proteins:
            fig_name += "_filtered"
        _save_fig(fig=fig, qdf=qdf, fig_dir=fig_dir, fig_name=fig_name)
        plt.close(fig)
    else:
        plt.show()
    return qdf


def plot_protein_quant_rolling_quantiles(
    combined_protein: pd.DataFrame,
    color_map: Optional[dict] = None,
    fig_dir: Optional[str] = None,
    dataset_name: str = "",
    filtered_proteins: Optional[list] = None,
    annot_fontsize: int = 12,
    label_map: Optional[dict] = None,
    int_col_keyword: str = "MaxLFQ Intensity",
    id_cols: list[str] = ["Protein ID", "Organism"],
    window: int = 200,
    step: int = 50,
    xlim: Optional[tuple] = None,
    ylim: Optional[tuple] = None,
    ref_y_line: Optional[list] = None,
    cond_A_keyword: str = "HYE124_A",
    cond_B_keyword: str = "HYE124_B",
    min_valid_per_cond: int = 2,
):
    """Plot rolling window quantiles of log2 ratio vs log2 intensity per organism."""
    plt.rcParams.update({"font.size": annot_fontsize})
    qdf, n_total_proteins = _prepare_qdf_from_combined_protein(
        combined_protein=combined_protein,
        int_col_keyword=int_col_keyword,
        filtered_proteins=filtered_proteins,
        id_cols=id_cols,
        cond_A_keyword=cond_A_keyword,
        cond_B_keyword=cond_B_keyword,
        min_valid_per_cond=min_valid_per_cond,
    )

    if label_map is None:
        label_map = {}
    if color_map is None:
        orgs = sorted(qdf["Organism"].dropna().unique().tolist())
        palette = sns.color_palette("tab10", n_colors=max(1, len(orgs)))
        color_map = {org: palette[i] for i, org in enumerate(orgs)}

    fig, ax = plt.subplots(figsize=(9, 5))
    for org in color_map:
        sub = (
            qdf[qdf["Organism"] == org]
            .sort_values("log2_Intensity_ref")
            .reset_index(drop=True)
        )
        if len(sub) < window:
            Logger.warning(
                "Organism %s has fewer proteins (%d) than window size (%d); skipping.",
                org,
                len(sub),
                window,
            )
            continue

        centers, medians, q1s, q3s = [], [], [], []
        for start in range(0, len(sub) - window + 1, step):
            chunk = sub.iloc[start : start + window]
            centers.append(chunk["log2_Intensity_ref"].median())
            medians.append(chunk["ratio"].median())
            q1s.append(chunk["ratio"].quantile(0.25))
            q3s.append(chunk["ratio"].quantile(0.75))

        centers = np.array(centers)
        medians = np.array(medians)
        q1s = np.array(q1s)
        q3s = np.array(q3s)

        color = color_map[org]
        display = label_map.get(org, org)
        ax.fill_between(centers, q1s, q3s, color=color, alpha=0.25)
        ax.plot(centers, medians, color=color, linewidth=1.8, label=display)
        ax.plot(centers, q1s, color=color, linewidth=0.8, linestyle="--")
        ax.plot(centers, q3s, color=color, linewidth=0.8, linestyle="--")

    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xlabel("log2 Intensity, Mean of MixA and MixB")
    ax.set_ylabel("log2 Ratio, MixB/MixA")
    ax.set_title(
        f"{dataset_name} | Total Protein IDs: {n_total_proteins}, Quantifiable: {len(qdf)}"
        f"\n(rolling window: {window} proteins, step: {step})"
    )
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if ref_y_line is not None:
        for y in ref_y_line:
            ax.axhline(y, color="grey", linewidth=0.8, linestyle="--")
    ax.legend(
        fontsize=annot_fontsize * 0.85,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
    )
    plt.tight_layout()

    if fig_dir:
        fig_name = f"protein_quant_rolling_{dataset_name.replace(' ', '_')}"
        if filtered_proteins:
            fig_name += "_filtered"
        _save_fig(fig=fig, qdf=qdf, fig_dir=fig_dir, fig_name=fig_name)
        plt.close(fig)
    else:
        plt.show()
    return qdf
