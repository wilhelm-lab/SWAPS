import pandas as pd
import logging
import numpy as np
from peak_detection_2d.utils import calc_fdr_and_thres

Logger = logging.getLogger(__name__)


def add_decoy_prefix(row: pd.Series):
    proteins = row["Proteins"]
    if not row["Decoy"]:
        return proteins

    return ";".join([f"REV__{p}" for p in proteins.split(";")])


def empirical_p_value(score, decoy_scores):
    """Compute empirical p-value for a given protein score."""
    return (np.sum(decoy_scores >= score) + 1) / (len(decoy_scores) + 1)


def calc_protein_score(
    maxquant_dict: pd.DataFrame,
    ps_result: pd.DataFrame,
    low_int_penalty: float = 0.0001,
    min_hit: int = 1,
    save_dir: str = None,
    metric: str = "arithm",
):
    """
    Calculate protein score for each protein.
    """
    ps_result_full = pd.merge(left=maxquant_dict, right=ps_result, on="mz_rank")
    ps_result_full["Proteins"] = ps_result_full[["Proteins", "Decoy"]].apply(
        add_decoy_prefix, axis=1
    )
    ps_result_full.loc[ps_result_full["sum_intensity"] < 100, "target_decoy_score"] = (
        low_int_penalty
    )
    ps_result_full["target_decoy_score"] = ps_result_full["target_decoy_score"].clip(
        low_int_penalty, 1
    )

    # Score calcualtion
    ## Explode the "Leading proteins" column and create a mapping to scores
    exploded_df = ps_result_full.assign(
        Protein_unique=ps_result_full["Proteins"].str.split(";")
    ).explode("Protein_unique")

    ## Group by protein and calculate means
    protein_score_df = (
        exploded_df.groupby("Protein_unique")
        .agg(
            {
                "target_decoy_score": [
                    (
                        "target_decoy_score_arithm_mean",
                        lambda x: np.mean(x),
                    ),  # Compute arithmetic mean
                    ("count", "count"),  # Count peptides per protein
                    (
                        "target_decoy_score_geom_mean",
                        lambda x: np.exp(np.mean(np.log(x))),
                    ),  # Compute geometric mean
                ]
            }
        )
        .reset_index()
    )

    ## Flatten column names
    protein_score_df.columns = [
        "Protein_unique",
        "target_decoy_score_arithm_mean",
        "count",
        "target_decoy_score_geom_mean",
    ]
    ## Add Decoy column based on protein name
    protein_score_df["Decoy"] = protein_score_df["Protein_unique"].str.contains("REV__")
    ## Remove REV__ prefix from protein names
    protein_score_df["Protein_name"] = protein_score_df["Protein_unique"].str.replace(
        "REV__", ""
    )
    match metric:
        case "arithm":
            col = "target_decoy_score_arithm_mean"
        case "geom":
            col = "target_decoy_score_geom_mean"
        case _:
            raise ValueError(f"Unknown metric: {metric}, please use 'arithm' or 'geom'")
    Logger.info("Number of proteins: %d", protein_score_df["Protein_name"].nunique())
    protein_tdc = protein_score_df.groupby("Protein_name").filter(
        lambda x: x[col].nunique() > 1
    )  # Remove groups with identical scores
    Logger.info(
        "After removing groups with identical scores, %d proteins remain",
        protein_tdc["Protein_name"].nunique(),
    )
    protein_tdc = (
        protein_tdc.sort_values(col, ascending=False)
        .groupby("Protein_name")
        .first()
        .reset_index()
    )
    protein_tdc_filtered = protein_tdc.loc[protein_tdc["count"] > min_hit]
    Logger.info(
        "After filtering out proteins with no more than %d hits, %d proteins remain",
        min_hit,
        protein_tdc_filtered["Protein_name"].nunique(),
    )
    protein_tdc_fdr = calc_fdr_and_thres(
        pred_df=protein_tdc_filtered.loc[protein_tdc_filtered["count"] > min_hit],
        score_col=col,
        return_plot=True,
        save_dir=save_dir,
        title="Protein-level FDR vs. Idenfitied Targets",
    )
    return protein_tdc_fdr
