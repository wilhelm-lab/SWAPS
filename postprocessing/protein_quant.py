import os
from typing import Literal
import logging
import pandas as pd
from triqler.qvality import getQvaluesFromScores

Logger = logging.getLogger(__name__)
# from peak_detection_2d.utils import calc_fdr_and_thres


def prepare_generic_input_from_dict_and_pept_act_sum(
    maxquant_dict: pd.DataFrame,
    pept_act_sum_df: pd.DataFrame,
    run_name: str,
    raw_file: str | None = None,
    agg_col: Literal["Proteins", "Species"] = "Proteins",
    score_thres_by_species: dict | None = None,
):
    if agg_col == "Species":
        assert raw_file is not None
        maxquant_dict.loc[
            maxquant_dict["Raw file"] == raw_file,
            "Taxonomy names",
        ] = maxquant_dict.loc[
            maxquant_dict["Raw file"] == raw_file,
            "Protein group IDs",
        ].apply(
            replace_protein_group_ids
        )
        maxquant_dict["Species"] = maxquant_dict["Taxonomy names"].apply(
            get_unique_values
        )
        input_to_join = "Species"
    else:
        input_to_join = "protein"
    maxquant_dict["ion"] = (
        maxquant_dict["Modified sequence"] + "_" + maxquant_dict["Charge"].astype(str)
    )
    generic_input = pd.merge(
        left=maxquant_dict[["mz_rank", "ion", agg_col, "Taxonomy names"]],
        right=pept_act_sum_df[["mz_rank", "sum_intensity", "target_decoy_score"]],
        on="mz_rank",
        how="inner",
    )
    filtered_rows = pd.DataFrame()
    for species in score_thres_by_species.keys():
        tmp = generic_input.loc[
            (generic_input["Taxonomy names"] == species)
            & (generic_input["target_decoy_score"] > score_thres_by_species[species])
        ]
        filtered_rows = pd.concat([filtered_rows, tmp], axis=0)
    # generic_input = generic_input[
    #     generic_input.apply(
    #         _apply_score_threshold,
    #         axis=1,
    #         score_thres_by_species=score_thres_by_species,
    #     )
    # ]
    filtered_rows.drop(
        labels=["mz_rank", "Taxonomy names", "target_decoy_score"], axis=1, inplace=True
    )
    filtered_rows.rename(columns={"sum_intensity": run_name}, inplace=True)
    if agg_col == "Proteins":
        filtered_rows.rename(columns={"Proteins": input_to_join}, inplace=True)

    return filtered_rows


# Apply different score thresholds based on Taxonomy names
def _apply_score_threshold(row, score_thres_by_species):
    taxonomy_names = row["Taxonomy names"]
    for name in taxonomy_names:
        if name in score_thres_by_species:
            if row["target_decoy_score"] > score_thres_by_species[name]:
                return True
    return False


def prepare_generic_input_from_result_dir(
    swap_result_dir,
    # run_name: str,
    score_thres_by_species: dict,
    ps_exp_folder: str = "eval_model_transfer",
    log_intensity_thres=2,
    agg_col="Proteins",
):

    run_name = os.path.basename(swap_result_dir).split("ref_")[1][0:6]
    Logger.info("Processing %s", run_name)
    maxquant_dict = pd.read_pickle(
        os.path.join(swap_result_dir, "maxquant_result_ref.pkl")
    )
    pept_act_sum_df = pd.read_csv(
        os.path.join(
            swap_result_dir,
            "peak_selection",
            ps_exp_folder,
            "pept_act_sum_ps_full_tdc_fdr_thres.csv",
        )
    )

    pept_act_sum_df = pept_act_sum_df.loc[
        (pept_act_sum_df["log_sum_intensity"] > log_intensity_thres)
        & (pept_act_sum_df["Decoy"] == 0)
    ]
    # pept_act_sum_df = pept_act_sum_df[
    #     pept_act_sum_df.apply(apply_score_threshold, axis=1)
    # ]
    return prepare_generic_input_from_dict_and_pept_act_sum(
        maxquant_dict,
        pept_act_sum_df,
        run_name,
        raw_file=run_name,
        agg_col=agg_col,
        score_thres_by_species=score_thres_by_species,
    )


def prepare_generic_input_from_result_dir_list(
    swap_result_dir_list: list,
    result_dir: str,
    agg_col="Proteins",
    score_thres_by_species_a={
        "Homo sapiens": 0.2,
        "Saccharomyces cerevisiae": 0.4,
        "Escherichia coli K-12": 0.9,
    },
    score_thres_by_species_b={
        "Homo sapiens": 0.2,
        "Saccharomyces cerevisiae": 0.9,
        "Escherichia coli K-12": 0.2,
    },
):
    if agg_col == "Proteins":
        input_to_join = "protein"
    else:
        input_to_join = agg_col
    generic_input = pd.DataFrame()
    # Define score thresholds for different taxonomy names

    for swaps_dir in swap_result_dir_list:
        mixture = os.path.basename(swaps_dir).split("ref_")[1][0]

        if mixture == "A":
            score_thres_by_species = score_thres_by_species_a
        elif mixture == "B":
            score_thres_by_species = score_thres_by_species_b
        df = prepare_generic_input_from_result_dir(
            swaps_dir,
            agg_col=agg_col,
            score_thres_by_species=score_thres_by_species,
        )
        if generic_input.shape[0] == 0:
            generic_input = df
        else:
            generic_input = pd.merge(
                left=generic_input, right=df, on=["ion", input_to_join], how="outer"
            )

    # concatenate_proteins groups
    generic_input_grouped = (
        generic_input.groupby("ion")
        .agg(
            {
                input_to_join: concatenate_proteins,  # Apply the custom function to 'protein'
                **{
                    col: "first"
                    for col in generic_input.columns
                    if col != "ion" and col != input_to_join
                },  # Keep the first value of other columns
            }
        )
        .reset_index()
    )
    generic_input_grouped = generic_input_grouped[
        [input_to_join, "ion"] + generic_input_grouped.columns[2:].tolist()
    ]
    generic_input_grouped.fillna(0, inplace=True)
    # Concatenate key and value in dict score_thres_by_species into a string
    score_thres_str = "_".join(
        f"{species}_{thres}" for species, thres in score_thres_by_species.items()
    )
    Logger.info("Score thresholds by species: %s", score_thres_str)
    generic_input_path = os.path.join(
        result_dir, f"generic_input_{agg_col}_score_{score_thres_str}.aq_reformat.tsv"
    )
    generic_input_grouped.to_csv(
        generic_input_path,
        sep="\t",
        index=False,
    )
    return generic_input_path


def concatenate_proteins(proteins):
    # Split each entry by ';' and flatten the list
    split_proteins = [p.strip() for sublist in proteins for p in sublist.split(";")]

    # Remove duplicates by converting to set, then join the unique values back with ';'
    unique_proteins = ";".join(sorted(set(split_proteins)))

    return unique_proteins


# Function to replace Protein group IDs with corresponding Taxonomy names
def replace_protein_group_ids(protein_group_ids, dict_id_to_taxonomy_map):
    # Split the 'Protein group IDs' string by ';'
    ids = protein_group_ids.split(";")

    # Replace each id with the corresponding 'Taxonomy names' using the dictionary
    replaced = [dict_id_to_taxonomy_map.get(int(i), "") for i in ids]

    # Join the replaced values back with ';'
    return ";".join(replaced)


def get_unique_values(cell_string):
    # If there's no ';' (only one entry), return the string itself
    cell_string = str(cell_string)
    if ";" not in cell_string:
        return cell_string
    else:
        # Split the string by ';' and get unique values using a set
        values = cell_string.split(";")
        unique_values = list(set(values))

        # Join the unique values back into a string separated by ';'
        return ";".join(unique_values)


# Function to populate 'Taxonomy names' with 'Homo sapiens' based on the number of Protein IDs
def populate_taxonomy_from_protein_ids(protein_ids):
    # Count the number of protein IDs by splitting with ';'
    n = len(protein_ids.split(";"))
    # Return 'Homo sapiens' repeated n times, concatenated with ';'
    return ";".join(["Homo sapiens"] * n)


# Function to append species based on conditions
def append_species_from_fasta_header(fasta_header):
    species = []

    if isinstance(fasta_header, str) and "HUMAN" in fasta_header:
        species.append("Homo sapiens")

    if isinstance(fasta_header, str) and (
        ("YEAST" in fasta_header) or ("Saccharomyces cerevisiae" in fasta_header)
    ):
        species.append("Saccharomyces cerevisiae")

    if isinstance(fasta_header, str) and "ECOLI" in fasta_header:
        species.append("Escherichia coli K-12")

    # Join all species found by ';' (if any)
    return ";".join(species)


def add_decoy_prefix(row: pd.Series):
    proteins = row["Leading proteins"]
    if row["Reverse"] != "+":
        return proteins

    return ";".join([f"REV__{p}" for p in proteins.split(";")])


def prepare_evidence_for_pickedgroupfdr(
    swaps_result_dir_list,
    out_dir,
    ps_exp_folder: str = "eval_model_transfer",
    keep_col_for_tmt: bool = False,
    keep_col_for_silac: bool = False,
    keep_col_for_directlfq: bool = True,
):
    col_for_id = [
        "Modified sequence",
        "Leading proteins",
        "Score",
        "Experiment",
        "Length",
        "Reverse",
        "PEP",
    ]
    col_for_quant = ["Charge", "Intensity", "Raw file", "id"]
    col_for_tmt = [
        "Reporter intensity corrected",
        "Reporter intensity",
        "Reporter intensity count",
    ]
    col_for_silac = ["Intensity L", "Intensity H"]
    col_for_directlfq = ["Protein group IDs", "Potential contaminant", "Proteins"]
    col_to_keep = col_for_id + col_for_quant
    if keep_col_for_tmt:
        col_to_keep += col_for_tmt
        Logger.warning("TMT columns updates are not implemented in SWAPS yet")
    if keep_col_for_silac:
        col_to_keep += col_for_silac
        Logger.warning("SILAC columns updates are not implemented in SWAPS yet")
    if keep_col_for_directlfq:
        col_to_keep += col_for_directlfq
        Logger.info("Keep %s for directLFQ", col_for_directlfq)
    evidence_all = pd.DataFrame()
    for swaps_result_dir in swaps_result_dir_list:
        run_name = swaps_result_dir.split("ref_")[1][0:6]
        Logger.info("Processing %s", run_name)
        maxquant_dict = pd.read_pickle(
            os.path.join(swaps_result_dir, "maxquant_result_ref.pkl")
        )
        maxquant_dict.loc[maxquant_dict["Decoy"], "Reverse"] = "+"
        maxquant_dict["Raw file"] = run_name
        maxquant_dict["Experiment"] = run_name
        pept_act = pd.read_csv(
            os.path.join(
                swaps_result_dir,
                "peak_selection",
                ps_exp_folder,
                "pept_act_sum_ps_full_tdc_fdr_thres.csv",
            )
        )
        # pept_act = calc_fdr_given_thres(pept_act)
        evidence = pd.merge(
            maxquant_dict,
            pept_act[["mz_rank", "sum_intensity", "target_decoy_score"]],
            on="mz_rank",
            how="inner",
        )
        # evidence = calc_fdr_and_thres(evidence)
        evidence["id"] = evidence["mz_rank"]
        evidence["Score"] = evidence["target_decoy_score"]
        evidence["Intensity"] = evidence["sum_intensity"]
        evidence["Leading proteins"] = evidence[["Leading proteins", "Reverse"]].apply(
            add_decoy_prefix, axis=1
        )
        # evidence["PEP"] = evidence["fdr"]  # Use q-value as PEP
        evidence = calc_pep_in_evidence(evidence)
        evidence = evidence[col_to_keep]
        evidence_all = pd.concat([evidence_all, evidence], axis=0)
    out_dir = os.path.join(out_dir, "evidence_swaps.txt")
    evidence_all.to_csv(out_dir, sep="\t")
    return out_dir, evidence_all


def calc_pep_in_evidence(
    evidence: pd.DataFrame, include_decoys=True, plot_regression_curve: bool = True
):
    target_scores = evidence.loc[evidence["Reverse"].isna(), "Score"]
    decoy_scores = evidence.loc[~evidence["Reverse"].isna(), "Score"]
    _, evidence["PEP"] = getQvaluesFromScores(
        target_scores,
        decoy_scores,
        includeDecoys=include_decoys,
        plotRegressionCurve=plot_regression_curve,
        pi0=len(decoy_scores) / (len(target_scores) + len(decoy_scores)),
        numBins=1000,
    )
    return evidence
