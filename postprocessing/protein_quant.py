import os
import pandas as pd

# from peak_detection_2d.utils import calc_fdr_and_thres


def prepare_generic_input_from_dict_and_pept_act_sum(
    maxquant_dict: pd.DataFrame,
    pept_act_sum_df: pd.DataFrame,
    run_name: str,
    raw_file: str | None = None,
    agg_col: str = "Proteins",
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
        left=maxquant_dict[["mz_rank", "ion", agg_col]],
        right=pept_act_sum_df[["mz_rank", "sum_intensity"]],
        on="mz_rank",
        how="inner",
    )
    generic_input.drop(labels=["mz_rank"], axis=1, inplace=True)
    generic_input.rename(columns={"sum_intensity": run_name}, inplace=True)
    if agg_col == "Proteins":
        generic_input.rename(columns={"Proteins": input_to_join}, inplace=True)

    return generic_input


def prepare_generic_input_from_result_dir(
    swap_result_dir,
    run_name: str,
    raw_file: str | None = None,
    score_thres=0,
    log_intensity_thres=2,
    agg_col="Proteins",
):
    maxquant_dict = pd.read_pickle(
        os.path.join(swap_result_dir, "maxquant_result_ref.pkl")
    )
    pept_act_sum_df = pd.read_csv(
        os.path.join(
            swap_result_dir, "peak_selection", "pept_act_sum_ps_full_tdc_fdr_thres.csv"
        )
    )
    pept_act_sum_df = pept_act_sum_df.loc[
        (pept_act_sum_df["log_sum_intensity"] > log_intensity_thres)
        & (pept_act_sum_df["target_decoy_score"] > score_thres)
        & (pept_act_sum_df["Decoy"] == 0)
    ]
    return prepare_generic_input_from_dict_and_pept_act_sum(
        maxquant_dict, pept_act_sum_df, run_name, raw_file=raw_file, agg_col=agg_col
    )


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


def prepare_evidence_for_pickedgroupfdr(swaps_result_dir, run_name, out_dir):
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
    maxquant_dict = pd.read_pickle(
        os.path.join(swaps_result_dir, "maxquant_result_ref.pkl")
    )
    maxquant_dict.loc[maxquant_dict["Decoy"], "Reverse"] = "+"
    maxquant_dict["Raw file"] = run_name
    maxquant_dict["Experiment"] = run_name
    pept_act = pd.read_csv(
        os.path.join(
            swaps_result_dir, "peak_selection", "pept_act_sum_ps_full_tdc_fdr_thres.csv"
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
    # evidence["PEP"] = evidence["fdr"]  # Use q-value as PEP
    evidence = evidence[col_for_id + col_for_quant]
    out_dir = os.path.join(out_dir, "evidence_" + run_name + ".txt")
    evidence.to_csv(out_dir, sep="\t")
    return out_dir, evidence
