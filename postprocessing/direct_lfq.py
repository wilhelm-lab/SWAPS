# run mq evidence
import directlfq.lfq_manager as lfq_manager
import directlfq.benchmarking as lfqbenchmark
import directlfq.utils as lfqutils
import fire
import logging
import pandas as pd
import os
from .singleton_direct_lfq import direct_lfq_config
from utils.config import get_cfg_defaults
import matplotlib.pyplot as plt
import shutil
import numpy as np


# TODO refactor script
def direct_lfq(direct_lfq_config_path):
    cfg = get_cfg_defaults(direct_lfq_config)
    cfg.merge_from_file(direct_lfq_config_path)
    assert len(cfg.SWAPS_RESULT_DIR_LIST) > 0
    os.makedirs(cfg.RESULT_DIR, exist_ok=True)

    samplemap_directlfq = os.path.join(cfg.RESULT_DIR, "samplemap.tsv")
    directlfq_file = os.path.join(
        cfg.RESULT_DIR,
        "evidence_swaps_int.txt.protgroup_annotated.tsv.maxquant_evidence.protein_intensities.tsv",
    )

    if os.path.exists(directlfq_file) and os.path.exists(samplemap_directlfq):
        logging.info("directLFQ results already exist, skipping")

    else:
        logging.info("Running directLFQ")
        # prepare inputs
        evidence_swaps_int = pd.DataFrame()
        for swap_dir in cfg.SWAPS_RESULT_DIR_LIST:
            maxquant_result_ref = pd.read_pickle(
                os.path.join(swap_dir, "maxquant_result_ref.pkl")
            )
            maxquant_result_ref = maxquant_result_ref.sort_values("mz_rank")
            # pept_act_sum_df = pd.read_csv(
            #     os.path.join(
            #         swap_dir, "results", "activation", cfg.INTENSITY_COLUMN + ".csv"
            #     )
            # )
            pept_act_sum_df = pd.read_csv(
                os.path.join(
                    swap_dir,
                    "results",
                    "peak_selection",
                    "pept_act_sum_ps_full_tdc_fdr_thres.csv",
                )
            )
            pept_act_sum_df = pept_act_sum_df.loc[1:]
            maxquant_result_ref["Intensity"] = pept_act_sum_df[
                cfg.INTENSITY_COLUMN
            ].values
            # evidence_swaps_int = pd.concat(
            #     [evidence_swaps_int, maxquant_result_ref], axis=0
            # )
            logging.info("Filtering activation results")
            if "log_sum_intensity" in pept_act_sum_df.columns:
                pept_act_sum_df["log_sum_intensity"] = np.log10(
                    pept_act_sum_df["log_sum_intensity"] + 1
                )
            pept_act_sum_df = pept_act_sum_df.loc[
                (pept_act_sum_df["log_sum_intensity"] > cfg.FILTER.LOG_INTENSITY_THRES)
                & (pept_act_sum_df["target_decoy_score"] > cfg.FILTER.SCORE_THRES)
                & (pept_act_sum_df["Decoy"] == 0)
            ]
            evidence_swaps_int = pd.merge(
                left=maxquant_result_ref,
                right=pept_act_sum_df,
                on="mz_rank",
                how="inner",
            )
            logging.info("evidence_swaps_int shape: %s", evidence_swaps_int.shape)
        if cfg.RAW_FILE_AS_EXPERIMENT:
            samplemap = evidence_swaps_int[["Raw file", "Experiment"]].drop_duplicates()
            samplemap.rename(
                columns={"Raw file": "sample", "Experiment": "condition"}, inplace=True
            )
            evidence_swaps_int["Experiment"] = evidence_swaps_int["Raw file"]
        else:
            samplemap = pd.DataFrame(cfg.SAMPLE_MAP, columns=["sample", "condition"])
        samplemap.to_csv(
            os.path.join(cfg.RESULT_DIR, "samplemap.tsv"),
            sep="\t",
            index=False,
        )
        evidence_swaps_int.to_csv(
            os.path.join(cfg.RESULT_DIR, "evidence_swaps_int.txt"),
            sep="\t",
            index=False,
        )

        # run direct_lfq
        lfq_manager.run_lfq(
            input_file=os.path.join(cfg.RESULT_DIR, "evidence_swaps_int.txt"),
            mq_protein_groups_txt=cfg.MQ_PROTEIN_GROUP_PATH,
            input_type_to_use="maxquant_evidence",
        )

    if cfg.REF_RESULT_FILE != "":
        ref_directlfq_file = os.path.join(
            cfg.RESULT_DIR,
            "evidence.txt.protgroup_annotated.tsv.maxquant_evidence.protein_intensities.tsv",
        )
        if os.path.exists(ref_directlfq_file):
            logging.info("Reference directLFQ results already exist, skipping")
        else:
            shutil.copy(cfg.REF_RESULT_FILE, cfg.RESULT_DIR)
            new_ref_path = os.path.join(
                cfg.RESULT_DIR, os.path.basename(cfg.REF_RESULT_FILE)
            )
            logging.info("Running directLFQ on reference data")
            lfq_manager.run_lfq(
                input_file=new_ref_path,
                mq_protein_groups_txt=cfg.MQ_PROTEIN_GROUP_PATH,
                input_type_to_use="maxquant_evidence",
            )

    # benchmark results
    logging.info("Benchmarking directLFQ results")
    # prepare group protein.txt
    try:
        organism_annotator = lfqbenchmark.OrganismAnnotatorMaxQuant(
            mapping_file=cfg.MQ_PROTEIN_GROUP_PATH, protein_column="Protein IDs"
        )
    except (
        ValueError
    ):  # if the protein group file does not have the species column, generate it TODO: needs to be updated
        ori_mq_search = pd.read_csv(
            cfg.MQ_PROTEIN_GROUP_PATH,
            sep="\t",
        )
        ori_mq_search["Species"] = ori_mq_search["Fasta headers"].apply(
            lambda x: (
                "Homo sapiens"
                if isinstance(x, str) and "HUMAN" in x
                else (
                    "Saccharomyces cerevisiae"
                    if isinstance(x, str) and "YEAST" in x
                    else (
                        "Escherichia coli"
                        if isinstance(x, str) and "ECOLI" in x
                        else ""
                    )
                )
            )
        )
        ori_mq_search.to_csv(
            os.path.join(cfg.RESULT_DIR, "proteinGroups_added_species.txt"),
            sep="\t",
            index=False,
        )
        protein_group = os.path.join(cfg.RESULT_DIR, "proteinGroups_added_species.txt")
        organism_annotator = lfqbenchmark.OrganismAnnotatorMaxQuant(
            mapping_file=protein_group, protein_column="Protein IDs"
        )
    samplemap_df_directlfq = lfqutils.load_samplemap(samplemap_directlfq)
    samples_used_directlfq = lfqutils.get_samples_used_from_samplemap_df(
        samplemap_df_directlfq, cond1=cfg.CONDITIONS[0], cond2=cfg.CONDITIONS[1]
    )

    restable_directlfq = lfqbenchmark.ResultsTableDirectLFQ(
        input_file=directlfq_file,
        input_name="directLFQ",
        samples_c1=samples_used_directlfq[0],
        samples_c2=samples_used_directlfq[1],
    )
    organism_annotator.annotate_table_with_organism(restable_directlfq)
    fig, axes = plt.subplots(1, 1)
    fcplotter_directLFQ = lfqbenchmark.MultiOrganismIntensityFCPlotter(
        ax=axes,
        resultstable_w_ratios=restable_directlfq,
        organisms_to_plot=cfg.FC.PLOT_ORGANISMS,
        # organisms_to_plot=["Homo sapien", "Yeast", "Ecoli"],
        fcs_to_expect=cfg.FC.EXPECTATION,
        title=cfg.FC.TITLE,
    )
    fig.savefig(os.path.join(cfg.RESULT_DIR, "FC_plot.png"), dpi=300)
    if cfg.REF_RESULT_FILE != "":
        restable_ref_directlfq = lfqbenchmark.ResultsTableDirectLFQ(
            input_file=ref_directlfq_file,
            input_name="ref_directLFQ",
            samples_c1=samples_used_directlfq[0],
            samples_c2=samples_used_directlfq[1],
        )
        organism_annotator.annotate_table_with_organism(restable_ref_directlfq)
        fig, axes = plt.subplots(1, 1)
        fcplotter_directLFQ_ref = lfqbenchmark.MultiOrganismIntensityFCPlotter(
            ax=axes,
            resultstable_w_ratios=restable_ref_directlfq,
            organisms_to_plot=cfg.FC.PLOT_ORGANISMS,
            # organisms_to_plot=["Homo sapien", "Yeast", "Ecoli"],
            fcs_to_expect=cfg.FC.EXPECTATION,
            title=cfg.FC.TITLE + ", Reference",
        )
        fig.savefig(os.path.join(cfg.RESULT_DIR, "FC_plot_ref.png"), dpi=300)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    fire.Fire(direct_lfq)
    # fire.Fire(lfq_manager.run_lfq) # TODO: change to direct_lfq
