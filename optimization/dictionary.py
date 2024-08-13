import logging
from typing import Union
import numpy as np
import pandas as pd
from scipy import sparse

import matplotlib.pyplot as plt
import sklearn.preprocessing as pp
import networkx as nx

from utils.plot import plot_comparison
from utils.tools import ExtractPeak


Logger = logging.getLogger(__name__)


def csr_cosine_similarities(csrmat, normalize=True):
    if normalize:
        csrmat = pp.normalize(csrmat.tocsc(), axis=0)
    return csrmat.T * csrmat


def csr_jaccard_similarities(csrmat):
    csrmat.data[:] = 1  # binarize the matrix
    cols_sum = csrmat.getnnz(
        axis=0
    )  # getnnz: Number of stored values, including explicit zeros in each column
    set_ab = csrmat.T * csrmat

    # for rows
    set_a = np.repeat(cols_sum, set_ab.getnnz(axis=0))
    # for columns
    set_b = cols_sum[set_ab.indices]

    jaccard_similarities = set_ab.copy()
    jaccard_similarities.data /= set_a + set_b - set_ab.data

    return jaccard_similarities


def _connect_collinear_candidate_pairs(collinear_pairs: pd.Series):
    """
    Given a series of collinear pairs, connect them into a graph and return the graph object

    :collinear_pairs: pd.Series, index being collinear pairs and values being correlation coefficient
    """
    G = nx.from_edgelist(collinear_pairs.index)
    # Find connected components
    connected_components_df = nx.to_pandas_adjacency(G)
    # Create an empty pandas series to store the connected precursors
    connected_precursors = pd.Series(index=connected_components_df.index, dtype=object)

    # Iterate over each precursor
    for precursor in connected_components_df.index:
        # Get the connected precursors for the current precursor
        connected = connected_components_df.columns[
            connected_components_df.loc[precursor] == 1
        ].tolist()
        # Store the connected precursors in the series
        connected_precursors[precursor] = connected
    connected_precursors.sort_index(inplace=True)
    return connected_precursors


class Dict:
    def __init__(
        self,
        candidate_by_rt: pd.DataFrame,
        one_scan: Union[pd.DataFrame, pd.Series],
        abundance_missing_thres: float = 0.4,
        rel_height: float = 0.75,
    ) -> None:
        self.n_candidate_by_rt = candidate_by_rt.shape[0]
        self.candidate_by_rt = candidate_by_rt
        Logger.info("Number of candidates by RT %s", self.n_candidate_by_rt)

        self.abundance_missing_thres = abundance_missing_thres
        ms1_int = pd.DataFrame(
            {"mzarray_obs": one_scan["mzarray"], "intensity": one_scan["intarray"]}
        )
        self.peak_results = ExtractPeak(
            np.array(ms1_int["mzarray_obs"]),
            np.array(ms1_int["intensity"]),
            rel_height=rel_height,
        )

        self.dict = None
        self.filtered_candidate_idx = None
        self.iso_abundance_by_id = None
        self.obs_peak_int = None
        self.match_candidate_with_peak()

        self.cos_sim_matrix = None
        self.high_corr_sol = None
        self.collinear_precursors = None
        self.collinear_sets = None

    def match_candidate_with_peak(self):
        candidate_by_rt_iso = (
            self.candidate_by_rt[["id", "IsoMZ", "IsoAbundance"]]
            .explode(["IsoMZ", "IsoAbundance"])
            .astype({"IsoMZ": "float64", "IsoAbundance": "float64"})
        )

        candidate_by_rt_iso = pd.merge_asof(
            candidate_by_rt_iso.sort_values("IsoMZ"),
            self.peak_results,
            left_on="IsoMZ",
            right_on="apex_mz",
            allow_exact_matches=True,
            direction="nearest",
        )
        candidate_by_rt_iso["matched"] = (
            candidate_by_rt_iso["IsoMZ"] >= candidate_by_rt_iso["start_mz"]
        ) & (candidate_by_rt_iso["IsoMZ"] <= candidate_by_rt_iso["end_mz"])
        self.iso_abundance_by_id = (
            candidate_by_rt_iso.groupby(["id", "matched"])["IsoAbundance"]
            .sum()
            .reset_index()
        )
        candidate_by_rt_iso_matched = candidate_by_rt_iso[
            (candidate_by_rt_iso["matched"])
        ]
        Logger.info(
            "Number of candidates after isotope match %s",
            len(candidate_by_rt_iso_matched["id"].unique()),
        )

        # filter candidates by isotope abundance
        if len(candidate_by_rt_iso_matched["id"].unique()) > 0:
            candidate_by_rt_iso_filtered = self.iso_abundance_by_id[
                self.iso_abundance_by_id["matched"]
                & (
                    self.iso_abundance_by_id["IsoAbundance"]
                    >= (1 - self.abundance_missing_thres)
                )
            ]
            self.filtered_candidate_idx = (
                candidate_by_rt_iso_filtered["id"].unique().tolist()
            )
            Logger.info(
                "Number of candidates after isotope abundance filter %s",
                len(self.filtered_candidate_idx),
            )
            if len(self.filtered_candidate_idx) > 0:
                candidate_by_rt_iso_filtered_sum_abundance = (
                    candidate_by_rt_iso_matched[
                        candidate_by_rt_iso_matched["id"].isin(
                            self.filtered_candidate_idx
                        )
                    ]
                    .groupby(["id", "apex_mz"])["IsoAbundance"]
                    .sum()
                    .reset_index()
                )
                candid_dict = candidate_by_rt_iso_filtered_sum_abundance.pivot(
                    index="apex_mz", columns="id", values="IsoAbundance"
                )
                self.dict = pd.merge(
                    left=self.peak_results[["apex_mz"]],
                    right=candid_dict,
                    left_on="apex_mz",
                    right_index=True,
                    how="left",
                ).set_index("apex_mz")
                self.dict.loc[0] = 1 - self.dict.sum(
                    axis=0
                )  # append mz for accumulated not observed iso abundance
                self.dict = self.dict.fillna(0)

                # drop mz values with no match in the candidates
                self.dict = self.dict.loc[~(self.dict == 0).all(axis=1)]
                self.peak_results = self.peak_results.loc[
                    self.peak_results["apex_mz"].isin(self.dict.index)
                ]
                Logger.debug(
                    "Number of peaks after filtering %s", self.peak_results.shape
                )
                Logger.debug("Number of mz in dict %s", self.dict.shape)
                # assert self.dict.shape[0] == self.peak_results.shape[0]
                Logger.debug(
                    "[Double check] Number of candidates by RT and abundance filter %s",
                    len(self.dict.columns.values),
                )

                self.obs_peak_int = pd.DataFrame(
                    {
                        "mzarray_obs": np.append(
                            self.peak_results["apex_mz"].values,
                            [0],  # append 0 mz for not observed iso abundance
                            axis=0,
                        ),
                        "intensity": np.append(
                            self.peak_results["peak_intensity_sum"].values, [0], axis=0
                        ),
                    }
                )

    def get_feature_corr(
        self,
        cos_sim_thres: float = 0.9,
        jaccard_sim_thres: float = 0.5,
        calc_jaccard_sim: bool = True,
        calc_cos_sim: bool = True,
    ):
        assert (
            calc_cos_sim or calc_jaccard_sim
        ), "At least one similarity measure is required."

        dict_csr = sparse.csr_matrix(self.dict.values)
        # high_corr_sol = pd.DataFrame(
        #     {"Cosine Similarity": [], "Jaccard Similarity": []}
        # )
        if calc_cos_sim:
            # self.corr_matrix = self.dict.corr()
            corr_csr = csr_cosine_similarities(dict_csr, normalize=True)
            self.cos_sim_matrix = pd.DataFrame(
                corr_csr.todense(), index=self.dict.columns, columns=self.dict.columns
            )

            cos_sim = self.cos_sim_matrix.where(
                np.triu(np.ones(self.cos_sim_matrix.shape), k=1).astype(bool)
            ).stack()
            # high_corr_sol["Cosine Similarity"] = cos_sim
            high_corr_sol_cos = cos_sim[cos_sim >= cos_sim_thres]  # type: ignore
            Logger.debug("Finished calculating cosine similarity.")

        if calc_jaccard_sim:
            jaccard_sim_csr = csr_jaccard_similarities(dict_csr)
            self.jaccard_sim_matrix = pd.DataFrame(
                jaccard_sim_csr.todense(),
                index=self.dict.columns,
                columns=self.dict.columns,
            )
            jaccard_sim = self.jaccard_sim_matrix.where(
                np.triu(np.ones(self.jaccard_sim_matrix.shape), k=1).astype(bool)
            ).stack()
            # high_corr_sol["Jaccard Similarity"] = jaccard_sim
            high_corr_sol_jaccard = jaccard_sim[jaccard_sim >= jaccard_sim_thres]  # type: ignore
            Logger.debug("Finished calculating Jaccard similarity.")
        if calc_cos_sim and calc_jaccard_sim:
            self.high_corr_sol = pd.concat(
                [high_corr_sol_cos, high_corr_sol_jaccard], axis=1
            )
        else:
            self.high_corr_sol = (
                high_corr_sol_cos if calc_cos_sim else high_corr_sol_jaccard
            )
        graph = nx.from_edgelist(self.high_corr_sol.index)

        # Get the connected nodes
        self.collinear_sets = list(nx.connected_components(graph))
        self.collinear_precursors = _connect_collinear_candidate_pairs(
            self.high_corr_sol
        )

        Logger.info("Number of collinear sets: %s ", len(self.collinear_sets))
        Logger.info(
            "Number of candidates involved in high correlation: %s",
            sum(len(s) for s in self.collinear_sets),
        )

    def plot_feature_corr(
        self, plot_hmap: bool = False, plot_collinear_hist: bool = False
    ):
        if plot_hmap:
            self.cos_sim_matrix.style.background_gradient(cmap="coolwarm")
            plt.matshow(self.cos_sim_matrix)
            plt.show()
        if plot_collinear_hist:
            self.high_corr_sol["Cosine Similarity"].plot.hist()
            ymin, ymax = plt.ylim()
            plt.vlines(x=0.95, ymin=ymin, ymax=ymax, linestyles="dashed")
            plt.vlines(x=0.9, ymin=ymin, ymax=ymax, linestyles="dashed")
            plt.vlines(
                x=0.5, ymin=ymin, ymax=ymax, linestyles="dashed", color="r"
            )  # TODO: change x: threshold
            plt.title("Distribution of Highly Correlation Coefficients")

    def plot_observe_iso_abundance(self):
        self.iso_abundance_by_id[self.iso_abundance_by_id["matched"]][
            "IsoAbundance"
        ].hist(bins=100)
        plt.title("Distribution of Observed Isotope Abundance")
        ymin, ymax = plt.ylim()
        plt.vlines(
            x=1 - self.abundance_missing_thres,
            ymin=ymin,
            ymax=ymax,
            label="Abundance Thres",
            color="r",
        )


def compare_isotope_pattern(MQ_withIso: pd.DataFrame, seq_A_idx: int, seq_B_idx: int):
    plot_comparison(
        y_true=MQ_withIso.loc[MQ_withIso["id"] == seq_A_idx, "IsoAbundance"].values[0],
        y_pred=MQ_withIso.loc[MQ_withIso["id"] == seq_B_idx, "IsoAbundance"].values[0],
        x_true=MQ_withIso.loc[MQ_withIso["id"] == seq_A_idx, "IsoMZ"].values[0],
        x_pred=MQ_withIso.loc[MQ_withIso["id"] == seq_B_idx, "IsoMZ"].values[0],
    )
    print()
