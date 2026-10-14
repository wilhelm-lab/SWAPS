"""Adapter that reshapes Sage's ``lfq.tsv`` + ``results.sage.tsv`` into the
FragPipe/IonQuant ``combined_ion.tsv`` layout consumed by the SWAPS-vs-baseline
benchmark plots (see mq_baseline.load_baseline_combined_ions, which dispatches here).

Sage's LFQ table has no per-run match type, so it is derived per (ion, run):
quantified = LFQ intensity > 0; identified = the ion is marked "Reference"/"Quant_Only"
in that run in the SWAPS ``dict_ref`` built from the same search -- small and already
q <= 0.01 filtered. Without a dict_ref it falls back to parsing the (~300 MB)
results.sage.tsv with the same cutoffs SWAPS' own Sage parser applies (label == 1,
spectrum/peptide/protein q <= 0.01), which gives near-identical identifications.

With a SWAPS *dict_ref* the table covers exactly the dictionary's ions: an lfq.tsv ion that
is not in it (typically one whose protein fails the protein-level q-value cutoff SWAPS
applies when building the dictionary) is discarded, and a dictionary ion with no lfq.tsv row
at all is kept with all-zero intensities and flagged ``In LFQ == False`` (the match-type plots
show it as "Not in LFQ").

The ion is the peptide when Sage combined charge states (``combine_charge_states``,
lfq.tsv ``charge == -1``: identified in a run if any of its charge states is), and the
(peptide, charge) pair when it did not (one lfq.tsv row per charge state) -- detected
from the lfq.tsv itself.
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

try:
    from swaps.prepare_dict.search_engine_output_parser import SAGE_Q_VALUE_CUTOFF
except ImportError:  # swaps/ itself on sys.path (notebook / script style)
    from prepare_dict.search_engine_output_parser import SAGE_Q_VALUE_CUTOFF

Logger = logging.getLogger(__name__)

SAGE_RESULTS_NAME = "results.sage.tsv"
SAGE_LFQ_NAME = "lfq.tsv"
_RUN_SUFFIX = ".d"
# dict_ref per-run cell values meaning "identified from that run's own MS/MS"
# (see swaps.postprocessing.helper.reassign_quant_only_as_msms)
_DICT_REF_MSMS = ("Reference", "Quant_Only")


@lru_cache(maxsize=8)
def _passing_psms(results_path: Union[str, Path], q_cutoff: float) -> pd.DataFrame:
    """Target PSMs passing *q_cutoff* on the spectrum, peptide and protein q-values --
    the same filter sbs_runner_ims.py / sage_parser apply, so the identifications
    match the SWAPS dict_ref's -- with a ``run`` column (filename without ``.d``).
    Cached (the ~300 MB results.sage.tsv read dominates, and the benchmark scripts
    reload the same baseline once per plot): treat the returned frame as read-only."""
    psms = pd.read_csv(
        results_path,
        sep="\t",
        usecols=["peptide", "charge", "proteins", "filename", "label", "spectrum_q", "peptide_q", "protein_q"],
    )
    passing = psms[
        (psms["label"] == 1)
        & (psms["spectrum_q"] <= q_cutoff)
        & (psms["peptide_q"] <= q_cutoff)
        & (psms["protein_q"] <= q_cutoff)
    ]
    return passing.assign(run=passing["filename"].str.removesuffix(_RUN_SUFFIX))


@lru_cache(maxsize=4)
def _read_dict_ref(path: str) -> pd.DataFrame:
    return pd.read_pickle(path)


def _identified_from_dict_ref(
    dict_ref: Union[pd.DataFrame, str, Path], runs: list[str], key: list[str]
) -> tuple[pd.DataFrame, pd.Series]:
    """(ion x run boolean table, protein per ion) from a SWAPS dict_ref, with the ion
    keyed by *key* (``["peptide"]`` or ``["peptide", "charge"]``): an ion is identified
    in a run if any of its dict_ref rows is (at the peptide level: any charge state)."""
    if not isinstance(dict_ref, pd.DataFrame):
        dict_ref = _read_dict_ref(str(dict_ref))
    dict_ref = dict_ref.rename(columns={"Modified sequence": "peptide", "Charge": "charge"})
    run_cols = [r for r in runs if r in dict_ref.columns]
    identified = dict_ref[run_cols].isin(_DICT_REF_MSMS).groupby([dict_ref[k].values for k in key]).any()
    identified.index.names = key
    proteins = dict_ref.drop_duplicates(key).set_index(key)["Proteins"]
    return identified, proteins


def identified_peptides_by_run(
    results_path: Union[str, Path], q_cutoff: float = SAGE_Q_VALUE_CUTOFF
) -> pd.DataFrame:
    """Boolean table (index: Sage modified peptide, columns: run name without ``.d``)
    of peptides with a passing target PSM in that run (see _passing_psms)."""
    passing = _passing_psms(str(results_path), q_cutoff)
    return pd.crosstab(passing["peptide"], passing["run"]) > 0


def lfq_q_cutoff_grid(results_path: Union[str, Path], step: float = 0.01) -> np.ndarray:
    """Cutoffs on the ``q_value`` of the ``lfq.tsv`` next to *results_path*, *step* apart,
    over the range the table actually covers: from the first multiple of *step* at which
    any target row passes up to the first at which all of them do (a cutoff there is
    the same as no ``lfq_q_cutoff``). For sage_to_combined_ion(lfq_q_cutoff=...) sweeps."""
    lfq = pd.read_csv(Path(results_path).parent / SAGE_LFQ_NAME, sep="\t", usecols=["proteins", "q_value"])
    q = lfq.loc[~lfq["proteins"].str.contains("rev_", na=False), "q_value"]
    first, last = (int(np.ceil(round(v / step, 6))) for v in (q.min(), q.max()))
    return np.round(np.arange(first, last + 1) * step, 6)


def sage_to_combined_ion(
    results_path: Union[str, Path],
    lfq_path: Optional[Union[str, Path]] = None,
    q_cutoff: float = SAGE_Q_VALUE_CUTOFF,
    dict_ref: Optional[Union[pd.DataFrame, str, Path]] = None,
    lfq_q_cutoff: Optional[float] = None,
) -> pd.DataFrame:
    """Sage lfq.tsv + identifications -> combined_ion-style table (one row per ion: per modified
    peptide when Sage combined charge states, per peptide+charge otherwise; see module docstring).

    *lfq_q_cutoff* is the quantification-side "FDR cutoff" handle: an lfq.tsv row whose
    ``q_value`` (Sage's quantification-level q-value, monotone in the LFQ ``score`` and
    independent of the PSM q-values; 0.0134-0.23 in practice) exceeds it contributes no
    "MBR" quantifications, i.e. its intensities are zeroed in every run where the peptide
    is not identified. Identified (MS/MS) run-peptide pairs keep their LFQ intensity
    regardless -- their identification is already q <= 0.01. None (default) keeps all.

    Identifications come from *dict_ref* (a SWAPS dict_ref DataFrame or a path to its
    pickle; see module docstring) when given, else from *results_path* (results.sage.tsv,
    q <= *q_cutoff*).

    Per run, ``Intensity`` is Sage's LFQ intensity (0 where not quantified) and
    ``Match Type`` is:

    * ``"MS/MS"``    -- the ion is identified in that run, whether or not LFQ
      quantified it. Identified-but-zero-LFQ stays
      "MS/MS" with intensity 0, exactly like IonQuant's combined_ion.tsv -- the plots
      already display that as "Zero Quant" (swaps.utils.plot.plot_match_type_comparison)
      and the MBR fill-rate definition keeps it out of the MBR-fillable slots.
    * ``"MBR"``      -- quantified (LFQ > 0) but not identified in that run, including
      ions that pass no identification cutoff in any run (a "Match").
    * ``"unmatched"`` -- neither.

    Rows: with *dict_ref*, exactly the dictionary's ions (lfq.tsv ions outside it are
    discarded; dictionary ions without an lfq.tsv row get all-zero intensities); without,
    every LFQ ion plus every identified ion LFQ has no row for. *lfq_path* defaults to
    ``lfq.tsv`` next to *results_path*. Columns follow combined_ion.tsv (``Modified
    Sequence``, ``Charge`` (-1 when charge states are combined), ``Protein``, ``<run>
    Match Type``, ``<run> Intensity``) plus ``In LFQ`` (the ion has an lfq.tsv row).
    """
    results_path = Path(results_path)
    lfq_path = Path(lfq_path) if lfq_path is not None else results_path.parent / SAGE_LFQ_NAME

    lfq = pd.read_csv(lfq_path, sep="\t")
    run_cols = [c for c in lfq.columns if c.endswith(_RUN_SUFFIX)]
    runs = [c.removesuffix(_RUN_SUFFIX) for c in run_cols]
    lfq = lfq.rename(columns=dict(zip(run_cols, runs)))
    lfq = lfq[~lfq["proteins"].str.contains("rev_", na=False)]
    per_charge = bool((lfq["charge"] != -1).any())
    key = ["peptide", "charge"] if per_charge else ["peptide"]
    lfq = lfq.set_index(key)
    lfq_q = lfq["q_value"]

    if dict_ref is not None:
        identified, id_proteins = _identified_from_dict_ref(dict_ref, runs, key)
    else:
        passing = _passing_psms(str(results_path), q_cutoff)
        identified = pd.crosstab([passing[k] for k in key], passing["run"]) > 0
        id_proteins = passing.drop_duplicates(key).set_index(key)["proteins"]
    ions = identified.index if dict_ref is not None else lfq.index.union(identified.index)
    n_lfq_only_dropped = len(lfq.index.difference(ions))
    in_lfq = ions.isin(lfq.index)

    intensity = lfq[runs].reindex(ions).fillna(0.0)
    identified = identified.reindex(index=ions, columns=runs, fill_value=False)
    if lfq_q_cutoff is not None:
        gated = (lfq_q.reindex(ions) > lfq_q_cutoff).values[:, None] & ~identified.values
        intensity = intensity.mask(gated, 0.0)
    quantified = intensity > 0

    match_type = pd.DataFrame(
        np.where(identified, "MS/MS", np.where(quantified, "MBR", "unmatched")),
        index=ions,
        columns=runs,
    )
    n_never_identified = int((quantified.any(axis=1) & ~identified.any(axis=1)).sum())
    Logger.info(
        "sage_to_combined_ion: %d %s (%d in LFQ, %d not in LFQ, %d LFQ-only discarded; ids "
        "from %s); %d quantified in some run but identified in none (all 'MBR')",
        len(ions), "ions (peptide+charge)" if per_charge else "peptides",
        int(in_lfq.sum()), int((~in_lfq).sum()), n_lfq_only_dropped,
        "dict_ref" if dict_ref is not None else f"results.sage.tsv q<={q_cutoff}",
        n_never_identified,
    )

    protein_by_ion = pd.concat([lfq["proteins"], id_proteins]).groupby(level=list(range(len(key)))).first()
    peptides = ions.get_level_values(0)
    out = pd.concat(
        [match_type.add_suffix(" Match Type"), intensity.add_suffix(" Intensity")], axis=1
    )
    out.insert(0, "In LFQ", in_lfq)
    out.insert(0, "Protein", protein_by_ion.reindex(ions))
    out.insert(0, "Charge", ions.get_level_values(1) if per_charge else -1)
    out.insert(0, "Peptide Sequence", peptides.str.replace(r"\[.*?\]", "", regex=True).str.lstrip("-"))
    out.insert(0, "Modified Sequence", peptides)
    return out.reset_index(drop=True)
