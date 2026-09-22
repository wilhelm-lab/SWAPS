"""Adapter that reshapes MaxQuant ``evidence.txt`` into the FragPipe/IonQuant
``combined_ion.tsv`` layout consumed by the SWAPS-vs-baseline benchmark plots
(one row per ion, ``<run> Match Type`` / ``<run> Intensity`` columns per run).
"""

import re
from pathlib import Path
from typing import Optional, Union

import pandas as pd

MQ_EVIDENCE_NAME = "evidence.txt"
_FASTA_ACC_RE = re.compile(r"^>(?:\w+\|)?([^|\s]+)(?:\|(\S+))?")
# MaxQuant runs launched from Windows record parameters.txt paths as
# "W:\ORIGINS\data\...", where W: is that Windows box's mapped drive letter for
# this cluster's /cmnfs/proj mount.
_WINDOWS_DRIVE_ROOT = "/cmnfs/proj"
# Extra roots to try the bare filename against when even the drive-letter
# translation doesn't exist (moved/renamed fasta dirs).
_FASTA_SEARCH_DIRS = [Path("/cmnfs/proj/ORIGINS/data/fasta"), Path("/cmnfs/proj/ORIGINS/data/species_mix/fasta")]


def _resolve_fasta_path(raw_path: str) -> Optional[Path]:
    """A ``Fasta file`` entry from parameters.txt as an existing local Path, or None.

    Handles plain POSIX paths as-is, Windows paths (``X:\\...``) via
    _WINDOWS_DRIVE_ROOT, and otherwise falls back to matching the basename
    under _FASTA_SEARCH_DIRS.
    """
    p = Path(raw_path)
    if p.exists():
        return p
    m = re.match(r"^[A-Za-z]:[\\/](.*)$", raw_path)
    if m:
        p = Path(_WINDOWS_DRIVE_ROOT) / m.group(1).replace("\\", "/")
        if p.exists():
            return p
    name = Path(raw_path.replace("\\", "/")).name
    for d in _FASTA_SEARCH_DIRS:
        if (d / name).exists():
            return d / name
    return None


def _fasta_paths_from_parameters(evidence_path: Path) -> list[Path]:
    """FASTA files listed in the ``parameters.txt`` next to *evidence_path* (';'-separated)."""
    params = evidence_path.parent / "parameters.txt"
    if not params.exists():
        return []
    with open(params) as f:
        for line in f:
            if line.startswith("Fasta file\t"):
                raw_paths = [p for p in line.rstrip("\n").split("\t", 1)[1].split(";") if p]
                resolved = [_resolve_fasta_path(p) for p in raw_paths]
                return [p for p in resolved if p is not None]
    return []


def accession_to_entry_name(fasta_paths: list[Path]) -> dict[str, str]:
    """{UniProt accession: entry name (e.g. 'IRS2_HUMAN')} from UniProt-style FASTA headers."""
    mapping: dict[str, str] = {}
    for path in fasta_paths:
        with open(path) as f:
            for line in f:
                if not line.startswith(">"):
                    continue
                m = _FASTA_ACC_RE.match(line)
                if m and m.group(2):
                    mapping[m.group(1)] = m.group(2)
    return mapping


def mq_evidence_to_combined_ion(
    evidence_path: Union[str, Path], fasta_paths: Optional[list[Path]] = None
) -> pd.DataFrame:
    """MaxQuant evidence.txt -> combined_ion-style table (one row per modified sequence x charge).

    Per run, ``Match Type`` is ``"MS/MS"`` when the ion has an MS/MS-identified
    evidence row, ``"MBR"`` when it only has MaxQuant ``*-MATCH`` (match-between-
    runs) rows, else ``"unmatched"``; ``Intensity`` is the max over duplicate
    evidence rows. MaxQuant leaves ``Modified sequence`` blank on ``*-MATCH``
    rows (it isn't re-derived for a matched, not re-identified, feature), so
    those are backfilled via ``Mod. peptide ID`` from an MS/MS row of the same
    modified peptide (present somewhere in the combined evidence.txt whenever
    MBR actually matched against it); rows where that lookup still fails are
    dropped (unresolvable modified sequence). ``Protein`` is the leading razor
    protein's entry name (``ACC_SPECIES``) when it resolves via the FASTA files
    (default: those in the sibling parameters.txt), so the ``split("_")[-1]``
    species convention of the HYE scripts holds.
    """
    evidence_path = Path(evidence_path)
    ev = pd.read_csv(
        evidence_path,
        sep="\t",
        usecols=[
            "Sequence", "Modified sequence", "Charge", "m/z", "Type", "Raw file",
            "Leading razor protein", "Intensity", "Reverse", "Potential contaminant",
            "Mod. peptide ID",
        ],
        low_memory=False,
    )
    ev = ev[ev["Reverse"].isna() & ev["Potential contaminant"].isna()]
    ev["Intensity"] = ev["Intensity"].fillna(0)
    ev["Match Type"] = ev["Type"].str.contains("MATCH", na=False).map({True: "MBR", False: "MS/MS"})

    modseq_by_pepid = (
        ev.dropna(subset=["Modified sequence", "Mod. peptide ID"])
        .drop_duplicates("Mod. peptide ID")
        .set_index("Mod. peptide ID")["Modified sequence"]
    )
    ev["Modified sequence"] = ev["Modified sequence"].fillna(ev["Mod. peptide ID"].map(modseq_by_pepid))
    ev = ev.dropna(subset=["Modified sequence"])

    keys = ["Modified sequence", "Charge"]
    # MS/MS before MBR, so groupby(...).agg("first") below prefers an MS/MS row
    # over an MBR row when duplicate evidence rows exist for the same ion+run.
    ev = ev.sort_values("Match Type", key=lambda s: s.map({"MS/MS": 0, "MBR": 1}))
    per_run = ev.groupby(keys + ["Raw file"]).agg(
        Intensity=("Intensity", "max"), MatchType=("Match Type", "first")
    ).reset_index()
    intensity = per_run.pivot(index=keys, columns="Raw file", values="Intensity")
    match_type = per_run.pivot(index=keys, columns="Raw file", values="MatchType").fillna("unmatched")
    intensity = intensity.fillna(0)

    out = pd.concat(
        [match_type.add_suffix(" Match Type"), intensity.add_suffix(" Intensity")], axis=1
    ).reset_index()

    ion_info = ev.groupby(keys).agg(
        **{
            "Peptide Sequence": ("Sequence", "first"),
            "M/Z": ("m/z", "first"),
            "Protein": ("Leading razor protein", "first"),
        }
    ).reset_index()
    out = ion_info.merge(out, on=keys, how="right").rename(columns={"Modified sequence": "Modified Sequence"})

    entry_names = accession_to_entry_name(fasta_paths or _fasta_paths_from_parameters(evidence_path))
    if entry_names:
        out["Entry Name"] = out["Protein"].map(entry_names)
        out["Protein"] = out["Entry Name"].fillna(out["Protein"])
    return out


def load_baseline_combined_ions(search_output: Union[str, Path]) -> pd.DataFrame:
    """Baseline (IonQuant/MaxQuant) combined-ion table for a SWAPS run's *search_output*.

    Accepts a ``combined_ion.tsv`` path, a FragPipe dir holding one, or a MaxQuant
    ``evidence.txt`` (or its ``txt/`` dir), dispatching to the matching reader.
    """
    p = Path(search_output)
    if p.is_dir():
        p = p / MQ_EVIDENCE_NAME if (p / MQ_EVIDENCE_NAME).exists() else p / "combined_ion.tsv"
    df = mq_evidence_to_combined_ion(p) if p.name == MQ_EVIDENCE_NAME else pd.read_csv(p, sep="\t")
    df.attrs["baseline_label"] = baseline_label(p)
    return df


def baseline_label(search_output: Union[str, Path]) -> str:
    """Display name of the baseline tool behind *search_output* (see load_baseline_combined_ions)."""
    p = Path(search_output)
    is_mq = p.name == MQ_EVIDENCE_NAME or (p.is_dir() and (p / MQ_EVIDENCE_NAME).exists())
    return "MaxQuant" if is_mq else "IonQuant"


def species_from_proteins(proteins: pd.Series, evidence_path: Union[str, Path]) -> pd.Series:
    """Species suffix (e.g. 'HUMAN') per row of a bare-accession ``Proteins`` column
    (';'-separated groups: first accession wins), via the FASTA files of a MaxQuant search."""
    evidence_path = Path(evidence_path)
    entry_names = accession_to_entry_name(_fasta_paths_from_parameters(evidence_path))
    first_acc = proteins.str.split(";").str[0]
    return first_acc.map(entry_names).str.split("_").str[-1]


def search_output_path_from_effective_config(quant_dir: Union[str, Path]) -> Path:
    """SEARCH_OUTPUT_PATH recorded in *quant_dir*'s effective_config.yaml -- the
    FragPipe combined_ion.tsv, or a MaxQuant evidence.txt, a SWAPS run was seeded
    from. Check ``.name == MQ_EVIDENCE_NAME`` on the result to tell which."""
    cfg = (Path(quant_dir) / "effective_config.yaml").read_text()
    return Path(re.search(r"SEARCH_OUTPUT_PATH:\s*(\S+)", cfg).group(1))


def add_species_column(
    df: pd.DataFrame,
    protein_col: str,
    evidence_path: Optional[Union[str, Path]] = None,
    organism_col: str = "Species",
) -> pd.DataFrame:
    """swaps.postprocessing.direct_lfq.add_filtered_organism_column, generalized: for a
    MaxQuant-seeded SWAPS run (e.g. a dict_ref-derived table), *protein_col* is a bare
    accession with no ``_SPECIES`` suffix to parse, so *evidence_path* (that run's own
    MaxQuant evidence.txt, see mq_baseline.baseline_label) resolves species via its FASTA
    files (species_from_proteins) instead. *evidence_path* is None (default) for a
    FragPipe-seeded run/table whose protein ids are already ``ACC_SPECIES`` entry names --
    same behavior as add_filtered_organism_column itself.
    """
    from .direct_lfq import ALLOWED_ORGANISMS, add_filtered_organism_column

    if evidence_path is None:
        return add_filtered_organism_column(df, protein_col=protein_col, organism_col=organism_col)
    out = df.copy()
    out[organism_col] = species_from_proteins(out[protein_col], evidence_path)
    return out[out[organism_col].isin(ALLOWED_ORGANISMS)].copy()
