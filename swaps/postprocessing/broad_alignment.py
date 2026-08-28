"""RT-binned, pairwise majority-vote shift correction ("broad alignment").

Per-candidate SIFT/template-match alignment in match_features.py picks the
wrong RT/IM shift for low-S/N peptides -- the image itself is too noisy to
match reliably. This module builds a calibration table of the RT/IM shift
between every ordered (reference_run, matched_run) raw-file pair, binned by
RT, from a cohort of peptides selected for likely reliability -- so
match_features.py can force that shift directly instead of re-discovering it
per candidate when MATCH_FEATURES_KWARGS.broad_alignment.enabled is set.

Calibration depends only on dict_ref (or dict_ref_with_activation.pkl) and
Stage-2 activation/*.parquet output -- never on match_features output -- so
it can run as a pre-pass between Stage 2 (activation-building) and Stage 3
(match_features), before any match_features output exists.
"""

import argparse
import logging
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from math import ceil
from typing import Any

import numpy as np
import pandas as pd

from .helper import get_pept_act_from_parquet, load_peptide_batch_df_from_partquet
from .image_processing import smooth_and_denoise_image
from .match_features import align_images_to_reference

Logger = logging.getLogger(__name__)

_SAMPLE_COLUMNS = [
    "mz_rank",
    "reference_run",
    "matched_run",
    "rt_position",
    "im_position",
    "shift_rt",
    "shift_im",
    "template_matching_score",
]
_TABLE_COLUMNS = [
    "reference_run",
    "matched_run",
    "rt_bin_index",
    "rt_bin_start",
    "rt_bin_end",
    "shift_rt",
    "shift_im",
    "n_samples",
    "is_fallback",
]


def select_calibration_peptides(
    dict_ref: pd.DataFrame,
    n_peptides: int = 500,
    rt_column: str = "RT_search_center",
    n_rt_bins: int = 20,
    random_seed: int = 42,
) -> np.ndarray:
    """RT-stratified mz_rank sample, biased toward reliably-identified peptides
    when reference_score/n_identifications/score_std are available.

    Uses only dict_ref columns -- no pp_reference/match_features output --
    since this is meant to run before match_features ever exists. Within each
    RT stratum, prefers higher reference_score (the search engine's own
    confidence in that peptide's reference identification), with
    n_identifications/score_std as secondary tie-breakers. Calibration itself
    only needs the selected peptides' activation images (always present after
    Stage 2), so these three columns are a ranking preference, not a hard
    requirement -- datasets whose dict_ref predates them (built before
    prepare_dict.py started computing these columns) fall back to an
    unranked, shuffled-within-bin RT-stratified sample instead of raising.
    """
    score_cols = ["reference_score", "n_identifications", "score_std"]
    required = ["mz_rank", rt_column]
    missing = [c for c in required if c not in dict_ref.columns]
    if missing:
        raise KeyError(f"select_calibration_peptides requires dict_ref columns {missing}")
    available_score_cols = [c for c in score_cols if c in dict_ref.columns]
    if len(available_score_cols) < len(score_cols):
        Logger.info(
            "select_calibration_peptides: dict_ref missing %s -- falling back to "
            "unranked RT-stratified sampling.",
            [c for c in score_cols if c not in available_score_cols],
        )

    df = dict_ref[required + available_score_cols].dropna(subset=[rt_column]).copy()
    if df.empty:
        return np.array([], dtype=int)

    n_bins = max(1, min(n_rt_bins, df[rt_column].nunique()))
    df["_rt_bin"] = pd.cut(df[rt_column], bins=n_bins, duplicates="drop")
    if available_score_cols:
        df = df.sort_values(
            available_score_cols,
            ascending=[False, False, True][: len(available_score_cols)],
            na_position="last",
        )
    else:
        df = df.sample(frac=1, random_state=random_seed)
    per_bin = max(1, ceil(n_peptides / df["_rt_bin"].nunique(dropna=True)))
    selected = df.groupby("_rt_bin", observed=True, group_keys=False).head(per_bin)

    mz_ranks = selected["mz_rank"].to_numpy()
    if len(mz_ranks) > n_peptides:
        rng = np.random.default_rng(random_seed)
        keep = np.sort(rng.choice(len(mz_ranks), size=n_peptides, replace=False))
        mz_ranks = mz_ranks[keep]
    return mz_ranks


_CALIBRATION_WORKER_CONTEXT: dict[str, Any] = {}


def _init_calibration_worker(
    dict_ref_by_mz: pd.DataFrame,
    raw_file_list: list[str],
    result_dir: str,
    rt_column: str,
    im_column: str = "IM_search_center",
    require_msms_both_runs: bool = False,
) -> None:
    _CALIBRATION_WORKER_CONTEXT["dict_ref_by_mz"] = dict_ref_by_mz
    _CALIBRATION_WORKER_CONTEXT["raw_file_list"] = raw_file_list
    _CALIBRATION_WORKER_CONTEXT["result_dir"] = result_dir
    _CALIBRATION_WORKER_CONTEXT["rt_column"] = rt_column
    _CALIBRATION_WORKER_CONTEXT["im_column"] = im_column
    _CALIBRATION_WORKER_CONTEXT["require_msms_both_runs"] = require_msms_both_runs


def _calibration_batch_worker(mz_chunk: np.ndarray) -> list[dict]:
    ctx = _CALIBRATION_WORKER_CONTEXT
    return _pairwise_align_chunk(
        mz_chunk,
        ctx["raw_file_list"],
        ctx["result_dir"],
        ctx["dict_ref_by_mz"],
        ctx["rt_column"],
        ctx["im_column"],
        require_msms_both_runs=ctx.get("require_msms_both_runs", False),
    )


def _load_calibration_images(
    mz_chunk: np.ndarray,
    raw_file: str,
    result_dir: str,
    dict_ref_by_mz: pd.DataFrame,
) -> dict[int, np.ndarray]:
    act_dir = os.path.join(result_dir, raw_file, "activation")
    act_df = load_peptide_batch_df_from_partquet(act_dir, mz_chunk)
    act_by_mz = {int(mz): sub for mz, sub in act_df.groupby("mz_rank", sort=False)}
    empty = act_df.iloc[0:0]
    return {
        int(mz): smooth_and_denoise_image(
            get_pept_act_from_parquet(
                act_by_mz.get(int(mz), empty),
                int(mz),
                dict_ref_by_mz,
                raw_file,
                return_offset=True,
            )[0]
        )
        for mz in mz_chunk
    }


_MSMS_STATUSES = {"Reference", "Quant_Only"}  # direct MS/MS identification in that
# run -- "Match" is an MBR-predicted position only (no MS/MS in that run for this
# peptide); see prepare_dict.py's pivot_psm_by_mz_rank label_logic.


def _pairwise_align_chunk(
    mz_chunk: np.ndarray,
    raw_file_list: list[str],
    result_dir: str,
    dict_ref_by_mz: pd.DataFrame,
    rt_column: str,
    im_column: str = "IM_search_center",
    require_msms_both_runs: bool = False,
) -> list[dict]:
    """One direct one-hop template match per (peptide, reference run, matched run).

    Template built fresh from the reference run's own image, matched
    directly against the matched run's own image -- no composition through
    a third run, no compounding of two independently-noisy legs.

    `require_msms_both_runs`, if True, skips a (peptide, ref_run, match_run)
    triplet unless the peptide has a direct MS/MS identification (status
    "Reference" or "Quant_Only" -- see _MSMS_STATUSES) in BOTH ref_run and
    match_run, not just an MBR "Match" position in either. Narrows the
    calibration signal to genuinely-observed positions on both sides of the
    pair, at the cost of fewer (and RT/run-pair-imbalanced) samples --
    peptides/pairs with no MS/MS-confirmed run image on both ends
    contribute nothing.
    """
    images_by_run = {
        raw_file: _load_calibration_images(mz_chunk, raw_file, result_dir, dict_ref_by_mz)
        for raw_file in raw_file_list
    }
    rows: list[dict] = []
    for mz in mz_chunk:
        rt_pos = float(dict_ref_by_mz.at[int(mz), rt_column])
        im_pos = float(dict_ref_by_mz.at[int(mz), im_column])
        for ref_run in raw_file_list:
            if require_msms_both_runs and dict_ref_by_mz.at[int(mz), ref_run] not in _MSMS_STATUSES:
                continue
            ref_image = images_by_run[ref_run][int(mz)]
            for match_run in raw_file_list:
                if match_run == ref_run:
                    continue
                if (
                    require_msms_both_runs
                    and dict_ref_by_mz.at[int(mz), match_run] not in _MSMS_STATUSES
                ):
                    continue
                match_image = images_by_run[match_run][int(mz)]
                alignment = align_images_to_reference(
                    images=[ref_image, match_image], reference_idx=0
                )
                rows.append(
                    {
                        "mz_rank": int(mz),
                        "reference_run": ref_run,
                        "matched_run": match_run,
                        "rt_position": rt_pos,
                        "im_position": im_pos,
                        "shift_rt": int(alignment.shifts[1][0]),
                        "shift_im": int(alignment.shifts[1][1]),
                        "template_matching_score": float(alignment.max_scores[1]),
                    }
                )
    return rows


def run_pairwise_calibration_alignment(
    calibration_mz_ranks: np.ndarray,
    raw_file_list: list[str],
    result_dir: str,
    dict_ref: pd.DataFrame,
    rt_column: str = "RT_search_center",
    im_column: str = "IM_search_center",
    chunk_size: int = 25,
    max_workers: int | None = None,
    require_msms_both_runs: bool = False,
) -> pd.DataFrame:
    """Direct one-hop pairwise shift samples for every (A, B) raw-file pair.

    Parallelized over peptide chunks via ProcessPoolExecutor, the same
    pattern match_features_batches_parallel uses. Each row also carries the
    peptide's own mz_rank and im_position (IM_search_center) alongside
    rt_position -- build_shift_table still only bins by RT (rt_column), but
    having im_position on every sample lets downstream analysis/diagnostics
    look at either axis without a second pass.

    `require_msms_both_runs` -- see _pairwise_align_chunk.
    """
    dict_ref_by_mz = (
        dict_ref.set_index("mz_rank")
        if dict_ref["mz_rank"].is_unique
        else dict_ref.drop_duplicates("mz_rank").set_index("mz_rank")
    )
    mz_ranks = np.asarray(calibration_mz_ranks)
    if len(mz_ranks) == 0:
        return pd.DataFrame(columns=_SAMPLE_COLUMNS)

    chunks = [mz_ranks[i : i + chunk_size] for i in range(0, len(mz_ranks), chunk_size)]
    rows: list[dict] = []
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_calibration_worker,
        initargs=(
            dict_ref_by_mz,
            raw_file_list,
            result_dir,
            rt_column,
            im_column,
            require_msms_both_runs,
        ),
    ) as executor:
        for chunk_rows in executor.map(_calibration_batch_worker, chunks):
            rows.extend(chunk_rows)
    return pd.DataFrame(rows, columns=_SAMPLE_COLUMNS)


def _top_confidence_fraction(group: pd.DataFrame, confidence_frac: float) -> pd.DataFrame:
    """Top confidence_frac of a pair's samples by template_matching_score.

    Percentile-based, not a fixed absolute threshold -- self-calibrates to
    each pair's own score scale instead of starving thin pairs on a dataset
    where absolute scores run low. NaN scores (what forced-shift rows would
    carry) are excluded, not merely sorted last.
    """
    scored = group.dropna(subset=["template_matching_score"])
    if scored.empty:
        return scored
    n_keep = max(1, int(ceil(len(scored) * confidence_frac)))
    return scored.nlargest(n_keep, "template_matching_score")


def _joint_mode(rt_shifts: np.ndarray, im_shifts: np.ndarray) -> tuple[int, int]:
    """Mode of the (shift_rt, shift_im) pair, not independent per-axis modes.

    Avoids synthesizing a shift combination that never actually occurred.
    """
    counts = Counter(zip(rt_shifts.tolist(), im_shifts.tolist()))
    (best_rt, best_im), _ = counts.most_common(1)[0]
    return int(best_rt), int(best_im)


def _build_pair_bins(
    reference_run: str,
    matched_run: str,
    confident: pd.DataFrame,
    bin_width_minutes: float,
    min_samples_per_bin: int,
    max_neighbor_search_bins: int,
) -> list[dict]:
    if confident.empty:
        return []

    rt_min = float(confident["rt_position"].min())
    rt_max = float(confident["rt_position"].max())
    n_bins = max(1, int(ceil((rt_max - rt_min) / bin_width_minutes)))
    bin_edges = np.linspace(rt_min, rt_max + 1e-9, n_bins + 1)
    bin_idx = np.clip(np.digitize(confident["rt_position"], bin_edges) - 1, 0, n_bins - 1)
    confident = confident.assign(_bin=bin_idx)

    per_bin: dict[int, tuple[int, int, int]] = {}
    for b, sub in confident.groupby("_bin"):
        shift_rt, shift_im = _joint_mode(sub["shift_rt"].to_numpy(), sub["shift_im"].to_numpy())
        per_bin[int(b)] = (shift_rt, shift_im, len(sub))

    global_shift_rt, global_shift_im = _joint_mode(
        confident["shift_rt"].to_numpy(), confident["shift_im"].to_numpy()
    )

    rows = []
    for b in range(n_bins):
        is_fallback = False
        if b in per_bin and per_bin[b][2] >= min_samples_per_bin:
            shift_rt, shift_im, n = per_bin[b]
        else:
            is_fallback = True
            neighbor = None
            for dist in range(1, max_neighbor_search_bins + 1):
                for cand in (b - dist, b + dist):
                    if cand in per_bin and per_bin[cand][2] >= min_samples_per_bin:
                        neighbor = per_bin[cand]
                        break
                if neighbor is not None:
                    break
            if neighbor is not None:
                shift_rt, shift_im, n = neighbor
            elif b in per_bin:
                shift_rt, shift_im, n = per_bin[b]
            else:
                shift_rt, shift_im, n = global_shift_rt, global_shift_im, 0
        rows.append(
            {
                "reference_run": reference_run,
                "matched_run": matched_run,
                "rt_bin_index": b,
                "rt_bin_start": float(bin_edges[b]),
                "rt_bin_end": float(bin_edges[b + 1]),
                "shift_rt": shift_rt,
                "shift_im": shift_im,
                "n_samples": n,
                "is_fallback": is_fallback,
            }
        )
    return rows


def build_shift_table(
    samples: pd.DataFrame,
    confidence_frac: float = 0.2,
    bin_width_minutes: float = 0.4,
    min_samples_per_bin: int = 10,
    max_neighbor_search_bins: int = 5,
) -> pd.DataFrame:
    """RT-binned, per-(reference_run, matched_run)-pair majority-vote shift table.

    Every bin in [min_bin, max_bin] per pair is filled -- so runtime lookup
    is a plain array index -- with sparse bins borrowing the nearest
    non-sparse neighbor, falling back to that pair's global joint-mode.
    """
    if samples.empty:
        return pd.DataFrame(columns=_TABLE_COLUMNS)

    rows: list[dict] = []
    for (reference_run, matched_run), group in samples.groupby(
        ["reference_run", "matched_run"]
    ):
        confident = _top_confidence_fraction(group, confidence_frac)
        rows.extend(
            _build_pair_bins(
                reference_run,
                matched_run,
                confident,
                bin_width_minutes,
                min_samples_per_bin,
                max_neighbor_search_bins,
            )
        )
    return pd.DataFrame(rows, columns=_TABLE_COLUMNS)


@dataclass
class ShiftLookup:
    bin_starts: dict[tuple[str, str], np.ndarray]
    shift_rt: dict[tuple[str, str], np.ndarray]
    shift_im: dict[tuple[str, str], np.ndarray]

    def lookup(
        self, reference_run: str, matched_run: str, rt_position: float
    ) -> tuple[int, int] | None:
        key = (reference_run, matched_run)
        starts = self.bin_starts.get(key)
        if starts is None or len(starts) == 0:
            return None
        idx = int(np.clip(np.searchsorted(starts, rt_position, side="right") - 1, 0, len(starts) - 1))
        return int(self.shift_rt[key][idx]), int(self.shift_im[key][idx])


def build_shift_lookup(table: pd.DataFrame) -> ShiftLookup:
    bin_starts: dict[tuple[str, str], np.ndarray] = {}
    shift_rt: dict[tuple[str, str], np.ndarray] = {}
    shift_im: dict[tuple[str, str], np.ndarray] = {}
    for (reference_run, matched_run), group in table.groupby(["reference_run", "matched_run"]):
        ordered = group.sort_values("rt_bin_start")
        key = (reference_run, matched_run)
        bin_starts[key] = ordered["rt_bin_start"].to_numpy()
        shift_rt[key] = ordered["shift_rt"].to_numpy()
        shift_im[key] = ordered["shift_im"].to_numpy()
    return ShiftLookup(bin_starts, shift_rt, shift_im)


def load_shift_table(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def _load_dict_ref(dict_ref_path: str) -> pd.DataFrame:
    if dict_ref_path.endswith((".pkl", ".pickle")):
        return pd.read_pickle(dict_ref_path)
    return pd.read_parquet(dict_ref_path)


def calibrate_broad_alignment(
    result_dir: str,
    raw_file_list: list[str],
    dict_ref_path: str,
    output_path: str | None = None,
    n_peptides: int = 500,
    rt_column: str = "RT_search_center",
    confidence_frac: float = 0.2,
    bin_width_minutes: float = 0.4,
    min_samples_per_bin: int = 10,
    max_neighbor_search_bins: int = 5,
    max_workers: int | None = None,
    require_msms_both_runs: bool = False,
) -> pd.DataFrame:
    """Build and persist the broad-alignment shift table for one dataset.

    Uses only dict_ref (or dict_ref_with_activation.pkl) and Stage-2
    activation/*.parquet output -- runnable between Stage 2 and Stage 3,
    before match_features output exists.

    `require_msms_both_runs` -- see _pairwise_align_chunk. Peptide selection
    (select_calibration_peptides) is unaffected; this only restricts which
    (peptide, reference_run, matched_run) triplets contribute a sample.
    """
    dict_ref = _load_dict_ref(dict_ref_path)
    calibration_mz_ranks = select_calibration_peptides(
        dict_ref, n_peptides=n_peptides, rt_column=rt_column
    )
    Logger.info(
        "broad_alignment calibration: %d peptides x %d raw files",
        len(calibration_mz_ranks),
        len(raw_file_list),
    )
    samples = run_pairwise_calibration_alignment(
        calibration_mz_ranks,
        raw_file_list,
        result_dir,
        dict_ref,
        rt_column=rt_column,
        max_workers=max_workers,
        require_msms_both_runs=require_msms_both_runs,
    )
    table = build_shift_table(
        samples,
        confidence_frac=confidence_frac,
        bin_width_minutes=bin_width_minutes,
        min_samples_per_bin=min_samples_per_bin,
        max_neighbor_search_bins=max_neighbor_search_bins,
    )
    resolved_output_path = output_path or os.path.join(
        result_dir, "broad_alignment_shift_table.parquet"
    )
    table.to_parquet(resolved_output_path, index=False)
    Logger.info("broad_alignment shift table written to %s", resolved_output_path)
    return table


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate the broad-alignment shift table.")
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--raw-file", action="append", required=True, dest="raw_file_list")
    parser.add_argument("--dict-ref-path", required=True)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--n-peptides", type=int, default=500)
    parser.add_argument("--rt-column", default="RT_search_center")
    parser.add_argument("--confidence-frac", type=float, default=0.2)
    parser.add_argument("--bin-width-minutes", type=float, default=0.4)
    parser.add_argument("--min-samples-per-bin", type=int, default=10)
    parser.add_argument("--max-neighbor-search-bins", type=int, default=5)
    parser.add_argument(
        "--require-msms-both-runs",
        action="store_true",
        help="Only use (peptide, ref_run, match_run) triplets where the peptide has a "
        "direct MS/MS identification (status Reference or Quant_Only) in BOTH runs, "
        "not an MBR-predicted Match position in either.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    _args = _parse_args()
    calibrate_broad_alignment(
        result_dir=_args.result_dir,
        raw_file_list=_args.raw_file_list,
        dict_ref_path=_args.dict_ref_path,
        output_path=_args.output_path,
        n_peptides=_args.n_peptides,
        rt_column=_args.rt_column,
        confidence_frac=_args.confidence_frac,
        bin_width_minutes=_args.bin_width_minutes,
        min_samples_per_bin=_args.min_samples_per_bin,
        max_neighbor_search_bins=_args.max_neighbor_search_bins,
        require_msms_both_runs=_args.require_msms_both_runs,
    )
