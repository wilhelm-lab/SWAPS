import os
import re
import sparse
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import pandas as pd
import duckdb
from typing import Optional, List, Dict
import logging

Logger = logging.getLogger(__name__)

_SORTED_ACTIVATION_FILENAME = "activation_sorted_by_mz.parquet"
_SORTED_ROW_GROUP_SIZE = 100_000

# Brain-region sample-tag convention shared by FragPipe combined_ion.tsv run
# columns and SWAPS run names, e.g. "..._P064051_Fresh2_5ug_..._4921[_1]".
SAMPLE_TAG_PATTERN = re.compile(r"P\d{6}_(?P<tag>.+?)_5ug")
REGION_REPLICATE_PATTERN = re.compile(r"^(?P<region>[A-Za-z]+)(?P<replicate>\d+)")

MSSTATS_COLUMNS = [
    "ProteinName",
    "PeptideSequence",
    "PrecursorCharge",
    "FragmentIon",
    "ProductCharge",
    "IsotopeLabelType",
    "Condition",
    "BioReplicate",
    "Run",
    "Intensity",
]


def parse_condition_bioreplicate(run_name: str) -> tuple[str, int]:
    """Extract (Condition, BioReplicate) from a run/column name.

    Expects the brain-region sample-tag convention ``P{6 digits}_<region><replicate>_5ug``
    (e.g. ``ssDDA_P064051_Fresh2_5ug_R1_BD6_1_4921``), shared by the FragPipe
    ``combined_ion.tsv`` run columns and the SWAPS run/column names.
    """
    tag_match = SAMPLE_TAG_PATTERN.search(run_name)
    if tag_match is None:
        raise ValueError(
            f"Could not find a 'P######_<tag>_5ug' sample tag in: {run_name!r}"
        )
    region_match = REGION_REPLICATE_PATTERN.match(tag_match.group("tag"))
    if region_match is None:
        raise ValueError(
            f"Could not parse region/replicate from tag: {tag_match.group('tag')!r}"
        )
    return region_match.group("region"), int(region_match.group("replicate"))


def build_mz_sorted_activation(
    activation_dir: str,
    row_group_size: int = _SORTED_ROW_GROUP_SIZE,
    delete_input_files: bool = True,
) -> str:
    """Merge all swa_frame_batch_*_activation.parquet files into one file sorted by mz_rank.

    This one-time preprocessing step enables DuckDB row-group skipping when
    workers query a contiguous mz_rank slice with BETWEEN.  I/O then scales
    with batch_size / N_total instead of requiring a full scan of all files.

    If the output file already exists the function returns immediately without
    rebuilding it.

    Parameters
    ----------
    activation_dir:
        Directory containing the per-frame-batch parquet files.
    row_group_size:
        Number of rows per parquet row group.  Smaller groups give finer
        skipping granularity at the cost of slightly larger file overhead.

    Returns
    -------
    str
        Absolute path of the written (or already-existing) sorted parquet file.
    """
    out_path = os.path.join(activation_dir, _SORTED_ACTIVATION_FILENAME)
    if os.path.exists(out_path):
        Logger.info("mz-sorted activation already exists, skipping: %s", out_path)
        return out_path

    batch_files = sorted(
        os.path.join(activation_dir, f)
        for f in os.listdir(activation_dir)
        if f.startswith("swa_frame_batch_") and f.endswith(".parquet")
    )
    # Read one file at a time to avoid exhausting OS file descriptors when many
    # runs are processed concurrently (DuckDB glob opens all ~128 files per
    # connection simultaneously, hitting ulimit with N concurrent workers).
    tables = [pq.read_table(f) for f in batch_files]
    combined = pa.concat_tables(tables, promote_options="none")
    del tables
    combined = combined.sort_by("mz_rank")
    pq.write_table(
        combined,
        out_path,
        compression="snappy",
        row_group_size=row_group_size,
    )
    del combined

    if delete_input_files:
        for filename in os.listdir(activation_dir):
            if filename.startswith("swa_frame_batch_"):
                os.remove(os.path.join(activation_dir, filename))
                Logger.info("Deleted input file: %s", filename)
    Logger.info("Written mz-sorted activation to %s", out_path)
    return out_path


def expand_group_ids_to_members(
    coord_frame_indices,
    coord_im_indices,
    coord_pept_indices,
    data,
    group_to_members: dict,
):
    """
    Duplicate rows whose coord_pept_indices entry is a confounder group id
    (a key in group_to_members) out to every member's real mz_rank. Used by
    load_peptide_batch_df_from_partquet to lazily re-expand a coSWA group's
    single stored activation trace, in memory, only for the batch currently
    being read -- the SWA write path stores exactly one row-set per group
    (see optimization.inference.collapse_candidates_by_confounder_group),
    never duplicated to disk. Rows whose coord_pept_indices entry is not a
    group id pass through unchanged.
    """
    coord_pept_indices = np.asarray(coord_pept_indices)
    if len(coord_pept_indices) == 0 or not group_to_members:
        return coord_frame_indices, coord_im_indices, coord_pept_indices, data
    is_group = np.isin(coord_pept_indices, list(group_to_members.keys()))
    if not is_group.any():
        return coord_frame_indices, coord_im_indices, coord_pept_indices, data

    solo_idx = np.flatnonzero(~is_group)
    group_idx = np.flatnonzero(is_group)
    rep_counts = np.array(
        [len(group_to_members[coord_pept_indices[i]]) for i in group_idx]
    )
    expanded_pept = np.concatenate(
        [coord_pept_indices[solo_idx]]
        + [group_to_members[coord_pept_indices[i]] for i in group_idx]
    )

    def _expand(arr):
        arr = np.asarray(arr)
        return np.concatenate([arr[solo_idx], np.repeat(arr[group_idx], rep_counts)])

    return (
        _expand(coord_frame_indices),
        _expand(coord_im_indices),
        expanded_pept,
        _expand(data),
    )


def load_peptide_batch_df_from_partquet(
    activation_dir: str,
    pept_indicies,  # list of int or np.ndarray
    group_to_members: Optional[dict] = None,
    con: duckdb.DuckDBPyConnection | None = None,
    expand_to_members: bool = True,
) -> pd.DataFrame:
    """Load activation rows for a batch of mz_ranks from the sorted parquet.

    Relies on the file produced by :func:`build_mz_sorted_activation`. Uses a
    single ``WHERE mz_rank IN (...)`` predicate: DuckDB's row-group min/max
    stats let it skip any row group whose range doesn't contain one of the
    requested values, so I/O scales with how much of the mz_rank space the
    batch actually touches. Measured on this dataset's activation parquets,
    this costs the same as the old BETWEEN(min, max) predicate for a
    contiguous batch (the production case), and is ~40x faster than a range
    scan for a batch scattered across the whole mz_rank space (e.g. a
    benchmark sample drawn from a filtered subset) -- a BETWEEN there would
    read almost the entire file since min/max span nearly all of it.

    Parameters
    ----------
    activation_dir:
        Directory containing ``activation_sorted_by_mz.parquet``.
    pept_indicies:
        Sequence of mz_rank values for this batch.
    group_to_members:
        Optional {confounder_group_id: [real mz_ranks present in THIS batch]}
        mapping (batch-scoped -- see match_features._group_members_in_batch).
        coSWA groups are stored on disk as a single row-set keyed by their
        group id, which is always > every real mz_rank (see
        prepare_dict.dict_add_confounder_group_id). When provided, group ids
        are folded into the same IN(...) predicate as pept_indicies and, if
        `expand_to_members` is True, each group's row-set is duplicated in
        memory to every member key -- lazily reconstructing, per batch, the
        same per-member view the SWA write path used to persist to disk
        (before that duplication was found to blow up
        build_mz_sorted_activation's memory footprint by up to ~20x on real
        datasets).
    con:
        Optional existing DuckDB connection.  A new one is created (and closed)
        when *None*.
    expand_to_members:
        When True (default), duplicate each group's row-set to every member's
        own mz_rank as described above. When False, group rows are returned
        as-is, still keyed by their confounder_group_id -- for callers that
        resolve a member's own mz_rank to its group id at lookup time instead
        (see match_features._act_lookup_key), avoiding both the duplication
        cost and the larger, more-duplicate-heavy mz_rank index that later
        per-candidate lookups would otherwise run against.
    """
    sorted_path = os.path.join(activation_dir, _SORTED_ACTIVATION_FILENAME)
    own_connection = con is None
    if con is None:
        con = duckdb.connect()
        con.execute("SET enable_progress_bar = false")
    ids = {int(v) for v in pept_indicies}
    if group_to_members:
        ids.update(int(g) for g in group_to_members.keys())
    ids_sql = ",".join(str(v) for v in ids)
    query = f"SELECT * FROM parquet_scan('{sorted_path}') WHERE mz_rank IN ({ids_sql})"
    df = con.execute(query).df()
    if own_connection:
        con.close()
    if group_to_members and expand_to_members:
        frame_idx, im_idx, mz_rank, data = expand_group_ids_to_members(
            df["frame_idx"].to_numpy(),
            df["im_idx"].to_numpy(),
            df["mz_rank"].to_numpy(),
            df["activation"].to_numpy(),
            group_to_members,
        )
        df = pd.DataFrame(
            {
                "frame_idx": frame_idx,
                "im_idx": im_idx,
                "mz_rank": mz_rank,
                "activation": data,
            }
        )
    return df


def get_pept_act_from_parquet(
    act_df,
    pept_idx,
    dict_ref,
    run_name,
    shape: Optional[tuple] = None,
    return_offset: bool = False,
    use_group_window: bool = False,
    window_override: Optional[tuple] = None,
):
    """`window_override`, if given, is `((rt_start, rt_end), (im_start, im_end))` --
    used in place of run_name's own dict_ref-predicted window entirely (skips both
    the group- and individual-window lookups below). broad_alignment consumers use
    this to crop a "Match" run at a window DERIVED from the reference run's own
    window shifted by the calibrated inter-run offset, instead of that run's own
    independently-predicted (and for Match runs, unverified) window -- see
    match_features.py's per-candidate loop."""
    # Support both a pre-indexed DataFrame (index == "mz_rank") and a plain one.
    if dict_ref.index.name == "mz_rank":
        row = dict_ref.loc[[pept_idx]]
    else:
        row = dict_ref.loc[dict_ref["mz_rank"] == pept_idx, :]
    if window_override is not None:
        (rt_start, rt_end), (im_start, im_end) = window_override
    # coSWA: confounder-group members crop to the merged (union) RT/IM window
    # the shared SWA solve spans, not each member's own narrower individual
    # window. Falls back to the individual window when group columns are absent
    # (solo candidates, or runs built without confounder merging).
    elif use_group_window and f"MS1_frame_idx_left_group_ref_{run_name}" in row.columns:
        rt_start = row[f"MS1_frame_idx_left_group_ref_{run_name}"].values[0]
        rt_end = row[f"MS1_frame_idx_right_group_ref_{run_name}"].values[0]
        im_start = row[f"mobility_values_index_left_group_ref_{run_name}"].values[0]
        im_end = row[f"mobility_values_index_right_group_ref_{run_name}"].values[0]
    else:
        rt_start = row[f"MS1_frame_idx_left_ref_{run_name}"].values[0]
        rt_end = row[f"MS1_frame_idx_right_ref_{run_name}"].values[0]
        im_start = row[f"mobility_values_index_left_ref_{run_name}"].values[0]
        im_end = row[f"mobility_values_index_right_ref_{run_name}"].values[0]
    # rt_exp_start = row["MS1_frame_idx_left_exp"].values[0] - rt_start
    # Bounded on BOTH sides, not just the lower one: with window_override (see
    # above), the window can come from the reference's own window shifted by a
    # calibrated inter-run offset rather than from this run's own predicted
    # RT -- a whole-image drift estimate, not a guarantee about where THIS
    # peptide's own independently-observed position falls. When it falls
    # outside the window entirely, falling back to the window's own center
    # (same fallback the pre-existing lower-bound check already used) keeps
    # the anchor sane instead of feeding _build_reference_template a wildly
    # out-of-range value -- upstream of match_template's crash on an empty
    # template.mean() (IndexError: too many indices for array, from an
    # inverted/negative-size template slice).
    _rt_exp = row[f"{run_name}_MS1_frame_idx_exp"].values[0]
    rt_exp_center = (
        (_rt_exp - rt_start)
        if rt_start < _rt_exp <= rt_end
        else ((rt_start + rt_end) // 2 - rt_start)
    )
    _im_exp = row[f"{run_name}_mobility_values_index_exp"].values[0]
    im_exp_center = (
        (_im_exp - im_start)
        if im_start < _im_exp <= im_end
        else ((im_start + im_end) // 2 - im_start)
    )
    pept_act = parquet_df_to_dense_frame(act_df, (rt_start, rt_end), (im_start, im_end))

    if shape is not None:
        target_h, target_w = shape

        pept_act = sym_pad_crop_2d(pept_act, target_h, target_w)
    if return_offset:
        return pept_act, rt_exp_center, im_exp_center, (rt_start, im_start)
    else:
        return pept_act, rt_exp_center, im_exp_center


def sym_pad_crop_2d(arr, target_h, target_w):
    src_h, src_w = arr.shape[0], arr.shape[1]

    # 1. Math (Standard CPU registers)
    h_diff, w_diff = target_h - src_h, target_w - src_w
    h_bef, w_bef = h_diff // 2, w_diff // 2

    # 2. Pre-allocate (The biggest cost)
    res = np.zeros((target_h, target_w) + arr.shape[2:], dtype=arr.dtype)

    # 3. Direct Slice Calculation (No loops, no lists)
    # Define source boundaries
    sh0, sh1 = (0, src_h) if h_diff >= 0 else (-h_bef, -h_bef + target_h)
    sw0, sw1 = (0, src_w) if w_diff >= 0 else (-w_bef, -w_bef + target_w)

    # Define destination boundaries
    dh0, dh1 = (h_bef, h_bef + src_h) if h_diff >= 0 else (0, target_h)
    dw0, dw1 = (w_bef, w_bef + src_w) if w_diff >= 0 else (0, target_w)

    # 4. The "One-Shot" Copy
    res[dh0:dh1, dw0:dw1] = arr[sh0:sh1, sw0:sw1]

    return res


def convert_sparse_to_parquet(npz_path):
    """
    Converts a sparse.COO NPZ to Parquet, replacing the extension.
    Uses uint32 for indices to prevent overflow/data loss.
    """
    # 1. Load the sparse matrix
    s_matrix = sparse.load_npz(npz_path)

    # sparse.COO.coords is shape (ndim, nnz)
    # Typically: [0]=frame, [1]=im, [2]=mz_rank
    coords = s_matrix.coords  # type: ignore

    # 2. Build the columnar dictionary
    # We use uint32 for all indices to safely handle large datasets
    coo_dict = {
        "frame_idx": coords[0].astype(np.uint32),
        "im_idx": coords[1].astype(np.uint32),
        "mz_rank": coords[2].astype(np.uint32),
        "activation": s_matrix.data.astype(np.float32),
    }

    # 3. Define the Schema
    schema = pa.schema(
        [
            ("frame_idx", pa.uint32()),
            ("im_idx", pa.uint32()),
            ("mz_rank", pa.uint32()),
            ("activation", pa.float32()),
        ]
    )

    # 4. Generate the new path and save
    parquet_path = npz_path.replace(".npz", ".parquet")
    table = pa.Table.from_pydict(coo_dict, schema=schema)
    pq.write_table(table, parquet_path, compression="snappy")

    print(f"Done: {os.path.basename(parquet_path)} ({s_matrix.nnz} points)")


# Usage
# convert_sparse_to_parquet('path/to/your_file.npz')
def parquet_df_to_dense_frame(
    df, frame_range: Optional[tuple] = None, im_range: Optional[tuple] = None
):
    if frame_range is None:
        frame_range = (df["frame_idx"].min(), df["frame_idx"].max())
    if im_range is None:
        im_range = (df["im_idx"].min(), df["im_idx"].max())
    # Create an empty grid
    shape = (im_range[1] - im_range[0] + 1, frame_range[1] - frame_range[0] + 1)
    grid = np.zeros(shape)
    df_filtered = df[
        (df["frame_idx"] >= frame_range[0])
        & (df["frame_idx"] <= frame_range[1])
        & (df["im_idx"] >= im_range[0])
        & (df["im_idx"] <= im_range[1])
    ].copy()
    # Map indices to 0-based grid coordinates
    rows = df_filtered["im_idx"].values - im_range[0]
    cols = df_filtered["frame_idx"].values - frame_range[0]
    vals = df_filtered["activation"].values
    # Fill the grid
    grid[rows, cols] = vals

    return grid.T


def build_pivot(pp_all, dict_ref):
    """
    Create a pivoted dataframe with mz_rank as index and Run_name-specific
    Match Type and Intensity columns.

    Parameters
    ----------
    pp_all : pd.DataFrame
        DataFrame containing ['mz_rank', 'Run_name', 'Match Type', 'intensity_sum'].
    dict_ref : dict or pd.DataFrame
        Reference containing mz_rank values to enforce full index coverage.

    Returns
    -------
    pd.DataFrame
        Pivoted dataframe with '{Run} Match Type' and '{Run} Intensity' columns.
        Rows whose "Match Type" is "MBR" but which belong to a coSWA
        confounder group flagged as spatially undistinguishable (see
        `undistinguishable_group_id`) are relabeled "MBR_undistinguished"
        instead, since such MBR hits cannot be reliably attributed to one
        specific group member and should not feed into downstream LFQ.
    """
    _has_undistinguishable_group_id = "undistinguishable_group_id" in pp_all.columns
    if _has_undistinguishable_group_id:
        pp_all = pp_all.copy()
        _is_undistinguishable = ~pp_all["undistinguishable_group_id"].astype(str).isin(
            ["-1", "-1.0"]
        )
        pp_all.loc[
            (pp_all["Match Type"] == "MBR") & _is_undistinguishable, "Match Type"
        ] = "MBR_undistinguished"
        pp_all_undistinguishable_group_id = pp_all.groupby("mz_rank").agg(
            {"undistinguishable_group_id": "first"}
        )
    pivot = pp_all.pivot_table(
        index="mz_rank",
        columns="Run_name",
        values=["Match Type", "intensity_sum"],
        aggfunc={"Match Type": "first", "intensity_sum": "sum"},
    )

    # Rename intensity column level
    pivot = pivot.rename(columns={"intensity_sum": "Intensity"})

    # Fill missing match types
    pivot["Match Type"] = pivot["Match Type"].fillna("unmatched")

    # Flatten column names
    pivot.columns = [f"{run} {metric}" for metric, run in pivot.columns]

    # Ensure all mz_rank from dict_ref exist
    pivot = pivot.reindex(dict_ref["mz_rank"])
    if _has_undistinguishable_group_id:
        pivot = pd.merge(
            pivot,
            pp_all_undistinguishable_group_id,
            left_index=True,
            right_index=True,
            how="left",
        )
    # Fill missing Match Type again after reindexing
    for col in pivot.columns:
        if "Match Type" in col:
            pivot[col] = pivot[col].fillna("unmatched")

    return pivot.reset_index()


def _wide_intensity_to_msstats_long(
    df: pd.DataFrame,
    intensity_cols: List[str],
    id_cols: Dict[str, str],
    intensity_suffix: str = " Intensity",
    drop_missing: bool = True,
    zero_as_missing: bool = False,
) -> pd.DataFrame:
    """Melt a wide per-run-intensity table into MSstats long format.

    Parameters
    ----------
    df : pd.DataFrame
        Wide table with one row per ion and one ``<run><intensity_suffix>``
        column per run, plus the identity columns named in `id_cols`.
    intensity_cols : list of str
        Columns in `df` holding per-run intensities.
    id_cols : dict
        Maps MSstats column name -> source column name, for
        ``{"ProteinName": ..., "PeptideSequence": ..., "PrecursorCharge": ...}``.
    intensity_suffix : str
        Suffix stripped from `intensity_cols` to recover the run name, which
        is then parsed with :func:`parse_condition_bioreplicate`.
    drop_missing : bool
        Drop rows with a missing intensity, i.e. ions not observed in that run.
    zero_as_missing : bool
        Treat an intensity of exactly 0.0 as missing (NaN) before dropping.
        FragPipe's ``combined_ion.tsv`` encodes "not observed" as 0.0 rather
        than NaN, unlike the SWAPS pivot table.

    Returns
    -------
    pd.DataFrame
        Long-format table with the standard MSstats columns
        (see `MSSTATS_COLUMNS`).
    """
    run_names = sorted({col[: -len(intensity_suffix)] for col in intensity_cols})
    run_meta = {
        name: (*parse_condition_bioreplicate(name), run_id)
        for run_id, name in enumerate(run_names, start=1)
    }
    condition_map = {name: meta[0] for name, meta in run_meta.items()}
    bioreplicate_map = {name: meta[1] for name, meta in run_meta.items()}
    run_id_map = {name: meta[2] for name, meta in run_meta.items()}

    long_df = df.melt(
        id_vars=list(id_cols.values()),
        value_vars=intensity_cols,
        var_name="_run_col",
        value_name="Intensity",
    )
    if zero_as_missing:
        long_df["Intensity"] = long_df["Intensity"].replace(0.0, np.nan)
    if drop_missing:
        long_df = long_df.dropna(subset=["Intensity"])

    run_name = long_df.pop("_run_col").str[: -len(intensity_suffix)]
    long_df["Condition"] = run_name.map(condition_map)
    long_df["BioReplicate"] = run_name.map(bioreplicate_map)
    long_df["Run"] = run_name.map(run_id_map)
    long_df["IsotopeLabelType"] = "L"
    long_df["FragmentIon"] = np.nan
    long_df["ProductCharge"] = np.nan

    long_df = long_df.rename(columns={src: dst for dst, src in id_cols.items()})
    return long_df[MSSTATS_COLUMNS].reset_index(drop=True)


def combined_ion_to_msstats(
    combined_ion: pd.DataFrame,
    protein_col: str = "Protein ID",
    peptide_col: str = "Modified Sequence",
    charge_col: str = "Charge",
    intensity_suffix: str = " Intensity",
    drop_missing: bool = True,
) -> pd.DataFrame:
    """Convert a FragPipe ``combined_ion.tsv`` wide table to MSstats long format.

    Each ``<run>{intensity_suffix}`` column becomes one set of rows; Condition
    and BioReplicate are parsed from the run name (see
    :func:`parse_condition_bioreplicate`), Run is a consecutive integer per
    unique run, and IsotopeLabelType is always ``"L"`` (no isotope labeling).
    FragmentIon/ProductCharge are NA since this is precursor (ion)-level, not
    fragment-level, data. FragPipe encodes "not observed in this run" as an
    intensity of 0.0 (rather than NaN), so those rows are treated as missing
    too.
    """
    intensity_cols = [c for c in combined_ion.columns if c.endswith(intensity_suffix)]
    if not intensity_cols:
        raise ValueError(
            f"No columns ending in {intensity_suffix!r} found in combined_ion."
        )
    id_cols = {
        "ProteinName": protein_col,
        "PeptideSequence": peptide_col,
        "PrecursorCharge": charge_col,
    }
    return _wide_intensity_to_msstats_long(
        combined_ion,
        intensity_cols,
        id_cols,
        intensity_suffix,
        drop_missing,
        zero_as_missing=True,
    )


def swaps_combined_ions_to_msstats(
    combined_ions: pd.DataFrame,
    dict_ref: pd.DataFrame,
    protein_col: str = "Proteins",
    peptide_col: str = "Modified sequence",
    charge_col: str = "Charge",
    ion_id_col: str = "mz_rank",
    intensity_suffix: str = " Intensity",
    drop_missing: bool = True,
) -> pd.DataFrame:
    """Convert a SWAPS ``swaps_combined_ions.parquet`` wide table to MSstats long format.

    `combined_ions` is the ``mz_rank`` + per-run ``"<run> Match Type"`` /
    ``"<run> Intensity"`` table produced by :func:`build_pivot`. Peptide
    identity (sequence/charge/protein) isn't carried in that table and is
    joined in from `dict_ref` (the pipeline's ``dict_ref.pkl``) on
    `ion_id_col`. Rows with no match in a given run have NaN intensity and
    are dropped, same as for `combined_ion_to_msstats`.
    """
    intensity_cols = [c for c in combined_ions.columns if c.endswith(intensity_suffix)]
    if not intensity_cols:
        raise ValueError(
            f"No columns ending in {intensity_suffix!r} found in combined_ions."
        )
    merged = combined_ions[[ion_id_col] + intensity_cols].merge(
        dict_ref[[ion_id_col, protein_col, peptide_col, charge_col]],
        on=ion_id_col,
        how="left",
    )
    id_cols = {
        "ProteinName": protein_col,
        "PeptideSequence": peptide_col,
        "PrecursorCharge": charge_col,
    }
    return _wide_intensity_to_msstats_long(
        merged, intensity_cols, id_cols, intensity_suffix, drop_missing
    )


def reassign_quant_only_as_msms(
    swaps_combined_ions: pd.DataFrame,
    dict_ref: pd.DataFrame,
    ion_id_col: str = "mz_rank",
    match_type_suffix: str = " Match Type",
    intensity_suffix: str = " Intensity",
    int_thres: Optional[float] = None,
) -> pd.DataFrame:
    """Relabel run-peptide pairs marked "Reference"/"Quant_Only" in `dict_ref` as "MS/MS".

    `dict_ref` has one column per raw file name whose cells encode the
    search-engine match status ("Reference", "Quant_Only", "Not_Match", ...,
    see `prepare_dict.pivot_psm_by_mz_rank`). Both "Reference" (the run chosen
    as that peptide's global identification anchor) and "Quant_Only" (every
    other run where the search engine also directly identified it) mean the
    run-peptide pair was identified from its own MS/MS, so the corresponding
    `"<run>{match_type_suffix}"` cell in `swaps_combined_ions` (produced by
    :func:`build_pivot`) should be reported as "MS/MS" rather than whatever
    SWAPS assigned it (e.g. "MS/MS Ref", "MBR").

    If `int_thres` is given, any "MS/MS" cell (including ones just relabeled
    above) whose matching `"<run>{intensity_suffix}"` value is missing or
    below `int_thres` is further relabeled "Zero Quant", since it was
    identified but not usably quantified.

    Matches rows on `ion_id_col` and returns a copy of `swaps_combined_ions`
    with the affected cells updated.
    """
    combined = swaps_combined_ions.copy()
    has_id_index = combined.index.name == ion_id_col
    if not has_id_index:
        combined = combined.set_index(ion_id_col)

    dict_ref_by_id = dict_ref.set_index(ion_id_col)
    run_cols = [
        col
        for col in dict_ref_by_id.columns
        if f"{col}{match_type_suffix}" in combined.columns
    ]
    for run in run_cols:
        msms_ids = dict_ref_by_id.index[
            dict_ref_by_id[run].isin(["Reference", "Quant_Only"])
        ]
        idx = combined.index.intersection(msms_ids)
        combined.loc[idx, f"{run}{match_type_suffix}"] = "MS/MS"

    if int_thres is not None:
        for run in run_cols:
            match_type_col = f"{run}{match_type_suffix}"
            intensity_col = f"{run}{intensity_suffix}"
            if intensity_col not in combined.columns:
                continue
            is_msms = combined[match_type_col] == "MS/MS"
            low_intensity = (
                combined[intensity_col].isna() | (combined[intensity_col] < int_thres)
            )
            combined.loc[is_msms & low_intensity, match_type_col] = "Zero Quant"

    if not has_id_index:
        combined = combined.reset_index()
    return combined
