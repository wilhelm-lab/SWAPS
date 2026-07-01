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


def load_peptide_batch_df_from_partquet(
    activation_dir: str,
    pept_indicies,  # list of int or np.ndarray
    con: duckdb.DuckDBPyConnection | None = None,
) -> pd.DataFrame:
    """Load activation rows for a contiguous mz_rank slice from the sorted parquet.

    Relies on the file produced by :func:`build_mz_sorted_activation`.
    DuckDB skips row groups outside [min(pept_indicies), max(pept_indicies)],
    so read I/O is proportional to the batch fraction rather than total data.

    Parameters
    ----------
    activation_dir:
        Directory containing ``activation_sorted_by_mz.parquet``.
    pept_indicies:
        Sequence of mz_rank values for this batch.  Should be a contiguous
        range so the BETWEEN predicate matches exactly the needed row groups.
    con:
        Optional existing DuckDB connection.  A new one is created (and closed)
        when *None*.
    """
    sorted_path = os.path.join(activation_dir, _SORTED_ACTIVATION_FILENAME)
    own_connection = con is None
    if con is None:
        con = duckdb.connect()
        con.execute("SET enable_progress_bar = false")
    mz_min = int(min(pept_indicies))
    mz_max = int(max(pept_indicies))
    df = con.execute(
        f"SELECT * FROM parquet_scan('{sorted_path}') "
        f"WHERE mz_rank BETWEEN {mz_min} AND {mz_max}"
    ).df()
    if own_connection:
        con.close()
    return df


def get_pept_act_from_parquet(
    act_df,
    pept_idx,
    dict_ref,
    run_name,
    shape: Optional[tuple] = None,
    return_offset: bool = False,
):
    # Support both a pre-indexed DataFrame (index == "mz_rank") and a plain one.
    if dict_ref.index.name == "mz_rank":
        row = dict_ref.loc[[pept_idx]]
    else:
        row = dict_ref.loc[dict_ref["mz_rank"] == pept_idx, :]
    rt_start = row[f"MS1_frame_idx_left_ref_{run_name}"].values[0]
    rt_end = row[f"MS1_frame_idx_right_ref_{run_name}"].values[0]
    im_start = row[f"mobility_values_index_left_ref_{run_name}"].values[0]
    im_end = row[f"mobility_values_index_right_ref_{run_name}"].values[0]
    # rt_exp_start = row["MS1_frame_idx_left_exp"].values[0] - rt_start
    rt_exp_center = (
        (row[f"{run_name}_MS1_frame_idx_exp"].values[0] - rt_start)
        if row[f"{run_name}_MS1_frame_idx_exp"].values[0] > rt_start
        else ((rt_start + rt_end) // 2 - rt_start)
    )
    im_exp_center = (
        (row[f"{run_name}_mobility_values_index_exp"].values[0] - im_start)
        if row[f"{run_name}_mobility_values_index_exp"].values[0] > im_start
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
    """

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
    intensity_cols = [
        c for c in combined_ion.columns if c.endswith(intensity_suffix)
    ]
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
    intensity_cols = [
        c for c in combined_ions.columns if c.endswith(intensity_suffix)
    ]
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
