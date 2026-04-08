import os
import sparse
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import duckdb
from typing import Optional, List
import logging

Logger = logging.getLogger(__name__)


def load_peptide_batch_df_from_partquet(parquet_path, pept_indicies: List):
    con = duckdb.connect()
    # if not isinstance(pept_indicies, list):
    #     pept_indicies = pept_indicies.tolist()
    pept_ids_str = (
        tuple(pept_indicies) if len(pept_indicies) > 1 else f"({pept_indicies[0]})"
    )
    query = f"""
    SELECT *
    FROM parquet_scan('{parquet_path}')
    WHERE mz_rank IN {pept_ids_str}
    """
    df = con.execute(query).df()
    return df


def get_pept_act_from_parquet(
    act_df,
    pept_idx,
    dict_ref,
    run_name,
    shape: Optional[tuple] = None,
    return_offset: bool = False,
):
    # con = duckdb.connect()
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
        aggfunc={"Match Type": "first", "intensity_sum": "max"},
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
