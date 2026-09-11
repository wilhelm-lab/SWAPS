"""Build activation_sorted_by_mz.parquet directly from a raw .d file's own
intensities, bypassing scan-wise activation (sparse coding) entirely.

Writes the exact same on-disk format/schema real SWA produces
(helper._SORTED_ACTIVATION_FILENAME, inference.init_parquet_scheme:
frame_idx uint16, im_idx uint16, mz_rank uint32, activation float32, sorted
by mz_rank) to the exact same location match_features.py already reads from
(RESULT_PATH/<raw_file>/activation/activation_sorted_by_mz.parquet) -- so
helper.load_peptide_batch_df_from_partquet / get_pept_act_from_parquet, and
therefore match_features.py itself, can be run completely unchanged against
it. Useful for a from-raw-data sanity check of the alignment/consensus
pipeline that isn't also exercising (or waiting on) the sparse-coding solve.

For each dict_ref candidate (mz_rank), sums raw intensity within its own
predicted RT/IM search window for this run -- MS1_frame_idx_left_ref_<run>/
_right_ref_<run>, mobility_values_index_left_ref_<run>/_right_ref_<run> --
at its monoisotopic m/z +/- ppm_tol. Reuses plot_raw_rt_im_image's own
alphatims query/scatter logic (_raw_rt_im_image) via a thin wrapper that
resolves those frame/scan INDEX bounds to the RT-minute/1-over-K0 bounds
that function expects; the two agree on where nearest-match aside, exactly
because the min/max values plugged in ARE the real Time_minute/mobility_
values already sitting at those exact index positions.

Unlike scan-wise activation's genuinely sparse (L1-regularized) solve, this
writes every raw, strictly-positive (frame_idx, im_idx) pixel within each
candidate's window -- a much denser representation, but behaviorally
identical to downstream readers: any (frame_idx, im_idx) pair absent from
the table reconstructs as 0 in helper.parquet_df_to_dense_frame either way.

If the run doesn't yet have its own MS1_frame_idx_left_ref_<run>/etc.
columns in dict_ref, they're added here (dict_add_index_to_raw_file/
dict_add_im_index/dict_add_rt_index -- the exact same calls
sbs_runner_ims.py's real Stage-2 loop makes) and the updated dict_ref is
saved back to dict_ref_with_activation.pkl, same as Stage 2 does. coSWA
confounder-group window columns (Group*) are NOT computed here -- this is a
single-candidate, no-optimization retrieval path with no group solve to
derive a merged window from; get_pept_act_from_parquet's use_group_window
falls back to each candidate's own individual window when those columns are
absent, which is what every candidate gets here regardless.
"""

import argparse
import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm

from optimization.inference import init_parquet_scheme
from postprocessing.helper import _SORTED_ACTIVATION_FILENAME, _SORTED_ROW_GROUP_SIZE
from postprocessing.plot_raw_rt_im_image import _raw_rt_im_image
from postprocessing.broad_alignment_image_based import _resolve_raw_file_paths
from prepare_dict.prepare_dict import dict_add_im_index, dict_add_index_to_raw_file, dict_add_rt_index
from utils.config import get_cfg_defaults, merge_cfg_from_file
from utils.ims_utils import export_im_and_ms1scans, load_dotd_data
from utils.singleton_swaps_optimization import swaps_optimization_cfg

logger = logging.getLogger(__name__)


def _index_bounds_to_native(
    ms1scans: pd.DataFrame, mobility_values_df: pd.DataFrame, rt_start: int, rt_end: int, im_start: int, im_end: int
) -> tuple[float, float, float, float]:
    """(rt_min, rt_max) minutes / (im_min, im_max) 1-over-K0 at exactly
    `rt_start`/`rt_end`/`im_start`/`im_end` -- plugging these back into
    _raw_rt_im_image's own Time_minute.between()/mobility_values.between()
    filtering selects EXACTLY the contiguous index range
    [rt_start, rt_end] x [im_start, im_end] back out again, since
    ms1scans/mobility_values_df are indexed by (and monotonic in)
    MS1_frame_idx/mobility_values_index with no gaps."""
    rt_min = float(ms1scans.loc[rt_start, "Time_minute"])
    rt_max = float(ms1scans.loc[rt_end, "Time_minute"])
    im_min = float(mobility_values_df.loc[im_start, "mobility_values"])
    im_max = float(mobility_values_df.loc[im_end, "mobility_values"])
    return rt_min, rt_max, im_min, im_max


def build_raw_activation_for_raw_file(
    dot_d_path: str,
    dict_ref: pd.DataFrame,
    result_dir: str,
    ppm_tol: float,
    export_hdf5_dir: Optional[str] = None,
    min_intensity: float = 0.0,
) -> tuple[pd.DataFrame, str]:
    """Builds and writes one raw file's activation_sorted_by_mz.parquet.
    Returns (dict_ref, possibly augmented with this run's own window
    columns; path to the written parquet)."""
    raw_file = os.path.basename(os.path.normpath(dot_d_path)).split(".")[0]
    act_dir = os.path.join(result_dir, raw_file, "activation")
    os.makedirs(act_dir, exist_ok=True)
    out_path = os.path.join(act_dir, _SORTED_ACTIVATION_FILENAME)

    logger.info("Loading %s", dot_d_path)
    data, _ = load_dotd_data(dot_d_path, swaps_result_dir=export_hdf5_dir or "")
    ms1scans, mobility_values_df = export_im_and_ms1scans(
        data, swaps_result_dir=os.path.join(result_dir, raw_file)
    )

    if f"MS1_frame_idx_left_ref_{raw_file}" not in dict_ref.columns:
        logger.info("Adding RT/IM index columns for %s to dict_ref", raw_file)
        dict_ref = dict_add_index_to_raw_file(dict_ref, mobility_values_df, ms1scans, raw_file)
        dict_ref = dict_add_im_index(
            dict_ref, mobility_values_df, "IM_search_left", "IM_search_center", "IM_search_right",
            idx_suffix=f"_ref_{raw_file}",
        )
        dict_ref = dict_add_rt_index(dict_ref, ms1scans, idx_suffix=f"_ref_{raw_file}")

    dict_ref_by_mz = dict_ref.drop_duplicates("mz_rank").set_index("mz_rank")

    frame_idx_parts, im_idx_parts, mz_rank_parts, activation_parts = [], [], [], []
    for mz_rank, row in tqdm.tqdm(
        dict_ref_by_mz.iterrows(), total=len(dict_ref_by_mz), desc=f"Raw activation for {raw_file}"
    ):
        rt_start = int(row[f"MS1_frame_idx_left_ref_{raw_file}"])
        rt_end = int(row[f"MS1_frame_idx_right_ref_{raw_file}"])
        im_start = int(row[f"mobility_values_index_left_ref_{raw_file}"])
        im_end = int(row[f"mobility_values_index_right_ref_{raw_file}"])
        if rt_end < rt_start or im_end < im_start:
            continue
        rt_min, rt_max, im_min, im_max = _index_bounds_to_native(
            ms1scans, mobility_values_df, rt_start, rt_end, im_start, im_end
        )
        img, _ = _raw_rt_im_image(
            data, ms1scans, mobility_values_df, float(row["m/z"]), rt_min, rt_max, im_min, im_max, ppm_tol
        )
        if img.size == 0:
            continue
        frame_positions, im_positions = np.nonzero(img > min_intensity)
        if len(frame_positions) == 0:
            continue
        frame_idx_parts.append(frame_positions + rt_start)
        im_idx_parts.append(im_positions + im_start)
        mz_rank_parts.append(np.full(len(frame_positions), mz_rank))
        activation_parts.append(img[frame_positions, im_positions])

    if not frame_idx_parts:
        raise ValueError(f"No raw signal retrieved for any candidate in {raw_file} -- check ppm_tol/dict_ref.")

    table_df = pd.DataFrame(
        {
            "frame_idx": np.concatenate(frame_idx_parts).astype(np.uint16),
            "im_idx": np.concatenate(im_idx_parts).astype(np.uint16),
            "mz_rank": np.concatenate(mz_rank_parts).astype(np.uint32),
            "activation": np.concatenate(activation_parts).astype(np.float32),
        }
    ).sort_values("mz_rank", ignore_index=True)

    schema = init_parquet_scheme(data_col_name="activation")
    table = pa.Table.from_pandas(table_df, schema=schema, preserve_index=False)
    pq.write_table(table, out_path, compression="snappy", row_group_size=_SORTED_ROW_GROUP_SIZE)
    logger.info("Raw activation for %s written to %s (%d rows)", raw_file, out_path, len(table_df))
    return dict_ref, out_path


def build_raw_activation(
    result_dir: str,
    raw_file_list: list[str],
    data_paths: list[str],
    exclude_dataset_names: list[str],
    dict_ref_path: Optional[str] = None,
    ppm_tol: Optional[float] = None,
    export_hdf5_dir: Optional[str] = None,
    min_intensity: float = 0.0,
) -> pd.DataFrame:
    """Runs build_raw_activation_for_raw_file for every raw file in
    raw_file_list, saving the (progressively augmented) dict_ref back to
    dict_ref_with_activation.pkl after each one -- same incremental-save
    pattern sbs_runner_ims.py's own Stage-2 loop uses, so a crash partway
    through a multi-run batch doesn't lose already-finished runs' index
    columns. Returns the final dict_ref.
    """
    raw_file_paths = _resolve_raw_file_paths(raw_file_list, data_paths, exclude_dataset_names)
    resolved_dict_ref_path = dict_ref_path or (
        os.path.join(result_dir, "dict_ref_with_activation.pkl")
        if os.path.exists(os.path.join(result_dir, "dict_ref_with_activation.pkl"))
        else os.path.join(result_dir, "dict_ref.pkl")
    )
    dict_ref = pd.read_pickle(resolved_dict_ref_path)

    cfg = get_cfg_defaults(swaps_optimization_cfg)
    merge_cfg_from_file(cfg, os.path.join(result_dir, "effective_config.yaml"))
    resolved_ppm_tol = float(ppm_tol) if ppm_tol is not None else float(cfg.PREPARE_DICT.PPM_TOL)

    for raw_file in raw_file_list:
        dict_ref, _ = build_raw_activation_for_raw_file(
            raw_file_paths[raw_file], dict_ref, result_dir, resolved_ppm_tol,
            export_hdf5_dir=export_hdf5_dir, min_intensity=min_intensity,
        )
        dict_ref.to_pickle(os.path.join(result_dir, "dict_ref_with_activation.pkl"))

    return dict_ref


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build activation_sorted_by_mz.parquet directly from raw .d files, bypassing scan-wise activation."
    )
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--raw-file", action="append", required=True, dest="raw_file_list")
    parser.add_argument("--data-path", action="append", required=True, dest="data_paths")
    parser.add_argument("--exclude-dataset-name", action="append", default=[], dest="exclude_dataset_names")
    parser.add_argument("--dict-ref-path", default=None, help="Overrides the default dict_ref_with_activation.pkl/dict_ref.pkl resolution.")
    parser.add_argument("--ppm-tol", type=float, default=None, help="Overrides cfg.PREPARE_DICT.PPM_TOL from effective_config.yaml.")
    parser.add_argument("--export-hdf5-dir", default=None)
    parser.add_argument("--min-intensity", type=float, default=0.0, help="Strictly-greater-than threshold for a pixel to be written (default: 0, i.e. any nonzero raw signal).")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    _args = _parse_args()
    build_raw_activation(
        result_dir=_args.result_dir,
        raw_file_list=_args.raw_file_list,
        data_paths=_args.data_paths,
        exclude_dataset_names=_args.exclude_dataset_names,
        dict_ref_path=_args.dict_ref_path,
        ppm_tol=_args.ppm_tol,
        export_hdf5_dir=_args.export_hdf5_dir,
        min_intensity=_args.min_intensity,
    )
