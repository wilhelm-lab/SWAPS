"""Repair tool: rebuild RESULT_PATH/dict_ref_with_activation.pkl from
RESULT_PATH/dict_ref.pkl plus each raw file's already-exported
ms1scans.csv/mobility_values.csv, WITHOUT reloading the raw .d file or
rerunning the scan-wise-activation sparse-coding solve
(optimization.inference.process_frames_parallel).

Use when dict_ref_with_activation.pkl has been lost, corrupted, or
partially overwritten -- e.g. an accidental SWA=True re-run that got
interrupted partway through its per-raw-file loop: dict_ref is saved after
EVERY raw file in that loop (see sbs_runner_ims.opt_scan_by_scan), so a run
killed partway through leaves only some raw files' MS1_frame_idx_left_ref_*/
IM_search_*_ref_* columns populated, which later crashes match_features_batch
with a KeyError on the missing columns for the untouched raw files. This
script re-derives those columns for every raw file directory under
RESULT_PATH from its already-exported per-run CSVs -- pure pandas
merge_asof, no AlphaTims/raw-file access -- and re-saves
dict_ref_with_activation.pkl. The activation/*.parquet files themselves
(the actual, expensive SWA output) are untouched and reused as-is.

Usage:
    python swaps/rebuild_dict_ref_with_activation.py <path-to-config.yaml>
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd

from sbs_runner_ims import _augment_dict_ref_with_run_indices
from utils.config import get_cfg_defaults, merge_cfg_from_file
from utils.singleton_swaps_optimization import swaps_optimization_cfg

Logger = logging.getLogger(__name__)


def rebuild_dict_ref_with_activation(cfg) -> pd.DataFrame:
    dict_ref_path = os.path.join(cfg.RESULT_PATH, "dict_ref.pkl")
    assert os.path.exists(dict_ref_path), (
        f"{dict_ref_path} not found -- this repairs dict_ref_with_activation.pkl "
        "from the pre-activation dict_ref.pkl; run prepare_dict first if that's "
        "missing too."
    )
    dict_ref = pd.read_pickle(dict_ref_path)
    Logger.info("Loaded pre-activation dict_ref: %d entries", len(dict_ref))

    # Same raw_file_list derivation match_features_batches_parallel uses
    # downstream (sbs_runner_ims.opt_scan_by_scan) -- every RESULT_PATH
    # subdirectory except the quantification output -- so the columns
    # rebuilt here line up exactly with what the pipeline will look up.
    raw_file_list = sorted(
        d
        for d in os.listdir(cfg.RESULT_PATH)
        if os.path.isdir(os.path.join(cfg.RESULT_PATH, d))
        and not d.startswith("quantification")
    )
    Logger.info(
        "Found %d raw file directories under %s", len(raw_file_list), cfg.RESULT_PATH
    )

    out_path = os.path.join(cfg.RESULT_PATH, "dict_ref_with_activation.pkl")
    for raw_file in raw_file_list:
        raw_dir = os.path.join(cfg.RESULT_PATH, raw_file)
        ms1scans_path = os.path.join(raw_dir, "ms1scans.csv")
        mobility_path = os.path.join(raw_dir, "mobility_values.csv")
        if not (os.path.exists(ms1scans_path) and os.path.exists(mobility_path)):
            Logger.warning(
                "Skipping %s: missing ms1scans.csv/mobility_values.csv "
                "(not a raw-file result dir, or its original SWA export "
                "never finished)",
                raw_file,
            )
            continue
        Logger.info("Augmenting dict_ref for %s", raw_file)
        ms1scans = pd.read_csv(ms1scans_path)
        mobility_values_df = pd.read_csv(mobility_path)
        dict_ref = _augment_dict_ref_with_run_indices(
            dict_ref, mobility_values_df, ms1scans, raw_file
        )
        # Save after every raw file, same as the original SWA loop, so an
        # interruption here is likewise resumable rather than losing
        # everything done so far.
        dict_ref.to_pickle(out_path)
        Logger.info("Done: %s", raw_file)

    Logger.info("Rebuilt %s (%d entries)", out_path, len(dict_ref))
    return dict_ref


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_path", help="Path to the configuration YAML file")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    cfg = get_cfg_defaults(swaps_optimization_cfg)
    merge_cfg_from_file(cfg, args.config_path)

    rebuild_dict_ref_with_activation(cfg)


if __name__ == "__main__":
    main()
