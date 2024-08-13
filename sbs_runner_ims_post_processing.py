"""Module providing a function calling the scan by scan optimization."""
import logging

import time
import fire
import os

# from utils.config_json import Config
from utils.config import get_cfg_defaults
from utils.singleton_swaps_optimization import swaps_optimization_cfg
from utils.ims_utils import combine_3d_act_and_sum_int, sum_pept_act_by_peptbatch

# os.environ["NUMEXPR_MAX_THREADS"] = "8"


def pp_scan_by_scan(config_path: str):
    """Scan by scan optimization for joint identification and quantification."""
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    cfg = get_cfg_defaults(swaps_optimization_cfg)
    cfg.merge_from_file(config_path)
    # conf = Config(config_path)
    # conf.make_result_dirs()

    # start analysis
    start_time_init = time.time()
    # _combine_3d_act(
    #     n_blocks_by_pept=conf.n_blocks_by_pept,
    #     n_batch=conf.n_batch,
    #     output_file=conf.output_file,
    #     result_dir=conf.result_dir,
    #     remove_batch_file=False,
    # )
    sum_pept_act_by_peptbatch(
        n_blocks_by_pept=cfg.OPTIMIZATION.N_BLOCKS_BY_PEPT,
        act_dir=os.path.join(cfg.RESULT_PATH, "results", "activation"),
    )
    minutes, seconds = divmod(time.time() - start_time_init, 60)
    logging.info(
        "Process scans - Script execution time: %dm %ds", int(minutes), int(seconds)
    )


if __name__ == "__main__":
    fire.Fire(pp_scan_by_scan)
