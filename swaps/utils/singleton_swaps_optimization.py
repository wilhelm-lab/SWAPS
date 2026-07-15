from yacs.config import CfgNode as ConfigurationNode

__C = ConfigurationNode()

# importing default as a global singleton
swaps_optimization_cfg = __C

# general setup
__C.DESCRIPTION = "Default config from the Singleton"
__C.DEBUG = False
__C.SWA = True  # whether to perform scan-wise activation; if False, skips SWA and saves dict_ref with empty activation, useful for debugging downstream steps without running SWA
__C.DATA_PATH = (
    []
)  # path(s) to raw data directories (.d or .mzml); accepts a single string or a list of strings
__C.EXCLUDE_DATASET_NAME = (
    []
)  # list of dataset names to exclude, e.g. ["20230515_HeLa_100ng_1", "125pg_ddaPASEF_a000_2R_150ms_500SPD_S2-C7_1_19515.d"]
__C.RESULT_PATH = ""  # path to save all intermediate and final results
__C.ADD_TIMESTAMP_TO_RESULT_PATH = False  # disable when reusing the same result path
__C.EXPORT_DATA_HDF5_DIR = (
    ""  # path to export HDF5 data; empty string exports to the data directory
)
__C.SEARCH_OUTPUT_PATH = ""  # path to search engine output; for FragPipe: output directory; for MaxQuant: path to evidence.txt; for Sage: path to results.sage.tsv
__C.USE_IMS = True
__C.N_CPU = (
    -1
)  # number of CPUs for multiprocessing; <0 means use all CPUs from SLURM_CPUS_PER_TASK (or 1 if not in a SLURM job)

# prepare dictionary
__C.PREPARE_DICT = ConfigurationNode()
__C.PREPARE_DICT.SEARCH_ENGINE = "maxquant"  # one of ["maxquant", "fragpipe", "sage"]
__C.PREPARE_DICT.RT_REF = (
    "pred"  # How to calc ref RT; supported: "pred" (AlphaPeptDeep model)
)
__C.PREPARE_DICT.IM_REF = (
    "ref"  # How to calc ref IM; supported: "ref" (from search engine output)
)
__C.PREPARE_DICT.PRED = ConfigurationNode()
__C.PREPARE_DICT.PRED.UPDATED_RT_MODEL_PATH = (
    ""  # path to a pre-trained RT model; empty string triggers retraining
)
__C.PREPARE_DICT.PRED.UPDATED_IM_MODEL_PATH = (
    ""  # path to a pre-trained IM model; empty string triggers retraining
)
__C.PREPARE_DICT.PRED.TRAIN_FRAC = 0.9
__C.PREPARE_DICT.PRED.RT_TRAIN_EPOCHS = (
    15  # For nanoflow higher epochs are needed, e.g. 30
)
__C.PREPARE_DICT.PRED.IM_TRAIN_EPOCHS = 8

__C.PREPARE_DICT.REF = ConfigurationNode()
__C.PREPARE_DICT.REF.RT_TOL = (
    -0.1
)  # RT tolerance in minutes; negative means infer from data
__C.PREPARE_DICT.REF.IM_LENGTH = (
    -1
)  # IM elution length in mobility index; negative means infer from data
__C.PREPARE_DICT.REF.DELTA_IM_95 = (
    -0.1
)  # delta IM covering 95% of data; negative means infer from data; only used when IM_REF == "pred"
__C.PREPARE_DICT.REF.SUMMARIZE_WITHOUT_MATCH = False

__C.PREPARE_DICT.SAGE = ConfigurationNode()
__C.PREPARE_DICT.SAGE.RT_WINDOW = (
    0.0  # RT elution window (min) for SAGE; 0 = auto (1% of max RT)
)
__C.PREPARE_DICT.SAGE.IM_WINDOW = (
    0.0  # IM elution window for SAGE; 0 = auto (0.1 1/K0 units)
)

__C.PREPARE_DICT.MZ_BIN_DIGITS = 2
__C.PREPARE_DICT.ISO_MIN_AB_THRES = 0.01
__C.PREPARE_DICT.PPM_TOL = 20  # ppm tolerance for matching precursors to MS1 peaks
__C.PREPARE_DICT.BIN_WIDTH = 0.01  # m/z bin width

__C.PREPARE_DICT.OK = ConfigurationNode()
__C.PREPARE_DICT.OK.DIR = ""  # path to Oktoberfest rescoring output directory; empty string disables rescoring
__C.PREPARE_DICT.OK.OUTPUT = "psms"  # one of ["psms", "peptides"]
__C.PREPARE_DICT.OK.FDR = 0.01  # FDR threshold for Oktoberfest rescoring

__C.PREPARE_DICT.MERGE_CONFOUNDERS = ConfigurationNode()
__C.PREPARE_DICT.MERGE_CONFOUNDERS.ENABLED = False  # merge confounder groups (coSWA) into one SWA candidate row; off by default for backward compatibility
__C.PREPARE_DICT.MERGE_CONFOUNDERS.GROUP_ID_OFFSET = (
    -1
)  # id offset added to a group's min mz_rank to form confounder_group_id; -1 = auto-derive from max mz_rank
__C.PREPARE_DICT.MERGE_CONFOUNDERS.EXCLUDE_CROSS_TARGET_DECOY = (
    True  # never merge a target with a decoy into the same confounder group
)

# optimization
__C.OPTIMIZATION = ConfigurationNode()
__C.OPTIMIZATION.N_BATCH = (
    -1
)  # number of batches of MS1 scans; -1 means use the same value as N_CPU
__C.OPTIMIZATION.DELTA_MOBILITY_INDEX_THRES = 80  # threshold for delta mobility index; only effective when extract_im_peak=True (currently hardcoded to False)
__C.OPTIMIZATION.IM_PEAK_EXTRACTION_WIDTH = 4  # width for IM peak extraction; only effective when extract_im_peak=True (currently hardcoded to False)

# match features kwargs (postprocessing peak detection/matching parameters)
__C.MATCH_FEATURES_KWARGS = ConfigurationNode()
__C.MATCH_FEATURES_KWARGS.apply_seg = True
__C.MATCH_FEATURES_KWARGS.align_images = True  # set False to skip template-matching alignment; rt/im shift and template_matching_score will be 0
__C.MATCH_FEATURES_KWARGS.dir_name = "quantification"
__C.MATCH_FEATURES_KWARGS.seg_mask_thres = ConfigurationNode()
__C.MATCH_FEATURES_KWARGS.seg_mask_thres.rt = 2  # min RT span of target labels
__C.MATCH_FEATURES_KWARGS.seg_mask_thres.im = 5  # min IM span of target labels
__C.MATCH_FEATURES_KWARGS.jump_dist_thres = ConfigurationNode()
__C.MATCH_FEATURES_KWARGS.jump_dist_thres.rt = 10  # max RT jump in pixels; 0 = disabled
__C.MATCH_FEATURES_KWARGS.jump_dist_thres.im = 10  # max IM jump in pixels; 0 = disabled
# denoise: three sequential ops applied across two pipeline stages.
#   smooth.at / clean.at / log_transform.at controls which stage each op executes.
#   "raw"       → applied to raw images before SIFT template-matching alignment
#   "consensus" → applied to the averaged consensus image before watershed segmentation
#   At peak-property extraction, raw_aligned images are compensated by applying all ops.
__C.MATCH_FEATURES_KWARGS.denoise = ConfigurationNode()
__C.MATCH_FEATURES_KWARGS.denoise.smooth = ConfigurationNode()
__C.MATCH_FEATURES_KWARGS.denoise.smooth.at = "consensus"
__C.MATCH_FEATURES_KWARGS.denoise.smooth.filter = "gaussian"
__C.MATCH_FEATURES_KWARGS.denoise.smooth.gaussian_kwargs = ConfigurationNode()
__C.MATCH_FEATURES_KWARGS.denoise.smooth.gaussian_kwargs.sigma = [1, 2]
__C.MATCH_FEATURES_KWARGS.denoise.smooth.gaussian_kwargs.mode = "nearest"
__C.MATCH_FEATURES_KWARGS.denoise.smooth.uniform_kwargs = ConfigurationNode()
__C.MATCH_FEATURES_KWARGS.denoise.smooth.uniform_kwargs.size = [3, 5]
__C.MATCH_FEATURES_KWARGS.denoise.clean = ConfigurationNode()
__C.MATCH_FEATURES_KWARGS.denoise.clean.at = "consensus"
__C.MATCH_FEATURES_KWARGS.denoise.clean.threshold = 0
__C.MATCH_FEATURES_KWARGS.denoise.clean.remove_kwargs = ConfigurationNode()
__C.MATCH_FEATURES_KWARGS.denoise.clean.remove_kwargs.min_size = 3
__C.MATCH_FEATURES_KWARGS.denoise.log_transform = ConfigurationNode()
__C.MATCH_FEATURES_KWARGS.denoise.log_transform.at = "raw"
__C.MATCH_FEATURES_KWARGS.denoise.log_transform.enabled = True
__C.MATCH_FEATURES_KWARGS.peak_consensus_kwargs = ConfigurationNode()
__C.MATCH_FEATURES_KWARGS.peak_consensus_kwargs.int_threshold = 1
__C.MATCH_FEATURES_KWARGS.peak_consensus_kwargs.h_rel = 0.001
__C.MATCH_FEATURES_KWARGS.peak_consensus_kwargs.norm_percentile = 95
__C.MATCH_FEATURES_KWARGS.peak_consensus_kwargs.compactness = 0.001
__C.MATCH_FEATURES_KWARGS.peak_consensus_kwargs.normalize_before_hmaxima = False
__C.MATCH_FEATURES_KWARGS.consensus_decoy_kwargs = ConfigurationNode()
__C.MATCH_FEATURES_KWARGS.consensus_decoy_kwargs.strategies = [
    "peptide_swap",
]  # ["peptide_swap", "off_target_shift"]
__C.MATCH_FEATURES_KWARGS.consensus_decoy_kwargs.n_peptide_swap_decoys = 1
__C.MATCH_FEATURES_KWARGS.consensus_decoy_kwargs.n_off_target_shift_decoys = 1
__C.MATCH_FEATURES_KWARGS.consensus_decoy_kwargs.off_target_min_offset_frac = 0.35
__C.MATCH_FEATURES_KWARGS.consensus_decoy_kwargs.off_target_max_overlap_fraction = 0.05
# peptide_swap only: prefer near-isobaric co-eluting candidates (confounders column
# in dict_ref) as decoy source; falls back to full batch if none are in-batch
__C.MATCH_FEATURES_KWARGS.consensus_decoy_kwargs.use_confounder_sampling = True

__C.FDR = ConfigurationNode()
__C.FDR.ENABLED = True  # if False, skip Mokapot/percolator FDR control entirely; pp_match_target_filtered is pp_match_target with only intensity filtering (FDR.INT_THRES) applied
__C.FDR.TRAIN = 0.05
__C.FDR.TEST = 0.01
__C.FDR.INT_THRES = 100  # intensity threshold for FDR; 0 means no threshold
__C.FDR.ONLY_SCORE_MATCH = True  # if True, exclude Reference/Quant_Only run-peptide pairs from rescoring and pass them directly as "MS/MS" in the output
__C.FDR.PERCOLATOR_POST_PROCESSING = "tdc"  # "tdc" (-Y, target-decoy competition) or "mix-max" (-y); selects percolator's post-processing method and the percolator_dir_name suffix
