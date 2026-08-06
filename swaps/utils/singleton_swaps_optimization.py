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
__C.MATCH_FEATURES_KWARGS.align_in_log_space = True  # whether the template-matching shift search itself runs on log2(1+x)-transformed images (True) or on linear images (False). Purely an alignment/registration knob -- independent of denoise.log_transform.enabled below, which controls the separate question of whether the consensus average and per-run descriptor images end up in log space. Search always finds a shift; that shift is then applied to the linear image regardless of this flag.
__C.MATCH_FEATURES_KWARGS.use_shift_crop_pad = False  # False (default): resize every image to the reference's shape via cv2.resize before template matching. True: skip resizing -- match_template runs directly on each run's native-shaped image, and the found integer shift is applied by exact slicing (pad where the image is smaller than the reference, crop where larger) instead of interpolation.
__C.MATCH_FEATURES_KWARGS.broad_alignment = ConfigurationNode()
__C.MATCH_FEATURES_KWARGS.broad_alignment.enabled = False  # if True, center each candidate's per-run RT/IM template-match search on a precalibrated, RT-binned majority-vote shift (swaps.postprocessing.broad_alignment) instead of searching the whole image -- fixes low-S/N peptides picking a spurious, far-away shift, while still letting the true local optimum (bounded by max_deviation) win over the table's own (imperfect, per-bin) estimate. The table auto-builds (if missing) at RESULT_PATH/broad_alignment_shift_table.parquet, shared across every quantification_* config variant of the same dataset. No-ops with a warning if align_images=False. rt_shift/im_shift/template_matching_score stay real (non-NaN) values and remain in the Mokapot FDR feature list.
__C.MATCH_FEATURES_KWARGS.broad_alignment.max_deviation = 5  # radius (in RT/IM pixel units of the activation image -- one row per MS1 frame, one column per mobility bin) of the constrained search window around the calibrated shift; 0 forces the exact calibrated shift (just rescoring there), larger values allow more local refinement at some risk of drifting toward a nearby wrong peak.
__C.MATCH_FEATURES_KWARGS.dir_name = "quantification"
__C.MATCH_FEATURES_KWARGS.batching = ConfigurationNode()
__C.MATCH_FEATURES_KWARGS.batching.batch_size_max = 500  # max peptides/confounder-group-members per worker batch
__C.MATCH_FEATURES_KWARGS.batching.oversize_multiplier = 3.0  # peptides/groups whose estimated image size exceeds this x the batch's median get carved into their own small batch, so a few oversized merge_confounders images can't compound in one worker (see match_features._build_peptide_batches); 0 or None disables the carve-out
__C.MATCH_FEATURES_KWARGS.batching.oversize_batch_size = 20  # max oversized solo peptides grouped into one carved-out batch; an oversized confounder group always gets its own batch regardless of this value
__C.MATCH_FEATURES_KWARGS.seg_mask_thres = ConfigurationNode()
__C.MATCH_FEATURES_KWARGS.seg_mask_thres.rt = 2  # min RT span of target labels
__C.MATCH_FEATURES_KWARGS.seg_mask_thres.im = 5  # min IM span of target labels
__C.MATCH_FEATURES_KWARGS.jump_dist_thres = ConfigurationNode()
__C.MATCH_FEATURES_KWARGS.jump_dist_thres.rt = 10  # max RT jump in pixels; 0 = disabled
__C.MATCH_FEATURES_KWARGS.jump_dist_thres.im = 10  # max IM jump in pixels; 0 = disabled
__C.MATCH_FEATURES_KWARGS.template_frac = 0.3  # half-width (as a fraction of each dim) of the template patch used for SIFT/consensus template matching; must be in (0, 0.5]
# denoise: smooth/clean are staged ops (smooth.at / clean.at select which pipeline
# stage each runs at); log_transform is NOT staged -- see below.
#   "raw"       → applied to each run's own image before template-matching alignment
#   "consensus" → applied to the averaged consensus image before watershed segmentation
#
# log_transform.enabled controls a single, fixed point in the pipeline that applies to
# BOTH the consensus image and every individual run's own image: log2(1+x) is applied
# exactly once, AFTER averaging (for the consensus descriptor) or AFTER alignment (for
# each run's own descriptor) -- never before, and never as part of the average itself.
# This matters because mean_i(log2(1+x_i)) != log2(1 + mean_i(x_i)): averaging
# already-logged images (a geometric-mean-like quantity, by Jensen's inequality)
# systematically under-represents the true consensus intensity/shape relative to
# log-transforming the linear-space average, and would make the consensus descriptor
# incomparable to a real run's own log2(1+x) descriptor. Keeping log_transform
# unstaged guarantees the consensus is always built by averaging in linear space first.
# Watershed segmentation (on the consensus) also has log_transform applied to it,
# after smooth/clean, consistent with the "consensus" stage order (smooth -> clean ->
# log_transform); this is separate from the alignment search space, see
# align_in_log_space above.
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
__C.FDR.TRAIN = 0.01  # percolator's own -F default; train_fdr was never actually passed to percolator until this was wired up, so 0.01 matches what every prior run actually used
__C.FDR.TEST = 0.01
__C.FDR.INT_THRES = 100  # intensity threshold for FDR; 0 means no threshold
__C.FDR.ONLY_SCORE_MATCH = True  # if True, exclude Reference/Quant_Only run-peptide pairs from rescoring and pass them directly as "MS/MS" in the output
__C.FDR.PERCOLATOR_POST_PROCESSING = "tdc"  # "tdc" (-Y, target-decoy competition) or "mix-max" (-y); selects percolator's post-processing method and the percolator_dir_name suffix
