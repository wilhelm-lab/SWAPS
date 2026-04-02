from yacs.config import CfgNode as ConfigurationNode

__C = ConfigurationNode()

# importing default as a global singleton
swaps_optimization_cfg = __C

# general setup
__C.DESCRIPTION = "Default config from the Singleton"
__C.DEBUG = False
__C.SWA = True  # whether to perform scan-wise activation, if False, the code will skip the SWA part and directly save the dict_ref with empty activation for downstream processing, which is useful for debugging and testing downstream steps without running SWA
__C.DATA_PATH = ""  # different results depending on data ending with .mzml or .d
__C.EXCLUDE_DATASET_NAME = []
__C.RESULT_PATH = ""  # path to save all intermediate and final results
__C.ADD_TIMESTAMP_TO_RESULT_PATH = False  # disenble when reusing the same result path
__C.EXPORT_DATA_HDF5_DIR = ""  # empty string for export to data directory
__C.SEARCH_OUTPUT_PATH = ""  # path to MaxQuant reference file, pickle
__C.USE_IMS = True
__C.N_CPU = (
    -1
)  # number of CPUs for multiprocessing, <0 means using all CPUs requested by the slurm job (if any), else 0

# prepare dictionary
__C.PREPARE_DICT = ConfigurationNode()
__C.PREPARE_DICT.SEARCH_ENGINE = "maxquant"  # one of ["maxquant", "fragpipe", "sage"]
__C.PREPARE_DICT.UPDATED_RT_MODEL_PATH = ""
__C.PREPARE_DICT.UPDATED_IM_MODEL_PATH = ""
__C.PREPARE_DICT.TRAIN_FRAC = 0.9
__C.PREPARE_DICT.RT_TRAIN_EPOCHS = 15  # For nanoflow higher epochs are needed, e.g. 30
__C.PREPARE_DICT.IM_TRAIN_EPOCHS = 8
__C.PREPARE_DICT.RT_REF = "pred"  # How to calc ref RT, one of ["pred", "exp", "mix"]
__C.PREPARE_DICT.IM_REF = (
    "ref"  # How to calc ref IM, one of ["exp", "pred", "mix", "ref", "pred_lr"]
)
__C.PREPARE_DICT.RT_TOL = (
    -0.1
)  # RT tolerance in minutes, negative means calc from data, float
__C.PREPARE_DICT.SUMMARIZE_WITHOUT_MATCH = False
__C.PREPARE_DICT.IM_LENGTH = (
    -1
)  # IM elution length in mobility index, negative means calc from data, int
__C.PREPARE_DICT.DELTA_IM_95 = (
    -0.1
)  # delta IM for 95% of the data, only used if IM_REF == "pred"
__C.PREPARE_DICT.MZ_BIN_DIGITS = 2  # legacy, not used
__C.PREPARE_DICT.ISO_MIN_AB_THRES = 0.01
__C.PREPARE_DICT.GENERATE_DECOY = False
__C.PREPARE_DICT.SAGE_RT_WINDOW = 0.0   # RT elution window (min) for SAGE; 0 = auto (1% of max RT)
__C.PREPARE_DICT.SAGE_IM_WINDOW = 0.0   # IM elution window for SAGE; 0 = auto (0.1 1/K0 units)
__C.PREPARE_DICT.RT_MAX = 0.0
__C.PREPARE_DICT.EXP_OK_DIR = ""  # path to the oktoberfest output directory for using rescoring output, then MQ evidence should be 100FDR
__C.PREPARE_DICT.EXP_OK_OUTPUT = "psms"  # one of ["psms", "peptides"], whether use ok psms or peptides table for filtering
__C.PREPARE_DICT.EXP_OK_FDR = 0.01  # FDR threshold for oktoberfest output
__C.PREPARE_DICT.REF_OK_DIR = ""  # path to the oktoberfest output directory for using rescoring output, then MQ evidence should be 100FDR
__C.PREPARE_DICT.REF_OK_OUTPUT = "psms"  # one of ["psms", "peptides"], whether use ok psms or peptides table for filtering
__C.PREPARE_DICT.REF_OK_FDR = 0.01  # FDR threshold for oktoberfest output
__C.PREPARE_DICT.PPM_TOL = 20  # ppm tolerance for matching precursors to MS1 peaks
__C.PREPARE_DICT.BIN_WIDTH = 0.01  # m/z bin
# optimization
__C.OPTIMIZATION = ConfigurationNode()
__C.OPTIMIZATION.N_BATCH = (
    -1
)  # number of batches of MS1 scans, -1 means set batches as the same as N_CPU
__C.OPTIMIZATION.DELTA_MOBILITY_INDEX_THRES = (
    80  # TODO: threshold for delta mobility, not used if extract_im_peak == False
)
__C.OPTIMIZATION.IM_PEAK_EXTRACTION_WIDTH = (
    4  # TODO: width for IM peak extraction, not used if extract_im_peak == False
)

# match features kwargs (postprocessing peak detection/matching parameters)
__C.MATCH_FEATURES_KWARGS = ConfigurationNode()
__C.MATCH_FEATURES_KWARGS.smooth_kwargs = ConfigurationNode()
__C.MATCH_FEATURES_KWARGS.smooth_kwargs.smooth_filter = "gaussian"
__C.MATCH_FEATURES_KWARGS.smooth_kwargs.gaussian_kwargs = ConfigurationNode()
__C.MATCH_FEATURES_KWARGS.smooth_kwargs.gaussian_kwargs.sigma = [2, 2]
__C.MATCH_FEATURES_KWARGS.smooth_kwargs.gaussian_kwargs.mode = "nearest"
__C.MATCH_FEATURES_KWARGS.smooth_kwargs.uniform_kwargs = ConfigurationNode()
__C.MATCH_FEATURES_KWARGS.smooth_kwargs.uniform_kwargs.size = [1, 5]
__C.MATCH_FEATURES_KWARGS.smooth_kwargs.threshold = 10
__C.MATCH_FEATURES_KWARGS.smooth_kwargs.remove_kwargs = ConfigurationNode()
__C.MATCH_FEATURES_KWARGS.smooth_kwargs.remove_kwargs.min_size = 5
__C.MATCH_FEATURES_KWARGS.peak_kwargs = ConfigurationNode()
__C.MATCH_FEATURES_KWARGS.peak_kwargs.int_threshold = 1
__C.MATCH_FEATURES_KWARGS.peak_kwargs.min_distance = 10
__C.MATCH_FEATURES_KWARGS.peak_kwargs.threshold_rel = 0.2
__C.MATCH_FEATURES_KWARGS.filter_kwargs = ConfigurationNode()
__C.MATCH_FEATURES_KWARGS.filter_kwargs.min_peak_area = 10
__C.MATCH_FEATURES_KWARGS.filter_kwargs.min_peak_sum_intensity = 500
__C.MATCH_FEATURES_KWARGS.apply_seg = True
