from yacs.config import CfgNode as ConfigurationNode

__C = ConfigurationNode()

# importing default as a global singleton
swaps_optimization_cfg = __C

# general setup
__C.DESCRIPTION = "Default config from the Singleton"
__C.DEBUG = False
__C.DATA_PATH = ""  # different results depending on data ending with .mzml or .d
__C.RESULT_PATH = ""  # path to save all intermediate and final results
__C.ADD_TIMESTAMP_TO_RESULT_PATH = False  # disenble when reusing the same result path
__C.EXPORT_DATA_HDF5_DIR = ""  # empty string for export to data directory
__C.MQ_EXP_PATH = ""  # path to MaxQuant experiment evidence file
__C.MQ_REF_PATH = ""  # path to MaxQuant reference file, pickle
__C.FILTER_EXP_BY_RAW_FILE = (
    []
)  # None for not filtering, "" for filtering by raw file that shares the same name with the data directory
# __C.FILTER_REF_BY_RAW_FIEL = ""  # None for not filtering, "" for filtering by raw file that shares the same name with the data directory
__C.DICT_PICKLE_PATH = ""
__C.NOTES = ""  # notes for the run
__C.USE_IMS = True
__C.RANDOM_SEED = 42  # for reproducibility when splitting train/test in DeepLC retrain
__C.N_CPU = 4  # number of CPUs for multiprocessing, <0 means using all CPUs requested by the slurm job (if any), else 0

# prepare dictionary
__C.PREPARE_DICT = ConfigurationNode()
__C.PREPARE_DICT.DICT_PICKLE_PATH = ""
__C.PREPARE_DICT.UPDATED_RT_MODEL_PATH = ""
__C.PREPARE_DICT.UPDATED_IM_MODEL_PATH = ""
__C.PREPARE_DICT.TRAIN_FRAC = 0.9
__C.PREPARE_DICT.RT_TRAIN_EPOCHS = 15
__C.PREPARE_DICT.IM_TRAIN_EPOCHS = 8
__C.PREPARE_DICT.ADD_IM_INDEX = True
__C.PREPARE_DICT.KEEP_MATCHED_PRECURSORS = False
__C.PREPARE_DICT.RT_REF = "exp"  # How to calc ref RT, one of ["pred", "exp", "mix"]
__C.PREPARE_DICT.IM_REF = (
    "ref"  # How to calc ref IM, one of ["exp", "pred", "mixs", "ref"]
)
__C.PREPARE_DICT.RT_TOL = (
    -0.1
)  # RT tolerance in minutes, negative means calc from data, float
__C.PREPARE_DICT.IM_LENGTH = (
    -1
)  # IM elution length in mobility index, negative means calc from data, int
__C.PREPARE_DICT.DELTA_IM_95 = (
    -0.1
)  # delta IM for 95% of the data, only used if IM_REF == "pred"
__C.PREPARE_DICT.FILTER_TRAIN_BY_RAW_FILE = ""  # None for not filtering, "" for filtering by raw file that shares the same name with the data directory
__C.PREPARE_DICT.FILTER_PRED_BY_RAW_FILE = ""  # None for not filtering, "" for filtering by raw file that shares the same name with the data directory
__C.PREPARE_DICT.MZ_BIN_DIGITS = 2
__C.PREPARE_DICT.ISO_MIN_AB_THRES = 0.01
__C.PREPARE_DICT.GENERATE_DECOY = False
__C.PREPARE_DICT.RT_MAX = 0.0


# optimization
__C.OPTIMIZATION = ConfigurationNode()
__C.OPTIMIZATION.N_BLOCKS_BY_PEPT = 4
__C.OPTIMIZATION.N_BATCH = (
    -1
)  # number of batches, -1 means set batches as the same as N_CPU
__C.OPTIMIZATION.HOW_BATCH = (  # method for splitting batches, from ["robin_round", "block"]
    "robin_round"
)
__C.OPTIMIZATION.DELTA_MOBILITY_INDEX_THRES = (
    80  # TODO: threshold for delta mobility, not used if extract_im_peak == False
)
__C.OPTIMIZATION.IM_PEAK_EXTRACTION_WIDTH = (
    4  # TODO: width for IM peak extraction, not used if extract_im_peak == False
)
__C.OPTIMIZATION.PEPTACT_SHAPE = [0, 0, 0]

# peak selection
__C.PEAK_SELECTION = ConfigurationNode()
__C.PEAK_SELECTION.ENABLE = False
__C.PEAK_SELECTION.DEBUG = False
__C.PEAK_SELECTION.TRAINING_DATA = []
__C.PEAK_SELECTION.EVAL_ON_TEST = True
__C.PEAK_SELECTION.INCLUDE_DECOYS = True

## DATASET
__C.PEAK_SELECTION.DATASET = ConfigurationNode()
__C.PEAK_SELECTION.DATASET.RESHAPE_METHOD = "resize"  # either resize and padding
__C.PEAK_SELECTION.DATASET.PADDING_SHAPE = (
    258,
    258,
)  # should be (multiple of 2^n_downhill)+2
__C.PEAK_SELECTION.DATASET.RESIZE_SHAPE = (192, 192)
__C.PEAK_SELECTION.DATASET.INPUT_CHANNELS = ["normal", "log", "hint"]
__C.PEAK_SELECTION.DATASET.TO_TENSOR = False  # TODO: not used
__C.PEAK_SELECTION.DATASET.MINMAX_SCALE = False
__C.PEAK_SELECTION.DATASET.ONLY_LOG_CHANNEL = False
__C.PEAK_SELECTION.DATASET.TRAIN_VAL_SIZE = 0.9
__C.PEAK_SELECTION.DATASET.TRAIN_SIZE = 0.9

__C.PEAK_SELECTION.DATASET.TRAIN_BATCH_SIZE = 256
__C.PEAK_SELECTION.DATASET.VAL_BATCH_SIZE = 512
__C.PEAK_SELECTION.DATASET.TEST_BATCH_SIZE = 512
__C.PEAK_SELECTION.DATASET.INFER_BATCH_SIZE = 256

## MODEL
__C.PEAK_SELECTION.MODEL = ConfigurationNode()
__C.PEAK_SELECTION.MODEL.TYPE = "mask_segmentation"
__C.PEAK_SELECTION.MODEL.NAME = "UNET"
__C.PEAK_SELECTION.MODEL.RESUME_PATH = ""
__C.PEAK_SELECTION.MODEL.KEEP_TRAINING = True
__C.PEAK_SELECTION.MODEL.PARAMS = ConfigurationNode()
__C.PEAK_SELECTION.MODEL.PARAMS.IN_CHANNELS = (
    -1
)  # place holder, will be updated in the code
__C.PEAK_SELECTION.MODEL.PARAMS.FIRST_OUT_CHANNELS = 32
__C.PEAK_SELECTION.MODEL.PARAMS.EXIT_CHANNELS = 1
__C.PEAK_SELECTION.MODEL.PARAMS.DOWNHILL = 5
__C.PEAK_SELECTION.MODEL.PARAMS.PADDING = 1
__C.PEAK_SELECTION.MODEL.PARAMS.IMAGE_SIZE = 258

### solver
__C.PEAK_SELECTION.MODEL.SOLVER = ConfigurationNode()
__C.PEAK_SELECTION.MODEL.SOLVER.TOTAL_EPOCHS = 100
__C.PEAK_SELECTION.MODEL.SOLVER.OPTIMIZER = ConfigurationNode()
__C.PEAK_SELECTION.MODEL.SOLVER.OPTIMIZER.BASE_LR = 0.001
__C.PEAK_SELECTION.MODEL.SOLVER.OPTIMIZER.NAME = "adam"
__C.PEAK_SELECTION.MODEL.SOLVER.OPTIMIZER.WEIGHT_DECAY = 0

#### sgd config
__C.PEAK_SELECTION.MODEL.SOLVER.OPTIMIZER.SGD = ConfigurationNode()
__C.PEAK_SELECTION.MODEL.SOLVER.OPTIMIZER.SGD.MOMENTUM = 0.9
__C.PEAK_SELECTION.MODEL.SOLVER.OPTIMIZER.SGD.NESTEROV = False

#### LR scheduler config
__C.PEAK_SELECTION.MODEL.SOLVER.SCHEDULER = ConfigurationNode()
__C.PEAK_SELECTION.MODEL.SOLVER.SCHEDULER.NAME = "reduce_on_plateau"
__C.PEAK_SELECTION.MODEL.SOLVER.SCHEDULER.PATIENCE = 3
__C.PEAK_SELECTION.MODEL.SOLVER.SCHEDULER.MIN_LR = 0.00000001
__C.PEAK_SELECTION.MODEL.SOLVER.SCHEDULER.LR_REDUCE_GAMMA = 0.1
__C.PEAK_SELECTION.MODEL.SOLVER.SCHEDULER.FACTOR = 0.1

##### OneCycleLR hyperparams
__C.PEAK_SELECTION.MODEL.SOLVER.SCHEDULER.PCT_START = 0.3
__C.PEAK_SELECTION.MODEL.SOLVER.SCHEDULER.ANNEAL_STRATEGY = "cos"
__C.PEAK_SELECTION.MODEL.SOLVER.SCHEDULER.DIV_FACTOR = 25
__C.PEAK_SELECTION.MODEL.SOLVER.SCHEDULER.MAX_LR = 0.01

#### Loss
__C.PEAK_SELECTION.MODEL.SOLVER.LOSS = ConfigurationNode()
__C.PEAK_SELECTION.MODEL.SOLVER.LOSS.NAME = "ComboLoss"
__C.PEAK_SELECTION.MODEL.SOLVER.LOSS.LOSSTYPES = ["bce", "wdice", "dice", "focal"]
__C.PEAK_SELECTION.MODEL.SOLVER.LOSS.SEG_CLS_WEIGHTS = [1, 1]
__C.PEAK_SELECTION.MODEL.SOLVER.LOSS.WEIGHTS = [1, 0, 4, 1]
__C.PEAK_SELECTION.MODEL.SOLVER.LOSS.CHANNEL_WEIGHTS = [1, 0.5]  # [normal, log, hint]
__C.PEAK_SELECTION.MODEL.SOLVER.LOSS.PER_IMAGE = True

##### focal loss related
__C.PEAK_SELECTION.MODEL.SOLVER.LOSS.FOCAL_LOSS = ConfigurationNode()
__C.PEAK_SELECTION.MODEL.SOLVER.LOSS.FOCAL_LOSS.GAMMA = 1
__C.PEAK_SELECTION.MODEL.SOLVER.LOSS.FOCAL_LOSS.ALPHA = -1

__C.PEAK_SELECTION.MODEL.SOLVER.EARLY_STOPPING = ConfigurationNode()
__C.PEAK_SELECTION.MODEL.SOLVER.EARLY_STOPPING.PATIENCE = 10
__C.PEAK_SELECTION.MODEL.SOLVER.EARLY_STOPPING.MODE = "max"

__C.PEAK_SELECTION.MODEL.EVALUATION = ConfigurationNode()
__C.PEAK_SELECTION.MODEL.EVALUATION.THRESHOLD = 0.5

## Confidence model
__C.PEAK_SELECTION.CONFMODEL = ConfigurationNode()
__C.PEAK_SELECTION.CONFMODEL.TYPE = "confidence_regression"
__C.PEAK_SELECTION.CONFMODEL.NAME = "CNNEncoderRegressor"
__C.PEAK_SELECTION.CONFMODEL.RESUME_PATH = ""
__C.PEAK_SELECTION.CONFMODEL.KEEP_TRAINING = True
__C.PEAK_SELECTION.CONFMODEL.DATASET_RESPLIT = False
__C.PEAK_SELECTION.CONFMODEL.BINARY_LABEL = False

__C.PEAK_SELECTION.CONFMODEL.PARAMS = ConfigurationNode()
__C.PEAK_SELECTION.CONFMODEL.PARAMS.IN_CHANNELS = 1
__C.PEAK_SELECTION.CONFMODEL.PARAMS.FIRST_OUT_CHANNELS = 16
__C.PEAK_SELECTION.CONFMODEL.PARAMS.DOWNHILL = 5
__C.PEAK_SELECTION.CONFMODEL.PARAMS.PADDING = 0
__C.PEAK_SELECTION.CONFMODEL.PARAMS.IMAGE_SIZE = (
    __C.PEAK_SELECTION.MODEL.PARAMS.IMAGE_SIZE
)
__C.PEAK_SELECTION.CONFMODEL.PARAMS.DROPOUT_RATE = 0.25
__C.PEAK_SELECTION.CONFMODEL.PARAMS.SIGMOID_OUTPUT = False

__C.PEAK_SELECTION.CONFMODEL.SOLVER = ConfigurationNode()
__C.PEAK_SELECTION.CONFMODEL.SOLVER.TOTAL_EPOCHS = 100

__C.PEAK_SELECTION.CONFMODEL.SOLVER.OPTIMIZER = ConfigurationNode()
__C.PEAK_SELECTION.CONFMODEL.SOLVER.OPTIMIZER.BASE_LR = 0.001
__C.PEAK_SELECTION.CONFMODEL.SOLVER.OPTIMIZER.NAME = "adam"
__C.PEAK_SELECTION.CONFMODEL.SOLVER.OPTIMIZER.WEIGHT_DECAY = 0.0

__C.PEAK_SELECTION.CONFMODEL.SOLVER.LOSS = ConfigurationNode()
__C.PEAK_SELECTION.CONFMODEL.SOLVER.LOSS.NAME = "L1Loss"

__C.PEAK_SELECTION.CONFMODEL.SOLVER.EARLY_STOPPING = ConfigurationNode()
__C.PEAK_SELECTION.CONFMODEL.SOLVER.EARLY_STOPPING.PATIENCE = 10
__C.PEAK_SELECTION.CONFMODEL.SOLVER.EARLY_STOPPING.MODE = "min"


## MODEL
__C.PEAK_SELECTION.CLSMODEL = ConfigurationNode()
__C.PEAK_SELECTION.CLSMODEL.TYPE = "mask_classification"
__C.PEAK_SELECTION.CLSMODEL.NAME = "UNET"
__C.PEAK_SELECTION.CLSMODEL.RESUME_PATH = ""
__C.PEAK_SELECTION.CLSMODEL.KEEP_TRAINING = True
__C.PEAK_SELECTION.CLSMODEL.PARAMS = ConfigurationNode()
__C.PEAK_SELECTION.CLSMODEL.PARAMS.IN_CHANNELS = (
    -1
)  # place holder, will be updated in the code
__C.PEAK_SELECTION.CLSMODEL.PARAMS.FIRST_OUT_CHANNELS = 32
__C.PEAK_SELECTION.CLSMODEL.PARAMS.EXIT_CHANNELS = 1
__C.PEAK_SELECTION.CLSMODEL.PARAMS.DOWNHILL = 5
__C.PEAK_SELECTION.CLSMODEL.PARAMS.PADDING = 1
__C.PEAK_SELECTION.CLSMODEL.PARAMS.IMAGE_SIZE = 258

### solver
__C.PEAK_SELECTION.CLSMODEL.SOLVER = ConfigurationNode()
__C.PEAK_SELECTION.CLSMODEL.SOLVER.TOTAL_EPOCHS = 100
__C.PEAK_SELECTION.CLSMODEL.SOLVER.OPTIMIZER = ConfigurationNode()
__C.PEAK_SELECTION.CLSMODEL.SOLVER.OPTIMIZER.BASE_LR = 0.001
__C.PEAK_SELECTION.CLSMODEL.SOLVER.OPTIMIZER.NAME = "adam"
__C.PEAK_SELECTION.CLSMODEL.SOLVER.OPTIMIZER.WEIGHT_DECAY = 0

#### sgd config
__C.PEAK_SELECTION.CLSMODEL.SOLVER.OPTIMIZER.SGD = ConfigurationNode()
__C.PEAK_SELECTION.CLSMODEL.SOLVER.OPTIMIZER.SGD.MOMENTUM = 0.9
__C.PEAK_SELECTION.CLSMODEL.SOLVER.OPTIMIZER.SGD.NESTEROV = False

#### LR scheduler config
__C.PEAK_SELECTION.CLSMODEL.SOLVER.SCHEDULER = ConfigurationNode()
__C.PEAK_SELECTION.CLSMODEL.SOLVER.SCHEDULER.NAME = "reduce_on_plateau"
__C.PEAK_SELECTION.CLSMODEL.SOLVER.SCHEDULER.PATIENCE = 3
__C.PEAK_SELECTION.CLSMODEL.SOLVER.SCHEDULER.MIN_LR = 0.00000001
__C.PEAK_SELECTION.CLSMODEL.SOLVER.SCHEDULER.LR_REDUCE_GAMMA = 0.1
__C.PEAK_SELECTION.CLSMODEL.SOLVER.SCHEDULER.FACTOR = 0.1

##### OneCycleLR hyperparams
__C.PEAK_SELECTION.CLSMODEL.SOLVER.SCHEDULER.PCT_START = 0.3
__C.PEAK_SELECTION.CLSMODEL.SOLVER.SCHEDULER.ANNEAL_STRATEGY = "cos"
__C.PEAK_SELECTION.CLSMODEL.SOLVER.SCHEDULER.DIV_FACTOR = 25
__C.PEAK_SELECTION.CLSMODEL.SOLVER.SCHEDULER.MAX_LR = 0.01

#### Loss
# __C.PEAK_SELECTION.CLSMODEL.SOLVER.LOSS = ConfigurationNode()
# __C.PEAK_SELECTION.CLSMODEL.SOLVER.LOSS.NAME = "ComboLoss"
# __C.PEAK_SELECTION.CLSMODEL.SOLVER.LOSS.LOSSTYPES = ["bce", "wdice", "dice", "focal"]
# __C.PEAK_SELECTION.CLSMODEL.SOLVER.LOSS.SEG_CLS_WEIGHTS = [1, 1]
# __C.PEAK_SELECTION.CLSMODEL.SOLVER.LOSS.WEIGHTS = [1, 0, 4, 1]
# __C.PEAK_SELECTION.CLSMODEL.SOLVER.LOSS.CHANNEL_WEIGHTS = [
#     1,
#     0.5,
# ]  # [normal, log, hint]
# __C.PEAK_SELECTION.CLSMODEL.SOLVER.LOSS.PER_IMAGE = True

##### focal loss related
# __C.PEAK_SELECTION.CLSMODEL.SOLVER.LOSS.FOCAL_LOSS = ConfigurationNode()
# __C.PEAK_SELECTION.CLSMODEL.SOLVER.LOSS.FOCAL_LOSS.GAMMA = 1
# __C.PEAK_SELECTION.CLSMODEL.SOLVER.LOSS.FOCAL_LOSS.ALPHA = -1

##### early stopping
__C.PEAK_SELECTION.CLSMODEL.SOLVER.EARLY_STOPPING = ConfigurationNode()
__C.PEAK_SELECTION.CLSMODEL.SOLVER.EARLY_STOPPING.PATIENCE = 10
__C.PEAK_SELECTION.CLSMODEL.SOLVER.EARLY_STOPPING.MODE = "max"

__C.PEAK_SELECTION.CLSMODEL.EVALUATION = ConfigurationNode()
__C.PEAK_SELECTION.CLSMODEL.EVALUATION.THRESHOLD = 0.5
# OneCycleLR hyperparams for conf model
__C.PEAK_SELECTION.CONFMODEL.SOLVER.SCHEDULER = ConfigurationNode()
__C.PEAK_SELECTION.CONFMODEL.SOLVER.SCHEDULER.NAME = "one_cycle"

__C.PEAK_SELECTION.CONFMODEL.SOLVER.SCHEDULER.PATIENCE = 3
__C.PEAK_SELECTION.CONFMODEL.SOLVER.SCHEDULER.MIN_LR = 0.00000001
__C.PEAK_SELECTION.CONFMODEL.SOLVER.SCHEDULER.LR_REDUCE_GAMMA = 0.1
__C.PEAK_SELECTION.CONFMODEL.SOLVER.SCHEDULER.FACTOR = 0.1

__C.PEAK_SELECTION.CONFMODEL.SOLVER.SCHEDULER.PCT_START = 0.3
__C.PEAK_SELECTION.CONFMODEL.SOLVER.SCHEDULER.ANNEAL_STRATEGY = "cos"
__C.PEAK_SELECTION.CONFMODEL.SOLVER.SCHEDULER.DIV_FACTOR = 25
__C.PEAK_SELECTION.CONFMODEL.SOLVER.SCHEDULER.MAX_LR = 0.01

__C.PEAK_SELECTION.CONFMODEL.EVAL = ConfigurationNode()
__C.PEAK_SELECTION.CONFMODEL.EVAL.METRIC = (
    "MAE"  # TODO: for compatibility with the old code
)


# result analysis
__C.RESULT_ANALYSIS = ConfigurationNode()
__C.RESULT_ANALYSIS.ENABLE = False
# __C.RESULT_ANALYSIS.MQ_EXP_PATH = ""
# __C.RESULT_ANALYSIS.FILTER_BY_RAW_FILE = ""  # None for not filtering, "" for filtering by raw file that shares the same name with the data directory
__C.RESULT_ANALYSIS.FILTER_BY_RT_OVERLAP = [
    "full_overlap"
]  # how to filter experiment result according to rt overlap, choices ["full_overlap", "partial_overlap", "no_overlap"]
__C.RESULT_ANALYSIS.FDR_THRESHOLD = 0.1
__C.RESULT_ANALYSIS.POST_PROCESSING = ConfigurationNode()
__C.RESULT_ANALYSIS.POST_PROCESSING.FILTER_BY_IM = False
