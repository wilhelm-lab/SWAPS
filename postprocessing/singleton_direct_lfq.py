from yacs.config import CfgNode as ConfigurationNode


__C = ConfigurationNode()

# importing default as a global singleton
direct_lfq_config = __C

# general setup
__C.DESCRIPTION = "Default config from the Singleton"
__C.SWAPS_RESULT_DIR_LIST = []
__C.REF_RESULT_FILE = ""
__C.INTENSITY_COLUMN = "pept_act_sum_filter_by_im"
__C.RESULT_DIR = ""
__C.MQ_PROTEIN_GROUP_PATH = ""
__C.RAW_FILE_AS_EXPERIMENT = True
__C.SAMPLE_MAP = [
    ["5ug_MixB_R1", "B"],
    ["5ug_MixA_R1", "A"],
    ["5ug_MixA_R2", "A"],
    ["5ug_MixA_R3", "A"],
    ["5ug_MixA_R4", "A"],
    ["5ug_MixB_R2", "B"],
    ["5ug_MixB_R3", "B"],
    ["5ug_MixB_R4", "B"],
]
__C.CONDITIONS = ["A", "B"]
__C.FC = ConfigurationNode()
__C.FC.PLOT_ORGANISMS = ["Homo sapiens", "Saccharomyces cerevisiae", "Escherichia coli"]
__C.FC.EXPECTATION = [0, 1, -2]
__C.FC.TITLE = "SWAPS"
