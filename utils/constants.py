from typing import Literal

# relevant columns in exp
_alpha_opt_metric = Literal["cos_dist", "RMSE"]
_loss = Literal["lasso", "sqrt_lasso"]
_algo = Literal["lasso_lars", "lasso_cd", "lars", "omp", "threshold"]
_alpha_criteria = Literal["min", "convergence"]
_pp_method = Literal["raw", "sqrt"]
_AlignMethods = Literal["2stepNN", "peakRange"]
exp_cols = [
    "Sequence",
    "Modified sequence",
    "Charge",
    "Intensity",
    "Type",
    "Raw file",
    "Retention time",
    "Calibrated retention time",
    "Retention length",
    "Calibrated retention time start",
    "Calibrated retention time finish",
    "Ion mobility length",
    "1/K0",
    "1/K0 length",
    "Calibrated 1/K0",
    "Reverse",
    "MS/MS scan number",
    "Proteins",
    "m/z",
    "PEP",
    "Score",
]
decoy_mutation_rule = {
    "G": "L",  #
    "A": "L",
    "V": "L",
    "L": "V",
    "I": "V",
    "F": "L",
    "M": "L",
    "P": "L",
    "W": "L",
    "S": "T",
    "C": "S",
    "T": "S",
    "Y": "S",
    "H": "S",
    "K": "L",
    "R": "L",
    "Q": "N",
    "E": "D",
    "N": "Q",
    "D": "E",
}
# G and K has same weight
# I and L has same weight
