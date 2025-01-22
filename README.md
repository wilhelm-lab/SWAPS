# Scan-Wise Activation and Peak Selection
## Introduction
**S**can-**W**ise **A**ctivation and **P**eak **Se**lection (SWAPS) is a novel modular MS1-centric PIP framework designed to elevate peptide identification and quantification by leveraging the utilization of MS1-data at all available dimensions, state-of-the-art peptide-prediction models, and an innovative deep-learning-based method for feature quality control (peak selection and confidence scoring)
![HUPO_abstract](https://github.com/user-attachments/assets/8ba72094-8eb5-4f90-a914-50cdd0c4688b)

## Usage
### Installation
SWAPS requires `Python 3.10`. To have a clean environment, first create a empty environment with the required python version. This can be down for example via conda:
```
conda create --name swaps python==3.10.12 --no-default-packages
```
enter the environment:
```
conda activate swaps
```
Then pip install swaps:
```
pip install git+https://github.com/wilhelm-lab/SWAPS.git
```
### Run SWAPS
#### Configuration
SWAPS takes a `.yaml` as a configuration argument. Examples can be found at `swaps/utils/exp_configs`. Please modify the file path as necessary. Detail documents of config definiation can be found at `swaps/utils/singleton_swaps_optimization.py`.

#### Runner
To run SWAPS, in the created environment, in command line:
```
sbs_runner_ims [path-to-config-file]
```

### SWAPS results structure
- `[RESULT_PATH]/`
  - `config_[TIMESTAMP].yaml` (config file copy for reproducibility)
  - `ms1scans.csv`
  - `mobility_values.csv` 
  - `contruct_dict/` (intermediate results during dictionary construction)
    - `RT_transfer_learn/`
      - ...
    - `IM_transfer_learn/`
      - ...
    - `BarChart_candidate_overlap.png`
    - `BarChart_exp_elution_counts.png`
    - ...
  - `maxquant_result_ref.pkl`  (SWAPS dictionary)

  - `peak_selection/` (results from peak selection, only exists if peak_selection is enabled)
    - `training_data/`
      - ...(sparse matrix (for hint channel) and annotated data from MS2 identification)
    - `exp_[TIMESTAMP]/`
      - `updated_peak_selection_config.yaml`
      - `logs_tensorflow/`
        - ... (for `tensorboard` visualization)
      - `model_backups/`
        - ... (weights of segmentation and scoring models)  
      - `results/`
        - `evaluation/`
          - ... (result on full dataset)
        - ... (other evaluation results on testset)
        - `loss.json`
        - `metric.json`
      - `pept_act_sum_ps.csv` (all candidate results (target+decoy) inferred intensity after peak selection)
      - `pept_act_sum_ps_full_fdr_thres.csv` (all candidate results (target+decoy) with FDR threshold
      - `pept_act_sum_ps_full_tdc_fdr_thres.csv` (candidates after target-decoy competition)
  

  - `results/`
    - `activation/`
      - `sparse matrix in batch and peptbatch.npz`
      - `pept_act_sum.csv`
      - `pept_act_sum_filter_by_im.csv` (peptide activation sum after filter by ion mobility, only exists if `__C.RESULT_ANALYSIS.POST_PROCESSING.FILTER_BY_IM==True`)
    - `evaluation/` (only exists if peak selection is disabled, evaluation compared to reference (MQ))
      - `CorrQuant.png`
      - `VennDiag.png`

---
For questions, please contact *zixuan.xiao@tum.de*

