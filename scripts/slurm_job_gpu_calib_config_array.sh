#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --gres=shard:6
#SBATCH --time=24:00:00
#SBATCH --output=/cmnfs/proj/ORIGINS/data/protMSD/slurm_out/%x_%j-%a.out
#SBATCH --job-name=2d_PS_cfg_train
#SBATCH --mem-per-cpu=32Gb
#SBATCH --mail-user=zixuan.xiao@tum.de
#SBATCH --mail-type=ALL
#SBATCH --partition=shared-gpu
#SBATCH --array=0


source $HOME/condaInit.sh
conda activate sbs
cd /cmnfs/proj/ORIGINS/protMSD/maxquant/ScanByScan

# Generate the list of YAML files and save it to a temporary file
JOBLIB_FILE="joblib_$SLURM_JOB_ID.txt"
find /cmnfs/proj/ORIGINS/protMSD/maxquant/ScanByScan/peak_detection_2d/config/exp_calib -name "*cls.yaml" > $JOBLIB_FILE

# Get the total number of YAML files
TOTAL_FILES=$(wc -l < $JOBLIB_FILE)

# Adjust the job array range dynamically
#SBATCH --array=0-$((TOTAL_FILES - 1))

# Get the path_cfg for the current task
CONFIG_PATH=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $JOBLIB_FILE)

# Run the Python script with the specified arguments
python -m peak_detection_2d.train_calib_model --best_seg_model_path=/cmnfs/proj/ORIGINS/data/brain/FreshFrozenBrain/SingleShot/DDA/frame0_1830_ssDDA_P064428_Fresh1_5ug_R1_BD5_1_4921_ScanByScan_RTtol0.9_threshold_missabthres0.5_convergence_NoIntercept_pred_mzBinDigits2_imPeakWidth4_deltaMobilityThres80/2d_peak_selection/exp_wdice_per_image_combo_two_channel_exp_log_lr001_wdice1_dh5/model_backups/bst_model_0.7765.pt --path_cfg="$CONFIG_PATH" || exit 91

echo "Script finished"
exit 0