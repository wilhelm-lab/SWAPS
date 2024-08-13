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
#SBATCH --array=0-1


source $HOME/condaInit.sh
conda activate sbs
cd /cmnfs/proj/ORIGINS/protMSD/maxquant/ScanByScan

# Generate the list of YAML files and save it to a temporary file
JOBLIB_FILE="joblib_$SLURM_JOB_ID.txt"
find /cmnfs/proj/ORIGINS/protMSD/maxquant/ScanByScan/peak_detection_2d/config/exp_hint_channel -name "*.yaml" > $JOBLIB_FILE

# Get the total number of YAML files
TOTAL_FILES=$(wc -l < $JOBLIB_FILE)

# Adjust the job array range dynamically
#SBATCH --array=0-$((TOTAL_FILES - 1))

# Get the path_cfg for the current task
CONFIG_PATH=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $JOBLIB_FILE)

# Run the Python script with the specified arguments
python -m peak_detection_2d.train --path_output=auto --path_cfg="$CONFIG_PATH" || exit 91

echo "Script finished"
exit 0