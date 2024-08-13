#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --output=/cmnfs/proj/ORIGINS/data/protMSD/slurm_out/%x_%j.out
#SBATCH --job-name=prepare_dataset
#SBATCH --mem-per-cpu=256Gb
#SBATCH --mail-user=zixuan.xiao@tum.de
#SBATCH --mail-type=ALL
#SBATCH --partition=compms-cpu-small



source $HOME/condaInit.sh
conda activate sbs
python /cmnfs/proj/ORIGINS/protMSD/maxquant/ScanByScan/ims_3d_prepare_peak_detection_mask_input.py || exit 91

echo "Script finished"
exit 0