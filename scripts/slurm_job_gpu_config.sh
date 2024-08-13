#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --gres=shard:6
#SBATCH --time=24:00:00
#SBATCH --output=/cmnfs/proj/ORIGINS/data/protMSD/slurm_out/%x_%j.out
#SBATCH --job-name=2d_PS_train
#SBATCH --mem-per-cpu=32Gb
#SBATCH --mail-user=zixuan.xiao@tum.de
#SBATCH --mail-type=ALL
#SBATCH --partition=compms-gpu-a40



source $HOME/condaInit.sh
conda activate sbs
cd /cmnfs/proj/ORIGINS/protMSD/maxquant/ScanByScan
python -m peak_detection_2d.train --path_output=exp01 --path_cfg=/cmnfs/proj/ORIGINS/protMSD/maxquant/ScanByScan/peak_detection_2d/config/exp/exp1.yaml || exit 91

echo "Script finished"
exit 0