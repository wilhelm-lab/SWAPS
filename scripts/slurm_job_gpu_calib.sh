#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --gres=shard:8
#SBATCH --time=24:00:00
#SBATCH --output=/cmnfs/proj/ORIGINS/data/protMSD/slurm_out/%x_%j.out
#SBATCH --job-name=2d_PS_train
#SBATCH --mem-per-cpu=64Gb
#SBATCH --mail-user=zixuan.xiao@tum.de
#SBATCH --mail-type=ALL
#SBATCH --partition=compms-gpu-a40



source $HOME/condaInit.sh
conda activate sbs
cd /cmnfs/proj/ORIGINS/protMSD/maxquant/ScanByScan
python -m peak_detection_2d.calibrate --model_dir=/cmnfs/proj/ORIGINS/data/brain/FreshFrozenBrain/SingleShot/DDA/frame0_1830_ssDDA_P064428_Fresh1_5ug_R1_BD5_1_4921_ScanByScan_RTtol0.9_threshold_missabthres0.5_convergence_NoIntercept_pred_mzBinDigits2_imPeakWidth4_deltaMobilityThres80/2d_peak_selection/exp_one_channel_exp_normal_lr05_bce1_out32/ --model_name=bst_model_0.7116.pt || exit 91

echo "Script finished"
exit 0