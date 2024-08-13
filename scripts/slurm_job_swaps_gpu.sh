#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/cmnfs/proj/ORIGINS/SWAPS_exp/slurm_out/%x_%j.out
#SBATCH --job-name=swaps_PS
#SBATCH --mem=150G
#SBATCH --mail-user=zixuan.xiao@tum.de
#SBATCH --mail-type=ALL
#SBATCH --partition=shared-gpu
#SBATCH --nodelist=gpu02.exbio.wzw.tum.de


#config_path='/cmnfs/proj/ORIGINS/protMSD/maxquant/ScanByScan/utils/config_exp_160ms.yaml'
config_path='/cmnfs/proj/ORIGINS/SWAPS_exp/tims_ramp_time/corrected_RT_tol_pred_exp_library_160ms_with_decoy_pred_20240812_160156_311148/config_20240812_160156_311148.yaml'
source $HOME/condaInit.sh
conda activate sbs
python /cmnfs/proj/ORIGINS/protMSD/maxquant/ScanByScan/sbs_runner_ims.py --config_path=$config_path || exit 91

echo "Script finished"
exit 0