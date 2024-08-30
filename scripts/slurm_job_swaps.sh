#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=24:00:00
#SBATCH --output=/cmnfs/proj/ORIGINS/SWAPS_exp/slurm_out/%x_%j.out
#SBATCH --job-name=swaps_ims
#SBATCH --mem=200G
#SBATCH --mail-user=zixuan.xiao@tum.de
#SBATCH --mail-type=ALL
#SBATCH --partition=compms-cpu-big

config_path='/cmnfs/proj/ORIGINS/protMSD/maxquant/ScanByScan/utils/config_hela_120min_frac_100ms.yaml'
#config_path='/cmnfs/proj/ORIGINS/protMSD/maxquant/ScanByScan/utils/config_test_construct_dict_opt.yaml'
#config_path='/cmnfs/proj/ORIGINS/SWAPS_exp/tims_ramp_time/corrected_RT_tol_pred_120min_library_80ms_with_decoy_pred_20240806_090347_182738/config_resize.yaml'
source $HOME/condaInit.sh
conda activate sbs
python /cmnfs/proj/ORIGINS/protMSD/maxquant/ScanByScan/sbs_runner_ims.py \
--config_path=$config_path || exit 91

echo "Script finished"
exit 0