#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=24:00:00
#SBATCH --output=/cmnfs/proj/ORIGINS/data/protMSD/slurm_out/%x_%j.out
#SBATCH --job-name=SBS_ims
#SBATCH --mem-per-cpu=30Gb
#SBATCH --mail-user=zixuan.xiao@tum.de
#SBATCH --mail-type=ALL
#SBATCH --partition=compms-cpu-big


config_path='/cmnfs/proj/ORIGINS/protMSD/maxquant/ScanByScan/utils/config_direct_lfq.yaml'
# evidence='/cmnfs/proj/ORIGINS/SWAPS_exp/mixture/ori_MQ_MQ_protein_quant/evidence.txt'
# evidence='/cmnfs/data/proteomics/timstof_hela_fractionation/combined/txt/evidence.txt'
# protein_group='/cmnfs/data/proteomics/timstof_hela_fractionation/Tenzer.matching_MaxQuant.txt/txt/proteinGroups.txt'

source $HOME/condaInit.sh
conda activate sbs
cd /cmnfs/proj/ORIGINS/protMSD/maxquant/ScanByScan
python -m postprocessing.direct_lfq --direct_lfq_config_path=$config_path || exit 91

echo "Script finished"
exit 0