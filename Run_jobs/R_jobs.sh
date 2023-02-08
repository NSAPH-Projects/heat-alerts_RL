#!/bin/bash
#SBATCH -J R_model
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p fasse_gpu
#SBATCH -t 0-10:00
#SBATCH --mem 50G
#SBATCH --gres gpu:1
#SBATCH --array 0-1 # 0-2
#SBATCH -o ./Run_jobs/slurm/r.%a.out
#SBATCH --mail-user=ellen_considine@g.harvard.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#
source ~/.bashrc
mymodules
cd heat-alerts_mortality_RL
source activate pt1.12_cuda11.6
python Run_jobs/pylauncher.py --job_file="./Run_jobs/R_tests" --i $SLURM_ARRAY_TASK_ID
