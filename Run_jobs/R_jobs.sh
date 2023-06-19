#!/bin/bash
#SBATCH -J R_model
#SBATCH -N 1
#SBATCH -n 10
#SBATCH -p fasse # fasse_gpu
#SBATCH -t 0-12:00 # 0-8:00
#SBATCH --mem 92G # 50G
#SBATCH --gres gpu:0 # gpu:1
#SBATCH --array 0-2
#SBATCH -o ./Run_jobs/slurm/r.%a.out
#SBATCH --mail-user=ellen_considine@g.harvard.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#
source ~/.bashrc
mymodules
cd heat-alerts_mortality_RL
source activate pt2.0.1_cuda11.8 # pt1.12_cuda11.6
python Run_jobs/pylauncher.py --job_file="./Run_jobs/R_tests" --i $SLURM_ARRAY_TASK_ID
