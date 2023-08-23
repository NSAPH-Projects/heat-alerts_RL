#!/bin/bash
#SBATCH -J Orl_short
#SBATCH -N 1
#SBATCH -n 2 # 1
#SBATCH -p fasse # test
#SBATCH -t 0-4:00
#SBATCH --mem 12G # 6G
#SBATCH --array 0-11
#SBATCH -o ./Run_jobs/slurm/orl_short.%a.out
#SBATCH --mail-user=ellen_considine@g.harvard.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#
source ~/.bashrc
mymodules
cd heat-alerts_mortality_RL
source activate heatrl
python Run_jobs/pylauncher.py --job_file="./Run_jobs/Online_tests_short" --i $SLURM_ARRAY_TASK_ID
