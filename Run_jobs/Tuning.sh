#!/bin/bash
#SBATCH -J Orl_tune
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p shared # fasse # test
#SBATCH -t 0-10:00 # 0-7:00 
#SBATCH --mem 6G
#SBATCH --array 0-3644 # 7289
#SBATCH -o ./Run_jobs/slurm/orl_tune.%a.out
#SBATCH --mail-user=ellen_considine@g.harvard.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#
source ~/.bashrc
mymodules
cd heat-alerts_mortality_RL
source activate heatrl
python Run_jobs/pylauncher.py --job_file="./Run_jobs/Online_tuning" --i $SLURM_ARRAY_TASK_ID
