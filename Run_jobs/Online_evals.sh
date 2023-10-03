#!/bin/bash
#SBATCH -J Orl_short
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p fasse 
#SBATCH -t 0-0:30 
#SBATCH --mem 3G 
#SBATCH --array 0-3599 # 1319 
#SBATCH -o ./Run_jobs/slurm/orl_short.%a.out
#SBATCH --mail-user=ellen_considine@g.harvard.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#
source ~/.bashrc
mymodules
cd heat-alerts_mortality_RL
source activate heatrl
python Run_jobs/pylauncher.py --job_file="./Run_jobs/Eval_jobs" --i $SLURM_ARRAY_TASK_ID
