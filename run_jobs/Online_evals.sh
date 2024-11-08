#!/bin/bash
#SBATCH -J Evals
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p shared # fasse 
#SBATCH -t 0-0:30 
#SBATCH --mem 3G 
#SBATCH --array 0-29 # 4967 # 9999
#SBATCH -o ./run_jobs/slurm/evals.%a.out
#SBATCH --mail-user=ellen_considine@g.harvard.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#
source ~/.bashrc
mymodules
cd heat-alerts_mortality_RL
source activate heatrl
python run_jobs/pylauncher.py --job_file="./run_jobs/Eval_jobs" --i $SLURM_ARRAY_TASK_ID
