#!/bin/bash
#SBATCH -J Orl_short
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p shared
#SBATCH -t 0-15:00
#SBATCH --mem 6G
#SBATCH --array 0-29 #9599
#SBATCH -o ./run_jobs/slurm/orl_rebut.%a.out # orl_short
#SBATCH --mail-user=ellen_considine@g.harvard.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#
source ~/.bashrc
mymodules
cd heat-alerts_mortality_RL
conda activate heatrl
python run_jobs/pylauncher.py --job_file="./run_jobs/Online_tests_short" --i $SLURM_ARRAY_TASK_ID
