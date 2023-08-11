#!/bin/bash
#SBATCH -J Orl
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p fasse # test
#SBATCH -t 0-2:00
#SBATCH --mem 60G
#SBATCH --array 0-3
#SBATCH -o ./Run_jobs/slurm/orl.%a.out
#SBATCH --mail-user=ellen_considine@g.harvard.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#
source ~/.bashrc
mymodules
cd heat-alerts_mortality_RL
source activate heatrl
python Run_jobs/pylauncher.py --job_file="./Run_jobs/Online_tests" --i $SLURM_ARRAY_TASK_ID
