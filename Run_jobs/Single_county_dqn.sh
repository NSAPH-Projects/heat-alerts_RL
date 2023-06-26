#!/bin/bash
#SBATCH -J SCrl
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p fasse # test
#SBATCH -t 0-1:00
#SBATCH --mem 15G
#SBATCH --array 0-95 # 0-17 
#SBATCH -o ./Run_jobs/slurm/scrl.%a.out
#SBATCH --mail-user=ellen_considine@g.harvard.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#
source ~/.bashrc
mymodules
cd heat-alerts_mortality_RL
source activate pt2.0.1_cuda11.8
python Run_jobs/pylauncher.py --job_file="./Run_jobs/Single_county_dqn_tests" --i $SLURM_ARRAY_TASK_ID
