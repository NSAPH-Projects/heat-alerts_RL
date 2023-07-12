#!/bin/bash
#SBATCH -J my_DQN
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p fasse_gpu
#SBATCH -t 0-50:00
#SBATCH --mem 35G
#SBATCH --gres gpu:1
#SBATCH --array 0 # 0-1
#SBATCH -o ./Run_jobs/slurm/dqn.%a.out
#SBATCH --mail-user=ellen_considine@g.harvard.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#
source ~/.bashrc
mymodules
cd heat-alerts_mortality_RL
source activate pt1.12_cuda11.6
python Run_jobs/pylauncher.py --job_file="./Run_jobs/Old_DQN_tests" --i $SLURM_ARRAY_TASK_ID
