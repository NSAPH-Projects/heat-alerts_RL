#!/bin/bash
#SBATCH -J my_DQN
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p fasse_gpu
#SBATCH -t 0-10:00
#SBATCH --mem 50G
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
singularity shell --bind /n/dominici_nsaph_l3/Lab/projects/heat-alerts_mortality_RL/ --nv d3rlpy_latest.sif --num_gpus=4
python Run_jobs/pylauncher.py --job_file="./Run_jobs/New_DQN_tests" --i $SLURM_ARRAY_TASK_ID
