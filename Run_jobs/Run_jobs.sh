#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p fasse_gpu
#SBATCH -t 1-00:00
#SBATCH --mem 32G
#SBATCH --gres gpu:1
#SBATCH --array 0-1
#SBATCH -o ./slurm/dqn.%a.out

#
source ~/.bashrc
mymodules
source activate pt1.12_cuda11.6
python scripts/pylauncher.py --job_file="./scripts/DQN_tests" --i $SLURM_ARRAY_TASK_ID
