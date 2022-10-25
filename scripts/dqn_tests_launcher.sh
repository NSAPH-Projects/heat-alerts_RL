#!/bin/bash

#SBATCH -N 1
#SBATCH -n 8
#SBATCH -p fasse_gpu
#SBATCH -t 11:59:00
#SBATCH --mem 32G
#SBATCH --gres gpu:1
#SBATCH --array 0-3
#SBATCH -o ./slurm/dqn.%a.out

#
source ~/.bashrc
conda activate cuda116
python scripts/pylauncher.py --job_file="./scripts/dqn_tests" --i $SLURM_ARRAY_TASK_ID
