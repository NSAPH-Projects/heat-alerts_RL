#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p fasse_gpu
#SBATCH -t 0-12:00
#SBATCH --mem 32G
#SBATCH --gres gpu:1
#SBATCH --array 0-1
#SBATCH -o ./slurm/dqn.%a.out
#
source ~/.bashrc



source activate pt1.12_cuda11.6
