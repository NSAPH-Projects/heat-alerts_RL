#!/bin/bash
#SBATCH -J MA_NN
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p fasse # test
#SBATCH -t 0-10:00
#SBATCH --mem 90G
#SBATCH --array 0 # 0-17 # 0-2
#SBATCH -o ./Run_jobs/slurm/ma_nn.%a.out
#SBATCH --mail-user=ellen_considine@g.harvard.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#
source ~/.bashrc
mymodules
cd heat-alerts_mortality_RL
source activate pt2.0.1_cuda11.8
python Run_jobs/pylauncher.py --job_file="./Run_jobs/MA_NNs_policy" --i $SLURM_ARRAY_TASK_ID
# singularity exec d3rlpy_latest.sif python Run_jobs/pylauncher.py --job_file="./Run_jobs/MA_NNs_policy" --i $SLURM_ARRAY_TASK_ID
