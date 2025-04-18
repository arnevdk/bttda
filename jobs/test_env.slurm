#!/bin/bash

#SBATCH --clusters=genius
#SBATCH --partition=gpu_p100_debug
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=9
#SBATCH --gpus-per-node=1

# Print environment info
conda info --envs

echo
printenv | grep SLURM
echo
printenv | grep CONDA
echo
printenv | grep TENSORLY
echo
printenv | grep CUDA
echo
printenv | grep CUPY
echo
echo $PATH
echo
nvidia-smi
 
scripts/test_env.py
