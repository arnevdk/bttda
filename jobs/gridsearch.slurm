#!/bin/bash

##SBATCH --clusters=genius
##SBATCH --partition=gpu_p100
##SBATCH --ntasks-per-node=9

#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100,gpu_h100
#SBATCH --ntasks-per-node=18

#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=0:30:00

scripts/gridsearch.py "$@"
