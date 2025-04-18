#!/bin/bash

#SBATCH --clusters=genius
#SBATCH --partition=gpu_p100
##SBATCH --partition=gpu_p100_debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=9

#SBATCH --gpus-per-node=1
#SBATCH --time=2:00:00

scripts/moabb_erp.py "$@"
