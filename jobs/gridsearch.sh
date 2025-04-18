#!/bin/bash

mkdir -p $2

#for theta in 0.0 1.0; do
#for theta in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1; do
for theta in 0 0.010 0.025 0.050 0.075 0.1 0.25 0.75 0.5 0.75 1; do
	#for subject in 1 2; do
	for subject in 1 2 3 4 5 6 7 8; do
		sbatch jobs/gridsearch.slurm  $theta $subject $1 $2
	done
done

