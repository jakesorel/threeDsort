#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=0:50:00   # walltime
#SBATCH -J "3d_simul"   # job name
#SBATCH --output=bash_out/output.out
#SBATCH --error=bash_out/error.out
#SBATCH -n 1
#SBATCH --partition=cpu
#SBATCH --mem=2G


eval "$(conda shell.bash hook)"
source activate synmorph

python run_cluster_bootstrap.py ${SLURM_ARRAY_TASK_ID} "$1"
#python run_cluster_bootstrap.py "$1" "$2"

#
#((jp1 = "$1" + 1))
#
#if [[ "$1" -lt "$2" ]]
#then
#    sbatch run_simulation.sh "$jp1" "$2"
#fi
