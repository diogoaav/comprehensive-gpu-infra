#!/bin/bash
#SBATCH --job-name=simple_dist_test
#SBATCH --output=/mnt/jfs/simple_dist_%j.log
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1

srun bash -c 'echo "Node: $(hostname), SLURM_PROCID: $SLURM_PROCID, SLURM_NODEID: $SLURM_NODEID, GPU: $(nvidia-smi -L)"' 