#!/bin/bash
#SBATCH --job-name=test_dist
#SBATCH --output=/mnt/jfs/training_logs/test_dist_%j.log
#SBATCH --error=/mnt/jfs/training_logs/test_dist_%j.err
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00

export NCCL_DEBUG=INFO

srun python3 /mnt/jfs/test_distributed.py 