#!/bin/bash
#SBATCH --job-name=gpt_training
#SBATCH --output=/mnt/jfs/training_logs/gpt_training_%j.log
#SBATCH --error=/mnt/jfs/training_logs/gpt_training_%j.err
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

# Create log directory
mkdir -p /mnt/jfs/training_logs
mkdir -p /mnt/jfs/checkpoints

# Set environment variables
export PYTHONPATH=/mnt/jfs:$PYTHONPATH
export CUDA_LAUNCH_BLOCKING=1

# Run the training script
srun python3 /mnt/jfs/train_gpt.py 