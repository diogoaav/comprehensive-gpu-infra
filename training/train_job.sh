#!/bin/bash
#SBATCH --job-name=gpt_train_fixed
#SBATCH --output=/mnt/jfs/training_logs/gpt_training_fixed_%j.log
#SBATCH --error=/mnt/jfs/training_logs/gpt_training_fixed_%j.err
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

# Create log directory
mkdir -p /mnt/jfs/training_logs

# Set environment variables for better debugging
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=INFO

echo "Job started at $(date)"
echo "Running on nodes: $SLURM_JOB_NODELIST"
echo "Number of tasks: $SLURM_NTASKS"

# Run the fixed training script
srun python3 /mnt/jfs/train_gpt_fixed.py

echo "Job completed at $(date)" 