#!/bin/bash
#SBATCH --job-name=dist_train
#SBATCH --output=/mnt/jfs/dist_train_%j.log
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

# Set up environment
export PYTHONPATH=/mnt/jfs:$PYTHONPATH

# Install required packages if not already available
srun pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Run the distributed training script
srun python3 /mnt/jfs/distributed_training.py 