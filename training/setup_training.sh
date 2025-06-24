#!/bin/bash

# Copy training files to shared storage
echo "Copying training files to JuiceFS..."
cp training/train_gpt.py /mnt/jfs/
cp training/train_job.sh /mnt/jfs/
cp training/inference.py /mnt/jfs/

# Install additional dependencies on all compute nodes
echo "Installing training dependencies..."
ssh root@10.10.10.2 "pip3 install tiktoken"
ssh root@10.10.10.5 "pip3 install tiktoken"
ssh root@10.10.10.4 "pip3 install tiktoken"

# Create necessary directories
mkdir -p /mnt/jfs/training_logs
mkdir -p /mnt/jfs/checkpoints

echo "Setup complete! You can now submit the training job:"
echo "sbatch /mnt/jfs/train_job.sh" 