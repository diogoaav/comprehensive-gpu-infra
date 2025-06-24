# GPT Training on OpenWebText

This directory contains a complete implementation for training a GPT model on the tokenized OpenWebText dataset using distributed training across multiple GPUs.

## Files

- `train_gpt.py`: Main training script with GPT model implementation
- `train_job.sh`: SLURM job submission script
- `inference.py`: Script for generating text with trained models
- `setup_training.sh`: Setup script to prepare the training environment

## Model Architecture

The implementation includes:
- Transformer architecture similar to GPT-2
- Causal self-attention with Flash Attention support
- Layer normalization and MLP blocks
- Configurable model size (layers, heads, embedding dimensions)

## Usage

1. **Setup the training environment:**
   ```bash
   bash training/setup_training.sh
   ```

2. **Submit the training job:**
   ```bash
   sbatch /mnt/jfs/train_job.sh
   ```

3. **Monitor training:**
   ```bash
   # Check job status
   squeue
   
   # View training logs
   tail -f /mnt/jfs/training_logs/gpt_training_*.log
   ```

4. **Generate text with trained model:**
   ```bash
   python3 /mnt/jfs/inference.py
   ```

## Configuration

The model is configured for demonstration with:
- 6 layers, 6 attention heads, 384 embedding dimensions (~25M parameters)
- 512 token context length
- 1000 training steps
- Batch size of 8 per GPU (24 total across 3 GPUs)

You can modify these parameters in `train_gpt.py` for larger models or longer training.

## Checkpoints

- Training checkpoints are saved to `/mnt/jfs/checkpoints/`
- Checkpoints include model weights, optimizer state, and configuration
- Final model is saved as `final_model.pt`

## Distributed Training

The script uses PyTorch DistributedDataParallel with:
- NCCL backend for efficient GPU communication
- Automatic data sharding across GPUs
- Synchronized gradient updates 