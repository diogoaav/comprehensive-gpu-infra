import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '10.10.10.2'  # Use compute-1's private IP
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def run_training(rank, world_size):
    print(f"Running training on rank {rank} of {world_size}")
    setup(rank, world_size)
    
    # Each node has GPU 0, so always use device 0 locally
    local_gpu_id = 0
    torch.cuda.set_device(local_gpu_id)
    
    # Create model and move it to the local GPU
    model = torch.nn.Linear(10, 10).cuda()
    ddp_model = DistributedDataParallel(model, device_ids=[local_gpu_id])
    
    # Training loop would go here
    print(f"Node: {os.uname()[1]}, Rank: {rank}, Local GPU: {local_gpu_id}, Model device: {next(ddp_model.parameters()).device}")
    
    cleanup()

if __name__ == "__main__":
    # Get SLURM variables
    node_id = int(os.environ.get('SLURM_NODEID', 0))
    local_rank = int(os.environ.get('SLURM_LOCALID', 0))
    world_size = int(os.environ.get('SLURM_NTASKS', 1))
    rank = int(os.environ.get('SLURM_PROCID', 0))
    
    print(f"Node: {os.uname()[1]}, NodeID: {node_id}, LocalRank: {local_rank}, Rank: {rank}, WorldSize: {world_size}")
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA current device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available")
        exit(1)
    
    run_training(rank, world_size) 