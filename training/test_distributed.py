import os
import torch
import torch.distributed as dist

def test_distributed():
    # Get SLURM environment
    rank = int(os.environ.get('SLURM_PROCID', 0))
    world_size = int(os.environ.get('SLURM_NTASKS', 1)) 
    local_rank = int(os.environ.get('SLURM_LOCALID', 0))
    
    print(f"Rank {rank}: Starting distributed test")
    print(f"Rank {rank}: World size = {world_size}")
    print(f"Rank {rank}: Local rank = {local_rank}")
    
    if world_size > 1:
        os.environ['MASTER_ADDR'] = '10.10.10.2'
        os.environ['MASTER_PORT'] = '29501'  # Different port
        
        print(f"Rank {rank}: Initializing process group...")
        
        try:
            dist.init_process_group(
                backend='nccl',
                rank=rank, 
                world_size=world_size,
                timeout=torch.distributed.timedelta(seconds=30)  # 30 second timeout
            )
            print(f"Rank {rank}: Process group initialized successfully!")
            
            # Test a simple all-reduce
            tensor = torch.ones(1).cuda() * rank
            print(f"Rank {rank}: Before all-reduce: {tensor.item()}")
            
            dist.all_reduce(tensor)
            print(f"Rank {rank}: After all-reduce: {tensor.item()}")
            
            dist.destroy_process_group()
            print(f"Rank {rank}: Distributed test completed successfully!")
            
        except Exception as e:
            print(f"Rank {rank}: Error in distributed setup: {e}")
    else:
        print(f"Rank {rank}: Single node mode")

if __name__ == "__main__":
    test_distributed() 