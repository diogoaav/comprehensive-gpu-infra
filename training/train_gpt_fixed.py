import os
import math
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class GPTConfig:
    block_size: int = 512  # Match your data
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(torch.tril(torch.ones(T, T, device=x.device)) == 0, float('-inf'))
        att = torch.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

class OpenWebTextDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, block_size):
        self.block_size = block_size
        self.data_files = []
        
        for filename in os.listdir(data_dir):
            if filename.endswith('.npz'):
                self.data_files.append(os.path.join(data_dir, filename))
        
        self.data_files.sort()
        print(f"Found {len(self.data_files)} tokenized files")
        
        if len(self.data_files) == 0:
            raise ValueError(f"No .npz files found in {data_dir}")
        
        # Load first file to get info
        first_data = np.load(self.data_files[0])
        first_input_ids = first_data['input_ids']
        print(f"First file shape: {first_input_ids.shape}")
        print(f"Sequence length in data: {first_input_ids.shape[1]}")
        
        self.sequences_per_file = first_input_ids.shape[0]
        self.total_sequences = len(self.data_files) * self.sequences_per_file
        print(f"Sequences per file: {self.sequences_per_file}")
        print(f"Total sequences: {self.total_sequences}")
        print(f"Total tokens: {self.total_sequences * first_input_ids.shape[1]}")

    def __len__(self):
        return self.total_sequences

    def __getitem__(self, idx):
        file_idx = idx // self.sequences_per_file
        seq_idx = idx % self.sequences_per_file
        
        data = np.load(self.data_files[file_idx])
        input_ids = data['input_ids'][seq_idx]
        
        # Convert to tensor and create targets (shifted by 1)
        x = torch.from_numpy(input_ids[:-1]).long()
        y = torch.from_numpy(input_ids[1:]).long()
        
        return x, y

def setup_distributed():
    """Initialize distributed training"""
    # Get SLURM environment variables
    rank = int(os.environ.get('SLURM_PROCID', 0))
    world_size = int(os.environ.get('SLURM_NTASKS', 1))
    local_rank = int(os.environ.get('SLURM_LOCALID', 0))
    
    print(f"Rank: {rank}, World size: {world_size}, Local rank: {local_rank}")
    
    # Set up environment for distributed training
    if world_size > 1:
        # Use the first compute node as master
        os.environ['MASTER_ADDR'] = '10.10.10.2'  # compute-1 private IP
        os.environ['MASTER_PORT'] = '29500'  # Different port to avoid conflicts
        
        print(f"Initializing distributed training...")
        print(f"Master: {os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}")
        
        dist.init_process_group(
            backend='nccl',
            rank=rank,
            world_size=world_size,
            timeout=torch.distributed.default_pg_timeout
        )
        
        # Set device
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        
        print(f"Rank {rank}: Distributed training initialized successfully")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    return rank, world_size, device

def main():
    # Setup distributed training
    rank, world_size, device = setup_distributed()
    
    print(f"Starting training on {world_size} GPUs...")
    
    # Configuration
    config = GPTConfig()
    
    # Create model
    model = GPT(config)
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params/1e6:.2f}M")
    
    # Wrap model for distributed training
    if world_size > 1:
        model = DDP(model, device_ids=[device.index])
    
    # Create dataset
    data_dir = "/mnt/jfs/pvc-b261f13f-bf8b-4dfb-b706-0c9e4553ba3d/datasets/openwebtext_tokenized"
    dataset = OpenWebTextDataset(data_dir, config.block_size)
    print(f"Dataset size: {len(dataset)} sequences")
    
    # Create data loader with distributed sampler
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank) if world_size > 1 else None
    dataloader = DataLoader(
        dataset,
        batch_size=8,  # Small batch size to start
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=2,
        pin_memory=True
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Training loop
    model.train()
    step = 0
    
    print("Starting training loop...")
    
    for epoch in range(2):  # Just 2 epochs for testing
        if sampler:
            sampler.set_epoch(epoch)
            
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits, loss = model(x, y)
            loss.backward()
            optimizer.step()
            
            step += 1
            
            if step % 10 == 0:
                print(f"Rank {rank}, Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")
            
            if step >= 100:  # Stop after 100 steps for testing
                print(f"Rank {rank}: Completed 100 training steps")
                break
                
        if step >= 100:
            break
    
    print(f"Rank {rank}: Training completed successfully!")
    
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main() 