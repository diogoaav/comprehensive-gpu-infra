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
    block_size: int = 512  # Match the tokenized data chunk size
    vocab_size: int = 50257  # GPT-2 vocab_size of 50257, padded to nearest multiple of 64 for efficiency
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

# ... [Keep all the model classes: LayerNorm, CausalSelfAttention, MLP, Block, GPT - same as before]

class OpenWebTextDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, block_size):
        self.block_size = block_size
        
        # Load all tokenized files (.npz format)
        self.data_files = []
        for filename in sorted(os.listdir(data_dir)):
            if filename.endswith('.npz'):
                self.data_files.append(os.path.join(data_dir, filename))
        
        print(f"Found {len(self.data_files)} tokenized files")
        
        if len(self.data_files) == 0:
            raise ValueError(f"No .npz files found in {data_dir}")
        
        # Load first file to get data structure
        first_data = np.load(self.data_files[0])
        first_input_ids = first_data['input_ids']
        
        print(f"First file shape: {first_input_ids.shape}")
        print(f"Sequence length in data: {first_input_ids.shape[1]}")
        
        # Each file contains pre-chunked sequences
        self.sequences_per_file = first_input_ids.shape[0]
        self.sequence_length = first_input_ids.shape[1]
        
        # Total number of sequences across all files
        self.total_sequences = len(self.data_files) * self.sequences_per_file
        
        print(f"Sequences per file: {self.sequences_per_file}")
        print(f"Total sequences: {self.total_sequences}")
        print(f"Total tokens: {self.total_sequences * self.sequence_length}")
        
    def __len__(self):
        return self.total_sequences
    
    def __getitem__(self, idx):
        # Calculate which file and which sequence within that file
        file_idx = idx // self.sequences_per_file
        seq_idx = idx % self.sequences_per_file
        
        # Load the appropriate file
        data = np.load(self.data_files[file_idx])
        input_ids = data['input_ids'][seq_idx]
        
        # Create input and target sequences (shift by 1 for language modeling)
        # Input: tokens[:-1], Target: tokens[1:]
        if len(input_ids) > self.block_size:
            # If sequence is longer than block_size, truncate
            x = torch.from_numpy(input_ids[:self.block_size].astype(np.int64))
            y = torch.from_numpy(input_ids[1:self.block_size+1].astype(np.int64))
        else:
            # If sequence is shorter, use available length
            x = torch.from_numpy(input_ids[:-1].astype(np.int64))
            y = torch.from_numpy(input_ids[1:].astype(np.int64))
        
        return x, y

# ... [Keep all the model classes - copy from the previous script]
# I'll include the essential ones here:

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return nn.functional.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

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
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = nn.functional.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
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
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
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
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

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
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

def setup_distributed():
    rank = int(os.environ.get('SLURM_PROCID', 0))
    world_size = int(os.environ.get('SLURM_NTASKS', 1))
    local_rank = int(os.environ.get('SLURM_LOCALID', 0))
    
    os.environ['MASTER_ADDR'] = '10.10.10.2'
    os.environ['MASTER_PORT'] = '12355'
    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank

def cleanup_distributed():
    dist.destroy_process_group()

def main():
    config = GPTConfig(
        block_size=512,      # Match the tokenized data
        vocab_size=50257,
        n_layer=6,
        n_head=6,
        n_embd=384,
        dropout=0.1,
    )
    
    batch_size = 4       # Reduced for 512 token sequences
    learning_rate = 3e-4
    max_iters = 1000
    eval_interval = 100
    save_interval = 500
    
    rank, world_size, local_rank = setup_distributed()
    
    data_dir = "/mnt/jfs/pvc-b261f13f-bf8b-4dfb-b706-0c9e4553ba3d/datasets/openwebtext_tokenized"
    checkpoint_dir = "/mnt/jfs/checkpoints"
    
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    dataset = OpenWebTextDataset(data_dir, config.block_size)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=sampler,
        num_workers=2,
        pin_memory=True
    )
    
    model = GPT(config)
    model = model.cuda()
    model = DDP(model, device_ids=[local_rank])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
    
    model.train()
    total_loss = 0.0
    start_time = time.time()
    
    if rank == 0:
        print(f"Starting training on {world_size} GPUs...")
        print(f"Model parameters: {model.module.get_num_params()/1e6:.2f}M")
        print(f"Dataset size: {len(dataset)} sequences")
    
    for iter_num, (x, y) in enumerate(dataloader):
        if iter_num >= max_iters:
            break
            
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        
        logits, loss = model(x, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if rank == 0 and (iter_num + 1) % eval_interval == 0:
            avg_loss = total_loss / eval_interval
            elapsed = time.time() - start_time
            tokens_per_sec = (iter_num + 1) * batch_size * world_size * x.size(1) / elapsed
            
            print(f"Step {iter_num + 1}/{max_iters} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Tokens/sec: {tokens_per_sec:.0f} | "
                  f"Time: {elapsed:.1f}s")
            
            total_loss = 0.0
        
        if rank == 0 and (iter_num + 1) % save_interval == 0:
            checkpoint = {
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config,
                'iter_num': iter_num + 1,
            }
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{iter_num + 1}.pt")
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    if rank == 0:
        checkpoint = {
            'model': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'config': config,
            'iter_num': max_iters,
        }
        final_path = os.path.join(checkpoint_dir, "final_model.pt")
        torch.save(checkpoint, final_path)
        print(f"Training complete! Final model saved to {final_path}")
    
    cleanup_distributed()

if __name__ == "__main__":
    main() 