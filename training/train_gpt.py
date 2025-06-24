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
    block_size: int = 1024
    vocab_size: int = 50257  # GPT-2 vocab_size of 50257, padded to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

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
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = nn.functional.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
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
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for argument 'parameters'"
        # ignore these, they are harmless. TODO: figure out why this happens
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
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
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

class OpenWebTextDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, block_size):
        self.block_size = block_size
        
        # Load all tokenized files
        self.data_files = []
        for filename in os.listdir(data_dir):
            if filename.endswith('.npy'):
                self.data_files.append(os.path.join(data_dir, filename))
        
        print(f"Found {len(self.data_files)} tokenized files")
        
        # Load first file to get data length estimate
        first_data = np.load(self.data_files[0])
        self.tokens_per_file = len(first_data)
        self.total_tokens = len(self.data_files) * self.tokens_per_file
        self.total_samples = self.total_tokens // block_size
        
        print(f"Total tokens: {self.total_tokens}, Total samples: {self.total_samples}")
        
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        # Calculate which file and offset
        file_idx = (idx * self.block_size) // self.tokens_per_file
        token_offset = (idx * self.block_size) % self.tokens_per_file
        
        # Load the appropriate file
        data = np.load(self.data_files[file_idx])
        
        # Extract sequence
        if token_offset + self.block_size + 1 <= len(data):
            tokens = data[token_offset:token_offset + self.block_size + 1]
        else:
            # Handle edge case: wrap to next file or pad
            tokens = np.concatenate([
                data[token_offset:],
                np.zeros(self.block_size + 1 - (len(data) - token_offset), dtype=data.dtype)
            ])
        
        x = torch.from_numpy(tokens[:-1].astype(np.int64))
        y = torch.from_numpy(tokens[1:].astype(np.int64))
        
        return x, y

def setup_distributed():
    """Initialize distributed training"""
    # Get SLURM environment variables
    rank = int(os.environ.get('SLURM_PROCID', 0))
    world_size = int(os.environ.get('SLURM_NTASKS', 1))
    local_rank = int(os.environ.get('SLURM_LOCALID', 0))
    
    # Set up distributed training
    os.environ['MASTER_ADDR'] = '10.10.10.2'  # compute-1
    os.environ['MASTER_PORT'] = '12355'
    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank

def cleanup_distributed():
    """Clean up distributed training"""
    dist.destroy_process_group()

def main():
    # Training configuration
    config = GPTConfig(
        block_size=512,      # Reduced for faster training
        vocab_size=50257,
        n_layer=6,           # Smaller model for demo
        n_head=6,
        n_embd=384,
        dropout=0.1,
    )
    
    # Training hyperparameters
    batch_size = 8       # Per GPU batch size
    learning_rate = 3e-4
    max_iters = 1000     # Number of training steps
    eval_interval = 100  # Evaluate every N steps
    save_interval = 500  # Save checkpoint every N steps
    
    # Set up distributed training
    rank, world_size, local_rank = setup_distributed()
    
    # Data paths
    data_dir = "/mnt/jfs/openwebtext/tokenized"
    checkpoint_dir = "/mnt/jfs/checkpoints"
    
    # Create checkpoint directory
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create dataset and dataloader
    dataset = OpenWebTextDataset(data_dir, config.block_size)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=sampler,
        num_workers=2,
        pin_memory=True
    )
    
    # Create model
    model = GPT(config)
    model = model.cuda()
    model = DDP(model, device_ids=[local_rank])
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
    
    # Training loop
    model.train()
    total_loss = 0.0
    start_time = time.time()
    
    if rank == 0:
        print(f"Starting training on {world_size} GPUs...")
        print(f"Model parameters: {model.module.get_num_params()/1e6:.2f}M")
        print(f"Dataset size: {len(dataset)} samples")
    
    for iter_num, (x, y) in enumerate(dataloader):
        if iter_num >= max_iters:
            break
            
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        
        # Forward pass
        logits, loss = model(x, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Logging
        if rank == 0 and (iter_num + 1) % eval_interval == 0:
            avg_loss = total_loss / eval_interval
            elapsed = time.time() - start_time
            tokens_per_sec = (iter_num + 1) * batch_size * world_size * config.block_size / elapsed
            
            print(f"Step {iter_num + 1}/{max_iters} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Tokens/sec: {tokens_per_sec:.0f} | "
                  f"Time: {elapsed:.1f}s")
            
            total_loss = 0.0
        
        # Save checkpoint
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
    
    # Final checkpoint
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