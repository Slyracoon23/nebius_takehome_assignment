#!/usr/bin/env python3
import os
import time
import argparse
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.cuda.amp as amp
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

# Simple synthetic text dataset
class SyntheticTextDataset(Dataset):
    def __init__(self, size=10000, seq_length=128, vocab_size=50257):
        self.size = size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        # Generate random token indices (between 0 and vocab_size-1)
        input_ids = torch.randint(0, self.vocab_size, (self.seq_length,))
        labels = input_ids.clone()  # Use same sequence for labels in language modeling
        return {"input_ids": input_ids, "labels": labels}

def get_gpu_memory_usage():
    """Get GPU memory usage in GB"""
    if not torch.cuda.is_available():
        return 0, 0
    
    device = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
    return allocated, reserved

def print_metrics(rank, iteration, tokens_per_sec, epoch_time, gpu_mem_allocated, gpu_mem_reserved):
    """Print training metrics"""
    print(f"[Rank {rank}] Iteration {iteration}: "
          f"{tokens_per_sec:.2f} tokens/sec, "
          f"Time: {epoch_time:.2f}s, "
          f"GPU Mem: {gpu_mem_allocated:.1f}GB/{gpu_mem_reserved:.1f}GB")

def setup_distributed(rank, world_size):
    """Initialize process group for distributed training"""
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
    
    # Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank % torch.cuda.device_count())
    
    print(f"Initialized process {rank} / {world_size}")

def cleanup():
    """Clean up distributed process group"""
    if dist.is_initialized():
        dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--seq-length", type=int, default=512, help="Sequence length")
    parser.add_argument("--iterations", type=int, default=50, help="Number of iterations to run")
    parser.add_argument("--model-size", type=str, default="small", help="Model size: tiny, small, medium")
    parser.add_argument("--world-size", type=int, default=1, help="Total number of GPUs")
    parser.add_argument("--node-rank", type=int, default=0, help="Rank of this node")
    parser.add_argument("--local-rank", type=int, default=0, help="Local rank (set by torch.distributed.launch)")
    args = parser.parse_args()
    
    # Local rank might be set by torchrun/torch.distributed.launch
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    world_size = args.world_size
    node_rank = args.node_rank
    
    # Calculate global rank
    gpus_per_node = torch.cuda.device_count()
    global_rank = node_rank * gpus_per_node + local_rank
    
    # Set up distributed training
    setup_distributed(global_rank, world_size)
    
    # Create model, optimizer, and loss function
    # Choose model size based on argument
    if args.model_size == "tiny":
        # ~30M parameters
        config = GPT2Config(
            n_embd=256,
            n_layer=6,
            n_head=8,
        )
    elif args.model_size == "small":
        # ~125M parameters
        config = GPT2Config(
            n_embd=768,
            n_layer=12,
            n_head=12,
        )
    elif args.model_size == "medium":
        # ~350M parameters
        config = GPT2Config(
            n_embd=1024,
            n_layer=24,
            n_head=16,
        )
    else:
        raise ValueError(f"Unknown model size: {args.model_size}")
        
    model = GPT2LMHeadModel(config)
    
    print(f"[Rank {global_rank}] Created {args.model_size} model with {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
    
    model.cuda()
    model = DDP(model, device_ids=[local_rank])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # Create synthetic dataset and data loader
    dataset = SyntheticTextDataset(size=10000, seq_length=args.seq_length)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=global_rank)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        sampler=sampler,
        num_workers=4
    )
    
    # Warm-up
    print(f"[Rank {global_rank}] Starting warm-up...")
    model.train()
    for _ in range(3):
        for batch in dataloader:
            input_ids = batch["input_ids"].cuda()
            labels = batch["labels"].cuda()
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            break
    
    # Synchronize before timing
    torch.cuda.synchronize()
    dist.barrier()
    
    # Benchmark
    print(f"[Rank {global_rank}] Starting benchmark...")
    total_tokens = 0
    start_time = time.time()
    model.train()
    
    scaler = amp.GradScaler()
    
    for iteration in range(args.iterations):
        iter_start = time.time()
        total_iter_tokens = 0
        
        for batch in dataloader:
            input_ids = batch["input_ids"].cuda()
            labels = batch["labels"].cuda()
            
            optimizer.zero_grad()
            
            # Use mixed precision for better performance
            with amp.autocast():
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Calculate tokens processed
            total_iter_tokens += input_ids.numel()
            total_tokens += input_ids.numel()
            
            # Just run for one batch per iteration to keep things simple
            break
        
        # Synchronize before timing
        torch.cuda.synchronize()
        
        # Calculate metrics
        iter_time = time.time() - iter_start
        tokens_per_sec = total_iter_tokens / iter_time
        epoch_time = time.time() - start_time
        gpu_mem_allocated, gpu_mem_reserved = get_gpu_memory_usage()
        
        # Report metrics every 10 iterations
        if iteration % 10 == 0 or iteration == args.iterations - 1:
            print_metrics(global_rank, iteration, tokens_per_sec, epoch_time, 
                         gpu_mem_allocated, gpu_mem_reserved)
    
    # Collect final stats across all GPUs
    torch.cuda.synchronize()
    dist.barrier()
    end_time = time.time()
    total_time = end_time - start_time
    
    # Gather throughput from all processes
    local_tokens_per_sec = torch.tensor([total_tokens / total_time], device='cuda')
    global_tokens_per_sec = [torch.zeros_like(local_tokens_per_sec) for _ in range(world_size)]
    dist.all_gather(global_tokens_per_sec, local_tokens_per_sec)
    
    if global_rank == 0:
        total_throughput = sum([t.item() for t in global_tokens_per_sec])
        print("\n" + "="*50)
        print(f"FINAL RESULTS ({args.model_size} model):")
        print(f"Total throughput: {total_throughput:.2f} tokens/sec")
        print(f"Per-GPU throughput: {total_throughput/world_size:.2f} tokens/sec")
        scaling_efficiency = total_throughput / (local_tokens_per_sec.item() * world_size) * 100
        print(f"Scaling efficiency: {scaling_efficiency:.1f}%")
        print("="*50)
    
    # Clean up
    cleanup()

if __name__ == "__main__":
    main()