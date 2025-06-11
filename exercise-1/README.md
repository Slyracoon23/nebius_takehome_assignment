# Exercise 1: LLM Fine-tuning PoC on Nebius AI Cloud

## 🎯 Exercise Overview

This exercise demonstrates a complete end-to-end multi-node fine-tuning solution for LLMs on Nebius AI Cloud, specifically designed for ML Engineers with limited cloud infrastructure experience. The goal is to showcase how to efficiently utilize PoC resources for function calling fine-tuning before scaling to production.

### Exercise Requirements
✅ **Multi-node fine-tuning**: 16 H100 GPUs across 2 nodes  
✅ **Storage utilization**: 2TB SSD network disk + 2TB shared filesystem  
✅ **ML Engineer friendly**: Minimal cloud expertise required  
✅ **Function calling**: Fine-tune Llama 3.1 for AI agent workflows  
✅ **Complete code examples**: Ready-to-run scripts and configurations  
✅ **Monitoring & observability**: Real-time training insights  

## 🏗️ Technical Architecture

### Resource Allocation
- **Compute**: 16 H100 GPUs (2 nodes × 8 GPUs each)
- **Storage**: 2TB shared filesystem for models/datasets + 2TB network disk for logs/checkpoints
- **Orchestration**: Kubernetes + SkyPilot (abstracts cloud complexity)
- **Framework**: PyTorch with torchtune for distributed training

```
┌─────────────────────────────────────────────────────────────┐
│                    Nebius AI Cloud                         │
├─────────────────┬─────────────────┬─────────────────────────┤
│   Node 1        │   Node 2        │   Shared Resources      │
│   8×H100        │   8×H100        │                         │
│   128 vCPU      │   128 vCPU      │  ┌─────────────────┐   │
│   1600GB RAM    │   1600GB RAM    │  │ 2TB Shared FS   │   │
│                 │                 │  │ (models/data)   │   │
│                 │                 │  └─────────────────┘   │
│                 │                 │  ┌─────────────────┐   │
│                 │                 │  │ 2TB Network     │   │
│                 │                 │  │ (logs/checkpts) │   │
│                 │                 │  └─────────────────┘   │
└─────────────────┴─────────────────┴─────────────────────────┘
```

### Why This Architecture?
- **SkyPilot**: Abstracts Kubernetes complexity for ML engineers
- **Shared Filesystem**: Enables seamless data access across nodes
- **LoRA Fine-tuning**: Efficient training with minimal memory overhead
- **Multi-node**: Demonstrates scaling capabilities for production

## 🚀 Exercise Execution Guide

### Step 1: Environment Setup (5 minutes)

**Prerequisites:**
- Access to provisioned Nebius cluster with 16 H100 GPUs
- Kubernetes cluster credentials configured
- Terminal access (Linux/macOS/WSL)
- Basic familiarity with command line

**Quick Setup:**
```bash
# Clone this repository
git clone <repo-url>
cd exercise-1

# Verify cluster access
kubectl get nodes
# You should see 2 nodes with 8 H100 GPUs each

# Install SkyPilot if not already installed
pip install "skypilot[kubernetes]"

# Verify SkyPilot can access your cluster
sky check kubernetes

# Set up environment variables
cp .env.example .env
# Edit .env with your HuggingFace token and Weights & Biases key
```

**Environment Configuration:**
```bash
# Create .env file with required tokens
cat > .env << EOF
export HF_TOKEN="your-huggingface-token"
export WANDB_API_KEY="your-wandb-api-key"
export MODEL_SIZE="8B"
export DATASET="NousResearch/hermes-function-calling-v1"
export MAX_SAMPLES="1500"
EOF

# Source environment variables
source .env
```

### Step 2: Launch Multi-node Training (1 command)

```bash
# Launch the distributed training job
sky launch function-calling-finetune.yaml \
  --cluster-name llama-function-calling \
  --env HF_TOKEN \
  --env WANDB_API_KEY

# This command will:
# 1. Provision 2 nodes with 8 H100 GPUs each
# 2. Set up shared filesystem mounts
# 3. Install training dependencies
# 4. Start distributed training job
```

### Step 3: Monitor Training Progress

**Real-time Monitoring:**
```bash
# Check cluster status
sky status

# Stream training logs
sky logs llama-function-calling --follow

# Monitor GPU utilization across all nodes
kubectl get nodes
sky exec llama-function-calling "nvidia-smi"

# Check resource usage
sky exec llama-function-calling "df -h"  # Storage usage
sky exec llama-function-calling "htop"   # CPU/Memory usage
```

**Key Metrics to Monitor:**
- GPU utilization across all 16 H100s (target: >85%)
- Training loss and convergence
- Memory usage per node (shared filesystem usage)
- Network I/O between nodes
- Training throughput (tokens/sec)
- Weights & Biases dashboard for training metrics

### Step 4: Validate Results

**Expected Training Outcomes:**
```bash
# Check final model location
sky exec llama-function-calling "ls -la /mnt/shared/checkpoints/"

# Test function calling capability
sky exec llama-function-calling "python test_function_calling.py"
```

## 💻 Code Examples

### Core Training Configuration

**File: `configs/function_calling_lora.yaml`**
```yaml
# Distributed training config for 16 H100 GPUs
model:
  _component_: torchtune.models.llama3_1.llama3_1_8b
  
tokenizer:
  _component_: torchtune.models.llama3_1.llama3_1_tokenizer
  path: meta-llama/Meta-Llama-3.1-8B-Instruct

checkpointer:
  _component_: torchtune.utils.FullModelMetaCheckpointer
  checkpoint_dir: /mnt/shared/checkpoints/
  
# LoRA configuration for efficient fine-tuning
lora_rank: 64
lora_alpha: 128
lora_dropout: 0.1
quantize_base: False

# Distributed training settings
batch_size: 2  # Per GPU
gradient_accumulation_steps: 16  # Effective batch size: 512
max_steps_per_epoch: 1000
epochs: 3

# Optimizer settings
optimizer:
  _component_: torch.optim.AdamW
  lr: 2e-4
  weight_decay: 0.01

lr_scheduler:
  _component_: torchtune.modules.get_cosine_schedule_with_warmup
  num_warmup_steps: 100

# Dataset configuration
dataset:
  _component_: torchtune.datasets.alpaca_cleaned_dataset
  source: glaiveai/glaive-function-calling-v2
  split: train
  max_seq_len: 4096

# Storage paths
output_dir: /mnt/shared/outputs/
log_dir: /mnt/network/logs/

# Monitoring
enable_activation_checkpointing: True
save_every_n_steps: 500
log_every_n_steps: 10
```

### Distributed Training Script

**File: `scripts/train_distributed.py`**
```python
#!/usr/bin/env python3
"""
Multi-node distributed training script for function calling fine-tuning.
Optimized for ML Engineers with minimal distributed training experience.
"""

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchtune.config import config_from_path
from torchtune.training import train

def setup_distributed():
    """Initialize distributed training environment."""
    # SkyPilot automatically sets these environment variables
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    # Initialize process group for multi-node communication
    dist.init_process_group(
        backend='nccl',  # Optimized for GPU communication
        rank=rank,
        world_size=world_size
    )
    
    # Set GPU device for this process
    torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank

def main():
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    
    # Load training configuration
    config = config_from_path("configs/function_calling_lora.yaml")
    
    # Adjust batch size for distributed training
    config.batch_size = config.batch_size // world_size
    
    # Create model and move to GPU
    model = config.model
    model = model.to(f'cuda:{local_rank}')
    
    # Wrap model for distributed training
    model = DDP(model, device_ids=[local_rank])
    
    # Initialize tokenizer and datasets
    tokenizer = config.tokenizer
    train_dataset = config.dataset
    
    # Setup optimizer and scheduler
    optimizer = config.optimizer(model.parameters())
    lr_scheduler = config.lr_scheduler(optimizer)
    
    # Start training
    if rank == 0:
        print(f"Starting distributed training on {world_size} GPUs...")
        print(f"Effective batch size: {config.batch_size * world_size}")
    
    train(
        model=model,
        tokenizer=tokenizer,
        dataset=train_dataset,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=config
    )
    
    # Cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

### Function Calling Test Script

**File: `test_function_calling.py`**
```python
#!/usr/bin/env python3
"""
Test the fine-tuned model's function calling capabilities.
"""

import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_function_calling():
    """Test the fine-tuned model with function calling examples."""
    
    # Load the fine-tuned model
    model_path = "/mnt/shared/checkpoints/function-calling-final"
    
    print("Loading fine-tuned model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Define test functions
    test_functions = [
        {
            "name": "get_weather",
            "description": "Get current weather information for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"]
            }
        },
        {
            "name": "schedule_meeting",
            "description": "Schedule a meeting with participants",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Meeting title"},
                    "participants": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of participant emails"
                    },
                    "duration": {"type": "integer", "description": "Meeting duration in minutes"},
                    "date": {"type": "string", "description": "Meeting date in YYYY-MM-DD format"}
                },
                "required": ["title", "participants", "date"]
            }
        }
    ]
    
    # Test cases
    test_cases = [
        "What's the weather like in New York?",
        "Schedule a meeting with john@example.com and sarah@example.com for tomorrow about project review",
        "Get the temperature in London in celsius",
        "Book a 30-minute meeting with the team for 2024-01-15"
    ]
    
    print("\n=== Function Calling Test Results ===")
    
    for i, prompt in enumerate(test_cases, 1):
        print(f"\nTest {i}: {prompt}")
        
        # Prepare input with function definitions
        messages = [
            {
                "role": "system",
                "content": f"You are a helpful assistant with access to functions. Here are the available functions: {json.dumps(test_functions, indent=2)}"
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        # Tokenize and generate response
        inputs = tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            return_tensors="pt"
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        print(f"Response: {response}")
        print("-" * 50)

if __name__ == "__main__":
    test_function_calling()
```

## 📊 Monitoring & Observability

### Real-time Metrics

**1. GPU and Resource Monitoring**
```bash
# Monitor GPU utilization in real-time
sky exec llama-function-calling "nvidia-smi -l 5"

# Check detailed GPU usage per node
sky exec llama-function-calling "nvidia-smi -q -d UTILIZATION,MEMORY,TEMPERATURE"

# Monitor storage usage (both shared filesystem and network storage)
sky exec llama-function-calling "df -h /mnt/shared /mnt/network"

# Check training metrics in Weights & Biases
# Visit https://wandb.ai and view your project dashboard
```

**2. Training Progress Monitoring**
```bash
# Stream live training logs
sky logs llama-function-calling --follow

# Check training metrics
sky exec llama-function-calling "tail -f /mnt/network/logs/training.log"

# Monitor resource usage
sky exec llama-function-calling "nvidia-smi -l 5"
```

**3. Key Performance Indicators**

| Metric | Target Value | Troubleshooting |
|--------|-------------|----------------|
| GPU Utilization | 85-95% | Check data loading pipeline |
| Training Loss | < 1.5 after 3 epochs | Adjust learning rate/batch size |
| Memory Usage | 70-80% per GPU | Reduce sequence length if needed |
| Throughput | ~1000 tokens/sec | Verify network connectivity |
| Convergence | Stable decrease | Monitor gradient norms |

## 🔍 Technical Choices Explained

### Why SkyPilot + Kubernetes?
- **Abstraction**: ML engineers don't need to manage Kubernetes directly
- **Scalability**: Easy to scale from 16 to 512 GPUs
- **Reliability**: Built-in fault tolerance and job recovery
- **Cost efficiency**: Automatic resource cleanup

### Why LoRA Fine-tuning?
- **Memory efficiency**: Reduces VRAM usage by 60-80%
- **Training speed**: Faster convergence than full fine-tuning
- **Model quality**: Maintains performance while reducing parameters
- **Deployment**: Smaller model artifacts for production

### Why Shared Filesystem?
- **Data consistency**: All nodes access same dataset version
- **Checkpointing**: Centralized model checkpoint storage
- **Debugging**: Easy log aggregation and analysis
- **Scalability**: Supports larger datasets and model ensembles

## 🛠️ Troubleshooting Guide

### Common Issues for ML Engineers

**1. Training Stalls or Slow Progress**
```bash
# Check GPU utilization
sky exec llama-function-calling "nvidia-smi"

# Verify data loading and I/O
sky exec llama-function-calling "iostat -x 1"

# Check available storage space
sky exec llama-function-calling "df -h /mnt/shared /mnt/network"

# Verify cluster connectivity
kubectl get nodes -o wide
sky exec llama-function-calling "ping -c 3 8.8.8.8"  # Internet connectivity

# Solution: Increase data loading workers
# Edit config: num_workers: 8
```

**2. Out of Memory Errors**
```bash
# Reduce batch size
sky exec llama-function-calling "sed -i 's/batch_size: 2/batch_size: 1/' configs/function_calling_lora.yaml"

# Or reduce sequence length
# Edit config: max_seq_len: 2048
```

**3. Model Quality Issues**
```bash
# Check dataset quality
sky exec llama-function-calling "head -n 100 /mnt/shared/datasets/train.jsonl"

# Adjust learning rate
# Edit config: lr: 1e-4  # Reduce if loss diverges

# Increase LoRA rank for complex tasks
# Edit config: lora_rank: 128
```

**4. Storage Issues**
```bash
# Check filesystem usage
sky exec llama-function-calling "df -h"

# Clean up old checkpoints
sky exec llama-function-calling "rm -rf /mnt/shared/checkpoints/step_*"
```

## 📈 Expected Results & Validation

### Training Timeline
- **Setup**: 10-15 minutes (automated)
- **Training**: 2-3 hours for 3 epochs
- **Validation**: 15 minutes
- **Total**: ~4 hours end-to-end

### Success Criteria
✅ **Training Loss**: Converges to < 1.5  
✅ **Function Accuracy**: > 85% on test cases  
✅ **GPU Efficiency**: > 85% utilization  
✅ **Model Size**: ~16GB (manageable for deployment)  
✅ **Inference Speed**: ~100 tokens/sec  

### Validation Tests
```bash
# Run comprehensive validation
sky exec llama-function-calling "python validate_model.py"

# Test specific function calling scenarios
sky exec llama-function-calling "python test_function_calling.py"

# Performance benchmarking
sky exec llama-function-calling "python benchmark_inference.py"
```

## 🎯 Exercise Completion Checklist

- [ ] Successfully provision 16 H100 GPUs across 2 nodes
- [ ] Utilize both 2TB shared filesystem and network storage
- [ ] Complete function calling fine-tuning with > 85% accuracy
- [ ] Demonstrate real-time monitoring and observability
- [ ] Validate model performance with provided test cases
- [ ] Document all technical choices and rationale
- [ ] Show cost-efficient resource utilization
- [ ] Prepare for customer demo presentation

## 💰 Resource Utilization & Cost

### Efficient Resource Usage
- **Compute**: Full utilization of 16 H100 GPUs (>85% average)
- **Storage**: 
  - Shared FS: Models (40GB), datasets (20GB), checkpoints (60GB)
  - Network disk: Logs (10GB), monitoring data (15GB)
- **Network**: High-bandwidth inter-node communication for distributed training

### Estimated Costs (4-hour PoC)
- **GPU time**: 16 H100 × 4 hours = 64 GPU-hours
- **Storage**: 2TB shared + 2TB network × 1 day
- **Networking**: Inter-node communication bandwidth
- **Total**: Contact Nebius sales for regional pricing

## 🚀 Scaling to Production

### From PoC to 512 H100 GPUs
```yaml
# Update cluster configuration
resources:
  accelerators: {H100: 8, H100:32}  # 32 nodes × 8 GPUs
  memory: 32x
  disk_size: 10000  # Scale storage proportionally
```

### Production Optimizations
- **Multi-region deployment**: For global availability
- **Advanced checkpointing**: For fault tolerance
- **vLLM integration**: For high-throughput inference
- **Custom datasets**: Replace with proprietary function calling data

## 📞 Support & Demo Preparation

### For Customer Demo
1. **Live Training Demo**: Show real-time GPU utilization
2. **Function Calling Test**: Demonstrate AI agent capabilities
3. **Scaling Discussion**: Explain path to 512 GPUs
4. **Cost Analysis**: ROI comparison with other cloud providers
5. **Technical Deep-dive**: Architecture and design choices

### Getting Help
- **Training Issues**: Check Weights & Biases dashboard and training logs
- **Infrastructure**: Use `sky status` and `kubectl get nodes`
- **Performance**: Monitor with `nvidia-smi` and resource usage commands
- **Storage Issues**: Check `/mnt/shared` and `/mnt/network` mount points
- **Code Issues**: Review training logs with `sky logs llama-function-calling`

## 📚 Additional Resources

- [Nebius AI Cloud Documentation](https://docs.nebius.com/)
- [SkyPilot Documentation](https://skypilot.readthedocs.io/)
- [Torchtune Fine-tuning Guide](https://pytorch.org/torchtune/)
- [Llama 3.1 Model Card](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
- [Glaive Function Calling Dataset](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2)
- [Nebius Solution Library](https://github.com/nebius/nebius-solution-library)

---

**🎓 Exercise Completion**: This PoC demonstrates Nebius AI Cloud's ability to handle production-scale ML workloads with enterprise-grade infrastructure, simplified tooling, and cost-effective resource utilization - perfect for scaling from startup to enterprise AI applications.