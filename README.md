# Nebius AI Solution Architect - Take Home Assignment

This repository contains solutions for the two exercises in the Nebius AI ML specialist Customer Solution Architect take-home assignment.

## Assignment Overview

As part of the application process for the AI ML specialist Customer Solution Architect role at Nebius, I was tasked with completing two technical exercises:

1. **Exercise 1**: LLM fine-tuning solution for a potential customer's PoC
2. **Exercise 2**: Internal solution for testing GPU clusters

## Demo Walkthrough

📹 **[Watch the solution walkthrough video](https://youtu.be/cLJHAJt8IlQ)** - A detailed explanation of both exercises and their implementations.

## Exercise 1: LLM Fine-tuning for Customer PoC

### Background
A VC-funded startup (20 headcount) working on process automation with AI agents wants to test Nebius platform before committing to 512 H100 GPUs for 6 months. They need to perform end-to-end fine-tuning of an open-source LLM for function calling during their PoC phase.

### Solution: Multi-node Function Calling Fine-tune

**File**: `exercise-1/function-calling-finetune.yaml`

#### Technical Implementation

- **Model**: Llama 3.1 8B Instruct
- **Task**: Function calling fine-tuning using LoRA (Low-Rank Adaptation)
- **Dataset**: NousResearch/hermes-function-calling-v1 (1,500 samples subset)
- **Infrastructure**: 
  - 2 nodes × 8 H100 GPUs each = 16 H100 GPUs total
  - Shared filesystem mounted at `/mnt/shared`
  - Kubernetes orchestration

#### Key Features

1. **Multi-node Distributed Training**: Uses PyTorch distributed training with proper node coordination
2. **Efficient Resource Utilization**: Leverages all 16 H100 GPUs across 2 nodes
3. **Shared Storage**: Model downloads and checkpoints stored on shared filesystem
4. **Production-ready**: Includes monitoring with Weights & Biases integration
5. **Error Handling**: Robust environment variable handling and fallback mechanisms

#### Technical Choices Explained

- **Framework**: TorchTune - Meta's official fine-tuning library for Llama models
- **Method**: LoRA fine-tuning - Memory efficient, faster training, smaller output models
- **Scheduler**: Kubernetes - Better for ML workloads, easier scaling, resource management
- **Storage**: Shared filesystem for model persistence across nodes

#### Usage

```bash
# Set environment variables
export HF_TOKEN="your_huggingface_token"
export WANDB_API_KEY="your_wandb_key"

# Launch the fine-tuning job
sky launch exercise-1/function-calling-finetune.yaml -c llama-function-calling --env HF_TOKEN --env WANDB_API_KEY
```

#### Monitoring

- **Weights & Biases**: Real-time training metrics and loss curves
- **Logs**: Detailed logging of distributed training progress
- **Checkpoints**: Automatic model saving to shared storage

### Customer Presentation Points

1. **Scalability**: Easy to scale from 16 to 512 GPUs
2. **Cost Efficiency**: LoRA reduces training time and storage requirements
3. **Production Ready**: Includes monitoring, checkpointing, and error handling
4. **ML Engineer Friendly**: Simple configuration, minimal infrastructure knowledge required

## Exercise 2: GPU Cluster Testing Solution

### Background
Internal solution for Cloud/Infrastructure engineers to perform acceptance testing of GPU clusters. Needs to be portable, lightweight, and use only open-source components.

### Solution: Distributed LLM Training Benchmark

**File**: `exercise-2/llm-distributed-benchmark.yaml`

#### Technical Implementation

- **Base Image**: `nvcr.io/nvidia/pytorch:24.07-py3`
- **Infrastructure**: 1 node × 8 H100 GPUs
- **Purpose**: Validate GPU cluster functionality through distributed training

#### Key Features

1. **Containerized**: Uses NVIDIA PyTorch container for consistency
2. **Automated Testing**: Includes benchmark scripts for systematic testing
3. **GPU Validation**: Tests all GPUs simultaneously with distributed workloads
4. **Portable**: Can run on any Kubernetes cluster with GPU support

#### Technical Choices Explained

- **Container Strategy**: NVIDIA PyTorch container provides optimized CUDA/cuDNN
- **Single Node**: Focuses on testing GPU interconnect and memory bandwidth
- **Lightweight**: Minimal dependencies, fast startup time
- **Open Source**: Uses only publicly available models and datasets

#### Usage

```bash
# Launch the benchmark
sky launch exercise-2/llm-distributed-benchmark.yaml -c gpu-benchmark-test
```

#### Testing Coverage

- GPU memory utilization
- Inter-GPU communication (NCCL)
- Distributed training coordination
- CUDA kernel performance
- Memory bandwidth validation

## Repository Structure

```
takehome_assignment/
├── exercise-1/
│   ├── function-calling-finetune.yaml    # Multi-node LLM fine-tuning
│   └── configs/
│       └── function_calling_lora.yaml    # LoRA configuration
├── exercise-2/
│   ├── llm-distributed-benchmark.yaml    # GPU cluster benchmark
│   ├── llm_distributed_benchmark.py      # Benchmark implementation
│   └── run_benchmarks.sh                 # Test execution script
├── modules/                               # Terraform/Infrastructure modules
├── k8s-env/                              # Kubernetes environment configs
└── README.md                             # This file
```

## Prerequisites

- SkyPilot installed and configured
- Access to Nebius Kubernetes cluster
- HuggingFace account with token (Exercise 1)
- Weights & Biases account (Exercise 1, optional)

 