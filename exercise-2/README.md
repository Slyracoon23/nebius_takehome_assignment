# Distributed LLM Training Benchmark for GPU Cluster Testing

This solution provides a portable, lightweight benchmark for testing multi-node GPU clusters through distributed language model training. It measures throughput, GPU utilization, and scaling efficiency to validate cluster performance.

## Overview

This benchmark solution is designed for Cloud/Infrastructure engineers to perform acceptance testing of GPU clusters. It uses open-source models and datasets, containerized execution, and automated validation workflows.

## Files

- `llm-distributed-benchmark.yaml`: SkyPilot configuration file for cluster deployment
- `llm_distributed_benchmark.py`: Main Python script for distributed training
- `run_benchmarks.sh`: Shell script to run benchmarks with different model sizes
- `benchmark-and-push`: Complete CI/CD pipeline script (combines launch + build + push)
- `Dockerfile`: Container definition using nvcr.io/nvidia/pytorch:24.07-py3 base image

## Quick Start

### Option 1: Complete Pipeline (Recommended)
```bash
# Full end-to-end pipeline: test cluster + build + push validated container
./benchmark-and-push --registry earlpotters/benchmark-and-push

# Or with environment variable
export REGISTRY_URL=earlpotters/benchmark-and-push
./benchmark-and-push
```

### Option 2: Direct SkyPilot Launch Only
```bash
sky launch llm-distributed-benchmark.yaml
```

### Option 3: Flexible Pipeline with Skip Options
```bash
# Test existing cluster without launching new job
./benchmark-and-push --skip-launch --cluster existing-cluster

# Just build and push (skip testing)
./benchmark-and-push --skip-launch --registry earlpotters/benchmark-and-push

# Dry run to see what would happen
./benchmark-and-push --dry-run --registry earlpotters/benchmark-and-push
```

## Complete Workflow

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   1. Launch     │    │   2. Execute     │    │   3. Validate   │
│   GPU Cluster   │───▶│   Distributed    │───▶│   Results &     │
│   via SkyPilot  │    │   Training       │    │   Build Image   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        │
                    ┌──────────────────┐                 │
                    │  Benchmark Tests │                 │
                    │  • Tiny Model    │                 │
                    │  • Small Model   │                 │
                    │  • Medium Model  │                 │
                    └──────────────────┘                 │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   6. Deploy     │    │   5. Push to     │    │   4. Build      │
│   Validated     │◀───│   Registry       │◀───│   Container     │
│   Container     │    │   (ghcr.io)      │    │   Image         │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Benchmark Details

The benchmark tests three different model sizes to validate cluster scaling:

1. **Tiny Model (~30M parameters)**
   - 256 embedding dimension
   - 6 layers
   - 8 attention heads
   - 32 batch size per GPU

2. **Small Model (~125M parameters)**
   - 768 embedding dimension
   - 12 layers
   - 12 attention heads
   - 8 batch size per GPU

3. **Medium Model (~350M parameters)**
   - 1024 embedding dimension
   - 24 layers
   - 16 attention heads
   - 4 batch size per GPU

## Metrics Collected

- **Training throughput** (tokens/second)
- **GPU memory usage** and utilization
- **Scaling efficiency** (how close to linear scaling)
- **Training time** per epoch
- **Multi-node communication** overhead

## Container Registry Integration

### Docker Hub Registry
Set up GitHub Actions for automated builds:

```yaml
# .github/workflows/gpu-cluster-test.yml
name: GPU Cluster Validation
on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test-and-build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run Complete GPU Cluster Pipeline
        run: ./benchmark-and-push --registry earlpotters/benchmark-and-push
        env:
          DOCKER_HUB_TOKEN: ${{ secrets.DOCKER_HUB_TOKEN }}
```

### Manual Registry Push
```bash
# Complete pipeline with registry push
export REGISTRY_URL=earlpotters/benchmark-and-push
./benchmark-and-push

# Or inline
./benchmark-and-push --registry earlpotters/benchmark-and-push
```

## Usage Scenarios

### 1. Acceptance Testing
```bash
# Test new GPU cluster deployment and build validated image
./benchmark-and-push --cluster gpu-cluster-test --registry earlpotters/benchmark-and-push

# Or test first, then build if satisfied
sky launch llm-distributed-benchmark.yaml -c gpu-cluster-test
./benchmark-and-push --skip-launch --cluster gpu-cluster-test --registry earlpotters/benchmark-and-push
```

### 2. Continuous Integration
```bash
# Complete automated pipeline
./benchmark-and-push --registry earlpotters/benchmark-and-push

# With specific cluster name
./benchmark-and-push --cluster ci-gpu-test --registry earlpotters/benchmark-and-push
```

### 3. Regression Testing
```bash
# Test cluster after updates with full validation
./benchmark-and-push --cluster regression-test --registry earlpotters/benchmark-and-push --logs

# Dry run to preview what would happen
./benchmark-and-push --cluster regression-test --dry-run --logs
```

## Requirements

- **Infrastructure**: Kubernetes cluster with GPU support
- **Hardware**: H100 GPUs (adaptable for other GPU types)
- **Software**: 
  - SkyPilot installed and configured
  - Docker with registry access
  - NVIDIA Container Runtime

## Customization

Customize the benchmark by editing:

- **Model sizes** in `llm_distributed_benchmark.py`
- **Batch sizes** in `run_benchmarks.sh`
- **Number of iterations** in `run_benchmarks.sh`
- **Sequence length** with the `--seq-length` parameter
- **Node count** in `llm-distributed-benchmark.yaml`

## Environment Variables

```bash
# Required for registry push
export REGISTRY_URL=earlpotters/benchmark-and-push
export DOCKER_HUB_TOKEN=your_docker_hub_token

# Optional cluster configuration
export CLUSTER_NAME=my-gpu-cluster
```

## Troubleshooting

### Check job status:
```bash
sky status
sky logs cluster-name
```

### View recent logs and pipeline status:
```bash
./benchmark-and-push --cluster cluster-name --logs --dry-run
```

### Run specific parts of the pipeline:
```bash
# Only test the cluster (no build/push)
./benchmark-and-push --skip-build --skip-push

# Only build and push (skip testing)
./benchmark-and-push --skip-launch --registry earlpotters/benchmark-and-push

# Manual container build
docker build -t llm-benchmark-validated .
```

### Get help and see all options:
```bash
./benchmark-and-push --help
```

## Container Base Image

Uses `nvcr.io/nvidia/pytorch:24.07-py3` as the base image, providing:
- PyTorch with CUDA support
- NVIDIA optimized libraries
- Multi-GPU and distributed training capabilities
- Production-ready ML environment