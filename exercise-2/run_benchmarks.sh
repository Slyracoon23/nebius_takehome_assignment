#!/bin/bash
set -e

# Get node info for distributed training
MASTER_ADDR=$(hostname -i)
WORLD_SIZE=$(( $SKYPILOT_NUM_NODES * $(nvidia-smi --list-gpus | wc -l) ))
echo "Master address: $MASTER_ADDR"
echo "World size (total GPUs): $WORLD_SIZE"
echo "Node rank: $SKYPILOT_NODE_RANK"
echo "Number of nodes: $SKYPILOT_NUM_NODES"
echo "GPUs per node: $(nvidia-smi --list-gpus | wc -l)"

# Run benchmark with different model sizes
echo "===== Running Tiny Model Benchmark ====="
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
  --nproc_per_node=8 \
  --nnodes=$SKYPILOT_NUM_NODES \
  --node_rank=$SKYPILOT_NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=29500 \
  /app/llm_distributed_benchmark.py \
  --world-size=$WORLD_SIZE \
  --node-rank=$SKYPILOT_NODE_RANK \
  --model-size=tiny \
  --batch-size=32 \
  --iterations=20

echo "===== Running Small Model Benchmark ====="
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
  --nproc_per_node=8 \
  --nnodes=$SKYPILOT_NUM_NODES \
  --node_rank=$SKYPILOT_NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=29500 \
  /app/llm_distributed_benchmark.py \
  --world-size=$WORLD_SIZE \
  --node-rank=$SKYPILOT_NODE_RANK \
  --model-size=small \
  --batch-size=8 \
  --iterations=20

# Only run medium model if we have enough memory
if [ $SKYPILOT_NUM_NODES -ge 2 ]; then
  echo "===== Running Medium Model Benchmark ====="
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes=$SKYPILOT_NUM_NODES \
    --node_rank=$SKYPILOT_NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=29500 \
    /app/llm_distributed_benchmark.py \
    --world-size=$WORLD_SIZE \
    --node-rank=$SKYPILOT_NODE_RANK \
    --model-size=medium \
    --batch-size=4 \
    --iterations=10
fi

echo "===== Benchmark Complete ====="