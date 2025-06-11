#!/bin/bash
set -e

# Nebius PoC Setup Script for Function Calling Fine-tuning
# This script sets up the ML environment assuming cluster access is already configured

echo "🚀 Starting Nebius PoC Environment Setup for Function Calling Fine-tuning"

# Check prerequisites
command -v kubectl >/dev/null 2>&1 || { echo "❌ kubectl not found. Please install it first."; exit 1; }
command -v sky >/dev/null 2>&1 || { echo "❌ SkyPilot not found. Install with: pip install 'skypilot[kubernetes]'"; exit 1; }

# Verify cluster access
echo "🔍 Verifying cluster access..."
if ! kubectl get nodes >/dev/null 2>&1; then
    echo "❌ Cannot access Kubernetes cluster. Please configure kubectl credentials first."
    exit 1
fi

echo "✅ Kubernetes cluster access verified"
kubectl get nodes

# Verify SkyPilot can access the cluster
echo "🔍 Verifying SkyPilot configuration..."
if ! sky check kubernetes >/dev/null 2>&1; then
    echo "❌ SkyPilot cannot access Kubernetes cluster. Running sky check..."
    sky check kubernetes
    exit 1
fi

echo "✅ SkyPilot configuration verified"

# Verify required files exist
echo "📂 Verifying required files..."
required_files=(
    "function-calling-finetune.yaml"
    "configs/function_calling_lora.yaml"
)

for file in "${required_files[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo "❌ Required file missing: $file"
        echo "Please make sure you're in the correct directory with all exercise files."
        exit 1
    fi
done

echo "✅ All required files found"

# Prompt for API keys
echo "🔐 API Key Configuration:"
read -p "Enter your Hugging Face token (for model access): " HF_TOKEN
read -p "Enter your Weights & Biases API key (for monitoring): " WANDB_API_KEY

# Validate tokens
if [[ -z "$HF_TOKEN" ]]; then
    echo "❌ HuggingFace token is required for model access"
    exit 1
fi

if [[ -z "$WANDB_API_KEY" ]]; then
    echo "⚠️  Weights & Biases key not provided. Training metrics won't be logged to W&B."
fi

# Create environment file
cat > .env << EOF
export HF_TOKEN="$HF_TOKEN"
export WANDB_API_KEY="$WANDB_API_KEY"
export MODEL_SIZE="8B"
export DATASET="NousResearch/hermes-function-calling-v1"
export MAX_SAMPLES="1500"
export CLUSTER_NAME="llama-function-calling"
EOF

echo "💾 Environment configuration saved to .env"



echo ""
echo "🎉 Setup complete! Next steps:"
echo ""
echo "1. Source your environment:"
echo "   source .env"
echo ""
echo "2. Launch fine-tuning:"
echo "   sky launch function-calling-finetune.yaml --env HF_TOKEN --env WANDB_API_KEY"
echo ""
echo "3. Monitor training progress:"
echo "   # Check cluster status"
echo "   sky status"
echo ""
echo "   # Stream training logs"
echo "   sky logs llama-function-calling --follow"
echo ""
echo "   # Check GPU utilization"
echo "   sky exec llama-function-calling 'nvidia-smi'"
echo ""
echo "   # Monitor storage usage"
echo "   sky exec llama-function-calling 'df -h /mnt/shared'"
echo ""
echo "4. Validate training completion:"
echo "   sky exec llama-function-calling 'ls -la /mnt/shared/checkpoints/'"
echo ""
echo "5. When training is complete, cleanup resources:"
echo "   sky down llama-function-calling -y"
echo ""
echo "📊 Monitor training metrics at: https://wandb.ai (if W&B key provided)"
echo "📁 Working directory: $(pwd)"
echo ""
echo "🚀 Ready to start your function calling fine-tuning PoC!"
echo ""
echo "💡 Additional monitoring commands:"
echo "   # Real-time GPU monitoring"
echo "   sky exec llama-function-calling 'nvidia-smi -l 5'"
echo ""
echo "   # Check training progress"
echo "   sky logs llama-function-calling --tail 50"