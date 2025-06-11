#!/bin/bash

# Setup script for local Apple Silicon fine-tuning
# Run this script to prepare your environment

echo "🍎 Setting up local Apple Silicon fine-tuning environment..."

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p models
mkdir -p checkpoints
mkdir -p lora_finetune_output
mkdir -p logs

# Install required packages
echo "📦 Installing/upgrading PyTorch with MPS support..."
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install torchtune if not already installed
echo "🔧 Installing torchtune..."
pip install torchtune

# Install additional dependencies
echo "📚 Installing additional dependencies..."
pip install transformers datasets accelerate wandb

echo ""
echo "🚨 IMPORTANT: You need to download the Llama 3.1 8B model manually!"
echo ""
echo "Steps to get the model:"
echo "1. Request access at: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct"
echo "2. Once approved, download using:"
echo "   huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct --local-dir ./models/Meta-Llama-3.1-8B-Instruct"
echo ""
echo "Or you can use a smaller model for testing:"
echo "   huggingface-cli download meta-llama/Meta-Llama-3.1-1B-Instruct --local-dir ./models/Meta-Llama-3.1-1B-Instruct"
echo ""

# Check if MPS is available
echo "🔍 Checking MPS availability..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'MPS built: {torch.backends.mps.is_built()}')
if torch.backends.mps.is_available():
    print('✅ MPS is ready to use!')
else:
    print('❌ MPS not available, will fall back to CPU')
"

echo ""
echo "🎯 Setup complete! Next steps:"
echo "1. Download the Llama model (see instructions above)"
echo "2. Update model paths in function_calling_lora_local.yaml if needed"
echo "3. Run training with: tune run lora_finetune_single_device --config function_calling_lora_local.yaml"
echo "" 