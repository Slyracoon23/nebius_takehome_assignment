# Use NVIDIA PyTorch NGC Container as base
FROM nvcr.io/nvidia/pytorch:24.07-py3

# Set working directory
WORKDIR /app

# Update package lists and install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages that are commonly needed
RUN pip install --no-cache-dir \
    transformers \
    datasets \
    accelerate \
    wandb \
    tensorboard

# Copy application files
COPY . /app/

# Make scripts executable
RUN chmod +x /app/llm_distributed_benchmark.py /app/run_benchmarks.sh

# Set default command
CMD ["/bin/bash"] 