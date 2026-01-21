#!/bin/bash

# ExoT2I Phase 1: End-to-End PoC Training Script (TPU v5e)
# Robust, idempotent version.

set -e

echo "=== [1/4] Environment Check ==="

# 1. Dependency Check
if ! python3 -c "import torch_xla" &> /dev/null; then
    echo "torch_xla not found. Installing..."
    pip install torch-xla -f https://storage.googleapis.com/tpu-pytorch/wheels/tpuv2/torch_xla-2.0-cp310-cp310-linux_x86_64.whl
fi
pip install -r requirements.txt

# 2. Login Check
if ! huggingface-cli whoami &> /dev/null; then
    echo "Warning: Not logged into Hugging Face. Run 'huggingface-cli login' first."
fi

echo "=== [2/4] Data Verification ==="

# Check prompts
if [ ! -f "prompt.json" ]; then
    echo "prompt.json missing. Generating..."
    if [ -z "$GROQ_API_KEY" ]; then
        echo "Error: GROQ_API_KEY is required to generate prompts."
        exit 1
    fi
    python3 prompt.py
else
    echo "prompt.json found. Skipping generation."
fi

# Check trajectories
if [ ! -d "data/train_trajectories" ] || [ -z "$(ls -A data/train_trajectories 2>/dev/null)" ]; then
    echo "Data trajectories missing. Generating..."
    python3 generate_data.py
else
    echo "Trajectories found. Skipping generation."
fi

echo "=== [3/4] Launching TPU Training ==="

export XRT_TPU_CONFIG="localservice;0;localhost:51011"
export PJRT_DEVICE=TPU

# Training logic inside training.py handles multi-processing/single-chip
python3 training.py

echo "=== [4/4] Pipeline Complete ==="
