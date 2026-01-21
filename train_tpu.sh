#!/bin/bash

# ExoT2I TPU Training Launch Script
# Optimized for Google Cloud TPU v5e-1

# 1. Environment Setup (Ensure torch_xla is installed)
# pip install torch-xla -f https://storage.googleapis.com/tpu-pytorch/wheels/tpuv2/torch_xla-2.0-cp310-cp310-linux_x86_64.whl

# 2. XLA Parameters
export XRT_TPU_CONFIG="localservice;0;localhost:51011"
export PJRT_DEVICE=TPU

# 3. Training Launch
# For v5e-1, nprocs in training.py controls the cores used.
# nprocs=1 is set for the single-chip v5e.
echo "Launching ExoT2I Training on TPU v5e-1..."
python3 training.py
