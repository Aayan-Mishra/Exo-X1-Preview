"""
ExoT2I Configuration File
Contains hyperparameters, model paths, and training settings for the unified reasoning-diffusion transformer.
"""

import os
from pathlib import Path

# Model Configuration
# Model Configuration (Roadmap: Hub PoC 4B -> Scale 9B -> Ultimate 32B)
# Current: 4B "Thinking" PoC
QWEN_MODEL_NAME = "Qwen/Qwen3-VL-4B-Thinking"
FLUX_MODEL_NAME = "black-forest-labs/FLUX.2-klein-base-4B"
CHECKPOINT_DIR = "models/final/exo-poc-4b"

# Future Scaling Targets
# QWEN_9B = "Qwen/Qwen3-VL-8B"-Thinking
# FLUX_9B = "black-forest-labs/FLUX.2-klein-base-9B"

# QWEN_32B = "Qwen/Qwen3-VL-32B"
# FLUX_32B = "Qwen/Qwen3-VL-32B-Thinking"

# Architecture Parameters
VISION_TOKENS_MIN = 256
VISION_TOKENS_MAX = 1280
LATENT_DIM = 16  # FLUX latent channels
LATENT_SIZE = 64  # 64x64 latents for 1024x1024 images
EDIT_MASK_SIZE = 32  # 32x32 edit mask
EDIT_DELTA_DIM = 4096  # Edit delta embedding dimension

# Training Hyperparameters
BATCH_SIZE = 8  # Increased for 4B PoC
LEARNING_RATE = 1e-5
NUM_EPOCHS = 10
GRADIENT_ACCUMULATION_STEPS = 4
WARMUP_STEPS = 1000

# Loss Weights (as specified)
LOSS_DIFFUSION_WEIGHT = 0.9
LOSS_EDIT_RECON_WEIGHT = 0.1
LOSS_PERCEPTUAL_WEIGHT = 0.05

# Diffusion Parameters
NUM_DIFFUSION_STEPS = 20
NUM_REFINE_ITERS = 4  # Reasoning steps per diffusion step
SIGMA_MIN = 0.002
SIGMA_MAX = 80.0

# Data Generation
NUM_TRAJECTORIES = 25000  # 25K FLUX trajectories
IMAGE_SIZE = 1024
CAPTION_LENGTH = 77  # CLIP-like caption length

# Golden Prompts for Dataset Generation
GOLDEN_25 = [
    # Portraits (5)
    "professional headshot smiling woman", "elderly man thoughtful",
    "child laughing beach", "businessman suit modern office", "artist paintbrush",

    # Products (5)
    "ceramic vase white minimal", "leather handbag luxury", "smartphone glass",
    "coffee mug steam", "sneakers urban street",

    # Architecture (5)
    "modern glass skyscraper", "traditional wooden house", "brutalist concrete",
    "japanese temple garden", "victorian mansion foggy",

    # Landscapes (5)
    "sunset mountain lake", "autumn forest path", "desert dunes golden hour",
    "snowy pine forest", "tropical beach turquoise",

    # Abstract (5)
    "abstract blue waves", "geometric gold black", "marble texture veins",
    "smoke wisps colorful", "liquid metal flow"
]

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOGS_DIR = PROJECT_ROOT / "logs"

# Training Stages
STAGES = {
    "projector": {"epochs": 2, "lr": 1e-4},
    "diffusion_head": {"epochs": 5, "lr": 5e-6},
    "joint_lora": {"epochs": 3, "lr": 1e-6}
}

# LoRA Configuration
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# Memory Optimization
USE_FLASH_ATTENTION = True
GRADIENT_CHECKPOINTING = True
BF16_TRAINING = True

# Inference
INFERENCE_BATCH_SIZE = 1
GUIDANCE_SCALE = 7.5
NUM_INFERENCE_STEPS = 20

# HF Hub Configuration
HF_REPO_ID = "your-username/ExoT2I-9B"  # Update with your HF repo
PUSH_TO_HUB = True