# ExoT2I: Unified Reasoning-Diffusion Transformer

A complete implementation of "Exo" - a unified reasoning-diffusion transformer that fuses Qwen3-VL-8B-Thinking with FLUX.2-klein-base-9B into a single 9.1B parameter model for internal reasoning-driven image refinement.

## Architecture Overview

```
Input: Text prompt + optional init image
↓
Qwen3-VL Vision Tower (DeepStack ViT + Interleaved-MRoPE) → Visual tokens (256-1280, spatially indexed)
↓
Shared Transformer Decoder (Qwen3-VL-8B trunk, 80 layers, LoRA trainable)
  ├── Causal Self-Attention (text_tokens + v_toks + timestep_token + iter_token)
  ├── Cross-Attention to current FLUX latents z_t (64×64×16)
  ├── KV cache for internal reasoning across T=4 refinement steps
  └── NO text output gradients (disable lm_head)
↓ Parallel Heads (1.1B total params):
  ├── Diffusion Head: Predicts ε_pred ∈ ℝ^(64×64×16) for FLUX denoising
  ├── Edit Head: Predicts sparse_mask ∈ ℝ^(32×32) + edit_delta ∈ ℝ^(N_selected×4096)
↓
FLUX.2-klein VAE (frozen): z → 1024×1024 RGB
```

## Key Features

- **Unified Model**: Single 9.1B parameter model combining reasoning and diffusion
- **Internal Reasoning**: 4 reasoning iterations per diffusion step with KV cache persistence
- **Spatial Alignment**: Vision tokens mapped to FLUX latent patches (4x upsampling)
- **Sparse Editing**: L1-regularized edit masks for localized refinements
- **Memory Optimized**: Flash Attention 2, gradient checkpointing, 32GB VRAM target
- **3-Stage Training**: Curriculum learning from projector → diffusion → joint LoRA

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Generate Images

```python
from inference import ExoT2IInference

generator = ExoT2IInference()
image = generator.generate(
    "A beautiful landscape with mountains and lakes",
    height=1024,
    width=1024,
    steps=20,
    refine_iters=4
)
image.save("output.png")
```

### Interactive Demo

```bash
python demo_gradio.py
```

Open http://localhost:7860 for the Gradio interface.

### Training

```bash
# Generate training data
python generate_data.py

# Train model (3 stages)
./train.sh
```

## File Structure

```
ExoT2I/
├── model.py              # QwenFluxForImageGeneration + heads
├── vision_projector.py   # v_toks ↔ FLUX latent alignment
├── training.py           # 3-stage training loop + dataloader
├── inference.py          # Single-pass generation
├── generate_data.py      # FLUX trajectory dataset creation
├── config.py             # Hyperparams, paths
├── requirements.txt
├── train.sh              # DeepSpeed launch script
├── demo_gradio.py        # Gradio UI for testing
└── README.md
```

## Technical Specifications

### Model Components
- **Backbone**: Qwen/Qwen3-VL-8B-Thinking (8B parameters)
- **Diffusion Model**: black-forest-labs/FLUX.2-klein-base-9B VAE only
- **Vision Projector**: 10M parameters for spatial alignment
- **Output Heads**: 1.1B parameters (diffusion + edit)
- **LoRA**: 16M trainable parameters on transformer trunk

### Training
- **Dataset**: 25K synthetic FLUX trajectories
- **Loss Weights**: 0.9 diffusion + 0.1 edit recon + 0.05 perceptual
- **Stages**: Projector (2 epochs) → Diffusion Head (5 epochs) → Joint LoRA (3 epochs)
- **Optimization**: DeepSpeed ZeRO-3, bfloat16, AdamW

### Inference
- **Sampling**: Rectified flow DDIM with 20 steps
- **Reasoning**: 4 internal iterations per diffusion step
- **Resolution**: 1024×1024 pixels
- **Performance**: Target 3 seconds on A100

## Memory Optimization

- Flash Attention 2 for efficient attention computation
- Gradient checkpointing during training
- KV cache reuse across reasoning iterations
- Parameter offloading with DeepSpeed ZeRO-3
- bfloat16 mixed precision training

## Research Compliance

- FLUX non-commercial license compliant
- Research use only
- No external data dependencies (synthetic dataset generation)
- Open-source implementation

## Citation

If you use this code, please cite:

```bibtex
@misc{exo-t2i-2024,
  title={ExoT2I: Unified Reasoning-Diffusion Transformer},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/ExoT2I}
}
```

## License

MIT License - see LICENSE file for details.