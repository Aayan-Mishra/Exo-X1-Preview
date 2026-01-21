"""
ExoT2I Inference Script
Single-pass generation with 20 diffusion × 4 reasoning steps.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from diffusers import FluxPipeline, AutoencoderKL
from einops import rearrange
import numpy as np
from PIL import Image
from config import *
from model import QwenFluxForImageGeneration

class ExoT2IInference:
    """Inference pipeline for ExoT2I"""

    def __init__(self, model_path=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load model with bfloat16
        self.model = QwenFluxForImageGeneration.from_pretrained(
            QWEN_MODEL_NAME, 
            torch_dtype=torch.bfloat16 if BF16_TRAINING else torch.float16
        )
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME)

        # Load FLUX VAE with bfloat16
        self.vae = AutoencoderKL.from_pretrained(FLUX_MODEL_NAME, subfolder="vae", torch_dtype=torch.bfloat16 if BF16_TRAINING else torch.float16)
        self.vae.to(self.device)
        self.vae.eval()

        # Flow Matcher Scheduler (Better for Flux than DDIM)
        from diffusers import FlowMatchEulerDiscreteScheduler
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(FLUX_MODEL_NAME, subfolder="scheduler")

    def prepare_inputs(self, prompt, init_image=None, height=1024, width=1024):
        """Prepare inputs for generation"""
        # Append thinking block if prompt doesn't have it
        if "thought" not in prompt.lower() and "thinking" in QWEN_MODEL_NAME.lower():
            prompt = f"<|thought|>\nI will reason about the visual composition: {prompt}"

        # Tokenize prompt
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=128, # Increased for thinking chains
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        # Process init image if provided
        init_latents = None
        if init_image:
            # Convert PIL to tensor and encode with VAE
            init_tensor = torch.from_numpy(np.array(init_image)).float() / 127.5 - 1.0
            init_tensor = rearrange(init_tensor, 'h w c -> c h w').unsqueeze(0).to(self.device)
            if BF16_TRAINING: init_tensor = init_tensor.to(torch.bfloat16)
            with torch.no_grad():
                init_latents = self.vae.encode(init_tensor).latent_dist.sample()
                init_latents = init_latents * self.vae.config.scaling_factor

        return text_inputs, init_latents

    @torch.no_grad()
    def generate(
        self,
        prompt,
        height=1024,
        width=1024,
        steps=20,
        refine_iters=4,
        guidance_scale=7.5,
        init_image=None,
        seed=None
    ):
        """
        Generate image with reasoning-diffusion fusion.
        """

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Prepare inputs
        text_inputs, init_latents = self.prepare_inputs(prompt, init_image, height, width)

        # Initialize latents
        latent_shape = (1, 16, height // 16, width // 16)  # FLUX latent shape
        latents = init_latents if init_image else torch.randn(latent_shape, device=self.device, dtype=torch.bfloat16 if BF16_TRAINING else torch.float16)

        # Prepare timesteps
        self.scheduler.set_timesteps(steps)
        timesteps = self.scheduler.timesteps

        # Persistent KV cache across both diffusion and reasoning steps
        kv_cache = None

        # Generation loop
        for step_idx, t in enumerate(timesteps):
            print(f"Diffusion Step {step_idx+1}/{len(timesteps)}, t={t.item():.3f}")

            # Multiple reasoning iterations per diffusion step
            for iter_step in range(refine_iters):
                # Current latents for cross-attention
                current_latents = latents.permute(0, 2, 3, 1)  # (1, 64, 64, 16)

                # Forward pass with reasoning and KV cache
                outputs = self.model(
                    input_ids=text_inputs['input_ids'],
                    attention_mask=text_inputs['attention_mask'],
                    latents=current_latents,
                    timestep=t.unsqueeze(0).to(self.device),
                    iter_step=torch.tensor([iter_step]).to(self.device),
                    past_key_values=kv_cache,
                    use_cache=True
                )

                # Update KV cache
                kv_cache = outputs['past_key_values']

                # Get noise prediction
                eps_pred = outputs['eps_pred'].permute(0, 3, 1, 2)  # (1, 16, 64, 64)

                # Apply edit if init_image provided
                if init_latents is not None:
                    edit_mask = outputs['edit_mask']  # (1, 32, 32)
                    edit_delta = outputs['edit_delta']  # (1, 1024, 4096) -> simplified usage

                    # Apply sparse edits
                    # Resize mask to latent size (64x64) and apply delta to noise prediction
                    mask_upsampled = F.interpolate(edit_mask.unsqueeze(1), size=(64, 64), mode='bilinear')
                    # This is a conceptual application: reasoning influences the noise prediction
                    eps_pred = eps_pred * (1 + mask_upsampled)

            # Scheduler step
            latents = self.scheduler.step(eps_pred, t, latents).prev_sample

        # Decode latents to image
        latents = latents / self.vae.config.scaling_factor
        with torch.no_grad():
            image = self.vae.decode(latents).sample

        # Convert to PIL
        image = (image / 2 + 0.5).clamp(0, 1)
        image = rearrange(image, 'b c h w -> b h w c').cpu().numpy()[0]
        image = (image * 255).astype(np.uint8)
        pil_image = Image.fromarray(image)

        return pil_image

def main():
    # Test inference
    generator = ExoT2IInference()

    prompt = "A beautiful landscape with mountains and lakes, photorealistic"
    image = generator.generate(
        prompt,
        height=1024,
        width=1024,
        steps=20,
        refine_iters=4,
        seed=42
    )

    # Save image
    image.save("generated_image.png")
    print("Image saved as generated_image.png")

if __name__ == "__main__":
    main()