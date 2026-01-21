"""
Data Generation for ExoT2I Training
Generates 1M FLUX trajectories with noisy→clean latents and synthetic captions.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from diffusers import FluxPipeline
from transformers import CLIPTokenizer
import os
import json
from tqdm import tqdm
from config import *

class FLUXTrajectoryDataset(Dataset):
    """Dataset of FLUX diffusion trajectories for training"""

    def __init__(self, num_trajectories=NUM_TRAJECTORIES, split='train'):
        self.num_trajectories = num_trajectories
        self.data_dir = DATA_DIR / f"{split}_trajectories"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Generate data if not exists
        if not self._data_exists():
            self._generate_data()

        # Load trajectories
        self.trajectories = self._load_trajectories()

    def _data_exists(self):
        return len(list(self.data_dir.glob("*.pt"))) >= self.num_trajectories

    def _generate_data(self):
        """Generate synthetic FLUX trajectories"""
        print(f"Generating {self.num_trajectories} trajectories...")

        # Load FLUX pipeline for latent generation
        pipeline = FluxPipeline.from_pretrained(FLUX_MODEL_NAME, torch_dtype=torch.float16)
        pipeline.vae.requires_grad_(False)
        pipeline.text_encoder.requires_grad_(False)
        pipeline.transformer.requires_grad_(False)

        # Load captions (from prompt.json if it exists, else use GOLDEN_25)
        prompt_json_path = PROJECT_ROOT / "prompt.json"
        if prompt_json_path.exists():
            print(f"Loading expanded prompts from {prompt_json_path}")
            with open(prompt_json_path, "r") as f:
                captions = json.load(f)
        else:
            print(f"Using default golden prompts (expanded prompts not found)")
            captions = GOLDEN_25
            
        captions = captions * (self.num_trajectories // len(captions) + 1)

        # Initialize tokenizer
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

        # Check for GPU (often not present on TPU VMs)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            pipeline.to(device)

        for i in tqdm(range(self.num_trajectories)):
            # Random caption
            caption = captions[i % len(captions)]
            
            # Generate valid target latent (z_0) for the prompt
            # Using 1 step or low guidance for speed in ground truth generation
            with torch.no_grad():
                # Get the "clean" result from FLUX as target
                clean_latent = pipeline(
                    caption, 
                    num_inference_steps=2, # Fast ground truth
                    output_type="latent"
                ).images[0] # (16, 64, 64)
            
            # Generate target noise
            epsilon = torch.randn_like(clean_latent)

            # Generate trajectory: linear interpolation (Rectified Flow)
            t_values = torch.linspace(0, 1, NUM_DIFFUSION_STEPS)

            trajectory = []
            for t_val in t_values:
                # z_t = (1 - t) * clean + t * noise
                noisy_latent = (1.0 - t_val) * clean_latent + t_val * epsilon
                trajectory.append({
                    'timestep': int(t_val * 1000),
                    'noisy_latent': noisy_latent,
                    'clean_latent': clean_latent,
                    'velocity': (epsilon - clean_latent)
                })

            # Random caption
            caption = captions[i % len(captions)]
            caption_tokens = tokenizer(caption, padding="max_length", max_length=77,
                                     truncation=True, return_tensors="pt")['input_ids'].squeeze()

            # Save trajectory
            data = {
                'trajectory': trajectory,
                'caption': caption,
                'caption_tokens': caption_tokens
            }

            torch.save(data, self.data_dir / f"trajectory_{i:06d}.pt")

    def _load_trajectories(self):
        """Load all trajectories into memory (for small datasets)"""
        trajectories = []
        for file in sorted(self.data_dir.glob("*.pt")):
            trajectories.append(torch.load(file))
        return trajectories

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        return self.trajectories[idx]

def create_trajectory_dataloader(batch_size=1, num_trajectories=10000):
    """Create DataLoader for trajectory training"""
    dataset = FLUXTrajectoryDataset(num_trajectories)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return dataloader

def generate_synthetic_images(num_images=1000):
    """Generate synthetic images using FLUX for testing"""
    pipeline = FluxPipeline.from_pretrained(FLUX_MODEL_NAME, torch_dtype=torch.float16)
    pipeline.enable_model_cpu_offload()

    prompts = ["A beautiful landscape"] * num_images

    images = []
    for prompt in tqdm(prompts):
        image = pipeline(prompt, num_inference_steps=20, guidance_scale=7.5).images[0]
        images.append(image)

    # Save images
    output_dir = DATA_DIR / "synthetic_images"
    output_dir.mkdir(exist_ok=True)

    for i, img in enumerate(images):
        img.save(output_dir / f"image_{i:04d}.png")

if __name__ == "__main__":
    # Generate small dataset for testing
    dataset = FLUXTrajectoryDataset(num_trajectories=100)
    print(f"Generated {len(dataset)} trajectories")

    # Test dataloader
    dataloader = create_trajectory_dataloader(batch_size=1, num_trajectories=10)
    for batch in dataloader:
        print(f"Batch keys: {batch[0].keys()}")
        print(f"Trajectory length: {len(batch[0]['trajectory'])}")
        break