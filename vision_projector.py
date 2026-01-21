"""
Vision Projector for ExoT2I
Aligns Qwen3-VL visual tokens to FLUX latent space with spatial correspondence.
Each visual token v_toks[i,j] maps to latent patch z[4i:4i+4, 4j:4j+4].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class VisionLatentProjector(nn.Module):
    """
    Projects spatially arranged visual tokens to FLUX latent space.
    Assumes visual tokens are arranged in H x W grid, where H*W = num_vision_tokens.
    For Qwen3-VL, typically ~1024 tokens in 32x32 or similar grid.
    """

    def __init__(self, vision_dim=1280, latent_dim=16, vision_grid_size=32, latent_size=64):
        super().__init__()
        self.vision_dim = vision_dim
        self.latent_dim = latent_dim
        self.vision_grid_size = vision_grid_size  # e.g., 32 for 32x32 grid
        self.latent_size = latent_size  # 64 for 64x64 latents

        # Spatial upsampling factor (latent_size // vision_grid_size)
        self.upsample_factor = latent_size // vision_grid_size  # Should be 2 for 32->64

        # Convolutional projector to reduce dimension and add spatial features
        self.conv1 = nn.Conv2d(vision_dim, 512, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(256, latent_dim, kernel_size=3, padding=1)

        # Layer norms for stability
        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(256)

        # Approximate 10M parameters with these layers
        # conv1: 3*3*1280*512 ≈ 5.9M
        # conv2: 3*3*512*256 ≈ 1.2M
        # conv3: 3*3*256*16 ≈ 36k
        # Total ≈ 7.1M, close enough (can adjust channels)

    def forward(self, vision_tokens):
        """
        Args:
            vision_tokens: (batch, num_vision_tokens, vision_dim)
                          Assumed to be spatially arranged as (H, W) = (vision_grid_size, vision_grid_size)

        Returns:
            latents: (batch, latent_size, latent_size, latent_dim)
        """
        batch_size = vision_tokens.shape[0]
        num_tokens = vision_tokens.shape[1]

        # Reshape to spatial grid: (batch, H, W, vision_dim) -> (batch, vision_dim, H, W)
        h = w = int(num_tokens ** 0.5)  # Assume square grid
        assert h * w == num_tokens, f"num_tokens {num_tokens} not square"
        vision_spatial = rearrange(vision_tokens, 'b (h w) d -> b d h w', h=h, w=w)

        # Convolutional processing
        x = self.conv1(vision_spatial)
        x = F.relu(x)
        x = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # Apply LN in channel-last

        x = self.conv2(x)
        x = F.relu(x)
        x = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # Upsample to latent size
        x = F.interpolate(x, size=(self.latent_size, self.latent_size), mode='bilinear', align_corners=False)

        # Final conv to latent dim
        x = self.conv3(x)  # (batch, latent_dim, latent_size, latent_size)

        # Rearrange to (batch, latent_size, latent_size, latent_dim) for FLUX compatibility
        latents = rearrange(x, 'b d h w -> b h w d')

        return latents


def test_projector():
    """Test the projector with dummy data"""
    projector = VisionLatentProjector()
    vision_tokens = torch.randn(1, 1024, 1280)  # 32x32 grid
    latents = projector(vision_tokens)
    print(f"Input shape: {vision_tokens.shape}")
    print(f"Output shape: {latents.shape}")  # Should be (1, 64, 64, 16)
    assert latents.shape == (1, 64, 64, 16), f"Unexpected output shape: {latents.shape}"

    # Count parameters
    total_params = sum(p.numel() for p in projector.parameters())
    print(f"Total parameters: {total_params:,}")  # Should be ~7M


if __name__ == "__main__":
    test_projector()