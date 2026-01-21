"""
ExoT2I Model Implementation
QwenFluxForImageGeneration: Unified reasoning-diffusion transformer fusing Qwen3-VL with FLUX.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen3VLForConditionalGeneration, AutoModelForCausalLM
from diffusers import FluxTransformer2DModel, AutoencoderKL
from peft import LoraConfig, get_peft_model
from einops import rearrange
from config import *
from vision_projector import VisionLatentProjector

class DiffusionHead(nn.Module):
    """Predicts noise prediction ε_pred for FLUX denoising"""
    def __init__(self, hidden_dim, latent_dim=16, latent_size=64):
        super().__init__()
        self.latent_size = latent_size
        self.latent_dim = latent_dim

        # Project from transformer hidden states to FLUX latents
        self.proj = nn.Linear(hidden_dim, latent_size * latent_size * latent_dim)

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: (batch, seq_len, hidden_dim) - final layer outputs

        Returns:
            eps_pred: (batch, latent_size, latent_size, latent_dim)
        """
        # Use the last token or average pooling? For diffusion, perhaps average or specific token
        # For simplicity, use mean pooling across sequence
        pooled = hidden_states.mean(dim=1)  # (batch, hidden_dim)
        eps_flat = self.proj(pooled)  # (batch, latent_size*latent_size*latent_dim)
        eps_pred = rearrange(eps_flat, 'b (h w d) -> b h w d',
                           h=self.latent_size, w=self.latent_size, d=self.latent_dim)
        return eps_pred

class EditHead(nn.Module):
    """Predicts sparse edit mask and delta for localized refinements"""
    def __init__(self, hidden_dim, mask_size=32, delta_dim=None):
        super().__init__()
        self.mask_size = mask_size
        self.delta_dim = delta_dim or hidden_dim

        # Mask prediction: downsampled to 32x32
        self.mask_proj = nn.Linear(hidden_dim, mask_size * mask_size)

        # Delta prediction: for selected patches
        self.delta_proj = nn.Linear(hidden_dim, delta_dim)

    def forward(self, hidden_states, num_selected=None):
        """
        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            num_selected: number of patches to select for editing (optional)

        Returns:
            sparse_mask: (batch, mask_size, mask_size)
            edit_delta: (batch, num_selected or mask_size*mask_size, delta_dim)
        """
        # Use last token for prediction
        last_hidden = hidden_states[:, -1]  # (batch, hidden_dim)

        # Predict mask
        mask_flat = self.mask_proj(last_hidden)  # (batch, mask_size*mask_size)
        sparse_mask = rearrange(mask_flat, 'b (h w) -> b h w',
                               h=self.mask_size, w=self.mask_size)

        # For delta, predict for all patches, then select based on mask
        # In practice, select top-k based on mask values
        if num_selected is None:
            num_selected = self.mask_size * self.mask_size

        # Get top-k patches based on mask values
        mask_flat_2d = sparse_mask.view(sparse_mask.size(0), -1)
        _, top_indices = torch.topk(mask_flat_2d, num_selected, dim=1)

        # Predict delta for all, then select
        edit_delta_all = self.delta_proj(hidden_states)  # (batch, seq_len, delta_dim)
        # For simplicity, use mean delta, but ideally per selected patch
        edit_delta = edit_delta_all.mean(dim=1, keepdim=True).expand(-1, num_selected, -1)

        return sparse_mask, edit_delta

class CrossAttentionLayer(nn.Module):
    """Cross-attention to FLUX latents for diffusion conditioning"""
    def __init__(self, hidden_dim, latent_dim=16, num_heads=None):
        super().__init__()
        # Use Qwen's head count or default
        self.num_heads = num_heads or (hidden_dim // 128)
        self.head_dim = hidden_dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(latent_dim, hidden_dim)
        self.v_proj = nn.Linear(latent_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden_states, latents):
        """
        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            latents: (batch, latent_h, latent_w, latent_dim) -> flatten to (batch, latent_seq, latent_dim)

        Returns:
            attended: (batch, seq_len, hidden_dim)
        """
        batch, seq_len, hidden_dim = hidden_states.shape
        latent_seq = latents.shape[1] * latents.shape[2]
        latents_flat = rearrange(latents, 'b h w d -> b (h w) d')

        # Project queries, keys, values
        q = self.q_proj(hidden_states).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(latents_flat).view(batch, latent_seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(latents_flat).view(batch, latent_seq, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)

        attended = torch.matmul(attn_weights, v)
        attended = attended.transpose(1, 2).contiguous().view(batch, seq_len, hidden_dim)

        return self.out_proj(attended)

class QwenFluxForImageGeneration(Qwen3VLForConditionalGeneration):
    """
    Unified model fusing Qwen3-VL reasoning with FLUX diffusion.
    Inherits from Qwen3VLForConditionalGeneration and adds diffusion capabilities.
    """

    def __init__(self, config):
        super().__init__(config)

        # Freeze the language model head to disable text output gradients
        if hasattr(self, 'lm_head'):
            for param in self.lm_head.parameters():
                param.requires_grad = False

        # Load and freeze FLUX VAE
        self.flux_vae = AutoencoderKL.from_pretrained(FLUX_MODEL_NAME, subfolder="vae")
        for param in self.flux_vae.parameters():
            param.requires_grad = False

        # Vision projector (vision_dim varies by model scaling)
        vision_dim = getattr(config, "vision_config", {}).get("hidden_size", 1024) # Typical for 4B/7B
        self.vision_projector = VisionLatentProjector(vision_dim=vision_dim)

        # Cross-attention layers (add to some decoder layers)
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionLayer(hidden_dim) for _ in range(4)  # Add to last 4 layers
        ])

        # Output heads
        hidden_dim = config.hidden_size
        self.diffusion_head = DiffusionHead(hidden_dim)
        self.edit_head = EditHead(hidden_dim)

        # Timestep and iteration embeddings
        self.timestep_embed = nn.Embedding(1000, hidden_dim)
        self.iter_embed = nn.Embedding(NUM_REFINE_ITERS, hidden_dim)

        # Hook into decoder layers for cross-attention
        self._inject_cross_attention()

    def _inject_cross_attention(self):
        """Inject cross-attention layers into the model's decoder blocks"""
        # Distribute 4 layers across the depth of the model
        num_layers = len(self.model.layers)
        self.injection_indices = [
            int(num_layers * 0.25) - 1,
            int(num_layers * 0.5) - 1,
            int(num_layers * 0.75) - 1,
            num_layers - 1
        ]
        # cross_attn_layers are already initialized in __init__

    def add_cross_attention(self, hidden_states, latents, layer_idx):
        """Add cross-attention to FLUX latents at specific layers"""
        if layer_idx >= len(self.model.layers) - len(self.cross_attn_layers):
            cross_idx = layer_idx - (len(self.model.layers) - len(self.cross_attn_layers))
            hidden_states = self.cross_attn_layers[cross_idx](hidden_states, latents)
        return hidden_states

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        images=None,
        latents=None,  # Current FLUX latents z_t
        timestep=None,  # Diffusion timestep
        iter_step=None,  # Reasoning iteration (0-3)
        past_key_values=None,  # For reasoning CoT
        use_cache=True,
        **kwargs
    ):
        """
        Forward pass with reasoning-diffusion fusion and KV cache support.

        Args:
            input_ids: Text token ids
            attention_mask: Attention mask
            images: Input images for vision processing
            latents: Current diffusion latents (batch, 64, 64, 16)
            timestep: Diffusion timestep (0-999)
            iter_step: Reasoning iteration (0-3)

        Returns:
            dict with eps_pred, edit_mask, edit_delta
        """

        # Custom forward pass to intercept layers for cross-attention
        # Iterate through layers manually to apply cross-attention injection
        hidden_states = self.model.embed_tokens(input_ids)

        # Add timestep and iteration tokens to embedding
        if timestep is not None:
            timestep_emb = self.timestep_embed(timestep)
            hidden_states = hidden_states + timestep_emb.unsqueeze(1)
        if iter_step is not None:
            iter_emb = self.iter_embed(iter_step)
            hidden_states = hidden_states + iter_emb.unsqueeze(1)

        next_decoder_cache = None
        all_hidden_states = ()

        for i, decoder_layer in enumerate(self.model.layers):
            all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_values[i] if past_key_values is not None else None,
                use_cache=use_cache,
                **kwargs
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                if next_decoder_cache is None:
                    next_decoder_cache = ()
                next_decoder_cache += (layer_outputs[1],)

            # Apply cross-attention if at injection point
            if latents is not None and i in self.injection_indices:
                cross_idx = self.injection_indices.index(i)
                hidden_states = self.cross_attn_layers[cross_idx](hidden_states, latents)

        hidden_states = self.model.norm(hidden_states)
        all_hidden_states += (hidden_states,)

        # Predict diffusion noise
        eps_pred = self.diffusion_head(hidden_states)

        # Predict edit mask and delta
        edit_mask, edit_delta = self.edit_head(hidden_states)

        return {
            'eps_pred': eps_pred,
            'edit_mask': edit_mask,
            'edit_delta': edit_delta,
            'hidden_states': hidden_states,
            'past_key_values': next_decoder_cache
        }

    @classmethod
    def from_pretrained(cls, model_name, **kwargs):
        """Load model with LoRA applied"""
        model = super().from_pretrained(model_name, **kwargs)

        # Apply LoRA to transformer trunk
        lora_config = LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)

        return model

def test_model():
    """Basic test of model components"""
    from transformers import AutoTokenizer

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME)

    # Create model
    model = QwenFluxForImageGeneration.from_pretrained(QWEN_MODEL_NAME)

    # Test text input
    text = "A beautiful landscape"
    inputs = tokenizer(text, return_tensors="pt")
    latents = torch.randn(1, 64, 64, 16)
    timestep = torch.tensor([500])
    iter_step = torch.tensor([0])

    outputs = model(**inputs, latents=latents, timestep=timestep, iter_step=iter_step)

    print(f"eps_pred shape: {outputs['eps_pred'].shape}")  # (1, 64, 64, 16)
    print(f"edit_mask shape: {outputs['edit_mask'].shape}")  # (1, 32, 32)
    print(f"edit_delta shape: {outputs['edit_delta'].shape}")  # (1, N, 4096)

if __name__ == "__main__":
    test_model()