"""
ExoT2I Training Script (TPU Optimized)
3-stage curriculum training: projector → diffusion head → joint LoRA
Target: Google Cloud TPU v5e-1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_scheduler
import wandb
from tqdm import tqdm
import lpips
from config import *
from model import QwenFluxForImageGeneration
from generate_data import FLUXTrajectoryDataset

# TPU Imports
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

class ExoT2ILoss(nn.Module):
    """Combined loss for ExoT2I training (Device Agnostic)"""

    def __init__(self, device):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        # LPIPS is heavy on TPU, but v5e can handle it in bfloat16
        self.lpips_loss = lpips.LPIPS(net='vgg').to(device)
        self.device = device

    def forward(self, outputs, targets):
        """
        Args:
            outputs: dict with eps_pred, edit_mask, edit_delta
            targets: dict with clean_latent, noisy_latent, target_vel, timestep
        """

        # Diffusion loss: MSE between predicted velocity and target velocity
        v_pred = outputs['eps_pred'].permute(0, 3, 1, 2)  # (batch, 16, 64, 64)
        target_vel = targets['target_vel']  # (batch, 16, 64, 64)
        
        # Ensure bfloat16 for TPU efficiency
        if BF16_TRAINING:
            v_pred = v_pred.to(torch.bfloat16)
            target_vel = target_vel.to(torch.bfloat16)

        diffusion_loss = self.mse_loss(v_pred, target_vel)

        # Edit reconstruction loss
        edit_mask = outputs['edit_mask']  # (batch, 32, 32)
        mask_sparsity_loss = self.l1_loss(edit_mask, torch.zeros_like(edit_mask))

        # Perceptual loss
        perceptual_loss = self.lpips_loss(v_pred.float(), target_vel.float()).mean()

        # Combined loss
        total_loss = (
            LOSS_DIFFUSION_WEIGHT * diffusion_loss +
            LOSS_EDIT_RECON_WEIGHT * (0.1 * mask_sparsity_loss) +
            LOSS_PERCEPTUAL_WEIGHT * perceptual_loss
        )

        loss_dict = {
            'total': total_loss.item(),
            'diffusion': diffusion_loss.item(),
            'mask_sparsity': mask_sparsity_loss.item(),
            'perceptual': perceptual_loss.item()
        }

        return total_loss, loss_dict

def train_stage(model, dataloader, optimizer, scheduler, loss_fn, stage_name, num_epochs, device):
    """Train for one stage on TPU"""
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0
        # Wrap dataloader for XLA
        para_loader = pl.ParallelLoader(dataloader, [device])
        progress_bar = tqdm(para_loader.per_device_loader(device), desc=f"{stage_name} Epoch {epoch+1}")

        for batch in progress_bar:
            # Prepare batch data
            caption_tokens = batch['caption_tokens'].to(device)
            trajectories = batch['trajectory']
            
            # Batch preparation (Logic remains same, device handled by parallel loader)
            noisy_latents = []
            target_vels = []
            timesteps = []
            iter_steps = []
            
            for i in range(len(caption_tokens)):
                traj = trajectories[i]
                t_idx = torch.randint(0, len(traj), (1,)).item()
                step_data = traj[t_idx]
                
                noisy_latents.append(step_data['noisy_latent'])
                target_vels.append(step_data['velocity'])
                timesteps.append(step_data['timestep'])
                iter_steps.append(torch.randint(0, NUM_REFINE_ITERS, (1,)).item())

            noisy_latents = torch.stack(noisy_latents).to(device)
            target_vels = torch.stack(target_vels).to(device)
            timesteps = torch.tensor(timesteps).to(device)
            iter_steps = torch.tensor(iter_steps).to(device)

            if BF16_TRAINING:
                noisy_latents = noisy_latents.to(torch.bfloat16)
                target_vels = target_vels.to(torch.bfloat16)

            # Forward pass
            outputs = model(
                input_ids=caption_tokens,
                latents=noisy_latents.permute(0, 2, 3, 1),
                timestep=timesteps,
                iter_step=iter_steps
            )

            # Compute loss
            targets = {
                'target_vel': target_vels,
                'timestep': timesteps
            }
            loss, loss_dict = loss_fn(outputs, targets)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # XLA Optimizer Step
            xm.optimizer_step(optimizer)
            scheduler.step()

            epoch_loss += loss.item()
            if xm.is_master_ordinal():
                progress_bar.set_postfix(loss=loss.item())
                if wandb.run:
                    wandb.log(loss_dict)

        if xm.is_master_ordinal():
            print(f"{stage_name} Epoch {epoch+1} Average Loss: {epoch_loss / len(dataloader):.4f}")

def setup_model_for_stage(model, stage):
    """Configure model for specific training stage"""
    for param in model.parameters():
        param.requires_grad = False

    if stage == "projector":
        for name, param in model.named_parameters():
            if "vision_projector" in name:
                param.requires_grad = True
    elif stage == "diffusion_head":
        for name, param in model.named_parameters():
            if "diffusion_head" in name or "vision_projector" in name:
                param.requires_grad = True
    elif stage == "joint_lora":
        for name, param in model.named_parameters():
            if any(k in name for k in ["lora", "diffusion_head", "edit_head"]):
                param.requires_grad = True

def _mp_fn(index, flags):
    """Multi-processing function for TPU cores"""
    device = xm.xla_device()
    
    # Initialize wandb only on master
    if xm.is_master_ordinal():
        wandb.init(project="ExoT2I", name="TPU-v5e-Training")

    # Load model on TPU
    model = QwenFluxForImageGeneration.from_pretrained(QWEN_MODEL_NAME)
    if BF16_TRAINING:
        model = model.to(torch.bfloat16)
    model = model.to(device)

    # Dataset
    dataset = FLUXTrajectoryDataset(num_trajectories=1000)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=0 # TPU workers handled by ParallelLoader
    )

    # Loss
    loss_fn = ExoT2ILoss(device)

    # Training Loop
    for stage, config in STAGES.items():
        if xm.is_master_ordinal():
            print(f"\n=== Starting {stage} stage on TPU ===")

        setup_model_for_stage(model, stage)

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config['lr'],
            weight_decay=0.01
        )

        num_training_steps = len(dataloader) * config['epochs']
        scheduler = get_scheduler(
            "cosine",
            optimizer=optimizer,
            num_warmup_steps=WARMUP_STEPS,
            num_training_steps=num_training_steps
        )

        train_stage(model, dataloader, optimizer, scheduler, loss_fn, stage, config['epochs'], device)

        # Save checkpoint (Master only)
        if xm.is_master_ordinal():
            checkpoint_dir = CHECKPOINT_DIR / stage
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            xm.save(model.state_dict(), checkpoint_dir / "model.pt")

    if xm.is_master_ordinal():
        print("Training completed on TPU!")
        if PUSH_TO_HUB:
            from merge_and_upload import merge_and_upload
            merge_and_upload()

if __name__ == "__main__":
    # For v5e-1, we can just run on 1 core or use xmp.spawn(8) for pods.
    # We use xmp.spawn for robustness across TPU configurations.
    flags = {}
    xmp.spawn(_mp_fn, args=(flags,), nprocs=1, start_method='fork')