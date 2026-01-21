"""
ExoT2I Merging and Upload Script
Fuses LoRA weights into the base models and uploads the final model to Hugging Face.
"""

import torch
from huggingface_hub import HfApi, create_repo
from peft import PeftModel
from model import QwenFluxForImageGeneration
from config import *

def merge_and_upload():
    print(f"Loading base model from {QWEN_MODEL_NAME}...")
    
    # 1. Load the fused architecture
    model = QwenFluxForImageGeneration.from_pretrained(
        QWEN_MODEL_NAME, 
        torch_dtype=torch.bfloat16,
        device_map="cpu"
    )
    
    # 2. Load the LoRA weights from the final stage
    lora_path = CHECKPOINT_DIR / "joint_lora"
    if lora_path.exists():
        print(f"Merging LoRA weights from {lora_path}...")
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()
    else:
        print(f"WARNING: No LoRA weights found at {lora_path}. Saving base architecture only.")

    # 3. Save the fully fused model locally
    final_output_path = PROJECT_ROOT / "models" / "final" / "exo"
    final_output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving fused model to {final_output_path}...")
    model.save_pretrained(final_output_path)
    
    # 4. Upload to Hugging Face
    if PUSH_TO_HUB:
        print(f"Uploading to Hugging Face Hub: {HF_REPO_ID}...")
        try:
            api = HfApi()
            create_repo(repo_id=HF_REPO_ID, repo_type="model", exist_ok=True)
            api.upload_folder(
                folder_path=str(final_output_path),
                repo_id=HF_REPO_ID,
                repo_type="model"
            )
            print(f"Successfully uploaded to https://huggingface.co/{HF_REPO_ID}")
        except Exception as e:
            print(f"Upload failed: {e}")
            print("Make sure you are logged in via 'huggingface-cli login'")

if __name__ == "__main__":
    merge_and_upload()
