#!/bin/bash
# ExoT2I Training Launch Script
# DeepSpeed ZeRO-3 with bfloat16 for 32GB VRAM target

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# DeepSpeed configuration
cat > ds_config.json << EOF
{
  "fp16": {
    "enabled": false
  },
  "bf16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9
  },
  "gradient_accumulation_steps": 4,
  "train_micro_batch_size_per_gpu": 1,
  "gradient_clipping": 1.0,
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": "auto",
      "weight_decay": 0.01
    }
  },
  "scheduler": {
    "type": "WarmupDecayLR",
    "params": {
      "warmup_min_lr": "auto",
      "warmup_max_lr": "auto",
      "warmup_num_steps": "auto",
      "total_num_steps": "auto"
    }
  },
  "activation_checkpointing": {
    "partition_activations": true,
    "cpu_checkpointing": true,
    "contiguous_memory_optimization": false,
    "number_checkpoints": null,
    "synchronize_checkpoint_boundary": false,
    "profile": false
  },
  "flops_profiler": {
    "enabled": false,
    "profile_step": 1,
    "module_depth": -1,
    "top_modules": 1,
    "detailed": true,
    "output_file": null
  }
}
EOF

# Launch training
deepspeed --num_gpus=1 training.py \
  --deepspeed ds_config.json \
  --deepspeed_config ds_config.json

echo "Training launched with DeepSpeed ZeRO-3"