#!/usr/bin/env bash
# Launch InstructPix2Pix fine-tuning on a GPU instance (H200/A100/A10G).
#
# Usage:
#   bash ~/vibetrader/train/run_training_gpu.sh

set -euo pipefail

PROJECT_DIR="$HOME/vibetrader"
cd "$PROJECT_DIR"

export PATH="$HOME/.local/bin:$PATH"

# Load env vars
if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

# Locate training script
TRAIN_SCRIPT="$PROJECT_DIR/diffusers/examples/instruct_pix2pix/train_instruct_pix2pix.py"
if [[ ! -f "$TRAIN_SCRIPT" ]]; then
    echo "ERROR: Training script not found at $TRAIN_SCRIPT"
    echo "Run  bash ~/vibetrader/train/brev_setup.sh  first."
    exit 1
fi

# Auto-detect GPU memory and set batch size accordingly
GPU_MEM_GB=$(python3 -c "
import torch
if torch.cuda.is_available():
    print(int(torch.cuda.get_device_properties(0).total_memory / 1e9))
else:
    print(0)
" 2>/dev/null || echo "0")

if [[ "$GPU_MEM_GB" -ge 80 ]]; then
    # H200 / A100-80GB: aggressive settings
    BATCH_SIZE=4
    GRAD_ACCUM=4
    echo "Detected ${GPU_MEM_GB}GB VRAM -> batch_size=$BATCH_SIZE, grad_accum=$GRAD_ACCUM (effective batch 16)"
elif [[ "$GPU_MEM_GB" -ge 40 ]]; then
    # A100-40GB
    BATCH_SIZE=2
    GRAD_ACCUM=8
    echo "Detected ${GPU_MEM_GB}GB VRAM -> batch_size=$BATCH_SIZE, grad_accum=$GRAD_ACCUM (effective batch 16)"
elif [[ "$GPU_MEM_GB" -ge 16 ]]; then
    # A10G / T4
    BATCH_SIZE=1
    GRAD_ACCUM=8
    echo "Detected ${GPU_MEM_GB}GB VRAM -> batch_size=$BATCH_SIZE, grad_accum=$GRAD_ACCUM (effective batch 8)"
else
    echo "WARNING: No GPU detected or <16GB VRAM. Using minimal settings."
    BATCH_SIZE=1
    GRAD_ACCUM=8
fi

# W&B
export WANDB_PROJECT="vibetrader"
export WANDB_RESUME="allow"
export WANDB_RUN_ID="gpu-$(date +%s)"

echo ""
echo "=== Starting training ==="
echo "Model:    stable-diffusion-v1-5"
echo "Dataset:  $PROJECT_DIR/data/dataset"
echo "Output:   $PROJECT_DIR/checkpoints"
echo "W&B run:  $WANDB_RUN_ID"
echo ""

accelerate launch "$TRAIN_SCRIPT" \
    --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
    --dataset_name="$PROJECT_DIR/data/dataset" \
    --original_image_column="original_image" \
    --edited_image_column="edited_image" \
    --edit_prompt_column="edit_prompt" \
    --resolution=256 \
    --train_batch_size="$BATCH_SIZE" \
    --gradient_accumulation_steps="$GRAD_ACCUM" \
    --gradient_checkpointing \
    --max_train_steps=2000 \
    --learning_rate=5e-05 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=100 \
    --mixed_precision="fp16" \
    --report_to=wandb \
    --checkpointing_steps=250 \
    --output_dir="$PROJECT_DIR/checkpoints" \
    --seed=42 \
    --resume_from_checkpoint="latest"
