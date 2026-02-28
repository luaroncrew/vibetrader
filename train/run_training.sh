#!/usr/bin/env bash
# Launch InstructPix2Pix fine-tuning on the vibetrader chart dataset.
#
# Prerequisites:
#   pip install -r requirements.txt
#   pip install git+https://github.com/huggingface/diffusers.git
#   wandb login
#
# Usage:
#   bash train/run_training.sh [--colab]  # pass --colab for Colab/CUDA settings

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Locate the training script from diffusers
TRAIN_SCRIPT=""
# Check common locations
for candidate in \
    "${PROJECT_DIR}/diffusers/examples/instruct_pix2pix/train_instruct_pix2pix.py" \
    "$(python3 -c 'import diffusers, os; print(os.path.dirname(diffusers.__file__))' 2>/dev/null)/../examples/instruct_pix2pix/train_instruct_pix2pix.py" \
    ; do
    if [[ -f "$candidate" ]]; then
        TRAIN_SCRIPT="$candidate"
        break
    fi
done

if [[ -z "$TRAIN_SCRIPT" ]]; then
    echo "Training script not found. Cloning diffusers repo..."
    cd "$PROJECT_DIR"
    git clone --depth 1 https://github.com/huggingface/diffusers.git
    TRAIN_SCRIPT="${PROJECT_DIR}/diffusers/examples/instruct_pix2pix/train_instruct_pix2pix.py"
fi

echo "Using training script: $TRAIN_SCRIPT"

# Default: MPS (Apple Silicon) settings
MIXED_PRECISION="no"  # MPS doesn't support fp16 training well in all cases
EXTRA_ARGS=""

# Override for CUDA (Colab / cloud GPU)
if [[ "${1:-}" == "--colab" ]]; then
    MIXED_PRECISION="fp16"
    EXTRA_ARGS=""
    echo "Running in CUDA/Colab mode with fp16"
else
    echo "Running in MPS (Apple Silicon) mode"
fi

# W&B project
export WANDB_PROJECT="vibetrader"

accelerate launch "$TRAIN_SCRIPT" \
    --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
    --dataset_name="${PROJECT_DIR}/data/dataset" \
    --original_image_column="original_image" \
    --edited_image_column="edited_image" \
    --edit_prompt_column="edit_prompt" \
    --resolution=256 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=8 \
    --gradient_checkpointing \
    --max_train_steps=2000 \
    --learning_rate=5e-05 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=100 \
    --mixed_precision="$MIXED_PRECISION" \
    --report_to=wandb \
    --checkpointing_steps=500 \
    --output_dir="${PROJECT_DIR}/checkpoints" \
    --seed=42 \
    $EXTRA_ARGS
