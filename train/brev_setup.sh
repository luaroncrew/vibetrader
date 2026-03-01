#!/usr/bin/env bash
# One-time setup script for the Brev GPU instance.
# Run this ONCE after rsync-ing the project:
#   bash ~/vibetrader/train/brev_setup.sh

set -euo pipefail

PROJECT_DIR="$HOME/vibetrader"
cd "$PROJECT_DIR"

echo "=== Setting up vibetrader on Brev ==="

# Ensure user-installed binaries are on PATH
export PATH="$HOME/.local/bin:$PATH"

# Load env vars
if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
    echo "Loaded .env"
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r train/requirements-gpu.txt

# Clone diffusers and install training dependencies
if [[ ! -d "$PROJECT_DIR/diffusers" ]]; then
    echo "Cloning diffusers..."
    git clone --depth 1 https://github.com/huggingface/diffusers.git "$PROJECT_DIR/diffusers"
fi

echo "Installing instruct_pix2pix training requirements..."
EXAMPLE_DIR="$PROJECT_DIR/diffusers/examples/instruct_pix2pix"
if [[ -f "$EXAMPLE_DIR/requirements.txt" ]]; then
    pip install -r "$EXAMPLE_DIR/requirements.txt"
fi

# Login to wandb
if [[ -n "${WANDB_API_KEY:-}" ]]; then
    echo "Logging into W&B..."
    wandb login "$WANDB_API_KEY"
fi

# Login to HuggingFace
if [[ -n "${HF_TOKEN:-}" ]]; then
    echo "Logging into HuggingFace..."
    huggingface-cli login --token "$HF_TOKEN"
fi

# Verify GPU
echo ""
echo "=== GPU Check ==="
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU:             {torch.cuda.get_device_name(0)}')
    mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f'VRAM:            {mem:.1f} GB')
"

echo ""
echo "=== Setup complete ==="
echo "Run training with:  bash ~/vibetrader/train/run_training_gpu.sh"
