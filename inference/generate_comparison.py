"""Generate model comparison data across checkpoints.

Auto-discovers checkpoints, runs each model on test images, and produces
outputs/comparisons/comparison.json + image directories for the frontend.

Usage:
    python -m inference.generate_comparison --max-samples 20 --device mps
"""

import argparse
import gc
import json
from pathlib import Path

import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from diffusers.models import UNet2DConditionModel
from PIL import Image

from inference.predict import predict
from inference.extract_signal import extract_signal


BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"


def discover_checkpoints(checkpoints_dir: str = "checkpoints") -> list[dict]:
    """Auto-discover local checkpoints and build ordered model list.

    Returns list of dicts: [{"name": ..., "path": ...}, ...]
    Ordered: checkpoint-500 -> checkpoint-1000 -> ... -> final
    Only uses checkpoints already on disk — never downloads anything.
    """
    ckpt_root = Path(checkpoints_dir)
    models = []

    if not ckpt_root.exists():
        print(f"Warning: {ckpt_root} does not exist, no checkpoints found")
        return models

    # Find intermediate checkpoints (checkpoint-500, checkpoint-1000, etc.)
    intermediates = []
    for d in sorted(ckpt_root.iterdir()):
        if d.is_dir() and d.name.startswith("checkpoint-"):
            try:
                step = int(d.name.split("-")[1])
                intermediates.append((step, d))
            except (ValueError, IndexError):
                continue

    # Sort by step number
    intermediates.sort(key=lambda x: x[0])
    for step, path in intermediates:
        models.append({"name": f"Step {step}", "path": str(path)})

    # Add final checkpoint (the root checkpoints/ dir itself if it has model files)
    final_markers = ["model_index.json", "unet"]
    if any((ckpt_root / m).exists() for m in final_markers):
        models.append({"name": "Final", "path": str(ckpt_root)})

    return models


def load_test_samples(metadata_path: str = "data/rendered/metadata.json",
                      max_samples: int = 20) -> list[dict]:
    """Load test samples from the rendered dataset metadata."""
    meta_path = Path(metadata_path)
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Metadata not found at {meta_path}. "
            "Run data/render_charts.py first."
        )

    with open(meta_path) as f:
        all_samples = json.load(f)

    # Take evenly spaced samples for diversity
    if len(all_samples) <= max_samples:
        samples = all_samples
    else:
        step = len(all_samples) / max_samples
        indices = [int(i * step) for i in range(max_samples)]
        samples = [all_samples[i] for i in indices]

    return samples


def generate_comparison(
    max_samples: int = 10,
    device: str = "auto",
    output_dir: str = "outputs/comparisons",
    checkpoints_dir: str = "checkpoints",
    metadata_path: str = "data/rendered/metadata.json",
    input_dir: str = "data/rendered/input",
    target_dir: str = "data/rendered/target",
):
    """Run each model on test images and save comparison data.

    Outer loop is models (load once, then run all samples), inner loop is samples.
    """
    out = Path(output_dir)
    out_inputs = out / "inputs"
    out_targets = out / "targets"
    out_preds = out / "predictions"
    out_inputs.mkdir(parents=True, exist_ok=True)
    out_targets.mkdir(parents=True, exist_ok=True)
    out_preds.mkdir(parents=True, exist_ok=True)

    # Discover models
    models = discover_checkpoints(checkpoints_dir)
    print(f"Found {len(models)} models: {[m['name'] for m in models]}")

    # Load test samples
    samples = load_test_samples(metadata_path, max_samples)
    print(f"Using {len(samples)} test samples")

    # Copy input and target images, build sample info
    sample_records = []
    for sample in samples:
        inp_src = Path(input_dir) / f"{sample['id']}.png"
        inp_dst = out_inputs / f"{sample['id']}.png"
        if inp_src.exists():
            Image.open(inp_src).save(inp_dst)

        tgt_src = Path(target_dir) / f"{sample['id']}.png"
        tgt_dst = out_targets / f"{sample['id']}.png"
        if tgt_src.exists():
            Image.open(tgt_src).save(tgt_dst)

        sample_records.append({
            "id": sample["id"],
            "ground_truth": sample["signal"],
            "pct_change": sample["pct_change"],
            "rsi": sample["rsi"],
            "macd": sample["macd"],
            "prompt": sample["prompt"],
        })

    # Resolve device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # Load base pipeline once — checkpoints only contain UNet weights,
    # so we load the full base pipeline and swap in UNet weights per checkpoint.
    print(f"Loading base pipeline ({BASE_MODEL}) on {device}...")
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        BASE_MODEL, torch_dtype=dtype, safety_checker=None,
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    if device == "mps":
        pipe.enable_attention_slicing()

    # Build comparison results
    model_results = []

    for mi, model_info in enumerate(models):
        model_name = model_info["name"]
        model_path = model_info["path"]
        unet_path = Path(model_path) / "unet"
        print(f"\n[{mi+1}/{len(models)}] Loading UNet from {unet_path}...")

        try:
            unet = UNet2DConditionModel.from_pretrained(
                str(unet_path), torch_dtype=dtype,
            )
            pipe.unet = unet.to(device)
        except Exception as e:
            print(f"  Failed to load {model_name}: {e}")
            model_results.append({
                "name": model_name,
                "path": model_path,
                "predictions": [None] * len(samples),
            })
            continue

        predictions = []
        for si, sample in enumerate(samples):
            input_path = Path(input_dir) / f"{sample['id']}.png"
            if not input_path.exists():
                print(f"  Skipping {sample['id']}: input not found")
                predictions.append(None)
                continue

            input_img = Image.open(input_path).convert("RGB")
            prompt = sample["prompt"]

            print(f"  [{si+1}/{len(samples)}] Predicting {sample['id']}...", end=" ")

            try:
                pred_img = predict(pipe, input_img, prompt)

                # Save prediction image
                pred_filename = f"{model_name.lower().replace(' ', '_')}_{sample['id']}.png"
                pred_path = out_preds / pred_filename
                pred_img.save(pred_path)

                # Extract signal
                signal = extract_signal(pred_img)

                predictions.append({
                    "image": f"predictions/{pred_filename}",
                    "signal": signal.action,
                    "confidence": signal.confidence,
                    "avg_rgb": signal.avg_rgb,
                })
                print(f"{signal.action} ({signal.confidence:.0%})")

            except Exception as e:
                print(f"Error: {e}")
                predictions.append(None)

        model_results.append({
            "name": model_name,
            "path": model_path,
            "predictions": predictions,
        })

    # Free memory
    del pipe
    gc.collect()

    # Build final JSON
    comparison = {
        "samples": sample_records,
        "models": model_results,
    }

    json_path = out / "comparison.json"
    with open(json_path, "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"\nComparison saved to {json_path}")
    print(f"  {len(sample_records)} samples x {len(model_results)} models")


def main():
    parser = argparse.ArgumentParser(
        description="Generate model comparison across checkpoints"
    )
    parser.add_argument(
        "--max-samples", type=int, default=10,
        help="Maximum number of test samples (default: 10)"
    )
    parser.add_argument(
        "--device", default="auto",
        help="Device: auto, mps, cuda, cpu (default: auto)"
    )
    parser.add_argument(
        "--output", default="outputs/comparisons",
        help="Output directory (default: outputs/comparisons)"
    )
    parser.add_argument(
        "--checkpoints", default="checkpoints",
        help="Checkpoints directory (default: checkpoints)"
    )
    parser.add_argument(
        "--metadata", default="data/rendered/metadata.json",
        help="Metadata JSON path (default: data/rendered/metadata.json)"
    )
    parser.add_argument(
        "--input-dir", default="data/rendered/input",
        help="Input images directory (default: data/rendered/input)"
    )
    parser.add_argument(
        "--target-dir", default="data/rendered/target",
        help="Target images directory (default: data/rendered/target)"
    )
    args = parser.parse_args()

    generate_comparison(
        max_samples=args.max_samples,
        device=args.device,
        output_dir=args.output,
        checkpoints_dir=args.checkpoints,
        metadata_path=args.metadata,
        input_dir=args.input_dir,
        target_dir=args.target_dir,
    )


if __name__ == "__main__":
    main()
