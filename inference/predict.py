"""Run inference with the fine-tuned InstructPix2Pix model."""

import argparse
from pathlib import Path

import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from PIL import Image

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def load_pipeline(checkpoint_path: str, device: str = "auto") -> StableDiffusionInstructPix2PixPipeline:
    """Load the fine-tuned InstructPix2Pix pipeline.

    Args:
        checkpoint_path: Path to the saved model checkpoint.
        device: Device to run on ('auto', 'mps', 'cuda', 'cpu').

    Returns:
        Loaded pipeline ready for inference.
    """
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    if device == "mps":
        pipe = pipe.to("mps")
        pipe.enable_attention_slicing()
    elif device == "cuda":
        pipe = pipe.to("cuda")
    else:
        pipe = pipe.to("cpu")

    return pipe


def predict(
    pipe: StableDiffusionInstructPix2PixPipeline,
    image: Image.Image,
    prompt: str,
    num_inference_steps: int = 20,
    image_guidance_scale: float = 1.5,
    guidance_scale: float = 7.0,
) -> Image.Image:
    """Generate a chart continuation prediction.

    Args:
        pipe: Loaded InstructPix2Pix pipeline.
        image: Input chart image.
        prompt: Instruction prompt with indicators.
        num_inference_steps: Diffusion steps.
        image_guidance_scale: How much to follow the input image.
        guidance_scale: How much to follow the text prompt.

    Returns:
        Generated chart image with signal marker.
    """
    # Ensure image is 256x256 RGB
    image = image.convert("RGB").resize((256, 256))

    result = pipe(
        prompt=prompt,
        image=image,
        num_inference_steps=num_inference_steps,
        image_guidance_scale=image_guidance_scale,
        guidance_scale=guidance_scale,
    )
    return result.images[0]


def predict_batch(
    pipe: StableDiffusionInstructPix2PixPipeline,
    image_paths: list[str],
    prompts: list[str],
    output_dir: str,
    log_wandb: bool = False,
    **kwargs,
):
    """Run prediction on a batch of images.

    Args:
        pipe: Loaded pipeline.
        image_paths: List of input image file paths.
        prompts: Corresponding prompts.
        output_dir: Where to save generated images.
        log_wandb: Whether to log to W&B.
        **kwargs: Additional args passed to predict().
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    wandb_images = []

    for i, (img_path, prompt) in enumerate(zip(image_paths, prompts)):
        input_img = Image.open(img_path).convert("RGB")
        generated = predict(pipe, input_img, prompt, **kwargs)

        output_path = out / f"pred_{i:04d}.png"
        generated.save(output_path)

        if log_wandb and HAS_WANDB:
            wandb_images.append(wandb.Image(
                generated,
                caption=f"{prompt} | src: {Path(img_path).stem}",
            ))

        print(f"[{i+1}/{len(image_paths)}] {img_path} -> {output_path}")

    if log_wandb and HAS_WANDB and wandb_images:
        wandb.log({"predictions": wandb_images})


def main():
    parser = argparse.ArgumentParser(description="Run chart prediction inference")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--input", required=True, help="Input image or directory")
    parser.add_argument("--prompt", default="Predict next 4 candles. RSI=50.0, MACD=0.0")
    parser.add_argument("--output", default="outputs/predictions")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--image-guidance", type=float, default=1.5)
    parser.add_argument("--guidance", type=float, default=7.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--wandb", action="store_true", help="Log to W&B")
    args = parser.parse_args()

    if args.wandb and HAS_WANDB:
        wandb.init(project="vibetrader", job_type="inference")

    print(f"Loading model from {args.checkpoint}...")
    pipe = load_pipeline(args.checkpoint, device=args.device)

    input_path = Path(args.input)
    if input_path.is_dir():
        image_paths = sorted(str(p) for p in input_path.glob("*.png"))
        prompts = [args.prompt] * len(image_paths)
    else:
        image_paths = [str(input_path)]
        prompts = [args.prompt]

    print(f"Running inference on {len(image_paths)} images...")
    predict_batch(
        pipe, image_paths, prompts, args.output,
        log_wandb=args.wandb,
        num_inference_steps=args.steps,
        image_guidance_scale=args.image_guidance,
        guidance_scale=args.guidance,
    )

    if args.wandb and HAS_WANDB:
        wandb.finish()


if __name__ == "__main__":
    main()
