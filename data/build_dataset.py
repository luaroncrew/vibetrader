"""Assemble rendered chart pairs into a HuggingFace Dataset."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from datasets import Dataset, Features, Image, Value
from PIL import Image as PILImage
from tqdm import tqdm


def build_dataset(rendered_dir: str, output_dir: str, balanced: bool = False):
    """Build HuggingFace Dataset from rendered image pairs.

    Args:
        rendered_dir: Directory containing input/, target/, and metadata.json.
        output_dir: Where to save the HuggingFace dataset.
        balanced: If True, downsample to equal BUY/SELL/HOLD counts.
    """
    rendered = Path(rendered_dir)
    with open(rendered / "metadata.json") as f:
        metadata = json.load(f)

    if balanced:
        by_signal = defaultdict(list)
        for entry in metadata:
            by_signal[entry["signal"]].append(entry)
        min_count = min(len(v) for v in by_signal.values())
        metadata = []
        for signal, entries in by_signal.items():
            metadata.extend(random.sample(entries, min_count))
        random.shuffle(metadata)
        print(f"Balanced dataset: {min_count} per class, {len(metadata)} total")

    records = []
    for entry in tqdm(metadata, desc="Building dataset"):
        pair_id = entry["id"]
        input_path = str(rendered / "input" / f"{pair_id}.png")
        target_path = str(rendered / "target" / f"{pair_id}.png")

        # Verify files exist
        if not Path(input_path).exists() or not Path(target_path).exists():
            continue

        records.append({
            "original_image": input_path,
            "edited_image": target_path,
            "edit_prompt": entry["prompt"],
        })

    # Create dataset with image features
    ds = Dataset.from_dict({
        "original_image": [r["original_image"] for r in records],
        "edited_image": [r["edited_image"] for r in records],
        "edit_prompt": [r["edit_prompt"] for r in records],
    })

    # Cast image columns
    ds = ds.cast_column("original_image", Image())
    ds = ds.cast_column("edited_image", Image())

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out))
    print(f"Dataset saved to {out} ({len(ds)} examples)")


def main():
    parser = argparse.ArgumentParser(description="Build HuggingFace dataset")
    parser.add_argument("--rendered", default="data/rendered")
    parser.add_argument("--output", default="data/dataset")
    parser.add_argument("--balanced", action="store_true",
                        help="Downsample to equal BUY/SELL/HOLD counts")
    args = parser.parse_args()

    build_dataset(args.rendered, args.output, balanced=args.balanced)


if __name__ == "__main__":
    main()
