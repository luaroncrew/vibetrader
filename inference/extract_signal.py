"""Extract trading signal from the colored marker in generated chart images."""

from dataclasses import dataclass

import numpy as np
from PIL import Image


# Marker location: bottom-right corner
MARKER_SIZE = 30
MARKER_MARGIN = 5
IMG_SIZE = 256

# Reference colors
REF_GREEN = np.array([0, 255, 0])
REF_RED = np.array([255, 0, 0])
REF_GRAY = np.array([128, 128, 128])


@dataclass
class Signal:
    """Trading signal extracted from a chart image."""
    action: str        # "BUY", "SELL", or "HOLD"
    confidence: float  # 0.0 to 1.0
    avg_rgb: tuple     # Average RGB of the marker region


def extract_marker_region(image: Image.Image) -> np.ndarray:
    """Crop the marker region from the image.

    Args:
        image: Generated chart image (256x256).

    Returns:
        Numpy array of the marker region pixels.
    """
    img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img)

    # Bottom-right corner
    x_start = IMG_SIZE - MARKER_MARGIN - MARKER_SIZE
    y_start = IMG_SIZE - MARKER_MARGIN - MARKER_SIZE
    x_end = IMG_SIZE - MARKER_MARGIN
    y_end = IMG_SIZE - MARKER_MARGIN

    return arr[y_start:y_end, x_start:x_end]


def extract_signal(image: Image.Image) -> Signal:
    """Read the signal marker from a generated chart image.

    The model is trained to place a colored square in the bottom-right:
    - Green (#00FF00) = BUY
    - Red (#FF0000) = SELL
    - Gray (#808080) = HOLD

    Args:
        image: Generated chart image.

    Returns:
        Signal with action, confidence, and average RGB.
    """
    marker = extract_marker_region(image)
    avg_rgb = marker.mean(axis=(0, 1))

    # Compute distances to reference colors
    dist_green = np.linalg.norm(avg_rgb - REF_GREEN)
    dist_red = np.linalg.norm(avg_rgb - REF_RED)
    dist_gray = np.linalg.norm(avg_rgb - REF_GRAY)

    distances = {"BUY": dist_green, "SELL": dist_red, "HOLD": dist_gray}
    action = min(distances, key=distances.get)
    min_dist = distances[action]

    # Confidence: inverse of distance, normalized
    # Max possible distance is ~441 (diagonal of 255,255,255 cube)
    max_dist = 441.67
    confidence = max(0.0, 1.0 - min_dist / max_dist)

    return Signal(
        action=action,
        confidence=round(confidence, 3),
        avg_rgb=tuple(int(c) for c in avg_rgb),
    )


def extract_signal_from_path(image_path: str) -> Signal:
    """Convenience: extract signal from an image file path."""
    return extract_signal(Image.open(image_path))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract signal from chart image")
    parser.add_argument("image", help="Path to generated chart image")
    args = parser.parse_args()

    signal = extract_signal_from_path(args.image)
    print(f"Action: {signal.action}")
    print(f"Confidence: {signal.confidence}")
    print(f"Avg RGB: {signal.avg_rgb}")
