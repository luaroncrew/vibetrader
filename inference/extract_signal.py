"""Extract trading signal from generated chart images by reading the marker square.

Target chart images contain a 30x30 colored marker square in the bottom-right
corner that encodes the trading signal:
    Green (0, 255, 0)   -> BUY
    Red   (255, 0, 0)   -> SELL
    Gray  (128, 128, 128) -> HOLD

Constants match data/render_charts.py:
    IMG_SIZE = 256, MARKER_SIZE = 30, MARKER_MARGIN = 5
    Square region: pixels [221:251, 221:251]
"""

from dataclasses import dataclass

import numpy as np
from PIL import Image


IMG_SIZE = 256
MARKER_SIZE = 30
MARKER_MARGIN = 5

# Top-left corner of the marker square
_MARKER_X0 = IMG_SIZE - MARKER_MARGIN - MARKER_SIZE  # 221
_MARKER_Y0 = IMG_SIZE - MARKER_MARGIN - MARKER_SIZE  # 221

# Known marker colors
MARKER_GREEN = np.array([0, 255, 0], dtype=np.float32)
MARKER_RED = np.array([255, 0, 0], dtype=np.float32)
MARKER_GRAY = np.array([128, 128, 128], dtype=np.float32)

_MARKERS = [
    ("BUY", MARKER_GREEN),
    ("SELL", MARKER_RED),
    ("HOLD", MARKER_GRAY),
]


@dataclass
class Signal:
    """Trading signal extracted from a chart image."""
    action: str                        # "BUY", "SELL", or "HOLD"
    confidence: float                  # 0.0 to 1.0
    avg_rgb: tuple[float, float, float]  # average RGB of the marker square


def extract_signal(image: Image.Image) -> Signal:
    """Read the trading signal from the marker square of a generated chart.

    Reads the 30x30 pixel region at the bottom-right corner, computes its
    average RGB, and classifies by Euclidean distance to known marker colors.

    Args:
        image: Generated chart image (256x256).

    Returns:
        Signal with action, confidence, and average RGB of the marker region.
    """
    img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32)

    # Extract the marker square region
    region = arr[_MARKER_Y0:_MARKER_Y0 + MARKER_SIZE,
                 _MARKER_X0:_MARKER_X0 + MARKER_SIZE]

    avg_rgb = region.mean(axis=(0, 1))  # shape (3,)

    # Compute Euclidean distance to each known marker color
    distances = [(label, float(np.linalg.norm(avg_rgb - color)))
                 for label, color in _MARKERS]

    # Sort by distance (closest first)
    distances.sort(key=lambda x: x[1])
    best_label, best_dist = distances[0]
    _, second_dist = distances[1]

    # Confidence: how much closer the best match is vs the second best.
    # When best_dist == 0, confidence is 1.0; when best == second, confidence ~ 0.5.
    if second_dist + best_dist == 0:
        confidence = 0.0
    else:
        confidence = 1.0 - best_dist / (second_dist + best_dist)

    return Signal(
        action=best_label,
        confidence=round(confidence, 3),
        avg_rgb=(round(float(avg_rgb[0]), 1),
                 round(float(avg_rgb[1]), 1),
                 round(float(avg_rgb[2]), 1)),
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
    print(f"Action:     {signal.action}")
    print(f"Confidence: {signal.confidence}")
    print(f"Avg RGB:    {signal.avg_rgb}")
