"""Extract trading signal from generated chart images by reading candle colors.

Instead of relying on a small marker square (which the model often gets wrong),
we analyse the rightmost candles in the generated chart. Green candles mean
price went up, red candles mean price went down.

Candle colors used in rendering (data/render_charts.py):
    Green body  (0, 200, 0)   Green wick  (0, 160, 0)
    Red body    (200, 0, 0)   Red wick    (160, 0, 0)
    Background  (0, 0, 0)
"""

from dataclasses import dataclass

import numpy as np
from PIL import Image


IMG_SIZE = 256
CHART_LEFT = 10
CHART_RIGHT = IMG_SIZE - 10
CHART_TOP = 10
CHART_BOTTOM = IMG_SIZE - 10

# Thresholds for classifying a pixel as green-ish or red-ish candle
# A pixel is "green" if G channel dominates, "red" if R channel dominates.
MIN_BRIGHTNESS = 60  # ignore near-black (background / wicks too dim)


@dataclass
class Signal:
    """Trading signal extracted from a chart image."""
    action: str        # "BUY", "SELL", or "HOLD"
    confidence: float  # 0.0 to 1.0
    green_pct: float   # fraction of candle pixels that are green
    red_pct: float     # fraction of candle pixels that are red


def extract_signal(image: Image.Image, right_frac: float = 0.25) -> Signal:
    """Read the signal from the rightmost candles of a generated chart.

    Analyses the right portion of the chart area. Counts green vs red
    candle pixels to determine the dominant direction.

    Args:
        image: Generated chart image (256x256).
        right_frac: Fraction of chart width to analyse from the right
                    (0.25 = last ~10 candles out of 40).

    Returns:
        Signal with action, confidence, and green/red percentages.
    """
    img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32)

    # Crop the right portion of the chart area
    right_start = int(CHART_RIGHT - (CHART_RIGHT - CHART_LEFT) * right_frac)
    region = arr[CHART_TOP:CHART_BOTTOM, right_start:CHART_RIGHT]

    r, g, b = region[:, :, 0], region[:, :, 1], region[:, :, 2]

    # A pixel is a candle pixel if it's bright enough (not background)
    brightness = np.maximum(r, np.maximum(g, b))
    is_candle = brightness >= MIN_BRIGHTNESS

    # Green pixel: G channel is the dominant and significantly > R
    is_green = is_candle & (g > r * 1.5) & (g > b * 1.5)
    # Red pixel: R channel is the dominant and significantly > G
    is_red = is_candle & (r > g * 1.5) & (r > b * 1.5)

    n_green = int(is_green.sum())
    n_red = int(is_red.sum())
    n_total = n_green + n_red

    if n_total == 0:
        return Signal(action="HOLD", confidence=0.0, green_pct=0.0, red_pct=0.0)

    green_pct = n_green / n_total
    red_pct = n_red / n_total

    # Decision thresholds
    # Strong majority (>60%) of one color → BUY/SELL, otherwise HOLD
    DOMINANCE_THRESHOLD = 0.60

    if green_pct >= DOMINANCE_THRESHOLD:
        action = "BUY"
        confidence = green_pct
    elif red_pct >= DOMINANCE_THRESHOLD:
        action = "SELL"
        confidence = red_pct
    else:
        action = "HOLD"
        confidence = 1.0 - abs(green_pct - red_pct)  # more mixed = more confident HOLD

    return Signal(
        action=action,
        confidence=round(confidence, 3),
        green_pct=round(green_pct, 3),
        red_pct=round(red_pct, 3),
    )


def extract_signal_from_path(image_path: str) -> Signal:
    """Convenience: extract signal from an image file path."""
    return extract_signal(Image.open(image_path))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract signal from chart image")
    parser.add_argument("image", help="Path to generated chart image")
    parser.add_argument("--right-frac", type=float, default=0.25,
                        help="Fraction of chart to analyse from right (default: 0.25)")
    args = parser.parse_args()

    signal = extract_signal_from_path(args.image)
    print(f"Action:     {signal.action}")
    print(f"Confidence: {signal.confidence}")
    print(f"Green:      {signal.green_pct:.1%}")
    print(f"Red:        {signal.red_pct:.1%}")
