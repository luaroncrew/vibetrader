"""Extract trading signal from generated chart images by reading candle colors.

Instead of relying on the small marker square (which the model often gets wrong),
we analyse the future candles in the generated chart. Green candles mean
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

# A pixel is "green" if G channel dominates, "red" if R channel dominates.
MIN_BRIGHTNESS = 60  # ignore near-black (background)


@dataclass
class Signal:
    """Trading signal extracted from a chart image."""
    action: str        # "BUY", "SELL", or "HOLD"
    confidence: float  # 0.0 to 1.0
    green_pct: float   # fraction of candle pixels that are green
    red_pct: float     # fraction of candle pixels that are red


def extract_signal(
    image: Image.Image,
    window_size: int = 40,
    future_candles: int = 4,
) -> Signal:
    """Read the signal from the future candles of a generated chart.

    Computes the exact pixel region where future candles are drawn
    (slots window_size .. window_size+future_candles-1) and analyses
    green vs red candle pixels to determine the dominant direction.

    Args:
        image: Generated chart image (256x256).
        window_size: Number of input candles (determines where future starts).
        future_candles: Number of future candles to analyse.

    Returns:
        Signal with action, confidence, and green/red percentages.
    """
    img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32)

    # Compute exact pixel region where future candles are drawn
    total_slots = window_size + future_candles
    chart_width = CHART_RIGHT - CHART_LEFT
    candle_width = max(1, int(chart_width / total_slots * 0.7))
    gap = max(1, int(chart_width / total_slots * 0.3))
    total_step = candle_width + gap

    # First future candle starts at slot index = window_size
    first_future_x = CHART_LEFT + window_size * total_step
    # Last future candle ends at slot index = total_slots - 1
    last_future_x_center = CHART_LEFT + (total_slots - 1) * total_step + candle_width // 2
    last_future_x_right = last_future_x_center + candle_width // 2

    # Add some margin on left side to capture wicks
    region_left = max(CHART_LEFT, first_future_x - gap)
    region_right = min(CHART_RIGHT, last_future_x_right + gap)

    region = arr[CHART_TOP:CHART_BOTTOM, region_left:region_right]

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
    # Strong majority (>60%) of one color -> BUY/SELL, otherwise HOLD
    DOMINANCE_THRESHOLD = 0.60

    if green_pct >= DOMINANCE_THRESHOLD:
        action = "BUY"
        confidence = green_pct
    elif red_pct >= DOMINANCE_THRESHOLD:
        action = "SELL"
        confidence = red_pct
    else:
        action = "HOLD"
        confidence = 1.0 - abs(green_pct - red_pct)

    return Signal(
        action=action,
        confidence=round(confidence, 3),
        green_pct=round(green_pct, 3),
        red_pct=round(red_pct, 3),
    )


def extract_signal_from_path(
    image_path: str,
    window_size: int = 40,
    future_candles: int = 4,
) -> Signal:
    """Convenience: extract signal from an image file path."""
    return extract_signal(Image.open(image_path), window_size=window_size,
                          future_candles=future_candles)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract signal from chart image")
    parser.add_argument("image", help="Path to generated chart image")
    parser.add_argument("--window-size", type=int, default=40,
                        help="Number of input candles (default: 40)")
    parser.add_argument("--future-candles", type=int, default=4,
                        help="Number of future candles (default: 4)")
    args = parser.parse_args()

    signal = extract_signal_from_path(args.image, window_size=args.window_size,
                                      future_candles=args.future_candles)
    print(f"Action:     {signal.action}")
    print(f"Confidence: {signal.confidence}")
    print(f"Green:      {signal.green_pct:.1%}")
    print(f"Red:        {signal.red_pct:.1%}")
