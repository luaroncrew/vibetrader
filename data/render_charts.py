"""Render chart image pairs with signal markers for InstructPix2Pix training."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm


# Chart rendering constants
IMG_SIZE = 256
CANDLE_AREA_LEFT = 10
CANDLE_AREA_RIGHT = IMG_SIZE - 10
CANDLE_AREA_TOP = 10
CANDLE_AREA_BOTTOM = IMG_SIZE - 10
MARKER_SIZE = 30
MARKER_MARGIN = 5

# Signal thresholds
UP_THRESHOLD = 0.02    # +2% = green
DOWN_THRESHOLD = -0.02  # -2% = red

# Colors
BG_COLOR = (0, 0, 0)
GREEN_CANDLE = (0, 200, 0)
RED_CANDLE = (200, 0, 0)
WICK_COLOR_GREEN = (0, 160, 0)
WICK_COLOR_RED = (160, 0, 0)
MARKER_GREEN = (0, 255, 0)
MARKER_RED = (255, 0, 0)
MARKER_GRAY = (128, 128, 128)


def normalize_prices(prices: np.ndarray, low: float, high: float) -> np.ndarray:
    """Normalize prices to [0, 1] range."""
    span = high - low
    if span == 0:
        return np.full_like(prices, 0.5)
    return (prices - low) / span


def price_to_y(normalized: float) -> int:
    """Convert normalized price [0,1] to pixel y coordinate (inverted)."""
    return int(CANDLE_AREA_BOTTOM - normalized * (CANDLE_AREA_BOTTOM - CANDLE_AREA_TOP))


def render_candlestick(
    df_window: pd.DataFrame,
    img_size: int = IMG_SIZE,
    draw_marker: bool = False,
    marker_color: tuple = MARKER_GRAY,
) -> Image.Image:
    """Render a candlestick chart as a PIL image.

    Args:
        df_window: DataFrame slice with open/high/low/close columns.
        img_size: Output image size (square).
        draw_marker: Whether to draw the signal marker.
        marker_color: RGB tuple for the marker.

    Returns:
        PIL Image of the chart.
    """
    img = Image.new("RGB", (img_size, img_size), BG_COLOR)
    draw = ImageDraw.Draw(img)

    n = len(df_window)
    if n == 0:
        return img

    # Price range for normalization
    all_low = df_window["low"].min()
    all_high = df_window["high"].max()

    # Candle width calculation
    chart_width = CANDLE_AREA_RIGHT - CANDLE_AREA_LEFT
    candle_width = max(1, int(chart_width / n * 0.7))
    gap = max(1, int(chart_width / n * 0.3))
    total_step = candle_width + gap

    for i, (_, row) in enumerate(df_window.iterrows()):
        x_center = CANDLE_AREA_LEFT + i * total_step + candle_width // 2

        o = normalize_prices(np.array([row["open"]]), all_low, all_high)[0]
        h = normalize_prices(np.array([row["high"]]), all_low, all_high)[0]
        l = normalize_prices(np.array([row["low"]]), all_low, all_high)[0]
        c = normalize_prices(np.array([row["close"]]), all_low, all_high)[0]

        y_open = price_to_y(o)
        y_high = price_to_y(h)
        y_low = price_to_y(l)
        y_close = price_to_y(c)

        is_green = row["close"] >= row["open"]
        body_color = GREEN_CANDLE if is_green else RED_CANDLE
        wick_color = WICK_COLOR_GREEN if is_green else WICK_COLOR_RED

        # Draw wick
        draw.line([(x_center, y_high), (x_center, y_low)], fill=wick_color, width=1)

        # Draw body
        body_top = min(y_open, y_close)
        body_bottom = max(y_open, y_close)
        if body_bottom == body_top:
            body_bottom = body_top + 1
        x_left = x_center - candle_width // 2
        x_right = x_center + candle_width // 2
        draw.rectangle([x_left, body_top, x_right, body_bottom], fill=body_color)

    # Draw signal marker in bottom-right corner
    if draw_marker:
        mx = img_size - MARKER_MARGIN - MARKER_SIZE
        my = img_size - MARKER_MARGIN - MARKER_SIZE
        draw.rectangle([mx, my, mx + MARKER_SIZE, my + MARKER_SIZE], fill=marker_color)

    return img


def compute_signal(current_close: float, future_close: float) -> tuple:
    """Determine signal based on price change.

    Returns:
        (signal_name, marker_color) tuple.
    """
    pct_change = (future_close - current_close) / current_close
    if pct_change > UP_THRESHOLD:
        return "BUY", MARKER_GREEN
    elif pct_change < DOWN_THRESHOLD:
        return "SELL", MARKER_RED
    else:
        return "HOLD", MARKER_GRAY


def render_dataset(
    csv_path: str,
    output_dir: str,
    window_size: int = 40,
    future_candles: int = 4,
    stride: int = 1,
):
    """Render all chart pairs from OHLCV CSV.

    Args:
        csv_path: Path to the OHLCV CSV file.
        output_dir: Directory to save images and metadata.
        window_size: Number of candles in each chart.
        future_candles: How many candles into the future to predict.
        stride: Step size for sliding window.
    """
    df = pd.read_csv(csv_path)
    out = Path(output_dir)
    (out / "input").mkdir(parents=True, exist_ok=True)
    (out / "target").mkdir(parents=True, exist_ok=True)

    metadata = []
    total = (len(df) - window_size - future_candles) // stride

    for idx in tqdm(range(0, len(df) - window_size - future_candles, stride), desc="Rendering"):
        # Input: candles [idx : idx + window_size]
        input_window = df.iloc[idx : idx + window_size]
        # Target: candles [idx + future_candles : idx + future_candles + window_size]
        target_window = df.iloc[idx + future_candles : idx + future_candles + window_size]

        # Signal: compare close at end of input vs end of target
        current_close = input_window.iloc[-1]["close"]
        future_close = target_window.iloc[-1]["close"]
        signal, marker_color = compute_signal(current_close, future_close)

        # Render images
        input_img = render_candlestick(input_window, draw_marker=False)
        target_img = render_candlestick(target_window, draw_marker=True, marker_color=marker_color)

        # Get indicator values for prompt
        rsi_val = input_window.iloc[-1].get("rsi", 50.0)
        macd_val = input_window.iloc[-1].get("MACD_12_26_9", 0.0)

        pair_id = f"{idx:06d}"
        input_img.save(out / "input" / f"{pair_id}.png")
        target_img.save(out / "target" / f"{pair_id}.png")

        metadata.append({
            "id": pair_id,
            "signal": signal,
            "pct_change": (future_close - current_close) / current_close,
            "rsi": round(float(rsi_val), 1),
            "macd": round(float(macd_val), 2),
            "prompt": f"Predict next {future_candles} candles. RSI={round(float(rsi_val), 1)}, MACD={round(float(macd_val), 2)}",
        })

    # Save metadata
    with open(out / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Rendered {len(metadata)} pairs to {out}")
    dist = {s: sum(1 for m in metadata if m["signal"] == s) for s in ["BUY", "SELL", "HOLD"]}
    print(f"Signal distribution: {dist}")


def main():
    parser = argparse.ArgumentParser(description="Render chart image pairs")
    parser.add_argument("--csv", default="data/btc_usdt_4h.csv")
    parser.add_argument("--output", default="data/rendered")
    parser.add_argument("--window", type=int, default=40)
    parser.add_argument("--future", type=int, default=4)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()

    render_dataset(args.csv, args.output, args.window, args.future, args.stride)


if __name__ == "__main__":
    main()
