"""Backtest the chart prediction model against historical data."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.render_charts import render_candlestick, compute_signal
from inference.predict import load_pipeline, predict
from inference.extract_signal import extract_signal
from inference.extract_signal_mistral import extract_signal_mistral


def run_backtest(
    csv_path: str,
    checkpoint_path: str,
    output_dir: str,
    window_size: int = 40,
    future_candles: int = 4,
    test_months: int = 3,
    max_samples: int = 100,
    device: str = "auto",
    log_wandb: bool = False,
    use_mistral: bool = False,
):
    """Run backtest on held-out historical data.

    Args:
        csv_path: Path to OHLCV CSV.
        checkpoint_path: Path to model checkpoint.
        output_dir: Where to save results.
        window_size: Candles per chart.
        future_candles: Candles to predict ahead.
        test_months: Months of data to hold out for testing.
        max_samples: Maximum number of test samples (for speed).
        device: Inference device.
        log_wandb: Log results to W&B.
        use_mistral: Use Mistral Pixtral for signal extraction instead of pixel counting.
    """
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Split: last N months for testing
    cutoff = df["timestamp"].max() - pd.DateOffset(months=test_months)
    test_df = df[df["timestamp"] >= cutoff].reset_index(drop=True)
    print(f"Test set: {len(test_df)} candles (last {test_months} months)")

    # Load model
    print(f"Loading model from {checkpoint_path}...")
    pipe = load_pipeline(checkpoint_path, device=device)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results = []
    total_possible = len(test_df) - window_size - future_candles
    step = max(1, total_possible // max_samples)

    if log_wandb and HAS_WANDB:
        wandb.init(project="vibetrader", job_type="backtest")

    wandb_images = []

    for i in tqdm(range(0, total_possible, step), desc="Backtesting"):
        input_window = test_df.iloc[i : i + window_size]
        target_window = test_df.iloc[i : i + window_size + future_candles]

        current_close = input_window.iloc[-1]["close"]
        future_close = target_window.iloc[-1]["close"]
        actual_signal, _ = compute_signal(current_close, future_close)
        actual_pct = (future_close - current_close) / current_close

        # Shared Y-normalization based on input candles only
        input_low = float(input_window["low"].min())
        input_high = float(input_window["high"].max())
        total_slots = window_size + future_candles

        # Render input chart
        input_img = render_candlestick(
            input_window, draw_marker=False,
            total_slots=total_slots, price_low=input_low, price_high=input_high,
        )

        # Build prompt from indicators
        rsi = input_window.iloc[-1].get("rsi", 50.0)
        macd = input_window.iloc[-1].get("MACD_12_26_9", 0.0)
        prompt = f"Predict next {future_candles} candles. RSI={round(float(rsi), 1)}, MACD={round(float(macd), 2)}"

        # Predict
        generated = predict(pipe, input_img, prompt)

        if use_mistral:
            signal = extract_signal_mistral(input_img, generated)
        else:
            signal = extract_signal(generated)

        correct = signal.action == actual_signal
        results.append({
            "idx": i,
            "predicted": signal.action,
            "actual": actual_signal,
            "confidence": signal.confidence,
            "pct_change": actual_pct,
            "correct": correct,
        })

        # Log sample images to W&B
        if log_wandb and HAS_WANDB and len(wandb_images) < 20:
            # Render actual target for comparison
            actual_img = render_candlestick(
                target_window, draw_marker=True,
                marker_color=(0, 255, 0) if actual_signal == "BUY"
                else (255, 0, 0) if actual_signal == "SELL"
                else (128, 128, 128),
                total_slots=total_slots, price_low=input_low, price_high=input_high,
            )
            wandb_images.append(wandb.Image(
                np.hstack([np.array(input_img), np.array(generated), np.array(actual_img)]),
                caption=f"pred={signal.action} actual={actual_signal} ({actual_pct:+.2%})",
            ))

    # Compute metrics
    results_df = pd.DataFrame(results)
    accuracy = results_df["correct"].mean()
    buy_accuracy = results_df[results_df["predicted"] == "BUY"]["correct"].mean() if len(results_df[results_df["predicted"] == "BUY"]) > 0 else 0
    sell_accuracy = results_df[results_df["predicted"] == "SELL"]["correct"].mean() if len(results_df[results_df["predicted"] == "SELL"]) > 0 else 0

    # Simulated returns: go long on BUY, short on SELL, flat on HOLD
    returns = []
    for _, row in results_df.iterrows():
        if row["predicted"] == "BUY":
            returns.append(row["pct_change"])
        elif row["predicted"] == "SELL":
            returns.append(-row["pct_change"])
        else:
            returns.append(0)
    results_df["strategy_return"] = returns
    total_return = (1 + results_df["strategy_return"]).prod() - 1
    buy_hold_return = (1 + results_df["pct_change"]).prod() - 1

    print(f"\n{'='*50}")
    print(f"BACKTEST RESULTS ({len(results_df)} samples)")
    print(f"{'='*50}")
    print(f"Accuracy:       {accuracy:.1%}")
    print(f"BUY accuracy:   {buy_accuracy:.1%}")
    print(f"SELL accuracy:  {sell_accuracy:.1%}")
    print(f"Strategy return: {total_return:.2%}")
    print(f"Buy & hold:      {buy_hold_return:.2%}")
    print(f"Signal dist:     {results_df['predicted'].value_counts().to_dict()}")

    # Save results
    results_df.to_csv(out / "backtest_results.csv", index=False)
    metrics = {
        "accuracy": accuracy,
        "buy_accuracy": buy_accuracy,
        "sell_accuracy": sell_accuracy,
        "strategy_return": total_return,
        "buy_hold_return": buy_hold_return,
        "n_samples": len(results_df),
    }
    with open(out / "backtest_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    if log_wandb and HAS_WANDB:
        wandb.log(metrics)
        if wandb_images:
            wandb.log({"backtest_samples": wandb_images})
        wandb.finish()

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Backtest chart prediction model")
    parser.add_argument("--csv", default="data/btc_usdt_4h.csv")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="outputs/backtest")
    parser.add_argument("--window", type=int, default=40)
    parser.add_argument("--future", type=int, default=4)
    parser.add_argument("--test-months", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--mistral", action="store_true", help="Use Mistral Pixtral for signal extraction")
    args = parser.parse_args()

    run_backtest(
        args.csv, args.checkpoint, args.output,
        args.window, args.future, args.test_months,
        args.max_samples, args.device, args.wandb,
        args.mistral,
    )


if __name__ == "__main__":
    main()
