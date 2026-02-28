"""Paper trading loop using chart prediction model (stretch goal)."""

import argparse
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

try:
    import ccxt
except ImportError:
    ccxt = None

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.render_charts import render_candlestick
from inference.predict import load_pipeline, predict
from inference.extract_signal import extract_signal


class PaperTrader:
    """Simple paper trading bot that uses chart predictions.

    Attributes:
        balance: Current paper balance in USD.
        position: Current BTC position (positive = long).
        trades: List of executed trades.
    """

    def __init__(
        self,
        checkpoint_path: str,
        symbol: str = "BTC/USDT",
        timeframe: str = "4h",
        initial_balance: float = 10000.0,
        window_size: int = 40,
        future_candles: int = 4,
        device: str = "auto",
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.position = 0.0
        self.position_entry_price = 0.0
        self.trades = []
        self.window_size = window_size
        self.future_candles = future_candles

        print(f"Loading model from {checkpoint_path}...")
        self.pipe = load_pipeline(checkpoint_path, device=device)

        if ccxt is None:
            raise ImportError("ccxt is required for live trading. Install with: pip install ccxt")
        config = {"enableRateLimit": True}
        api_key = os.environ.get("BINANCE_API_KEY")
        secret = os.environ.get("BINANCE_SECRET")
        if api_key and secret:
            config["apiKey"] = api_key
            config["secret"] = secret
        self.exchange = ccxt.binance(config)

    def fetch_recent_candles(self) -> pd.DataFrame:
        """Fetch the most recent candles for chart rendering."""
        candles = self.exchange.fetch_ohlcv(
            self.symbol, self.timeframe, limit=self.window_size + 10
        )
        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df.tail(self.window_size).reset_index(drop=True)

    def get_signal(self, df: pd.DataFrame) -> dict:
        """Generate a trading signal from recent candles."""
        chart = render_candlestick(df, draw_marker=False)
        prompt = f"Predict next {self.future_candles} candles. RSI=50.0, MACD=0.0"
        generated = predict(self.pipe, chart, prompt)
        signal = extract_signal(generated)
        return {
            "action": signal.action,
            "confidence": signal.confidence,
            "avg_rgb": signal.avg_rgb,
        }

    def execute_signal(self, signal: dict, current_price: float):
        """Execute a paper trade based on the signal."""
        action = signal["action"]
        confidence = signal["confidence"]
        timestamp = datetime.utcnow().isoformat()

        if action == "BUY" and self.position <= 0:
            # Close short if any, then go long
            if self.position < 0:
                pnl = (self.position_entry_price - current_price) * abs(self.position)
                self.balance += pnl
                self.trades.append({
                    "time": timestamp, "action": "CLOSE_SHORT",
                    "price": current_price, "size": abs(self.position), "pnl": pnl,
                })
            # Open long: use 90% of balance
            size = (self.balance * 0.9) / current_price
            self.position = size
            self.position_entry_price = current_price
            self.trades.append({
                "time": timestamp, "action": "BUY",
                "price": current_price, "size": size,
                "confidence": confidence,
            })
            print(f"[{timestamp}] BUY {size:.6f} BTC @ ${current_price:.2f} (conf={confidence:.2f})")

        elif action == "SELL" and self.position >= 0:
            # Close long if any, then go short
            if self.position > 0:
                pnl = (current_price - self.position_entry_price) * self.position
                self.balance += pnl
                self.trades.append({
                    "time": timestamp, "action": "CLOSE_LONG",
                    "price": current_price, "size": self.position, "pnl": pnl,
                })
            # Open short
            size = (self.balance * 0.9) / current_price
            self.position = -size
            self.position_entry_price = current_price
            self.trades.append({
                "time": timestamp, "action": "SELL",
                "price": current_price, "size": size,
                "confidence": confidence,
            })
            print(f"[{timestamp}] SELL {size:.6f} BTC @ ${current_price:.2f} (conf={confidence:.2f})")

        else:
            print(f"[{timestamp}] HOLD (signal={action}, position={self.position:.6f})")

    def get_portfolio_value(self, current_price: float) -> float:
        """Calculate total portfolio value."""
        unrealized = 0
        if self.position > 0:
            unrealized = (current_price - self.position_entry_price) * self.position
        elif self.position < 0:
            unrealized = (self.position_entry_price - current_price) * abs(self.position)
        return self.balance + unrealized

    def run(self, interval_seconds: int = 14400, max_iterations: int = 100):
        """Run the paper trading loop.

        Args:
            interval_seconds: Seconds between checks (default 4h = 14400s).
            max_iterations: Maximum number of trading iterations.
        """
        print(f"Starting paper trader: {self.symbol} {self.timeframe}")
        print(f"Initial balance: ${self.initial_balance:.2f}")
        print(f"Check interval: {interval_seconds}s")

        if HAS_WANDB:
            wandb.init(project="vibetrader", job_type="paper_trading")

        for i in range(max_iterations):
            try:
                df = self.fetch_recent_candles()
                current_price = df.iloc[-1]["close"]
                signal = self.get_signal(df)

                self.execute_signal(signal, current_price)

                portfolio_value = self.get_portfolio_value(current_price)
                pnl_pct = (portfolio_value - self.initial_balance) / self.initial_balance

                print(f"  Portfolio: ${portfolio_value:.2f} ({pnl_pct:+.2%})")

                if HAS_WANDB:
                    wandb.log({
                        "portfolio_value": portfolio_value,
                        "pnl_pct": pnl_pct,
                        "btc_price": current_price,
                        "signal": signal["action"],
                        "confidence": signal["confidence"],
                        "position": self.position,
                    })

                if i < max_iterations - 1:
                    print(f"  Sleeping {interval_seconds}s until next check...")
                    time.sleep(interval_seconds)

            except KeyboardInterrupt:
                print("\nStopping trader...")
                break
            except Exception as e:
                print(f"  Error: {e}")
                time.sleep(60)

        # Summary
        portfolio_value = self.get_portfolio_value(df.iloc[-1]["close"] if len(df) > 0 else 0)
        print(f"\n{'='*50}")
        print(f"TRADING SUMMARY")
        print(f"{'='*50}")
        print(f"Total trades: {len(self.trades)}")
        print(f"Final balance: ${portfolio_value:.2f}")
        print(f"Return: {(portfolio_value - self.initial_balance) / self.initial_balance:+.2%}")

        if HAS_WANDB:
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Paper trading bot")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="4h")
    parser.add_argument("--balance", type=float, default=10000.0)
    parser.add_argument("--interval", type=int, default=14400)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    trader = PaperTrader(
        checkpoint_path=args.checkpoint,
        symbol=args.symbol,
        timeframe=args.timeframe,
        initial_balance=args.balance,
        device=args.device,
    )
    trader.run(interval_seconds=args.interval, max_iterations=args.max_iter)


if __name__ == "__main__":
    main()
