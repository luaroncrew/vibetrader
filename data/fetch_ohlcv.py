"""Fetch BTC/USDT OHLCV candles via ccxt and compute indicators."""

import argparse
import os
import time
from pathlib import Path

import ccxt
import pandas as pd
import pandas_ta as ta


def fetch_ohlcv(
    symbol: str = "BTC/USDT",
    timeframe: str = "4h",
    since_days: int = 730,
    exchange_id: str = "binance",
) -> pd.DataFrame:
    """Download OHLCV data from exchange.

    Args:
        symbol: Trading pair.
        timeframe: Candle interval.
        since_days: How many days back to fetch.
        exchange_id: ccxt exchange id.

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume.
    """
    config = {"enableRateLimit": True}
    api_key = os.environ.get("BINANCE_API_KEY")
    secret = os.environ.get("BINANCE_SECRET")
    if api_key and secret:
        config["apiKey"] = api_key
        config["secret"] = secret
    exchange = getattr(ccxt, exchange_id)(config)
    since = exchange.milliseconds() - since_days * 24 * 60 * 60 * 1000

    all_candles = []
    while True:
        candles = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not candles:
            break
        all_candles.extend(candles)
        since = candles[-1][0] + 1
        if len(candles) < 1000:
            break
        time.sleep(exchange.rateLimit / 1000)

    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add RSI and MACD indicators."""
    df["rsi"] = ta.rsi(df["close"], length=14)
    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    df = pd.concat([df, macd], axis=1)
    df = df.dropna().reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(description="Fetch OHLCV data")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="4h")
    parser.add_argument("--days", type=int, default=730)
    parser.add_argument("--output", default="data/btc_usdt_4h.csv")
    args = parser.parse_args()

    print(f"Fetching {args.symbol} {args.timeframe} candles for {args.days} days...")
    df = fetch_ohlcv(args.symbol, args.timeframe, args.days)
    print(f"Fetched {len(df)} candles")

    print("Computing indicators...")
    df = add_indicators(df)
    print(f"After indicator computation: {len(df)} candles")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
