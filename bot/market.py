"""Market data adapter for Binance/CCXT."""

from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd
import pandas_ta as ta

try:
    import ccxt
except ImportError:  # pragma: no cover
    ccxt = None


@dataclass(slots=True)
class MarketSnapshot:
    candles: pd.DataFrame
    last_price: float
    rsi: float
    macd: float


class MarketDataClient:
    def __init__(self, exchange_id: str, testnet: bool = True):
        if ccxt is None:
            raise ImportError("ccxt is required for market connectivity. Install with: pip install ccxt")

        config: dict[str, object] = {"enableRateLimit": True}
        api_key = os.getenv("BINANCE_API_KEY")
        secret = os.getenv("BINANCE_SECRET")
        if api_key and secret:
            config["apiKey"] = api_key
            config["secret"] = secret

        exchange_cls = getattr(ccxt, exchange_id)
        self.exchange = exchange_cls(config)
        if exchange_id == "binance" and testnet:
            self.exchange.set_sandbox_mode(True)

    def fetch_recent_snapshot(self, symbol: str, timeframe: str, window_size: int) -> MarketSnapshot:
        candles = self.exchange.fetch_ohlcv(symbol, timeframe, limit=window_size + 50)
        df = pd.DataFrame(
            candles,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = self._add_indicators(df).tail(window_size).reset_index(drop=True)

        last_row = df.iloc[-1]
        return MarketSnapshot(
            candles=df,
            last_price=float(last_row["close"]),
            rsi=float(last_row.get("rsi", 50.0)),
            macd=float(last_row.get("MACD_12_26_9", 0.0)),
        )

    @staticmethod
    def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["rsi"] = ta.rsi(df["close"], length=14)
        macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
        df = pd.concat([df, macd], axis=1)
        return df.dropna().reset_index(drop=True)
