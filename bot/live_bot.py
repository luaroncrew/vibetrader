"""Live trading bot using vibetrader's chart prediction pipeline.

Supports:
  - Paper mode (default): simulated portfolio, live price data, no real orders
  - Testnet mode (--testnet): real Binance testnet orders (requires testnet keys)
  - Live mode (--live): real Binance orders (requires BINANCE_API_KEY/SECRET)

Usage:
  # Paper trade BTC/USDT and ETH/USDT every 4h
  python -m bot.live_bot --checkpoint checkpoints/ --pairs BTC/USDT ETH/USDT

  # Live trading (Binance spot)
  python -m bot.live_bot --checkpoint checkpoints/ --pairs BTC/USDT --live

  # One-shot: run once and exit
  python -m bot.live_bot --checkpoint checkpoints/ --pairs BTC/USDT --once
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import ccxt
except ImportError:
    ccxt = None

from data.fetch_ohlcv import add_indicators
from data.render_charts import render_candlestick
from inference.predict import load_pipeline, predict
from inference.extract_signal import extract_signal
from inference.extract_signal import Signal as PixelSignal
from inference.extract_signal_mistral import extract_signal_mistral
from inference.extract_signal_mistral import Signal as MistralSignal
from bot.db import (
    DEFAULT_DB,
    init_db,
    log_signal,
    log_trade,
    save_position,
    get_positions,
    get_recent_trades,
    get_trade_summary,
)

# Union type for both signal flavours
AnySignal = Union[PixelSignal, MistralSignal]


@dataclass
class BotConfig:
    """Full trading bot configuration."""

    # --- Model ---
    checkpoint_path: str = "checkpoints/"
    device: str = "auto"
    use_mistral: bool = False

    # --- Trading pairs & timeframe ---
    pairs: List[str] = field(default_factory=lambda: ["BTC/USDT"])
    timeframe: str = "4h"

    # --- Schedule ---
    interval_seconds: int = 14400  # 4 hours

    # --- Risk management ---
    position_size_pct: float = 0.10    # fraction of available balance per entry
    stop_loss_pct: float = 0.03        # close long if price drops this much from entry
    take_profit_pct: float = 0.06      # close long if price rises this much from entry
    min_confidence: float = 0.65       # ignore signals below this confidence

    # --- Chart settings (must match training) ---
    window_size: int = 40
    future_candles: int = 4

    # --- Mode ---
    paper: bool = True           # paper trading (simulated)
    testnet: bool = False        # Binance testnet (real API calls, fake money)
    live: bool = False           # live Binance spot trading (real money)
    initial_balance: float = 10000.0

    # --- Storage ---
    db_path: str = "data/trades.db"


class LiveTradingBot:
    """Automated trading bot driven by vibetrader's diffusion inference pipeline.

    Spot-only (no shorting on SELL signals):
      - BUY signal  → open long (if not already long)
      - SELL signal → close long (if we have one), then stay flat
      - HOLD signal → do nothing
      - Stop-loss / take-profit → close long automatically
    """

    def __init__(self, config: BotConfig):
        self.config = config
        self.db_path = Path(config.db_path)

        if config.live and not config.paper:
            self.mode = "live"
        elif config.testnet:
            self.mode = "testnet"
        else:
            self.mode = "paper"

        init_db(self.db_path)

        # Paper trading state (in-memory, persisted to DB)
        self.paper_balance: float = config.initial_balance
        self.paper_positions: Dict[str, dict] = {}

        # Load inference model
        print(f"[{self.mode.upper()}] Loading model from {config.checkpoint_path}...")
        self.pipe = load_pipeline(config.checkpoint_path, device=config.device)
        print(f"[{self.mode.upper()}] Model loaded.")

        # Configure exchange
        if ccxt is None:
            raise ImportError("ccxt is required. Install: pip install ccxt")

        exchange_cfg: dict = {"enableRateLimit": True}

        api_key = os.environ.get("BINANCE_API_KEY")
        # Support both naming conventions (fetch_ohlcv.py uses BINANCE_SECRET)
        api_secret = os.environ.get("BINANCE_API_SECRET") or os.environ.get("BINANCE_SECRET")

        if self.mode == "live":
            if not api_key or not api_secret:
                raise ValueError(
                    "BINANCE_API_KEY and BINANCE_API_SECRET must be set for live trading."
                )
            exchange_cfg["apiKey"] = api_key
            exchange_cfg["secret"] = api_secret

        elif self.mode == "testnet":
            testnet_key = os.environ.get("BINANCE_TESTNET_API_KEY") or api_key
            testnet_secret = os.environ.get("BINANCE_TESTNET_API_SECRET") or api_secret
            if not testnet_key or not testnet_secret:
                raise ValueError(
                    "Testnet requires BINANCE_TESTNET_API_KEY and BINANCE_TESTNET_API_SECRET "
                    "(or BINANCE_API_KEY/SECRET pointing at testnet keys)."
                )
            exchange_cfg["apiKey"] = testnet_key
            exchange_cfg["secret"] = testnet_secret
            exchange_cfg["urls"] = {
                "api": {
                    "public": "https://testnet.binance.vision/api",
                    "private": "https://testnet.binance.vision/api",
                }
            }

        else:  # paper – use public endpoints only (no auth needed for OHLCV)
            if api_key and api_secret:
                exchange_cfg["apiKey"] = api_key
                exchange_cfg["secret"] = api_secret

        self.exchange = ccxt.binance(exchange_cfg)

        # Restore paper positions from DB so a restart doesn't lose state
        if self.mode == "paper":
            self._restore_paper_positions()

        self._print_config()

    # ------------------------------------------------------------------ #
    # Initialisation helpers
    # ------------------------------------------------------------------ #

    @property
    def live(self) -> bool:
        return self.mode in ("live", "testnet")

    def _print_config(self) -> None:
        print(f"\n[{self.mode.upper()}] Bot ready")
        print(f"  Pairs      : {self.config.pairs}")
        print(f"  Timeframe  : {self.config.timeframe}")
        print(f"  Interval   : {self.config.interval_seconds}s ({self.config.interval_seconds/3600:.1f}h)")
        print(f"  Pos size   : {self.config.position_size_pct:.0%} of available balance")
        print(f"  Stop-loss  : {self.config.stop_loss_pct:.0%}")
        print(f"  Take-profit: {self.config.take_profit_pct:.0%}")
        print(f"  Min conf   : {self.config.min_confidence:.2f}")
        if self.mode == "paper":
            print(f"  Paper bal  : ${self.paper_balance:,.2f}")

    def _restore_paper_positions(self) -> None:
        """Load previously saved paper positions from DB."""
        for pos in get_positions(self.db_path, mode="paper"):
            self.paper_positions[pos["pair"]] = {
                "side": pos["side"],
                "entry_price": pos["entry_price"],
                "size_base": pos["size_base"],
                "size_quote": pos["size_quote"],
            }
        if self.paper_positions:
            print(f"  Restored {len(self.paper_positions)} paper position(s) from DB.")

    # ------------------------------------------------------------------ #
    # Data fetching
    # ------------------------------------------------------------------ #

    def fetch_candles(self, pair: str) -> pd.DataFrame:
        """Fetch recent OHLCV candles and compute RSI/MACD indicators."""
        # Fetch extra candles so indicators have enough history after dropna()
        limit = self.config.window_size + 60
        raw = self.exchange.fetch_ohlcv(pair, self.config.timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = add_indicators(df)
        return df.tail(self.config.window_size).reset_index(drop=True)

    def get_current_price(self, pair: str) -> float:
        ticker = self.exchange.fetch_ticker(pair)
        return float(ticker["last"])

    # ------------------------------------------------------------------ #
    # Inference pipeline
    # ------------------------------------------------------------------ #

    def get_signal(self, pair: str, df: pd.DataFrame) -> AnySignal:
        """Run the full vibetrader inference pipeline for one pair."""
        total_slots = self.config.window_size + self.config.future_candles
        price_low = float(df["low"].min())
        price_high = float(df["high"].max())

        chart_img = render_candlestick(
            df,
            draw_marker=False,
            total_slots=total_slots,
            price_low=price_low,
            price_high=price_high,
        )

        rsi = float(df.iloc[-1].get("rsi", 50.0))
        macd = float(df.iloc[-1].get("MACD_12_26_9", 0.0))
        prompt = (
            f"Predict next {self.config.future_candles} candles. "
            f"RSI={round(rsi, 1)}, MACD={round(macd, 2)}"
        )

        generated_img = predict(self.pipe, chart_img, prompt)

        if self.config.use_mistral:
            signal = extract_signal_mistral(chart_img, generated_img)
        else:
            signal = extract_signal(generated_img)

        # Persist signal to DB
        log_signal(
            self.db_path,
            pair=pair,
            action=signal.action,
            confidence=signal.confidence,
            green_pct=getattr(signal, "green_pct", 0.0),
            red_pct=getattr(signal, "red_pct", 0.0),
            reasoning=getattr(signal, "reasoning", ""),
            mode=self.mode,
        )

        return signal

    # ------------------------------------------------------------------ #
    # Risk management
    # ------------------------------------------------------------------ #

    def _get_position(self, pair: str) -> Optional[dict]:
        """Return current open position for pair (paper or live)."""
        if self.mode == "paper":
            return self.paper_positions.get(pair)
        for pos in get_positions(self.db_path, mode=self.mode):
            if pos["pair"] == pair:
                return pos
        return None

    def check_exit_conditions(self, pair: str, current_price: float) -> Optional[str]:
        """Return 'STOP_LOSS', 'TAKE_PROFIT', or None."""
        pos = self._get_position(pair)
        if not pos or pos.get("side") != "long":
            return None

        entry = pos["entry_price"]
        pct = (current_price - entry) / entry

        if pct <= -self.config.stop_loss_pct:
            return "STOP_LOSS"
        if pct >= self.config.take_profit_pct:
            return "TAKE_PROFIT"
        return None

    # ------------------------------------------------------------------ #
    # Trade execution – paper
    # ------------------------------------------------------------------ #

    def _paper_open_long(self, pair: str, price: float, signal: AnySignal) -> None:
        """Open a paper long position."""
        quote_amount = self.paper_balance * self.config.position_size_pct
        if quote_amount < 1.0:
            print(f"  [{pair}] Insufficient paper balance (${self.paper_balance:.2f})")
            return

        size_base = quote_amount / price
        self.paper_balance -= quote_amount
        self.paper_positions[pair] = {
            "side": "long",
            "entry_price": price,
            "size_base": size_base,
            "size_quote": quote_amount,
        }
        save_position(self.db_path, pair, "long", price, size_base, quote_amount, "paper")
        log_trade(
            self.db_path, pair, "BUY", price, size_base, quote_amount,
            None, signal.action, signal.confidence, "paper",
        )
        print(
            f"  [{pair}] PAPER BUY  {size_base:.6f} @ ${price:,.4f}"
            f"  (${quote_amount:,.2f})"
        )

    def _paper_close_long(self, pair: str, price: float, signal: AnySignal, reason: str) -> None:
        """Close a paper long position."""
        pos = self.paper_positions.get(pair)
        if not pos or pos.get("side") != "long":
            print(f"  [{pair}] No open long to close")
            return

        size_base = pos["size_base"]
        proceeds = size_base * price
        pnl = proceeds - pos["size_quote"]
        pnl_pct = pnl / pos["size_quote"]

        self.paper_balance += proceeds
        del self.paper_positions[pair]
        save_position(self.db_path, pair, None, 0, 0, 0, "paper")
        log_trade(
            self.db_path, pair, "SELL", price, size_base, proceeds,
            pnl, signal.action, signal.confidence, "paper", reason,
        )
        print(
            f"  [{pair}] PAPER SELL {size_base:.6f} @ ${price:,.4f}"
            f"  P&L: ${pnl:+,.2f} ({pnl_pct:+.1%})  [{reason}]"
        )

    def execute_paper_trade(
        self, pair: str, action: str, price: float, signal: AnySignal, reason: str = "signal"
    ) -> None:
        if action == "BUY":
            pos = self.paper_positions.get(pair)
            if pos and pos.get("side") == "long":
                print(f"  [{pair}] Already long – skipping BUY")
                return
            self._paper_open_long(pair, price, signal)

        elif action in ("SELL", "CLOSE"):
            self._paper_close_long(pair, price, signal, reason)

    # ------------------------------------------------------------------ #
    # Trade execution – live / testnet
    # ------------------------------------------------------------------ #

    def execute_live_trade(
        self, pair: str, action: str, price: float, signal: AnySignal, reason: str = "signal"
    ) -> None:
        try:
            if action == "BUY":
                pos = self._get_position(pair)
                if pos and pos.get("side") == "long":
                    print(f"  [{pair}] Already long – skipping BUY")
                    return

                # Determine quote available
                quote_currency = pair.split("/")[1]
                balance = self.exchange.fetch_balance()
                available = float(balance["free"].get(quote_currency, 0))
                quote_amount = available * self.config.position_size_pct

                if quote_amount < 1.0:
                    print(f"  [{pair}] Insufficient balance ({quote_currency}: {available:.4f})")
                    return

                # ccxt market buy by quote amount
                order = self.exchange.create_market_buy_order(
                    pair,
                    self.exchange.amount_to_precision(pair, quote_amount / price),
                )
                filled_price = float(order.get("average") or price)
                filled_size = float(order.get("filled") or quote_amount / price)
                filled_cost = float(order.get("cost") or filled_size * filled_price)

                save_position(self.db_path, pair, "long", filled_price, filled_size, filled_cost, self.mode)
                log_trade(
                    self.db_path, pair, "BUY", filled_price, filled_size, filled_cost,
                    None, signal.action, signal.confidence, self.mode, reason,
                )
                print(
                    f"  [{pair}] {self.mode.upper()} BUY  {filled_size:.6f}"
                    f" @ ${filled_price:,.4f}  (${filled_cost:,.2f})"
                )

            elif action in ("SELL", "CLOSE"):
                pos = self._get_position(pair)
                if not pos or pos.get("side") != "long":
                    print(f"  [{pair}] No open long to close")
                    return

                size_base = pos["size_base"]
                order = self.exchange.create_market_sell_order(
                    pair,
                    self.exchange.amount_to_precision(pair, size_base),
                )
                filled_price = float(order.get("average") or price)
                filled_size = float(order.get("filled") or size_base)
                proceeds = float(order.get("cost") or filled_size * filled_price)
                pnl = proceeds - pos["size_quote"]

                save_position(self.db_path, pair, None, 0, 0, 0, self.mode)
                log_trade(
                    self.db_path, pair, "SELL", filled_price, filled_size, proceeds,
                    pnl, signal.action, signal.confidence, self.mode, reason,
                )
                print(
                    f"  [{pair}] {self.mode.upper()} SELL {filled_size:.6f}"
                    f" @ ${filled_price:,.4f}  P&L: ${pnl:+,.2f}  [{reason}]"
                )

        except Exception as exc:
            print(f"  [{pair}] Trade execution failed: {exc}")

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #

    def _paper_portfolio_value(self, prices: Dict[str, float]) -> float:
        total = self.paper_balance
        for pair, pos in self.paper_positions.items():
            if pos.get("side") == "long":
                p = prices.get(pair, pos["entry_price"])
                total += pos["size_base"] * p
        return total

    def run_once(self) -> Dict[str, float]:
        """Process all configured pairs once. Returns {pair: current_price}."""
        now = datetime.utcnow().isoformat()
        print(f"\n[{now}] Cycle start  mode={self.mode}  pairs={self.config.pairs}")

        prices: Dict[str, float] = {}

        for pair in self.config.pairs:
            print(f"\n  ── {pair} ──")
            try:
                df = self.fetch_candles(pair)
                if len(df) < self.config.window_size:
                    print(f"  [{pair}] Only {len(df)} candles – need {self.config.window_size}, skipping")
                    continue

                current_price = float(df.iloc[-1]["close"])
                prices[pair] = current_price
                print(f"  [{pair}] Price: ${current_price:,.4f}")

                # 1. Check stop-loss / take-profit before running inference
                exit_trigger = self.check_exit_conditions(pair, current_price)
                if exit_trigger:
                    print(f"  [{pair}] {exit_trigger} triggered @ ${current_price:,.4f}")
                    dummy_signal = PixelSignal(
                        action="HOLD", confidence=1.0, green_pct=0.0, red_pct=0.0
                    )
                    if self.mode == "paper":
                        self.execute_paper_trade(pair, "CLOSE", current_price, dummy_signal, exit_trigger)
                    else:
                        self.execute_live_trade(pair, "CLOSE", current_price, dummy_signal, exit_trigger)
                    continue

                # 2. Run inference pipeline
                signal = self.get_signal(pair, df)
                green = getattr(signal, "green_pct", None)
                red = getattr(signal, "red_pct", None)
                extra = f"green={green:.1%} red={red:.1%}" if green is not None else getattr(signal, "reasoning", "")[:60]
                print(
                    f"  [{pair}] Signal: {signal.action}  conf={signal.confidence:.2f}  {extra}"
                )

                # 3. Skip low-confidence signals
                if signal.action != "HOLD" and signal.confidence < self.config.min_confidence:
                    print(
                        f"  [{pair}] Confidence {signal.confidence:.2f} < {self.config.min_confidence:.2f} – skipping"
                    )
                    continue

                # 4. Execute trade
                if signal.action == "BUY":
                    if self.mode == "paper":
                        self.execute_paper_trade(pair, "BUY", current_price, signal)
                    else:
                        self.execute_live_trade(pair, "BUY", current_price, signal)

                elif signal.action == "SELL":
                    # Spot-only: SELL means close existing long (no shorting)
                    pos = self._get_position(pair)
                    if pos and pos.get("side") == "long":
                        if self.mode == "paper":
                            self.execute_paper_trade(pair, "CLOSE", current_price, signal)
                        else:
                            self.execute_live_trade(pair, "CLOSE", current_price, signal)
                    else:
                        print(f"  [{pair}] SELL signal but no open position – staying flat")

                else:
                    print(f"  [{pair}] HOLD – no action")

            except Exception as exc:
                print(f"  [{pair}] Error: {exc}")

        # Portfolio summary
        if self.mode == "paper" and prices:
            total = self._paper_portfolio_value(prices)
            initial = self.config.initial_balance
            ret = (total - initial) / initial
            n_pos = len(self.paper_positions)
            print(
                f"\n  Portfolio: ${total:,.2f} ({ret:+.2%})"
                f"  Cash: ${self.paper_balance:,.2f}"
                f"  Open positions: {n_pos}"
            )

        return prices

    def run(self) -> None:
        """Main loop: run_once() then sleep until next candle."""
        interval = self.config.interval_seconds
        print(f"\n[{self.mode.upper()}] Bot started. Press Ctrl+C to stop.")

        iteration = 0
        while True:
            try:
                iteration += 1
                print(f"\n{'='*60}")
                print(f"Iteration #{iteration}  ({datetime.utcnow().isoformat()})")
                self.run_once()
                print(f"\n  Sleeping {interval}s until next cycle...")
                time.sleep(interval)
            except KeyboardInterrupt:
                print("\n\n[BOT] Keyboard interrupt – stopping.")
                break
            except Exception as exc:
                print(f"\n[BOT] Unhandled error: {exc}")
                print("[BOT] Retrying in 60s...")
                time.sleep(60)

        # Final summary
        stats = get_trade_summary(self.db_path, mode=self.mode)
        print(f"\n{'='*60}")
        print(f"Session summary  (mode={self.mode})")
        print(f"  Closed trades : {stats['total_closed']}")
        print(f"  Total P&L     : ${stats['total_pnl']:+,.4f}")
        print(f"  Win rate      : {stats['win_rate']:.1%}")
        print(f"  Avg P&L/trade : ${stats['avg_pnl']:+,.4f}")
        if self.mode == "paper":
            print(f"  Final cash    : ${self.paper_balance:,.2f}")
        print("Bot stopped.")


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Vibetrader live trading bot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    p.add_argument("--checkpoint", default="checkpoints/", help="Path to model checkpoint")
    p.add_argument("--device", default="auto", help="Inference device (auto/cpu/cuda/mps)")
    p.add_argument("--mistral", action="store_true", help="Use Mistral Pixtral for signal extraction")

    # Pairs & schedule
    p.add_argument("--pairs", nargs="+", default=["BTC/USDT"], help="Trading pairs")
    p.add_argument("--timeframe", default="4h", help="Candlestick timeframe")
    p.add_argument(
        "--interval", type=int, default=14400,
        help="Seconds between cycles (default: 14400 = 4h)"
    )

    # Risk management
    p.add_argument("--position-size", type=float, default=0.10, help="Fraction of balance per trade")
    p.add_argument("--stop-loss", type=float, default=0.03, help="Stop-loss fraction (e.g. 0.03 = 3%%)")
    p.add_argument("--take-profit", type=float, default=0.06, help="Take-profit fraction (e.g. 0.06 = 6%%)")
    p.add_argument("--min-confidence", type=float, default=0.65, help="Minimum signal confidence to trade")

    # Mode
    mode_group = p.add_mutually_exclusive_group()
    mode_group.add_argument("--live", action="store_true", help="Live Binance trading (real money!)")
    mode_group.add_argument("--testnet", action="store_true", help="Binance testnet trading")

    # Paper trading
    p.add_argument("--balance", type=float, default=10000.0, help="Initial paper balance (USD)")

    # Misc
    p.add_argument("--db", default="data/trades.db", help="SQLite database path")
    p.add_argument("--once", action="store_true", help="Run one cycle and exit (no loop)")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    config = BotConfig(
        checkpoint_path=args.checkpoint,
        device=args.device,
        use_mistral=args.mistral,
        pairs=args.pairs,
        timeframe=args.timeframe,
        interval_seconds=args.interval,
        position_size_pct=args.position_size,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
        min_confidence=args.min_confidence,
        paper=not (args.live or args.testnet),
        testnet=args.testnet,
        live=args.live,
        initial_balance=args.balance,
        db_path=args.db,
    )

    bot = LiveTradingBot(config)

    if args.once:
        bot.run_once()
    else:
        bot.run()


if __name__ == "__main__":
    main()
