"""CLI status command: show current positions, recent trades, and P&L.

Usage:
  python -m bot.status
  python -m bot.status --db data/trades.db --trades 20 --signals 10
  python -m bot.status --pair BTC/USDT
  python -m bot.status --json          # machine-readable output
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.db import (
    DEFAULT_DB,
    get_positions,
    get_recent_signals,
    get_recent_trades,
    get_trade_summary,
)


def _fmt_pnl(pnl) -> str:
    if pnl is None:
        return "—"
    sign = "+" if pnl >= 0 else ""
    return f"{sign}${pnl:,.4f}"


def print_status(
    db_path: Path = DEFAULT_DB,
    n_trades: int = 10,
    n_signals: int = 5,
    pair: str = None,
    mode: str = None,
    as_json: bool = False,
) -> dict:
    """Print a human-readable status report. Returns raw data dict."""

    positions = get_positions(db_path, mode=mode)
    trades = get_recent_trades(db_path, n=n_trades, pair=pair)
    signals = get_recent_signals(db_path, n=n_signals, pair=pair)
    summary = get_trade_summary(db_path, mode=mode)

    if as_json:
        data = {
            "positions": positions,
            "recent_trades": trades,
            "recent_signals": signals,
            "summary": summary,
        }
        print(json.dumps(data, indent=2, default=str))
        return data

    SEP = "─" * 68

    # ── Open positions ───────────────────────────────────────────────────
    print(f"\n{'═'*68}")
    print("  VIBETRADER  –  STATUS")
    print(f"{'═'*68}")

    print(f"\n  OPEN POSITIONS  ({len(positions)} total)")
    print(f"  {SEP}")
    if not positions:
        print("  (none)")
    else:
        hdr = f"  {'Pair':<14} {'Mode':<8} {'Side':<6} {'Entry':>12} {'Size (base)':>14} {'Quote':>12}"
        print(hdr)
        print(f"  {SEP}")
        for pos in positions:
            print(
                f"  {pos['pair']:<14} {pos.get('mode','?'):<8} {pos['side']:<6}"
                f" {pos['entry_price']:>12,.4f} {pos['size_base']:>14,.6f}"
                f" ${pos['size_quote']:>11,.2f}"
            )

    # ── Recent signals ───────────────────────────────────────────────────
    print(f"\n  RECENT SIGNALS  (last {n_signals})")
    print(f"  {SEP}")
    if not signals:
        print("  (none)")
    else:
        hdr = f"  {'Timestamp':<24} {'Pair':<14} {'Action':<6} {'Conf':>6} {'Green':>7} {'Red':>7}"
        print(hdr)
        print(f"  {SEP}")
        for s in signals:
            ts = str(s["timestamp"])[:19]
            g = f"{s['green_pct']:.1%}" if s.get("green_pct") is not None else "  —  "
            r = f"{s['red_pct']:.1%}" if s.get("red_pct") is not None else "  —  "
            print(
                f"  {ts:<24} {s['pair']:<14} {s['action']:<6}"
                f" {s['confidence']:>6.2f} {g:>7} {r:>7}"
            )

    # ── Recent trades ────────────────────────────────────────────────────
    print(f"\n  RECENT TRADES  (last {n_trades})")
    print(f"  {SEP}")
    if not trades:
        print("  (none)")
    else:
        hdr = (
            f"  {'Timestamp':<24} {'Pair':<14} {'Action':<6}"
            f" {'Price':>12} {'Size (base)':>13} {'P&L':>12} {'Notes'}"
        )
        print(hdr)
        print(f"  {SEP}")
        for t in trades:
            ts = str(t["timestamp"])[:19]
            notes = (t.get("notes") or "")[:20]
            print(
                f"  {ts:<24} {t['pair']:<14} {t['action']:<6}"
                f" {t['price']:>12,.4f} {t['size_base']:>13,.6f}"
                f" {_fmt_pnl(t.get('pnl')):>12}  {notes}"
            )

    # ── Aggregate summary ────────────────────────────────────────────────
    print(f"\n  CLOSED-TRADE SUMMARY{' (mode=' + mode + ')' if mode else ''}")
    print(f"  {SEP}")
    print(f"  Closed trades : {summary['total_closed']}")
    print(f"  Total P&L     : {_fmt_pnl(summary['total_pnl'])}")
    print(f"  Win rate      : {summary['win_rate']:.1%}")
    print(f"  Avg P&L/trade : {_fmt_pnl(summary['avg_pnl'])}")
    print(f"\n{'═'*68}\n")

    return {
        "positions": positions,
        "recent_trades": trades,
        "recent_signals": signals,
        "summary": summary,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Show vibetrader bot status",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--db", default=str(DEFAULT_DB), help="SQLite database path")
    p.add_argument("--trades", type=int, default=10, help="Number of recent trades to show")
    p.add_argument("--signals", type=int, default=5, help="Number of recent signals to show")
    p.add_argument("--pair", default=None, help="Filter by trading pair")
    p.add_argument("--mode", default=None, choices=["paper", "live", "testnet"], help="Filter by mode")
    p.add_argument("--json", action="store_true", dest="as_json", help="Output raw JSON")
    args = p.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        print("The bot has not run yet, or use --db to specify a different path.")
        sys.exit(0)

    print_status(
        db_path=db_path,
        n_trades=args.trades,
        n_signals=args.signals,
        pair=args.pair,
        mode=args.mode,
        as_json=args.as_json,
    )


if __name__ == "__main__":
    main()
