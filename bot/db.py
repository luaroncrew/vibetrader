"""SQLite database layer for trade and signal logging."""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_DB = Path("data/trades.db")


def init_db(db_path: Path = DEFAULT_DB) -> None:
    """Initialize database with required tables."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT    NOT NULL,
            pair        TEXT    NOT NULL,
            action      TEXT    NOT NULL,
            confidence  REAL,
            green_pct   REAL,
            red_pct     REAL,
            reasoning   TEXT,
            mode        TEXT    NOT NULL
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp           TEXT    NOT NULL,
            pair                TEXT    NOT NULL,
            action              TEXT    NOT NULL,
            price               REAL    NOT NULL,
            size_base           REAL    NOT NULL,
            size_quote          REAL    NOT NULL,
            pnl                 REAL,
            signal_action       TEXT,
            signal_confidence   REAL,
            mode                TEXT    NOT NULL,
            notes               TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS positions (
            pair        TEXT    PRIMARY KEY,
            side        TEXT,
            entry_price REAL,
            size_base   REAL,
            size_quote  REAL,
            opened_at   TEXT,
            mode        TEXT
        )
    """)

    conn.commit()
    conn.close()


def log_signal(
    db_path: Path,
    pair: str,
    action: str,
    confidence: float,
    green_pct: float = 0.0,
    red_pct: float = 0.0,
    reasoning: str = "",
    mode: str = "paper",
) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO signals (timestamp, pair, action, confidence, green_pct, red_pct, reasoning, mode) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (datetime.utcnow().isoformat(), pair, action, confidence, green_pct, red_pct, reasoning, mode),
    )
    conn.commit()
    conn.close()


def log_trade(
    db_path: Path,
    pair: str,
    action: str,
    price: float,
    size_base: float,
    size_quote: float,
    pnl: Optional[float],
    signal_action: str,
    signal_confidence: float,
    mode: str,
    notes: str = "",
) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO trades (timestamp, pair, action, price, size_base, size_quote, pnl, "
        "signal_action, signal_confidence, mode, notes) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (
            datetime.utcnow().isoformat(), pair, action, price, size_base, size_quote, pnl,
            signal_action, signal_confidence, mode, notes,
        ),
    )
    conn.commit()
    conn.close()


def save_position(
    db_path: Path,
    pair: str,
    side: Optional[str],
    entry_price: float,
    size_base: float,
    size_quote: float,
    mode: str,
) -> None:
    """Upsert or delete a position record."""
    conn = sqlite3.connect(db_path)
    if side is None:
        conn.execute("DELETE FROM positions WHERE pair=? AND mode=?", (pair, mode))
    else:
        conn.execute(
            "INSERT OR REPLACE INTO positions (pair, side, entry_price, size_base, size_quote, opened_at, mode) "
            "VALUES (?,?,?,?,?,?,?)",
            (pair, side, entry_price, size_base, size_quote, datetime.utcnow().isoformat(), mode),
        )
    conn.commit()
    conn.close()


def get_positions(db_path: Path, mode: Optional[str] = None) -> List[Dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    if mode:
        rows = conn.execute(
            "SELECT * FROM positions WHERE side IS NOT NULL AND mode=?", (mode,)
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM positions WHERE side IS NOT NULL").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_recent_trades(db_path: Path, n: int = 20, pair: Optional[str] = None) -> List[Dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    if pair:
        rows = conn.execute(
            "SELECT * FROM trades WHERE pair=? ORDER BY timestamp DESC LIMIT ?", (pair, n)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?", (n,)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_recent_signals(db_path: Path, n: int = 20, pair: Optional[str] = None) -> List[Dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    if pair:
        rows = conn.execute(
            "SELECT * FROM signals WHERE pair=? ORDER BY timestamp DESC LIMIT ?", (pair, n)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM signals ORDER BY timestamp DESC LIMIT ?", (n,)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_trade_summary(db_path: Path, mode: Optional[str] = None) -> Dict:
    """Aggregate trade stats: total trades, total pnl, win rate."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    query = "SELECT * FROM trades WHERE action IN ('SELL', 'CLOSE') AND pnl IS NOT NULL"
    params: tuple = ()
    if mode:
        query += " AND mode=?"
        params = (mode,)
    rows = conn.execute(query, params).fetchall()
    conn.close()

    if not rows:
        return {"total_closed": 0, "total_pnl": 0.0, "win_rate": 0.0, "avg_pnl": 0.0}

    pnls = [r["pnl"] for r in rows]
    wins = sum(1 for p in pnls if p > 0)
    return {
        "total_closed": len(pnls),
        "total_pnl": round(sum(pnls), 4),
        "win_rate": round(wins / len(pnls), 4),
        "avg_pnl": round(sum(pnls) / len(pnls), 4),
    }
