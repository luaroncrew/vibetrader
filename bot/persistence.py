"""SQLite persistence layer for signals, orders, fills, positions, and metrics."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from bot.contracts import BotEvent, FillRecord, OrderRecord, PositionRecord, SignalContract


class Persistence:
    def __init__(self, sqlite_path: str):
        self.sqlite_path = sqlite_path
        Path(sqlite_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS signals (
                    signal_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    action TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    price REAL NOT NULL,
                    prompt TEXT NOT NULL,
                    source TEXT NOT NULL,
                    reasons_json TEXT NOT NULL,
                    diagnostics_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS orders (
                    order_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    signal_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    intent TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    requested_price REAL NOT NULL,
                    mode TEXT NOT NULL,
                    status TEXT NOT NULL,
                    exchange_order_id TEXT,
                    average_fill_price REAL,
                    filled_quantity REAL,
                    fees_usd REAL NOT NULL,
                    metadata_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS fills (
                    fill_id TEXT PRIMARY KEY,
                    order_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    fees_usd REAL NOT NULL,
                    metadata_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS positions (
                    symbol TEXT PRIMARY KEY,
                    updated_at TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    mark_price REAL NOT NULL,
                    realized_pnl REAL NOT NULL,
                    unrealized_pnl REAL NOT NULL,
                    mode TEXT NOT NULL,
                    status TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS bot_events (
                    event_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    level TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    details_json TEXT NOT NULL
                );
                """
            )

    def save_signal(self, signal: SignalContract) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO signals
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    signal.signal_id,
                    signal.timestamp,
                    signal.symbol,
                    signal.timeframe,
                    signal.action,
                    signal.confidence,
                    signal.price,
                    signal.prompt,
                    signal.source,
                    json.dumps(signal.reasons),
                    json.dumps(signal.diagnostics),
                ),
            )

    def save_order(self, order: OrderRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO orders
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    order.order_id,
                    order.created_at,
                    order.signal_id,
                    order.symbol,
                    order.side,
                    order.intent,
                    order.quantity,
                    order.requested_price,
                    order.mode,
                    order.status,
                    order.exchange_order_id,
                    order.average_fill_price,
                    order.filled_quantity,
                    order.fees_usd,
                    json.dumps(order.metadata),
                ),
            )

    def save_fill(self, fill: FillRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO fills
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fill.fill_id,
                    fill.order_id,
                    fill.timestamp,
                    fill.symbol,
                    fill.side,
                    fill.quantity,
                    fill.price,
                    fill.fees_usd,
                    json.dumps(fill.metadata),
                ),
            )

    def save_position(self, position: PositionRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO positions
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    position.symbol,
                    position.updated_at,
                    position.quantity,
                    position.entry_price,
                    position.mark_price,
                    position.realized_pnl,
                    position.unrealized_pnl,
                    position.mode,
                    position.status,
                ),
            )

    def save_event(self, event: BotEvent) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO bot_events
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.timestamp,
                    event.level,
                    event.event_type,
                    event.message,
                    json.dumps(event.details),
                ),
            )

    def get_position(self, symbol: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM positions WHERE symbol = ?",
                (symbol,),
            ).fetchone()
            return dict(row) if row else None

    def list_recent_orders(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM orders ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_realized_pnl_sum(self) -> float:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(realized_pnl), 0.0) AS total FROM positions"
            ).fetchone()
            return float(row["total"]) if row else 0.0
