"""Typed contracts for trading signals and bot state."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal
import uuid


SignalAction = Literal["BUY", "SELL", "HOLD"]
ExecutionMode = Literal["paper", "binance_testnet", "binance_live"]
OrderSide = Literal["buy", "sell"]
OrderIntent = Literal["open_long", "close_long", "open_short", "close_short", "hold"]
OrderStatus = Literal["accepted", "rejected", "filled", "paper_filled", "cancelled", "error"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass(slots=True)
class SignalContract:
    """Machine-safe signal produced by the model pipeline."""

    signal_id: str
    timestamp: str
    symbol: str
    timeframe: str
    action: SignalAction
    confidence: float
    price: float
    prompt: str
    source: str
    reasons: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RiskDecision:
    approved: bool
    intent: OrderIntent
    side: OrderSide | None
    quantity: float
    notional_usd: float
    leverage: float
    reasons: list[str] = field(default_factory=list)
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    reduce_only: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class OrderRecord:
    order_id: str
    created_at: str
    signal_id: str
    symbol: str
    side: OrderSide
    intent: OrderIntent
    quantity: float
    requested_price: float
    mode: ExecutionMode
    status: OrderStatus
    exchange_order_id: str | None = None
    average_fill_price: float | None = None
    filled_quantity: float | None = None
    fees_usd: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FillRecord:
    fill_id: str
    order_id: str
    timestamp: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    fees_usd: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PositionRecord:
    symbol: str
    updated_at: str
    quantity: float
    entry_price: float
    mark_price: float
    realized_pnl: float
    unrealized_pnl: float
    mode: ExecutionMode
    status: Literal["flat", "long", "short"]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BotEvent:
    event_id: str
    timestamp: str
    level: Literal["INFO", "WARN", "ERROR", "CRITICAL"]
    event_type: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
