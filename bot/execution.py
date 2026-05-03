"""Execution adapters for paper and Binance trading."""

from __future__ import annotations

import os

try:
    import ccxt
except ImportError:  # pragma: no cover
    ccxt = None

from bot.contracts import FillRecord, OrderRecord, PositionRecord, new_id, utc_now_iso


class BaseExecutionClient:
    def submit_order(self, order: OrderRecord) -> tuple[OrderRecord, FillRecord]:
        raise NotImplementedError


class PaperExecutionClient(BaseExecutionClient):
    def __init__(self, fee_bps: float, slippage_bps: float):
        self.fee_bps = fee_bps
        self.slippage_bps = slippage_bps

    def submit_order(self, order: OrderRecord) -> tuple[OrderRecord, FillRecord]:
        direction = 1 if order.side == "buy" else -1
        slip_multiplier = 1 + (direction * self.slippage_bps / 10_000)
        fill_price = order.requested_price * slip_multiplier
        fees = fill_price * order.quantity * (self.fee_bps / 10_000)
        order.status = "paper_filled"
        order.average_fill_price = fill_price
        order.filled_quantity = order.quantity
        order.fees_usd = fees
        fill = FillRecord(
            fill_id=new_id("fill"),
            order_id=order.order_id,
            timestamp=utc_now_iso(),
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            fees_usd=fees,
            metadata={"execution": "paper"},
        )
        return order, fill


class BinanceExecutionClient(BaseExecutionClient):
    def __init__(self, testnet: bool = True):
        if ccxt is None:
            raise ImportError("ccxt is required for Binance execution. Install with: pip install ccxt")
        api_key = os.getenv("BINANCE_API_KEY")
        secret = os.getenv("BINANCE_SECRET")
        if not api_key or not secret:
            raise RuntimeError("BINANCE_API_KEY and BINANCE_SECRET are required for Binance execution")

        self.exchange = ccxt.binance({
            "apiKey": api_key,
            "secret": secret,
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })
        self.exchange.set_sandbox_mode(testnet)

    def submit_order(self, order: OrderRecord) -> tuple[OrderRecord, FillRecord]:
        response = self.exchange.create_order(
            symbol=order.symbol,
            type="market",
            side=order.side,
            amount=order.quantity,
            params={},
        )
        filled = float(response.get("filled") or order.quantity)
        avg_price = float(response.get("average") or response.get("price") or order.requested_price)
        fee_cost = 0.0
        fees = response.get("fees") or []
        if fees:
            fee_cost = sum(float(fee.get("cost", 0.0)) for fee in fees)
        order.status = "filled"
        order.exchange_order_id = str(response.get("id"))
        order.average_fill_price = avg_price
        order.filled_quantity = filled
        order.fees_usd = fee_cost
        fill = FillRecord(
            fill_id=new_id("fill"),
            order_id=order.order_id,
            timestamp=utc_now_iso(),
            symbol=order.symbol,
            side=order.side,
            quantity=filled,
            price=avg_price,
            fees_usd=fee_cost,
            metadata={"execution": "binance", "exchange_order_id": order.exchange_order_id},
        )
        return order, fill


def flat_position(symbol: str, mode: str) -> PositionRecord:
    return PositionRecord(
        symbol=symbol,
        updated_at=utc_now_iso(),
        quantity=0.0,
        entry_price=0.0,
        mark_price=0.0,
        realized_pnl=0.0,
        unrealized_pnl=0.0,
        mode=mode,
        status="flat",
    )
