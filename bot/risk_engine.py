"""Risk engine and automatic kill-switch logic."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import floor

from bot.config import RiskConfig
from bot.contracts import PositionRecord, RiskDecision


@dataclass(slots=True)
class PortfolioState:
    cash_usd: float
    equity_usd: float
    start_equity_usd: float
    daily_start_equity_usd: float
    consecutive_losses: int
    current_position: PositionRecord


class RiskEngine:
    def __init__(self, config: RiskConfig):
        self.config = config

    def evaluate(self, signal, portfolio: PortfolioState) -> RiskDecision:
        reasons: list[str] = []
        current_qty = portfolio.current_position.quantity
        price = signal.price

        if signal.confidence < self.config.min_confidence:
            reasons.append(
                f"confidence {signal.confidence:.3f} below threshold {self.config.min_confidence:.3f}"
            )
            return RiskDecision(False, "hold", None, 0.0, 0.0, 1.0, reasons)

        total_drawdown = 1.0 - (portfolio.equity_usd / max(portfolio.start_equity_usd, 1e-9))
        daily_drawdown = 1.0 - (
            portfolio.equity_usd / max(portfolio.daily_start_equity_usd, 1e-9)
        )
        if total_drawdown >= self.config.max_total_drawdown_pct:
            reasons.append("total drawdown limit breached")
            return RiskDecision(False, "hold", None, 0.0, 0.0, 1.0, reasons)
        if daily_drawdown >= self.config.max_daily_drawdown_pct:
            reasons.append("daily drawdown limit breached")
            return RiskDecision(False, "hold", None, 0.0, 0.0, 1.0, reasons)
        if portfolio.consecutive_losses >= self.config.max_consecutive_losses:
            reasons.append("consecutive loss limit breached")
            return RiskDecision(False, "hold", None, 0.0, 0.0, 1.0, reasons)

        risk_budget = min(
            portfolio.equity_usd * self.config.max_notional_fraction,
            self.config.max_position_notional_usd,
        )
        quantity = floor((risk_budget / price) * 1_000_000) / 1_000_000
        if quantity <= 0:
            reasons.append("computed quantity rounded to zero")
            return RiskDecision(False, "hold", None, 0.0, 0.0, 1.0, reasons)

        if signal.action == "BUY":
            if current_qty < 0:
                return RiskDecision(
                    True,
                    "close_short",
                    "buy",
                    abs(current_qty),
                    abs(current_qty) * price,
                    1.0,
                    ["closing short before changing direction"],
                    self.config.stop_loss_pct,
                    self.config.take_profit_pct,
                    True,
                )
            if current_qty > 0:
                reasons.append("already long")
                return RiskDecision(False, "hold", None, 0.0, 0.0, 1.0, reasons)
            return RiskDecision(
                True,
                "open_long",
                "buy",
                quantity,
                quantity * price,
                1.0,
                ["long entry approved"],
                self.config.stop_loss_pct,
                self.config.take_profit_pct,
                False,
            )

        if signal.action == "SELL":
            if current_qty > 0:
                return RiskDecision(
                    True,
                    "close_long",
                    "sell",
                    current_qty,
                    current_qty * price,
                    1.0,
                    ["closing long on sell signal"],
                    self.config.stop_loss_pct,
                    self.config.take_profit_pct,
                    True,
                )
            if self.config.allow_short:
                if current_qty < 0:
                    reasons.append("already short")
                    return RiskDecision(False, "hold", None, 0.0, 0.0, 1.0, reasons)
                return RiskDecision(
                    True,
                    "open_short",
                    "sell",
                    quantity,
                    quantity * price,
                    1.0,
                    ["short entry approved"],
                    self.config.stop_loss_pct,
                    self.config.take_profit_pct,
                    False,
                )
            reasons.append("shorting disabled")
            return RiskDecision(False, "hold", None, 0.0, 0.0, 1.0, reasons)

        reasons.append("hold signal")
        return RiskDecision(False, "hold", None, 0.0, 0.0, 1.0, reasons)

    @staticmethod
    def utc_day_key() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")
