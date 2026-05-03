"""End-to-end automated trading bot runtime."""

from __future__ import annotations

import json
import time
from dataclasses import asdict

from bot.config import BotConfig
from bot.contracts import BotEvent, OrderRecord, PositionRecord, new_id, utc_now_iso
from bot.control import ControlPlane
from bot.execution import BinanceExecutionClient, PaperExecutionClient, flat_position
from bot.market import MarketDataClient
from bot.model_runtime import ModelRuntime
from bot.monitoring import Monitor
from bot.persistence import Persistence
from bot.risk_engine import PortfolioState, RiskEngine


class TradingBot:
    def __init__(self, config: BotConfig):
        self.config = config
        self.persistence = Persistence(config.persistence.sqlite_path)
        self.control = ControlPlane(config.persistence.control_dir)
        self.monitor = Monitor(
            config.monitoring.event_log_path,
            config.monitoring.enable_wandb,
            config.monitoring.wandb_project,
        )
        self.market = MarketDataClient(
            exchange_id=config.market.exchange_id,
            testnet=config.binance_testnet,
        )
        self.model = ModelRuntime(
            checkpoint_path=config.model.checkpoint_path,
            device=config.model.device,
            num_inference_steps=config.model.num_inference_steps,
            image_guidance_scale=config.model.image_guidance_scale,
            guidance_scale=config.model.guidance_scale,
        )
        self.risk = RiskEngine(config.risk)
        self.execution = (
            PaperExecutionClient(config.risk.fee_bps, config.risk.slippage_bps)
            if config.mode == "paper" or config.dry_run
            else BinanceExecutionClient(testnet=config.binance_testnet)
        )
        self.cash_usd = config.initial_balance_usd
        self.start_equity_usd = config.initial_balance_usd
        self.daily_start_equity_usd = config.initial_balance_usd
        self.daily_key = self.risk.utc_day_key()
        self.consecutive_losses = 0
        self.position = self._load_or_create_position()

    def _load_or_create_position(self) -> PositionRecord:
        row = self.persistence.get_position(self.config.market.symbol)
        if row:
            return PositionRecord(**row)
        position = flat_position(self.config.market.symbol, self.config.mode)
        self.persistence.save_position(position)
        return position

    def _event(self, level: str, event_type: str, message: str, details: dict | None = None) -> None:
        event = BotEvent(
            event_id=new_id("evt"),
            timestamp=utc_now_iso(),
            level=level,  # type: ignore[arg-type]
            event_type=event_type,
            message=message,
            details=details or {},
        )
        self.persistence.save_event(event)
        self.monitor.log_event(event)

    def _refresh_daily_baseline_if_needed(self) -> None:
        current_day = self.risk.utc_day_key()
        if current_day != self.daily_key:
            self.daily_key = current_day
            self.daily_start_equity_usd = self.current_equity(self.position.mark_price or self.position.entry_price or 0.0)

    def current_equity(self, mark_price: float) -> float:
        if self.position.quantity > 0:
            unrealized = (mark_price - self.position.entry_price) * self.position.quantity
        elif self.position.quantity < 0:
            unrealized = (self.position.entry_price - mark_price) * abs(self.position.quantity)
        else:
            unrealized = 0.0
        self.position.unrealized_pnl = unrealized
        self.position.mark_price = mark_price
        return self.cash_usd + unrealized

    def _update_position_from_fill(self, fill, intent: str) -> None:
        qty = fill.quantity
        price = fill.price
        fees = fill.fees_usd
        realized_change = 0.0

        if intent == "open_long":
            self.cash_usd -= qty * price + fees
            self.position.quantity = qty
            self.position.entry_price = price
        elif intent == "close_long":
            realized_change = (price - self.position.entry_price) * self.position.quantity - fees
            self.cash_usd += self.position.quantity * price - fees
            self.position.realized_pnl += realized_change
            self.position.quantity = 0.0
            self.position.entry_price = 0.0
        elif intent == "open_short":
            self.cash_usd += qty * price - fees
            self.position.quantity = -qty
            self.position.entry_price = price
        elif intent == "close_short":
            realized_change = (self.position.entry_price - price) * abs(self.position.quantity) - fees
            self.cash_usd -= abs(self.position.quantity) * price + fees
            self.position.realized_pnl += realized_change
            self.position.quantity = 0.0
            self.position.entry_price = 0.0

        if realized_change < 0:
            self.consecutive_losses += 1
        elif realized_change > 0:
            self.consecutive_losses = 0

        self.position.updated_at = utc_now_iso()
        self.position.status = (
            "long" if self.position.quantity > 0 else "short" if self.position.quantity < 0 else "flat"
        )

    def run_once(self) -> dict:
        self._refresh_daily_baseline_if_needed()
        if self.control.is_killed():
            self._event("CRITICAL", "kill_switch", "Kill switch is active; refusing to trade")
            raise RuntimeError("Kill switch is active")
        if self.control.is_paused():
            self._event("WARN", "paused", "Bot is paused; skipping iteration")
            return {"status": "paused"}

        snapshot = self.market.fetch_recent_snapshot(
            self.config.market.symbol,
            self.config.market.timeframe,
            self.config.market.window_size,
        )
        equity = self.current_equity(snapshot.last_price)

        signal = self.model.infer_signal(
            symbol=self.config.market.symbol,
            timeframe=self.config.market.timeframe,
            candles=snapshot.candles,
            future_candles=self.config.market.future_candles,
            rsi=snapshot.rsi,
            macd=snapshot.macd,
        )
        self.persistence.save_signal(signal)

        portfolio = PortfolioState(
            cash_usd=self.cash_usd,
            equity_usd=equity,
            start_equity_usd=self.start_equity_usd,
            daily_start_equity_usd=self.daily_start_equity_usd,
            consecutive_losses=self.consecutive_losses,
            current_position=self.position,
        )
        decision = self.risk.evaluate(signal, portfolio)

        if not decision.approved or not decision.side:
            if any("breached" in reason for reason in decision.reasons):
                self.control.trigger_kill("; ".join(decision.reasons))
                self._event("CRITICAL", "risk_kill_switch", "Risk guard triggered kill switch", {
                    "reasons": decision.reasons,
                    "signal": signal.to_dict(),
                })
            else:
                self._event("INFO", "signal_rejected", "Risk engine rejected signal", {
                    "signal": signal.to_dict(),
                    "decision": decision.to_dict(),
                })
            self.monitor.log_metrics({
                "equity_usd": equity,
                "signal_confidence": signal.confidence,
                "position_qty": self.position.quantity,
            })
            return {"status": "rejected", "signal": signal.to_dict(), "decision": decision.to_dict()}

        order = OrderRecord(
            order_id=new_id("ord"),
            created_at=utc_now_iso(),
            signal_id=signal.signal_id,
            symbol=signal.symbol,
            side=decision.side,
            intent=decision.intent,
            quantity=decision.quantity,
            requested_price=signal.price,
            mode=self.config.mode,
            status="accepted",
            metadata={"risk": decision.to_dict(), "dry_run": self.config.dry_run},
        )
        self.persistence.save_order(order)

        order, fill = self.execution.submit_order(order)
        self.persistence.save_order(order)
        self.persistence.save_fill(fill)

        self._update_position_from_fill(fill, decision.intent)
        self.current_equity(snapshot.last_price)
        self.persistence.save_position(self.position)

        result = {
            "status": "executed",
            "signal": signal.to_dict(),
            "decision": decision.to_dict(),
            "order": order.to_dict(),
            "fill": fill.to_dict(),
            "position": self.position.to_dict(),
            "cash_usd": self.cash_usd,
            "equity_usd": self.current_equity(snapshot.last_price),
        }
        self._event("INFO", "order_executed", "Order executed", result)
        self.monitor.log_metrics({
            "equity_usd": result["equity_usd"],
            "cash_usd": self.cash_usd,
            "position_qty": self.position.quantity,
            "signal_confidence": signal.confidence,
        })
        return result

    def run_loop(self, max_iterations: int | None = None) -> None:
        self.monitor.start({"config": json.loads(json.dumps(asdict(self.config), default=str))})
        self._event("INFO", "startup", "Trading bot started", {"mode": self.config.mode, "dry_run": self.config.dry_run})
        count = 0
        try:
            while max_iterations is None or count < max_iterations:
                self.run_once()
                count += 1
                time.sleep(self.config.market.poll_interval_seconds)
        finally:
            self._event("INFO", "shutdown", "Trading bot stopped", {"iterations": count})
            self.monitor.stop()
