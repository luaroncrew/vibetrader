"""Configuration loader for the trading bot."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path

from bot.contracts import ExecutionMode


TRUE_VALUES = {"1", "true", "yes", "on"}


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in TRUE_VALUES


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None else default


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


@dataclass(slots=True)
class MonitoringConfig:
    enable_wandb: bool = False
    wandb_project: str = "vibetrader"
    event_log_path: str = "runtime/events.jsonl"


@dataclass(slots=True)
class PersistenceConfig:
    sqlite_path: str = "runtime/trader.db"
    control_dir: str = "runtime/control"


@dataclass(slots=True)
class RiskConfig:
    min_confidence: float = 0.62
    max_notional_fraction: float = 0.10
    max_position_notional_usd: float = 2000.0
    max_daily_drawdown_pct: float = 0.05
    max_total_drawdown_pct: float = 0.12
    max_consecutive_losses: int = 4
    fee_bps: float = 10.0
    slippage_bps: float = 5.0
    stop_loss_pct: float = 0.015
    take_profit_pct: float = 0.03
    allow_short: bool = False


@dataclass(slots=True)
class MarketConfig:
    symbol: str = "BTC/USDT"
    timeframe: str = "4h"
    window_size: int = 40
    future_candles: int = 4
    poll_interval_seconds: int = 300
    exchange_id: str = "binance"


@dataclass(slots=True)
class ModelConfig:
    checkpoint_path: str = "checkpoints"
    device: str = "auto"
    num_inference_steps: int = 20
    image_guidance_scale: float = 1.5
    guidance_scale: float = 7.0
    signal_source: str = "pixel"


@dataclass(slots=True)
class BotConfig:
    mode: ExecutionMode = "paper"
    dry_run: bool = True
    binance_testnet: bool = True
    initial_balance_usd: float = 10000.0
    market: MarketConfig = field(default_factory=MarketConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    @property
    def is_live_mode(self) -> bool:
        return self.mode == "binance_live"

    @property
    def safe_default_mode(self) -> bool:
        return self.mode in {"paper", "binance_testnet"}


def load_config() -> BotConfig:
    mode = os.getenv("VIBETRADER_MODE", "paper").strip()
    if mode not in {"paper", "binance_testnet", "binance_live"}:
        raise ValueError(
            "VIBETRADER_MODE must be one of: paper, binance_testnet, binance_live"
        )

    config = BotConfig(
        mode=mode,  # type: ignore[arg-type]
        dry_run=env_bool("VIBETRADER_DRY_RUN", mode != "binance_live"),
        binance_testnet=env_bool("BINANCE_TESTNET", True),
        initial_balance_usd=env_float("VIBETRADER_INITIAL_BALANCE_USD", 10000.0),
        market=MarketConfig(
            symbol=os.getenv("VIBETRADER_SYMBOL", "BTC/USDT"),
            timeframe=os.getenv("VIBETRADER_TIMEFRAME", "4h"),
            window_size=env_int("VIBETRADER_WINDOW_SIZE", 40),
            future_candles=env_int("VIBETRADER_FUTURE_CANDLES", 4),
            poll_interval_seconds=env_int("VIBETRADER_POLL_INTERVAL_SECONDS", 300),
            exchange_id=os.getenv("VIBETRADER_EXCHANGE_ID", "binance"),
        ),
        model=ModelConfig(
            checkpoint_path=os.getenv("VIBETRADER_CHECKPOINT", "checkpoints"),
            device=os.getenv("VIBETRADER_DEVICE", "auto"),
            num_inference_steps=env_int("VIBETRADER_INFERENCE_STEPS", 20),
            image_guidance_scale=env_float("VIBETRADER_IMAGE_GUIDANCE", 1.5),
            guidance_scale=env_float("VIBETRADER_GUIDANCE", 7.0),
            signal_source=os.getenv("VIBETRADER_SIGNAL_SOURCE", "pixel"),
        ),
        risk=RiskConfig(
            min_confidence=env_float("VIBETRADER_MIN_CONFIDENCE", 0.62),
            max_notional_fraction=env_float("VIBETRADER_MAX_NOTIONAL_FRACTION", 0.10),
            max_position_notional_usd=env_float("VIBETRADER_MAX_POSITION_NOTIONAL_USD", 2000.0),
            max_daily_drawdown_pct=env_float("VIBETRADER_MAX_DAILY_DRAWDOWN_PCT", 0.05),
            max_total_drawdown_pct=env_float("VIBETRADER_MAX_TOTAL_DRAWDOWN_PCT", 0.12),
            max_consecutive_losses=env_int("VIBETRADER_MAX_CONSECUTIVE_LOSSES", 4),
            fee_bps=env_float("VIBETRADER_FEE_BPS", 10.0),
            slippage_bps=env_float("VIBETRADER_SLIPPAGE_BPS", 5.0),
            stop_loss_pct=env_float("VIBETRADER_STOP_LOSS_PCT", 0.015),
            take_profit_pct=env_float("VIBETRADER_TAKE_PROFIT_PCT", 0.03),
            allow_short=env_bool("VIBETRADER_ALLOW_SHORT", False),
        ),
        persistence=PersistenceConfig(
            sqlite_path=os.getenv("VIBETRADER_SQLITE_PATH", "runtime/trader.db"),
            control_dir=os.getenv("VIBETRADER_CONTROL_DIR", "runtime/control"),
        ),
        monitoring=MonitoringConfig(
            enable_wandb=env_bool("VIBETRADER_ENABLE_WANDB", False),
            wandb_project=os.getenv("VIBETRADER_WANDB_PROJECT", "vibetrader"),
            event_log_path=os.getenv("VIBETRADER_EVENT_LOG_PATH", "runtime/events.jsonl"),
        ),
    )

    Path(config.persistence.control_dir).mkdir(parents=True, exist_ok=True)
    Path(config.persistence.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
    Path(config.monitoring.event_log_path).parent.mkdir(parents=True, exist_ok=True)
    return config
