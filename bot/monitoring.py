"""Monitoring hooks for runtime events and metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bot.contracts import BotEvent

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class Monitor:
    def __init__(self, event_log_path: str, enable_wandb: bool, wandb_project: str):
        self.event_log_path = Path(event_log_path)
        self.enable_wandb = enable_wandb and HAS_WANDB
        self.wandb_project = wandb_project
        self._wandb_started = False

    def start(self, config: dict[str, Any]) -> None:
        if self.enable_wandb and not self._wandb_started:
            wandb.init(project=self.wandb_project, job_type="trading_bot", config=config)
            self._wandb_started = True

    def stop(self) -> None:
        if self._wandb_started:
            wandb.finish()
            self._wandb_started = False

    def log_event(self, event: BotEvent) -> None:
        with self.event_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_dict()) + "\n")

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        if self._wandb_started:
            wandb.log(metrics)
