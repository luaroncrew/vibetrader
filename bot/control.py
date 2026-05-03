"""Manual pause and kill-switch controls."""

from __future__ import annotations

from pathlib import Path


class ControlPlane:
    def __init__(self, control_dir: str):
        self.control_dir = Path(control_dir)
        self.control_dir.mkdir(parents=True, exist_ok=True)
        self.pause_file = self.control_dir / "pause"
        self.kill_file = self.control_dir / "kill"

    def is_paused(self) -> bool:
        return self.pause_file.exists()

    def is_killed(self) -> bool:
        return self.kill_file.exists()

    def set_paused(self, paused: bool) -> None:
        if paused:
            self.pause_file.touch()
        elif self.pause_file.exists():
            self.pause_file.unlink()

    def trigger_kill(self, reason: str | None = None) -> None:
        self.kill_file.write_text((reason or "kill switch engaged").strip() + "\n", encoding="utf-8")

    def clear_kill(self) -> None:
        if self.kill_file.exists():
            self.kill_file.unlink()
