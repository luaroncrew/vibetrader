"""CLI entrypoint for automated trading bot."""

from __future__ import annotations

import argparse
import json

from bot.config import load_config
from bot.control import ControlPlane
from bot.runtime import TradingBot


def main() -> None:
    parser = argparse.ArgumentParser(description="Automated VibeTrader bot")
    parser.add_argument("--once", action="store_true", help="Run a single trading iteration")
    parser.add_argument("--max-iterations", type=int, default=None, help="Limit loop iterations")
    parser.add_argument(
        "--control",
        choices=["pause", "resume", "kill", "clear-kill", "status"],
        default=None,
        help="Operate the control plane without starting the bot",
    )
    args = parser.parse_args()

    config = load_config()
    control = ControlPlane(config.persistence.control_dir)

    if args.control:
        if args.control == "pause":
            control.set_paused(True)
        elif args.control == "resume":
            control.set_paused(False)
        elif args.control == "kill":
            control.trigger_kill("manual operator kill switch")
        elif args.control == "clear-kill":
            control.clear_kill()
        print(json.dumps({
            "paused": control.is_paused(),
            "killed": control.is_killed(),
            "control_dir": config.persistence.control_dir,
        }, indent=2))
        return

    bot = TradingBot(config)
    if args.once:
        print(json.dumps(bot.run_once(), indent=2))
    else:
        bot.run_loop(max_iterations=args.max_iterations)


if __name__ == "__main__":
    main()
