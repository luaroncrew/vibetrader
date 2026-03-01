"""Plot profit/loss distribution of BUY and SELL trades from backtest results."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("outputs/backtest_250/backtest_results.csv")

# Filter only BUY and SELL trades (exclude HOLD)
trades = df[df["predicted"].isin(["BUY", "SELL"])].copy()
trades["strategy_return_pct"] = trades["strategy_return"] * 100

profitable = trades[trades["strategy_return"] > 0]["strategy_return_pct"]
losing = trades[trades["strategy_return"] <= 0]["strategy_return_pct"]

fig, ax = plt.subplots(figsize=(12, 6))

bin_edges = np.linspace(
    trades["strategy_return_pct"].min() - 0.5,
    trades["strategy_return_pct"].max() + 0.5,
    40,
)

ax.hist(profitable, bins=bin_edges, color="#2ecc71", alpha=0.85, label=f"Profit ({len(profitable)} trades)", edgecolor="white", linewidth=0.5)
ax.hist(losing, bins=bin_edges, color="#e74c3c", alpha=0.85, label=f"Loss ({len(losing)} trades)", edgecolor="white", linewidth=0.5)

ax.axvline(x=0, color="black", linestyle="--", linewidth=1, alpha=0.7)

mean_return = trades["strategy_return_pct"].mean()
median_return = trades["strategy_return_pct"].median()
ax.axvline(x=mean_return, color="#3498db", linestyle="-", linewidth=1.5, alpha=0.8, label=f"Mean: {mean_return:.2f}%")
ax.axvline(x=median_return, color="#f39c12", linestyle="-", linewidth=1.5, alpha=0.8, label=f"Median: {median_return:.2f}%")

win_rate = len(profitable) / len(trades) * 100
avg_profit = profitable.mean() if len(profitable) > 0 else 0
avg_loss = losing.mean() if len(losing) > 0 else 0

textstr = (
    f"Total trades: {len(trades)}\n"
    f"Win rate: {win_rate:.1f}%\n"
    f"Avg profit: +{avg_profit:.2f}%\n"
    f"Avg loss: {avg_loss:.2f}%"
)
ax.text(0.97, 0.95, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.9))

ax.set_xlabel("Return per Trade (%)", fontsize=12)
ax.set_ylabel("Number of Trades", fontsize=12)
ax.set_title("BTC/USDT Backtest — Profit & Loss Distribution (BUY/SELL trades, n=250)", fontsize=13)
ax.legend(loc="upper left", fontsize=10)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("pnl_distribution.png", dpi=150)
print("Saved pnl_distribution.png")
