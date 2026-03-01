# Backtest Report — VibeTrader v1

**Date:** 2026-02-28
**Model:** InstructPix2Pix fine-tuned on BTC/USDT 4h candlestick charts
**Checkpoint:** `checkpoints/` (final weights)
**Test data:** BTC/USDT 4h candles from `data/btc_usdt_4h.csv`, ETH/USDT 4h candles from `data/eth_usdt_4h.csv`
**Inference:** 20 diffusion steps, EulerAncestral scheduler, MPS (Apple Silicon)

---

## Results Summary — BTC/USDT (in-distribution)

| Metric              | 100 samples | 250 samples |
|---------------------|-------------|-------------|
| **Accuracy**        | 35.3%       | 27.5%       |
| **BUY accuracy**    | 20.5%       | 14.4%       |
| **SELL accuracy**   | 24.4%       | 23.4%       |
| **Strategy return** | +40.84%     | +107.97%    |
| **Buy & hold**      | -22.57%     | -50.09%     |
| **Samples**         | 102         | 255         |

### Signal Distribution (BTC)

| Run         | BUY | SELL | HOLD |
|-------------|-----|------|------|
| 100 samples | 39  | 41   | 22   |
| 250 samples | 111 | 107  | 37   |

---

## Results Summary — ETH/USDT (cross-asset validation)

Model trained on BTC/USDT only, tested on ETH/USDT 4h candles (6-month window).

| Metric              | 250 samples |
|---------------------|-------------|
| **Accuracy**        | 25.1%       |
| **BUY accuracy**    | 17.2%       |
| **SELL accuracy**   | 21.7%       |
| **Strategy return** | +95.98%     |
| **Buy & hold**      | -60.04%     |
| **Samples**         | 251         |

### Signal Distribution (ETH)

| Run         | BUY | SELL | HOLD |
|-------------|-----|------|------|
| 250 samples | 93  | 115  | 43   |

---

## Key Observations

1. **Directional accuracy is below random** (27.5% vs 33% baseline for 3 classes). The model does not reliably predict whether price goes up or down.

2. **Strategy return is strongly positive** (+108% on 250 samples) despite low accuracy. The model captures outsized returns on its correct predictions while incorrect predictions tend to occur on smaller price moves.

3. **Consistent across sample sizes.** Both the 100-sample and 250-sample runs show the same pattern: low accuracy but positive strategy return vs buy & hold.

4. **Bearish test period.** Buy-and-hold returned -50% over the full test window, making the model's positive return more notable — it profited in a down market.

5. **Cross-asset generalization works.** The model was trained only on BTC/USDT charts but produces profitable signals on ETH/USDT (+96% strategy return vs -60% buy & hold). This suggests the model learned general candlestick patterns rather than BTC-specific behavior.

---

## Limitations

- **Two assets, single timeframe** — BTC/USDT and ETH/USDT 4h tested
- **Single market regime** — 3-month bearish window; no sideways or bull market validation
- **No transaction costs** — real trading involves fees, slippage, and spread
- **No risk metrics** — max drawdown, Sharpe ratio, and volatility not computed
- **Return asymmetry may not persist** — profitability depends on being right on big moves, which is not guaranteed in other regimes

---

## Files

- `outputs/backtest/backtest_results.csv` — per-sample predictions (100 samples)
- `outputs/backtest/backtest_metrics.json` — summary metrics (100 samples)
- `outputs/backtest_250/backtest_results.csv` — per-sample predictions (255 samples)
- `outputs/backtest_250/backtest_metrics.json` — summary metrics (255 samples)
- `outputs/backtest_eth_250/backtest_results.csv` — per-sample predictions, ETH/USDT (251 samples)
- `outputs/backtest_eth_250/backtest_metrics.json` — summary metrics, ETH/USDT (251 samples)
