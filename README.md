# VibeTrader

**Predicting cryptocurrency price action with diffusion models.**

VibeTrader fine-tunes an [InstructPix2Pix](https://arxiv.org/abs/2211.09800) image-editing diffusion model on rendered candlestick charts to generate future price candles. Instead of predicting a number, it *draws* what the chart will look like next -- and then a signal extractor reads the drawing to produce a BUY, SELL, or HOLD decision.

This README includes a brief reference to *Exploring Diffusion Models for Generative Forecasting of Financial Charts* by Taegyeong Lee, Jiwon Park, Kyunga Bang, Seunghyun Hwang, and Ung-Jin Jang, arXiv:2509.02308v1 (2025), because the project buzzed on Twitter.

<p align="center">
  <img src="outputs/comparisons/inputs/000000.png" width="256" />
  &nbsp;&rarr;&nbsp;
  <img src="outputs/comparisons/targets/000000.png" width="256" />
</p>
<p align="center"><i>Input chart (40 candles) &rarr; Target chart with 4 predicted candles and a BUY marker</i></p>

---

## What makes this novel

Most ML trading systems frame the problem as **time-series regression or classification**: feed numerical OHLCV data into an LSTM / transformer and output a price or label. VibeTrader takes a fundamentally different approach.

### 1. Charts as images, not numbers

Price data is rendered into 256x256 candlestick chart images. The model sees exactly what a human trader would see on a screen -- green and red candles, wicks, patterns -- and learns to visually continue the chart.

### 2. Diffusion models for chart generation

We fine-tune Stable Diffusion 1.5 (via InstructPix2Pix) to perform **conditional image generation**: given an input chart of 40 candles plus a text prompt containing RSI and MACD values, the model generates a new chart with 4 additional future candles and a colored signal marker.

This means the model doesn't just classify -- it *imagines* a plausible visual future for the chart.

### 3. Dual signal extraction

Two methods read the generated image to produce a trading decision:

- **Pixel analysis** -- counts green vs. red pixels in the predicted candle region. Fast (<1ms), no API calls required.
- **Mistral Pixtral vision model** -- sends both the input and generated charts to a vision LLM that interprets candlestick patterns and returns a structured JSON signal with natural language reasoning.

### 4. Accuracy is low, but returns are positive

This is the most counterintuitive finding. On BTC/USDT 4h backtests:

| Metric | Value |
|--------|-------|
| Directional accuracy | 27.5% |
| Strategy return | **+107.97%** |
| Buy & hold return | -50.09% |

The model is wrong more often than a coin flip on direction, yet produces outsized returns. The explanation: when it's right, it tends to be right on **large moves**. When it's wrong, the moves are typically small. This asymmetry produces a positive expected value per trade.

### 5. Cross-asset generalization

The model is trained **only on BTC/USDT**, yet produces profitable signals on ETH/USDT (+95.98% strategy return vs -60.04% buy & hold) without any retraining. It learned general candlestick patterns, not asset-specific behavior.

---

## P&L Distribution

Distribution of returns on all BUY and SELL trades from the BTC/USDT 250-sample backtest:

<p align="center">
  <img src="pnl_distribution.png" width="800" />
</p>

| Stat | Value |
|------|-------|
| Total trades (BUY + SELL) | 218 |
| Win rate | 56.4% |
| Average profit (winners) | +1.61% |
| Average loss (losers) | -1.27% |
| Mean return per trade | +0.36% |
| Median return per trade | +0.20% |

The distribution shows a positive skew: winning trades are slightly larger on average than losing trades, and the model wins more often than it loses.

---

## Backtest Results

### BTC/USDT (in-distribution)

| Metric | 100 samples | 250 samples |
|--------|-------------|-------------|
| Accuracy | 35.3% | 27.5% |
| BUY accuracy | 20.5% | 14.4% |
| SELL accuracy | 24.4% | 23.4% |
| Strategy return | +40.84% | +107.97% |
| Buy & hold | -22.57% | -50.09% |

### ETH/USDT (cross-asset, zero-shot)

| Metric | 250 samples |
|--------|-------------|
| Accuracy | 25.1% |
| Strategy return | +95.98% |
| Buy & hold | -60.04% |

---

## How it works

```
OHLCV data                    Rendered charts               Diffusion model
(BTC/USDT 4h)                 (256x256 PNG)                (InstructPix2Pix)

 timestamp,open,...   ──>   ┌──────────────┐    ──>    ┌──────────────┐
 2024-01-01,42000...        │ ██ ▌█        │           │ ██ ▌█  █▌█   │
 2024-01-01,42100...        │█▌ █▌ ██      │           │█▌ █▌ ██ █ █  │
 ...                        │              │           │          ■   │
                            └──────────────┘           └──────────────┘
                             40 input candles           + 4 predicted candles
                                                        + signal marker
                                                            │
                                                            ▼
                                                    Signal extraction
                                                   (pixel or Mistral)
                                                            │
                                                            ▼
                                                   BUY / SELL / HOLD
```

### Pipeline steps

1. **Fetch data** -- Download OHLCV candles from Binance via CCXT, compute RSI and MACD indicators.
2. **Render charts** -- Convert sliding windows of 40 candles into 256x256 images. Create input/target pairs where the target includes 4 future candles and a colored signal marker (green=BUY, red=SELL, gray=HOLD).
3. **Build dataset** -- Assemble image pairs into HuggingFace Dataset format with text prompts.
4. **Train** -- Fine-tune InstructPix2Pix (SD 1.5 UNet) on the chart pairs for 2000 steps.
5. **Predict** -- Given a new chart image and prompt, run 20-step diffusion inference to generate a chart with predicted future candles.
6. **Extract signal** -- Analyze the generated image to determine BUY/SELL/HOLD.
7. **Backtest** -- Evaluate on held-out data: go long on BUY, short on SELL, flat on HOLD.

---

## Project structure

```
vibetrader/
├── data/
│   ├── fetch_ohlcv.py          # Download candles from Binance
│   ├── render_charts.py        # Render OHLCV to chart images
│   └── build_dataset.py        # Build HuggingFace dataset
├── train/
│   ├── run_training.sh         # Training script (Apple Silicon)
│   └── run_training_gpu.sh     # Training script (NVIDIA GPU)
├── inference/
│   ├── predict.py              # Diffusion inference pipeline
│   ├── extract_signal.py       # Pixel-based signal extraction
│   └── extract_signal_mistral.py  # Mistral Pixtral signal extraction
├── bot/
│   ├── backtest.py             # Historical backtesting
│   └── trader.py               # Live paper trading loop
├── app.py                      # Flask backend
├── frontend-v3/                # Interactive prediction UI
├── checkpoints/                # Fine-tuned model weights
├── outputs/                    # Backtest results and comparisons
└── pnl_distribution.png        # P&L distribution chart
```

---

## Quickstart

### Prerequisites

- Python 3.10+
- ~4 GB disk space for model weights
- Apple Silicon (MPS) or NVIDIA GPU recommended; CPU works but is slow

### Install

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Fetch data

```bash
python data/fetch_ohlcv.py --symbol BTC/USDT --timeframe 4h --days 730
```

### Render training charts

```bash
python data/render_charts.py --csv data/btc_usdt_4h.csv --output data/rendered
python data/build_dataset.py --rendered data/rendered --output data/dataset
```

### Train

```bash
# Apple Silicon
bash train/run_training.sh

# NVIDIA GPU
bash train/run_training_gpu.sh
```

### Run backtest

```bash
python -m bot.backtest --csv data/btc_usdt_4h.csv --checkpoint checkpoints/ --output outputs/backtest --max-samples 250
```

### Run the automated bot

Safe default is paper trading. The live execution path exists, but it is opt-in and requires explicit environment configuration.

```bash
export VIBETRADER_MODE=paper
export VIBETRADER_CHECKPOINT=checkpoints
python -m bot.run_bot --once
```

Run continuously:

```bash
python -m bot.run_bot
```

Control the bot manually:

```bash
python -m bot.run_bot --control pause
python -m bot.run_bot --control resume
python -m bot.run_bot --control kill
python -m bot.run_bot --control clear-kill
python -m bot.run_bot --control status
```

### Binance execution modes

The bot supports three execution modes through `VIBETRADER_MODE`:

- `paper`: default, fully simulated fills and balances
- `binance_testnet`: CCXT Binance sandbox market orders, still safer than live
- `binance_live`: real Binance spot market orders, only when explicitly configured

Recommended startup:

```bash
export BINANCE_API_KEY=...
export BINANCE_SECRET=...
export BINANCE_TESTNET=true
export VIBETRADER_MODE=binance_testnet
export VIBETRADER_DRY_RUN=false
python -m bot.run_bot --once
```

Live mode is intentionally not the default:

```bash
export BINANCE_API_KEY=...
export BINANCE_SECRET=...
export BINANCE_TESTNET=false
export VIBETRADER_MODE=binance_live
export VIBETRADER_DRY_RUN=false
python -m bot.run_bot --once
```

### Trading runtime architecture

The automated bot now implements:

- machine-safe signal contracts in `bot/contracts.py`
- market ingestion via CCXT + RSI/MACD enrichment in `bot/market.py`
- diffusion-model inference wrapped into a deterministic signal schema in `bot/model_runtime.py`
- a separate risk engine with confidence gating, sizing, drawdown limits, loss streak limits, and shorting controls in `bot/risk_engine.py`
- execution adapters for paper and Binance in `bot/execution.py`
- SQLite persistence for signals, orders, fills, positions, and events in `bot/persistence.py`
- monitoring hooks via JSONL event logs and optional W&B in `bot/monitoring.py`
- manual pause/kill files in `runtime/control/`
- automatic kill switch on risk breaches

### Persistence and controls

By default the bot stores runtime state here:

- SQLite DB: `runtime/trader.db`
- Event log: `runtime/events.jsonl`
- Control plane files: `runtime/control/pause` and `runtime/control/kill`

### Environment variables

Key configuration knobs:

```bash
VIBETRADER_MODE=paper|binance_testnet|binance_live
VIBETRADER_DRY_RUN=true|false
VIBETRADER_CHECKPOINT=checkpoints
VIBETRADER_SYMBOL=BTC/USDT
VIBETRADER_TIMEFRAME=4h
VIBETRADER_POLL_INTERVAL_SECONDS=300
VIBETRADER_INITIAL_BALANCE_USD=10000
VIBETRADER_MIN_CONFIDENCE=0.62
VIBETRADER_MAX_NOTIONAL_FRACTION=0.10
VIBETRADER_MAX_POSITION_NOTIONAL_USD=2000
VIBETRADER_MAX_DAILY_DRAWDOWN_PCT=0.05
VIBETRADER_MAX_TOTAL_DRAWDOWN_PCT=0.12
VIBETRADER_MAX_CONSECUTIVE_LOSSES=4
VIBETRADER_ALLOW_SHORT=false
VIBETRADER_ENABLE_WANDB=false
BINANCE_API_KEY=...
BINANCE_SECRET=...
BINANCE_TESTNET=true|false
```

### Notes on execution safety

- The execution path assumes trading-only API keys.
- Default operation is safe-mode paper trading.
- If `VIBETRADER_DRY_RUN=true`, the bot routes through the paper execution adapter even if a Binance mode is selected.
- The current Binance execution implementation is spot-market oriented. If you want isolated margin or futures behavior later, that should be added as a separate execution adapter rather than widened implicitly.

### Launch web UI

```bash
python app.py
# Open http://localhost:8923
```

---

## Limitations

- **Two assets tested** -- BTC/USDT and ETH/USDT on 4h timeframe only.
- **Single market regime** -- Backtested on a 3-month bearish window. Bull and sideways markets are not validated.
- **No transaction costs** -- Real trading involves fees, slippage, and spread that would reduce returns.
- **No risk metrics** -- Max drawdown, Sharpe ratio, and volatility are not yet computed.
- **Return asymmetry may not persist** -- Profitability depends on being correct on large moves, which is regime-dependent.

---

## License

Research project. Not financial advice. Use at your own risk.
