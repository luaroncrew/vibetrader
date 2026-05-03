# VibeTrader Repository Report

Date: 2026-05-03
Repository: `vibetrader`
Branch inspected: `add-repo-report-pdf-e2ffb2`

## Executive Summary

VibeTrader is a research-oriented Python project that experiments with using an image-editing diffusion model to forecast short-term cryptocurrency chart movement. Instead of predicting numeric prices directly, it renders OHLCV market data as candlestick-chart images, fine-tunes an InstructPix2Pix/Stable Diffusion pipeline to extend those charts by four candles, and then converts the generated image into a BUY, SELL, or HOLD signal.

The repository contains the full conceptual pipeline: data ingestion, chart rendering, dataset assembly, model training helpers, inference, signal extraction, backtesting, a paper-trading loop, and a Flask-based interactive UI. It also includes sample CSV data and saved backtest outputs. The main limitation is that several required runtime artifacts are intentionally excluded from Git, so a fresh clone is not operational without additional generated data and model checkpoints.

## Project Purpose

The project is built around a nonstandard thesis:

1. Convert price action into images rather than tabular sequences.
2. Fine-tune a diffusion image model to "draw" plausible future candles.
3. Extract a discrete trading signal from the generated continuation.
4. Evaluate whether that signal produces tradable returns.

This makes the repository closer to an applied ML research prototype than a production trading system.

## High-Level Architecture

Core pipeline:

1. `data/fetch_ohlcv.py`
   Downloads OHLCV data via `ccxt`, computes RSI and MACD, and saves CSV files.
2. `data/render_charts.py`
   Converts sliding windows of candles into 256x256 chart images and labels each target image with a BUY/SELL/HOLD marker based on future return thresholds.
3. `data/build_dataset.py`
   Packages rendered image pairs and text prompts into a Hugging Face dataset suitable for InstructPix2Pix training.
4. `train/run_training*.sh`
   Launches fine-tuning against Stable Diffusion v1.5 / InstructPix2Pix.
5. `inference/predict.py`
   Loads the fine-tuned diffusion pipeline and generates chart continuations.
6. `inference/extract_signal.py` and `inference/extract_signal_mistral.py`
   Convert generated images into trading decisions through either pixel heuristics or a Mistral vision model.
7. `bot/backtest.py`
   Replays historical windows and measures signal quality and simulated returns.
8. `app.py` + `frontend-v3/index.html`
   Exposes a lightweight UI for browsing examples and running predictions interactively.

## Major Directories and Files

### Top level

- `README.md`
  Strong narrative overview of the method, sample results, and quickstart commands.
- `requirements.txt`
  Python dependency list for research, inference, chart rendering, Flask, and Mistral integration.
- `app.py`
  Flask API and static-file server for the interactive web UI.
- `demo.ipynb`
  Notebook-based exploration and ad hoc experimentation.
- `plot_pnl_distribution.py` and `pnl_distribution.png`
  Utility script and pre-rendered visualization for trade return distribution.

### `data/`

- `fetch_ohlcv.py`
  Pulls exchange data from Binance through `ccxt`; enriches it with RSI/MACD via `pandas-ta`.
- `render_charts.py`
  The core data-generation script. It renders both input and target images, fixes chart scale per sample, and derives labels from percentage thresholds.
- `build_dataset.py`
  Creates a Hugging Face dataset with `original_image`, `edited_image`, and `edit_prompt` columns.
- `btc_usdt_4h.csv`, `eth_usdt_4h.csv`
  Sample OHLCV datasets already checked into Git.

### `inference/`

- `predict.py`
  Loads a `StableDiffusionInstructPix2PixPipeline`, selects device automatically, and runs generation.
- `extract_signal.py`
  Fast heuristic extractor based on green/red dominance in the predicted candle region.
- `extract_signal_mistral.py`
  Alternative extractor that sends both input and generated charts to Mistral Pixtral for JSON-structured reasoning.
- `generate_comparison.py`
  Intended to compare predictions across multiple checkpoints and prepare frontend comparison assets.

### `bot/`

- `backtest.py`
  Historical evaluator that computes directional metrics and simulated long/short returns.
- `trader.py`
  Paper-trading loop that fetches live candles and executes mock trades from the model signal.

### `train/`

- `run_training.sh`
  Apple Silicon / MPS-oriented launcher that can optionally adapt to CUDA/Colab.
- `run_training_gpu.sh`
  GPU-first launcher with simple VRAM-based batch-size heuristics.
- `brev_setup.sh`
  One-time environment bootstrap for a remote GPU box.
- `requirements-gpu.txt`
  Leaner dependency list for GPU training setups.

### Frontends

- `frontend/index.html`
  Older static interface for browsing comparison pairs and filtering by signal.
- `frontend-v3/index.html`
  Newer single-file UI served by Flask for interactive predictions and result inspection.

### `outputs/`

Contains saved backtest metrics, CSV result tables, a markdown backtest report, and sample comparison images. These are useful reference artifacts, but they are not sufficient to rerun the full app from a clean checkout.

## Dependencies

Main Python dependencies from `requirements.txt`:

- ML / generation: `torch`, `diffusers`, `transformers`, `accelerate`, `safetensors`
- Data handling: `pandas`, `numpy`, `datasets`, `tqdm`
- Market data and indicators: `ccxt`, `pandas-ta`
- Visualization and rendering: `Pillow`, `matplotlib`, `mplfinance`
- Observability / experiment tracking: `wandb`
- Serving / integration: `flask`, `mistralai`

Operationally important external dependencies:

- Hugging Face model weights for Stable Diffusion / InstructPix2Pix
- Local fine-tuned checkpoint artifacts under `checkpoints/`
- Rendered dataset artifacts under `data/rendered/` and `data/dataset/`
- Optional secrets such as `MISTRAL_API_KEY`, `WANDB_API_KEY`, `HF_TOKEN`, and Binance credentials

## Operational Workflow

Typical end-to-end usage:

1. Fetch raw OHLCV candles into CSV.
2. Compute indicators and render image pairs for each sliding price window.
3. Build a Hugging Face dataset for image-edit training.
4. Fine-tune the InstructPix2Pix model.
5. Run inference on held-out samples.
6. Convert generated charts into discrete trade signals.
7. Backtest the resulting signals.
8. Optionally expose the model through Flask for interactive inspection or through the paper trader for live simulation.

In practice, the workflow assumes access to heavyweight local artifacts that are not stored in the repository itself.

## Strengths

- Distinctive research angle: the image-based forecasting approach is unusual and clearly articulated.
- End-to-end coverage: the repo contains the full conceptual loop from data ingestion to UI.
- Readable codebase: the project is small enough to understand quickly, and the filenames generally match their responsibilities.
- Concrete artifacts: committed sample data and backtest outputs make the project easier to inspect without running training immediately.
- Multiple signal extraction strategies: heuristic and vision-LLM paths provide a useful comparison axis.
- Simple deployment surface: Flask plus static HTML keeps the UI easy to run once artifacts exist.

## Risks and Notable Gaps

### 1. Fresh-clone operability is incomplete

The app and several workflows depend on files that are ignored and absent in this checkout:

- `checkpoints/`
- `data/rendered/`
- `data/dataset/`

That means `python app.py` will not work in a clean clone until those artifacts are recreated or restored. The README describes the intended flow, but the repository is not self-sufficient.

### 2. Stale code paths reference a removed field

`extract_signal.py` returns a `Signal` with `action`, `confidence`, `green_pct`, and `red_pct`. However:

- `bot/trader.py`
- `inference/generate_comparison.py`
- notebook snippets in `demo.ipynb`

still reference `signal.avg_rgb`, which no longer exists. Those paths will raise runtime errors when exercised.

### 3. Live trading prompt quality is inconsistent with training

The paper trader does not compute fresh RSI/MACD features before inference. It hardcodes:

- `RSI=50.0`
- `MACD=0.0`

This weakens fidelity relative to the training and backtest pipeline, where prompts include actual indicator values.

### 4. Evaluation logic is simplified

Backtesting simulates long/short returns without modeling:

- fees
- slippage
- spread
- latency
- drawdown controls
- position sizing beyond a simple directional rule

This is acceptable for research exploration, but it inflates the gap between experimental returns and likely live performance.

### 5. Reproducibility and engineering maturity are limited

The repo has no visible:

- automated tests
- CI workflow
- pinned lockfile
- container definition
- artifact bootstrap script for local inference

There is also an inconsistency between `.gitignore` and tracked output/data files: some generated artifacts are committed even though their parent paths are ignored, which can confuse contributors about the intended source of truth.

### 6. Dependency setup relies on side effects

The training scripts may clone `diffusers` into the repository at runtime. That is pragmatic, but it couples execution to mutable local state and makes exact training provenance harder to reproduce.

## Overall Assessment

VibeTrader is a compact and interesting ML research prototype with a strong narrative and a clear experimental pipeline. It is easiest to think of as a proof-of-concept laboratory for visual financial forecasting rather than a production trading application.

The repository is strong on idea clarity and prototype coverage, but weaker on reproducibility, runtime completeness, and production hardening. The biggest practical blockers for a new operator are missing model/data artifacts and a few stale code paths that will fail at runtime. If those were addressed, the project would be substantially easier to evaluate and extend.

## Files Consulted

- `README.md`
- `requirements.txt`
- `app.py`
- `data/fetch_ohlcv.py`
- `data/render_charts.py`
- `data/build_dataset.py`
- `inference/predict.py`
- `inference/extract_signal.py`
- `inference/extract_signal_mistral.py`
- `inference/generate_comparison.py`
- `bot/backtest.py`
- `bot/trader.py`
- `train/run_training.sh`
- `train/run_training_gpu.sh`
- `train/brev_setup.sh`
- `outputs/BACKTEST_REPORT.md`
- `.gitignore`
