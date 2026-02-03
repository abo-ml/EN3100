# Design and Application of Machine Learning Models for Financial Market Prediction: A Comparative Study of Traditional and Deep Learning Approaches

## Overview
This repository accompanies an academic dissertation exploring a progressive sequence of forecasting and trading system designs for liquid financial assets. The project walks through five model iterations ranging from linear baselines to attention-based ensembles with risk-aware position sizing. The codebase emphasises reproducible research, walk-forward validation, and extensibility toward microstructure, sentiment, and reinforcement learning enhancements.

The default assets include:
- **AAPL** (US equity)
- **EURUSD=X** (FX proxy via Yahoo Finance)
- **XAUUSD=X** (Gold spot proxy)
- **^GSPC** (S&P 500 index benchmark)

All components operate on daily OHLCV bars sourced via `yfinance`. Hooks are provided for alternative data vendors (Alpha Vantage, Polygon, Interactive Brokers) and advanced modules (order flow imbalance, ICT/SMT liquidity concepts, reinforcement learning). TODO markers highlight where sensitive credentials, private market data, or broker integrations must be supplied by the student.

## Scope / decision gate
The mixed-asset workflow remains the default path for the dissertation. The new per-asset and optional 20-stock equity universe flows are additive experiment tracks that reuse the same walk-forward protocol and feature set. Use them when you want asset-specific diagnostics or a broader equity cross-section; otherwise stick with the baseline panel workflow for continuity with earlier results.

## Repository Structure
```
├── .github/
│   └── workflows/          # CI/CD workflows (tests.yml)
├── app/
│   └── streamlit_app.py    # Optional visualisation UI
├── configs/
│   └── universe.yaml       # Config-driven pipeline settings
├── data/
│   ├── raw/                # Downloaded OHLCV data
│   └── processed/          # Aligned and feature-engineered datasets
├── docs/
│   ├── COLAB_RUNBOOK.md    # Google Colab quick-start pipelines
│   ├── EQUATIONS.md        # Mathematical reference for all formulas
│   ├── EXPERIMENTS.md      # Experiment matrix and validation notes
│   └── local_setup.md      # Local and Colab setup guide
├── notebooks/
│   ├── Iteration_1.ipynb.py
│   ├── Iteration_2.ipynb.py
│   ├── Iteration_3.ipynb.py
│   ├── Iteration_4.ipynb.py
│   ├── Iteration_5.ipynb.py
│   └── final_comparison.ipynb.py
├── project_assets/         # Project management files (Gantt, Risk Register)
├── reports/
│   ├── figures/            # Generated plots
│   ├── iteration_1_results.md
│   ├── iteration_2_results.md
│   ├── iteration_3_results.md
│   ├── iteration_4_results.md
│   └── iteration_5_results.md
├── scripts/
│   ├── plot_walk_forward.py
│   └── smoke_check.sh      # Basic syntax/import checks
├── src/
│   ├── advanced/           # Future-work stubs (order flow, pattern recognition, RL)
│   ├── data/               # Data acquisition & alignment
│   ├── evaluation/         # Metrics, walk-forward, reporting utilities
│   ├── experiments/        # Per-asset & equity universe evaluation scripts
│   ├── features/           # Feature engineering and target construction
│   ├── models/             # Iteration-specific training scripts
│   ├── risk/               # Monte Carlo risk analysis
│   ├── universe/           # S&P 500 universe utilities
│   └── utils/              # Shared path utilities
├── tests/                  # Unit tests for data downloads and placeholders
├── tools/
│   └── md_to_doc_pdf.py    # Report conversion utility
├── market_forecasting.py   # Standalone demonstration module
├── CHANGELOG.md            # Release notes
├── MERGE_GUIDE.md          # Conflict resolution guidance
├── requirements.txt
└── README.md
```

## Getting Started
### 1. Set up the environment
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download market data
Export your Alpha Vantage API key (insert the key provided to you for coursework) and fetch daily OHLCV data.
```bash
export ALPHAVANTAGE_API_KEY="<YOUR_ALPHA_VANTAGE_KEY>"
# Optional: set the alternate variable name used in some notebooks/scripts.
export ALPHA_VANTAGE_API_KEY="$ALPHAVANTAGE_API_KEY"
```
Then pull the required tickers. Alpha Vantage is the default provider with automatic fallbacks to Yahoo Finance for symbols the API does not cover (e.g. broad market indices beginning with `^`).
Fetch daily OHLCV data and save to `data/raw/`.
```bash
python -m src.data.download_data \
    --tickers AAPL EURUSD=X XAUUSD=X ^GSPC \
    --start 2013-01-01 \
    --end 2023-12-31 \
    --provider alpha_vantage
```
Optional arguments allow changing the interval, retry policy, output format, and overriding the API key via `--api-key`. **TODO:** replace the dummy `fetch_orderbook_snapshot` implementation with a real broker API call (Interactive Brokers, Alpaca, etc.) when credentials are available.

### 3. Align data sources
Merge OHLCV, placeholder order flow, sentiment, and macro features into a single dataset.
```bash
python -m src.data.align_data
```
This step produces `data/processed/combined_features.csv`. Extend `align_data.py` to ingest proprietary sentiment feeds, macro benchmarks, or alternative data.

### 4. Engineer features & targets
Generate technical, regime, and microstructure-inspired features along with supervised learning targets.
```bash
python -m src.features.engineer_features
```
Outputs include `data/processed/model_features.parquet`. TODO markers indicate where true VWAP/TWAP, chart pattern detection, and ICT/SMT annotations should be implemented.

### 5. Run model iterations
Each iteration script performs walk-forward validation, trains the designated models, logs metrics to `reports/`, and saves diagnostic plots.
```bash
python -m src.models.iteration1_baseline
python -m src.models.iteration1_1_svr
python -m src.models.iteration2_ensemble
python -m src.models.iteration2_1_lightgbm
python -m src.models.iteration3_lstm
python -m src.models.iteration4_transformer
python -m src.models.iteration5_meta_ensemble
python -m src.risk.monte_carlo  # requires iteration 5 to have run
```
Refer to the notebooks in `notebooks/` for guided walkthroughs, narrative commentary, and exploratory analysis aligned with each iteration.

### 6. Running the project in Google Colab (or similar hosted notebooks)
When using Colab you typically upload or clone the repository into `/content`. The centralised
path utilities in `src/utils/paths.py` keep all artefacts anchored to the project root so the
commands below work regardless of the current working directory.

```python
# 1. (Optional) mount Google Drive if you want persistence beyond the session
from google.colab import drive
drive.mount("/content/drive")

# 2. Unzip or clone the repository into /content
!unzip -o "/content/EN3100.zip" -d "/content"  # adjust the filename as needed

# 3. Change into the project directory
%cd "/content/EN3100"

# 4. Install dependencies
!pip install -r requirements.txt -q

# 5. Set credentials for data providers
import os
os.environ["ALPHAVANTAGE_API_KEY"] = "<YOUR_KEY>"  # or ALPHA_VANTAGE_API_KEY

# 6. Run the same pipeline as documented above
!python -m src.data.download_data --tickers AAPL EURUSD=X XAUUSD=X ^GSPC \
    --start 2013-01-01 --end 2023-12-31 --provider alpha_vantage
!python -m src.data.align_data
!python -m src.features.engineer_features
!python -m src.models.iteration1_baseline
```

Key points for hosted notebooks:

- Always `cd` into the inner project folder before running CLI commands. With the shared
  path utilities, outputs will be directed to `data/` and `reports/` even if you launch a
  command one level higher, but switching directories avoids confusion when exploring files.
- Persist `data/` and `reports/` by copying them to Google Drive (or downloading them) at the
  end of the session if you want to resume without repeating the downloads.
- Keep API keys in environment variables or secret managers. Never hard-code credentials into
  notebooks before sharing or committing them.

## Per-asset evaluation (4-asset universe)
Run the existing iterations per ticker while keeping the same walk-forward settings and feature set:
```bash
python -m src.data.download_data --tickers AAPL EURUSD=X XAUUSD=X ^GSPC --start 2013-01-01 --end 2023-12-31 --provider alpha_vantage
python -m src.data.align_data --tickers AAPL EURUSD=X XAUUSD=X ^GSPC
python -m src.features.engineer_features
python -m src.experiments.per_asset_evaluation
```
The `--tickers`/`--ticker-file` flags on `align_data` let you restrict alignment without altering the default mixed-asset behaviour.

## Optional 20-stock S&P 500 experiment
Sample a reproducible equity universe, download data, align only those tickers, engineer features, and run per-stock reports:
```bash
python -m src.experiments.download_equity_universe --n 20 --start 2013-01-01 --end 2023-12-31 --provider yfinance
python -m src.data.align_data --ticker-file data/reference/equity_universe_20.txt
python -m src.features.engineer_features
python -m src.experiments.per_asset_equity_evaluation
```
Ticker lists are stored under `data/reference/`, with aligned data in `data/processed/` and plots in `reports/figures/`. The mixed-asset workflow stays unchanged if you skip these optional steps.

For a comprehensive evaluation across all iterations on the 20-stock universe:
```bash
python -m src.experiments.evaluate_20_stock_all_iterations
```
This produces summary statistics, pivot tables, and iteration comparisons for the equity universe.

## Config-driven runner (application layer)
Use the lightweight runner to drive the same pipelines from a config file:
```bash
python -m src.experiments.run_pipeline --config configs/universe.yaml
```
Adjust `configs/universe.yaml` to choose `mode` (`core4`, `sp500_sample`, or `custom`), provider, dates, and optional tags for outputs. The runner appends tags to report filenames when `--tag` is supplied, leaving defaults unchanged otherwise.

## Streamlit UI (optional visualisation)
Launch a simple UI to trigger the runner and view existing reports/figures:
```bash
streamlit run app/streamlit_app.py
```
The UI is for demonstration/visualisation only; dissertation results should continue to use the deterministic CLI flows above.

## Running Tests
Run the test suite with pytest to verify data downloads and placeholder guards:
```bash
pip install pytest
pytest tests/ -v
```
Tests include:
- `test_download_data.py`: validates provider fallback and CLI parser
- `test_placeholders.py`: ensures unimplemented advanced modules raise `NotImplementedError`

A smoke-check script is also available for quick syntax verification:
```bash
bash scripts/smoke_check.sh
```

## Iteration Roadmap
1. **Iteration 1 – Linear Baselines:** Persistence, linear regression, and logistic regression models validate the pipeline and establish benchmark metrics.
2. **Iteration 1.1 – SVR Baseline:** Extends Iteration 1 with an RBF Support Vector Regressor to benchmark kernel-based nonlinearity against the linear regressor on identical walk-forward splits.
3. **Iteration 2 – Tree-Based Ensembles:** Random Forest, optional XGBoost, and SVM classifiers capture nonlinear relationships and deliver feature importance insights.
4. **Iteration 2.1 – LightGBM Upgrade:** Extends Iteration 2 with a gradient-boosted tree regressor (LightGBM) plus feature importances, allowing a direct comparison to the Random Forest/XGBoost ensemble.
5. **Iteration 3 – LSTM Sequence Model:** Recurrent neural networks learn temporal dependencies over rolling windows and introduce deep learning considerations (scaling, early stopping, regularisation).
6. **Iteration 4 – Transformer Encoder:** Attention mechanisms handle long-range dependencies and multimodal inputs (price, sentiment, order flow placeholders) for enhanced forecasting power.
7. **Iteration 5 – Meta-Ensemble with Risk Layer:** Stacks the strongest models, applies dynamic volatility-aware position sizing, and outlines execution scheduling stubs (VWAP/TWAP, pairs trading hook).
8. **Iteration 5 – Monte Carlo Risk:** A standalone Monte Carlo module bootstraps Iteration 5 strategy returns to stress-test equity curve variability, drawdown risk, and tail outcomes.

## Validation & Fairness Across Market Regimes
Robustness is evaluated via chronological walk-forward validation across all assets. Metrics are aggregated within each split to inspect performance under varying volatility regimes, drawdowns, and market conditions. The `engineer_features.py` module labels regimes (e.g., high/low volatility buckets, risk-off periods) enabling analysis of whether models degrade during stress events. Future work should further stratify results by asset class, liquidity profiles, and macroeconomic cycles to avoid regime overfitting.

## Credentials & Sensitive Data
- **Order book / Level 2 feeds:** Insert broker API credentials (Interactive Brokers TWS, Alpaca) where the TODO markers appear. Store keys securely (environment variables, secrets manager).
- **Sentiment APIs:** Add keys for Twitter, news providers, or alternative sentiment platforms via `os.environ["SENTIMENT_API_KEY"]`.
- **Execution / Trading APIs:** Connect the VWAP/TWAP planners to broker SDKs once paper trading permissions are granted.
- **Security:** Keep secrets in environment variables or a local `.env` file (already ignored in version control). Never commit credentials, and rotate any key that may have been exposed.

### TODO checklist (API and private data wiring)
- `src/data/download_data.py::fetch_orderbook_snapshot`: replace the commented Alpaca placeholders (`APCA-API-KEY-ID`, `APCA-API-SECRET-KEY`) and return live Level 2 depth.
- `src/data/download_data.py` CLI: supply your Alpha Vantage key via `--api-key` or the `ALPHAVANTAGE_API_KEY/ALPHA_VANTAGE_API_KEY` environment variable when using the Alpha Vantage provider.
- `src/data/align_data.py`: pass `--sentiment-csv` or populate `fetch_sentiment_scores` with your API-backed sentiment feed.
- `src/advanced/sentiment.py`: implement the real sentiment loaders or API calls; keep credentials outside version control.
- `src/advanced/orderflow_scalping.py`: populate order book ingestion and OFI calculations using broker APIs.
- `src/advanced/pattern_recognition.py`: replace placeholders with rule-based or ML-driven chart-pattern detectors once you have annotated data.

### If you hit a merge conflict on GitHub
- Prefer the version that keeps the Alpha Vantage+yfinance downloader and shared path utilities (for example, **accept the incoming change** in `src/data/download_data.py`).
- Confirm post-merge that data still lands in `data/raw` and reports in `reports/`, and re-run `python -m compileall src`.
- See `MERGE_GUIDE.md` for a concise checklist.

## Reports & Interpretation
Each iteration stores Markdown summaries in `reports/iteration_X_results.md` and plots in `reports/figures/`. Review these artefacts alongside the notebooks to interpret model strengths, weaknesses, and improvement paths. Iteration 5 additionally logs cumulative PnL, Sharpe ratio, max drawdown, and hit rate for the dynamic strategy, while `src/risk/monte_carlo.py` produces equity/drawdown histograms and fan charts for stress-testing that strategy.

## Mathematical Reference (All Equations)
Every transformation, feature, target, and metric used across the pipeline is written explicitly in `docs/EQUATIONS.md`. Consult that file when documenting methodology or validating that the implementation matches the theoretical definitions (e.g., RSI, MACD, OFI, walk-forward scaling, RMSE/MAE/R², Sharpe/max drawdown, position sizing, Monte Carlo bootstrap).

## Additional Documentation
The `docs/` folder contains supplementary guides and references:
- **`COLAB_RUNBOOK.md`**: Ready-to-run pipelines for Google Colab (4-asset and 20-stock flows)
- **`EQUATIONS.md`**: Complete mathematical reference for all features, metrics, and transformations
- **`EXPERIMENTS.md`**: Experiment matrix detailing the three evaluation modes (mixed panel, per-asset, equity universe)
- **`local_setup.md`**: Step-by-step setup guide for Windows, macOS/Linux, and Colab environments

See also `CHANGELOG.md` for release notes and `MERGE_GUIDE.md` for conflict resolution guidance.

## Asset Scope Guidance
- **Single-asset runs (e.g., AAPL only):** useful for isolating model behaviour on one market, speeding up experimentation, and diagnosing feature relevance without cross-asset noise. Risk: overfitting to idiosyncratic patterns.
- **Single asset class (e.g., multiple equities or multiple FX pairs):** captures shared regime structure within a domain while keeping features comparable. Good middle ground for building specialised models per asset class.
- **Random sample of ~20 S&P 500 stocks:** broadens cross-sectional variation and can improve generalisation of tabular models (Linear/LightGBM) if features are scaled by ticker. It also enables pseudo cross-sectional learning but may dilute performance on any single name. When sampling, keep walk-forward splits chronological per ticker and consider per-ticker normalisation to avoid large-cap/low-vol names dominating the loss.

## Future Extensions
- **Microstructure Alpha:** Implement the order flow scalping module with real-time order book data, OFI features, and execution tactics.
- **Pattern Recognition & ICT Concepts:** Replace the placeholder pattern detectors with rule-based or vision-based recognition of liquidity grabs, fair value gaps, and Asia session behaviours.
- **Sentiment Fusion:** Integrate natural language processing pipelines for tweets, macro headlines, and alternative data. Consider transformer-based text encoders feeding into the time-series models.
- **Reinforcement Learning:** Extend `advanced/reinforcement_learning.py` to train DQN/PPO agents that learn execution policies under realistic transaction cost models.

## Acknowledgements
This project leverages open-source libraries (`pandas`, `scikit-learn`, `tensorflow`, `yfinance`) and builds upon academic literature on time-series forecasting, financial econometrics, and algorithmic trading. The structure is designed for rigorous experimentation, reproducibility, and transparency required for a dissertation project.
