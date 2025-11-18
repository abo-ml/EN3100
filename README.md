# Design and Application of Machine Learning Models for Financial Market Prediction: A Comparative Study of Traditional and Deep Learning Approaches

## Overview
This repository accompanies an academic dissertation exploring a progressive sequence of forecasting and trading system designs for liquid financial assets. The project walks through five model iterations ranging from linear baselines to attention-based ensembles with risk-aware position sizing. The codebase emphasises reproducible research, walk-forward validation, and extensibility toward microstructure, sentiment, and reinforcement learning enhancements.

The default assets include:
- **AAPL** (US equity)
- **EURUSD=X** (FX proxy via Yahoo Finance)
- **XAUUSD=X** (Gold spot proxy)
- **^GSPC** (S&P 500 index benchmark)

All components operate on daily OHLCV bars sourced via `yfinance`. Hooks are provided for alternative data vendors (Alpha Vantage, Polygon, Interactive Brokers) and advanced modules (order flow imbalance, ICT/SMT liquidity concepts, reinforcement learning). TODO markers highlight where sensitive credentials, private market data, or broker integrations must be supplied by the student.

## Repository Structure
```
├── data/
│   ├── raw/                # Downloaded OHLCV data
│   └── processed/          # Aligned and feature-engineered datasets
├── notebooks/
│   ├── Iteration_1.ipynb.py
│   ├── Iteration_2.ipynb.py
│   ├── Iteration_3.ipynb.py
│   ├── Iteration_4.ipynb.py
│   └── Iteration_5.ipynb.py
├── reports/
│   ├── figures/            # Generated plots
│   ├── iteration_1_results.md
│   ├── iteration_2_results.md
│   ├── iteration_3_results.md
│   ├── iteration_4_results.md
│   └── iteration_5_results.md
├── src/
│   ├── advanced/           # Future-work stubs (order flow, pattern recognition, RL)
│   ├── data/               # Data acquisition & alignment
│   ├── evaluation/         # Metrics, walk-forward, reporting utilities
│   ├── features/           # Feature engineering and target construction
│   └── models/             # Iteration-specific training scripts
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
    --end 2023-12-31
```
Optional arguments allow changing the interval, retry policy, and output format. **TODO:** replace the dummy `fetch_orderbook_snapshot` implementation with a real broker API call (Interactive Brokers, Alpaca, etc.) when credentials are available.

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
python -m src.models.iteration2_ensemble
python -m src.models.iteration3_lstm
python -m src.models.iteration4_transformer
python -m src.models.iteration5_meta_ensemble
```
Refer to the notebooks in `notebooks/` for guided walkthroughs, narrative commentary, and exploratory analysis aligned with each iteration.

## Iteration Roadmap
1. **Iteration 1 – Linear Baselines:** Persistence, linear regression, and logistic regression models validate the pipeline and establish benchmark metrics.
2. **Iteration 2 – Tree-Based Ensembles:** Random Forest, optional XGBoost, and SVM classifiers capture nonlinear relationships and deliver feature importance insights.
3. **Iteration 3 – LSTM Sequence Model:** Recurrent neural networks learn temporal dependencies over rolling windows and introduce deep learning considerations (scaling, early stopping, regularisation).
4. **Iteration 4 – Transformer Encoder:** Attention mechanisms handle long-range dependencies and multimodal inputs (price, sentiment, order flow placeholders) for enhanced forecasting power.
5. **Iteration 5 – Meta-Ensemble with Risk Layer:** Stacks the strongest models, applies dynamic volatility-aware position sizing, and outlines execution scheduling stubs (VWAP/TWAP, pairs trading hook).

## Validation & Fairness Across Market Regimes
Robustness is evaluated via chronological walk-forward validation across all assets. Metrics are aggregated within each split to inspect performance under varying volatility regimes, drawdowns, and market conditions. The `engineer_features.py` module labels regimes (e.g., high/low volatility buckets, risk-off periods) enabling analysis of whether models degrade during stress events. Future work should further stratify results by asset class, liquidity profiles, and macroeconomic cycles to avoid regime overfitting.

## Credentials & Sensitive Data
- **Order book / Level 2 feeds:** Insert broker API credentials (Interactive Brokers TWS, Alpaca) where the TODO markers appear. Store keys securely (environment variables, secrets manager).
- **Sentiment APIs:** Add keys for Twitter, news providers, or alternative sentiment platforms via `os.environ["SENTIMENT_API_KEY"]`.
- **Execution / Trading APIs:** Connect the VWAP/TWAP planners to broker SDKs once paper trading permissions are granted.

## Reports & Interpretation
Each iteration stores Markdown summaries in `reports/iteration_X_results.md` and plots in `reports/figures/`. Review these artefacts alongside the notebooks to interpret model strengths, weaknesses, and improvement paths. Iteration 5 additionally logs cumulative PnL, Sharpe ratio, max drawdown, and hit rate for the dynamic strategy.

## Future Extensions
- **Microstructure Alpha:** Implement the order flow scalping module with real-time order book data, OFI features, and execution tactics.
- **Pattern Recognition & ICT Concepts:** Replace the placeholder pattern detectors with rule-based or vision-based recognition of liquidity grabs, fair value gaps, and Asia session behaviours.
- **Sentiment Fusion:** Integrate natural language processing pipelines for tweets, macro headlines, and alternative data. Consider transformer-based text encoders feeding into the time-series models.
- **Reinforcement Learning:** Extend `advanced/reinforcement_learning.py` to train DQN/PPO agents that learn execution policies under realistic transaction cost models.

## Acknowledgements
This project leverages open-source libraries (`pandas`, `scikit-learn`, `tensorflow`, `yfinance`) and builds upon academic literature on time-series forecasting, financial econometrics, and algorithmic trading. The structure is designed for rigorous experimentation, reproducibility, and transparency required for a dissertation project.
