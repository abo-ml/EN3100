# Local and Colab Setup Guide

This guide shows how to create a virtual environment, install dependencies, set the Alpha Vantage key, and run the EN3100 pipeline on Windows (PowerShell), macOS/Linux (bash), and Google Colab.

## 1. Clone or unzip the repository
- **Windows:** Open PowerShell and run
  ```powershell
  cd C:\Users\<you>\source\repos
  git clone https://github.com/<your-username>/EN3100.git
  cd EN3100
  ```
- **macOS/Linux:**
  ```bash
  mkdir -p ~/projects && cd ~/projects
  git clone https://github.com/<your-username>/EN3100.git
  cd EN3100
  ```
- **Google Colab:** Upload the repo zip to `/content`, then run
  ```bash
  !unzip -o "/content/EN3100.zip" -d /content
  %cd /content/EN3100
  ```

## 2. Create and activate a virtual environment
- **Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .venv\Scripts\Activate.ps1
  ```
- **macOS/Linux (bash):**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```
- **Google Colab:** Colab already provides an environment; you can skip the venv and install directly.

## 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Set the Alpha Vantage API key
- **Windows (PowerShell):**
  ```powershell
  $env:ALPHA_VANTAGE_API_KEY = "YOUR_KEY_HERE"
  ```
- **macOS/Linux (bash):**
  ```bash
  export ALPHA_VANTAGE_API_KEY="YOUR_KEY_HERE"
  ```
- **Google Colab:**
  ```python
  import os
  os.environ["ALPHA_VANTAGE_API_KEY"] = "YOUR_KEY_HERE"
  ```

## 5. Run the pipeline from the project root
1. **Download data** (Alpha Vantage by default, with yfinance fallback):
   ```bash
   python -m src.data.download_data --tickers AAPL EURUSD=X XAUUSD=X ^GSPC --start 2013-01-01 --end 2023-12-31 --format csv
   ```
2. **Align sources:**
   ```bash
   python -m src.data.align_data
   ```
3. **Engineer features:**
   ```bash
   python -m src.features.engineer_features
   ```
4. **Run models:**
   ```bash
   python -m src.models.iteration1_baseline
   python -m src.models.iteration2_ensemble
   python -m src.models.iteration3_lstm
   python -m src.models.iteration4_transformer
   python -m src.models.iteration5_meta_ensemble
   ```

All outputs (datasets, metrics, plots) will be written under `data/processed/` and `reports/` using the shared path utilities.
