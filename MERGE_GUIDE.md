# Merge guidance for EN3100

If you see a merge conflict on GitHub between your older local edits and the current repository version, prefer **the version that retains the shared path utilities and dual Alpha Vantage / yfinance downloader**. That version ensures:

- Data and reports write to `data/` and `reports/` at the project root (prevents nested folders in Colab).
- Alpha Vantage support (with `--provider` and API-key handling) plus yfinance fallback.
- Order-book and sentiment stubs remain aligned with the rest of the pipeline.

## Resolving conflicts in `src/data/download_data.py`
1. **Accept the incoming change** (the branch containing the Alpha Vantage + path-utils integration) when resolving conflicts. This keeps the CLI options (`--provider`, `--api-key`, `--pause`) and the cleaned `date` handling you need.
2. After merging, confirm the file still:
   - Resets the DatetimeIndex to a `date` column and sorts chronologically.
   - Adds the `ticker` column and forward-fills minor gaps.
   - Saves outputs under `data/raw` using the shared path helpers.

## Quick checklist before pushing
- Run `python -m compileall src` to catch syntax errors.
- If you changed downloader settings, re-run `python -m src.data.download_data --tickers AAPL EURUSD=X XAUUSD=X ^GSPC --start 2013-01-01 --end 2023-12-31 --provider alpha_vantage --format csv` in an environment with internet access.
- Verify the metrics markdown files in `reports/` after any re-training.

Following this keeps the repository consistent for Colab zip downloads and avoids regression to the older single-source downloader.
