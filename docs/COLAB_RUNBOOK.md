# Google Colab Runbook for EN3100

This guide contains two end-to-end pipelines that run with the repository mounted at `/content/EN3100`. Both flows rely on the built-in `yfinance` provider (no API keys required) and assume the code is executed from the project root.

## Initial Setup: Extracting the ZIP

When you upload or clone the repository into Colab, the extracted directory name may vary (e.g., `EN3100`, `EN3100-main`, etc.). Use this robust pattern to detect and change to the correct directory:

```python
import os

# Detect the correct project directory after unzipping
possible_dirs = ['EN3100', 'EN3100-main', 'EN3100-master']
project_dir = None

for dirname in possible_dirs:
    path = f'/content/{dirname}'
    if os.path.isdir(path):
        project_dir = path
        break

if project_dir is None:
    # List contents of /content to help debug
    print("ERROR: Could not find EN3100 project directory.")
    print("Contents of /content:", os.listdir('/content'))
    print("Please check the extracted folder name and update the path manually.")
else:
    os.chdir(project_dir)
    print(f"Changed to project directory: {project_dir}")
```

## Pipeline A: 4-asset per-asset evaluation
```bash
%cd "/content/EN3100"
!pip install -r requirements.txt -q
!python -m src.data.download_data --tickers AAPL EURUSD=X XAUUSD=X ^GSPC --start 2013-01-01 --end 2023-12-31 --provider yfinance
!python -m src.data.align_data
!python -m src.features.engineer_features
!python -m src.experiments.per_asset_evaluation
```

## Pipeline B: 20-stock S&P 500 equity universe
```bash
%cd "/content/EN3100"
!pip install -r requirements.txt -q
!python -m src.experiments.download_equity_universe --n 20 --sector "Information Technology" --price-min 30 --price-max 150 --start 2013-01-01 --end 2023-12-31 --provider yfinance
!python -m src.data.align_data --ticker-file data/reference/equity_universe_20.txt
!python -m src.features.engineer_features
!python -m src.experiments.per_asset_equity_evaluation
```

### Notes
- Outputs are written to `data/processed/` and `reports/` (figures under `reports/figures/`). Existing files are overwritten safely on rerun.
- The default mixed-asset workflow remains unchanged if these optional steps are skipped.
- Keep any secrets in environment variables or a local `.env` file (ignored by git); no credentials are required for the `yfinance` commands above.
- If the `%cd` command fails to find the directory, use the detection script above to identify the correct folder name.
