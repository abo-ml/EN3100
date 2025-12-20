# Experiment Matrix and Validation Notes

This repository now supports three experiment modes that reuse the same walk-forward protocol, feature set, and metrics:

- **(A) Mixed 4-asset panel global models:** the original workflow trains a single model across AAPL, EURUSD=X, XAUUSD=X, and ^GSPC using chronological walk-forward splits.
- **(B) Single-asset re-estimation for each of the 4 assets:** the new `src/experiments/per_asset_evaluation.py` script reruns Iterations 1, 1.1, 2, and 2.1 for each ticker individually, producing per-asset reports and a directional-accuracy plot.
- **(C) Optional 20-stock S&P 500 equity universe:** `src/experiments/download_equity_universe.py` samples a reproducible equity universe, downloads raw data, and allows alignment/feature generation. `src/experiments/per_asset_equity_evaluation.py` then scores the same model set per stock with heatmaps and aggregated directional accuracy.

## Fairness & Protocol
- **Walk-forward only:** all experiments use the same `walk_forward_splits` helper with chronological training windows and held-out tails; no shuffling.
- **Train-only scaling:** standardisation is fit on the training fold and applied to the corresponding test fold to avoid lookahead bias.
- **Identical metrics:** RMSE, MAE, R², and directional accuracy are reported consistently across assets and models.
- **Feature parity:** feature engineering is unchanged between global and per-asset runs; the per-asset scripts simply filter the DataFrame by ticker before invoking the existing iteration runners.
- **No lookahead leakage:** targets are constructed with future returns (`next_day_return`) and directions; all rolling statistics are computed before slicing into train/test.

## Outputs
- Reports live under `reports/` with figures in `reports/figures/`:
  - `per_asset_metrics.(csv|md)` and `per_asset_directional_accuracy.png` for the 4-asset universe.
  - `per_asset_equity_metrics.(csv|md)`, `equity_avg_directional_accuracy.png`, and `equity_da_heatmap.png` for the optional equity universe.
- Raw universes are stored in `data/reference/`, and aligned/features in `data/processed/` to keep compatibility with the default mixed-asset workflow.
