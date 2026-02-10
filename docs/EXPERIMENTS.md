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

## Deep Learning Hyperparameter Tuning

LSTM and Transformer models (Iterations 3 and 4) can exhibit high RMSE (>0.12) without proper hyperparameter optimization. The `src/experiments/deep_learning_tuning.py` module implements Bayesian optimization using Optuna to search over literature-based hyperparameter spaces.

### Tuning Plan

The tuning script explores the following hyperparameter spaces based on academic literature on LSTM/GRU models for stock forecasting:

#### LSTM Search Space

| Parameter | Range | Literature Reference |
|-----------|-------|---------------------|
| Number of layers | 1-3 | Optimal architectures use 1-3 LSTM layers |
| Units per layer | 32-150 | Optimal: 96 or 128 neurons |
| Dropout rate | 0.0-0.5 | Optimal: 0.3-0.4 |
| Sequence length | 30-90 days | Captures short to medium-term patterns |
| Learning rate | 0.0005-0.01 | Log-uniform sampling |
| Batch size | 32, 64, 128 | Standard choices for time-series |
| L2 regularization | 0.0-0.01 | Kernel and recurrent weight regularization |
| Activation | tanh | Literature suggests tanh for LSTM gates |

#### Transformer Search Space

| Parameter | Range | Literature Reference |
|-----------|-------|---------------------|
| Number of encoder layers | 1-3 | Shallow transformers work well for time-series |
| Number of attention heads | 2, 4, 8 | Multi-head attention options |
| Model dimension (d_model) | 32, 64, 128 | Embedding size |
| Feed-forward dimension | 64, 128, 256 | Hidden layer size in FFN |
| Dropout rate | 0.0-0.5 | Applied to attention and FFN outputs |
| Sequence length | 30-90 days | Same as LSTM |
| Learning rate | 0.0005-0.01 | Log-uniform sampling |
| Batch size | 32, 64, 128 | Standard choices |

### Regularization Methods

1. **Dropout:** Applied between LSTM layers and in Transformer encoder blocks
2. **Early Stopping:** Patience-based (default: 10 epochs) with best weights restoration
3. **L2 Regularization:** Applied to LSTM kernel and recurrent weights

### Running the Tuning Script

```bash
# Tune both LSTM and Transformer with 50 trials each
python -m src.experiments.deep_learning_tuning --n-trials 50

# Tune only LSTM
python -m src.experiments.deep_learning_tuning --model lstm --n-trials 30

# Tune only Transformer
python -m src.experiments.deep_learning_tuning --model transformer --n-trials 30
```

### Configuration

Tuning parameters are defined in `configs/tuning.yaml`. Key settings include:
- `n_trials`: Number of Optuna optimization trials per model
- `n_splits`: Number of walk-forward validation splits
- `train_min_period`: Minimum training period (default: 252 days = 1 year)

### Outputs

Tuning results are saved to `reports/tuning/`:
- `best_hyperparameters.yaml`: Best parameters for each model
- `tuning_report.md`: Summary report with trial statistics
