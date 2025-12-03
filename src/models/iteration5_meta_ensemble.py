"""Iteration 5: Meta-ensemble with risk-aware position sizing."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import directional_accuracy, mae, max_drawdown, rmse, sharpe_ratio
from src.evaluation.reporting import plot_equity_curve, save_metrics_report
from src.evaluation.walkforward import aggregate_metrics, walk_forward_splits
from src.models.iteration1_baseline import feature_columns, load_dataset
from src.models.iteration3_lstm import build_lstm_regressor
from src.models.iteration4_transformer import build_transformer_model
from src.utils import REPORTS_DIR

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

REPORT_PATH = REPORTS_DIR / "iteration_5_results.md"
REPORT_PATH = Path("reports/iteration_5_results.md")
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level="INFO")

WINDOW = 60
MAX_LEVERAGE = 2.0


def create_sequences_with_index(df: pd.DataFrame, features: List[str], target_col: str, window: int) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    sequences = []
    targets = []
    indices = []
    for ticker, group in df.groupby("ticker"):
        group = group.sort_values("date")
        feat = group[features].values
        target = group[target_col].values
        group_indices = group.index.values
        if len(group) <= window:
            continue
        for i in range(window, len(group)):
            sequences.append(feat[i - window : i])
            targets.append(target[i])
            indices.append(group_indices[i])
    if not sequences:
        return np.empty((0, window, len(features))), np.empty((0,)), []
    return np.array(sequences), np.array(targets), indices


def predictions_to_df(indices: List[int], values: np.ndarray, name: str) -> pd.DataFrame:
    return pd.DataFrame({"row_index": indices, name: values})


def vwap_execution_plan(total_notional: float, num_slices: int) -> pd.DataFrame:
    """Placeholder VWAP schedule splitting orders by historical volume profile."""

    weights = np.linspace(1, num_slices, num_slices)
    weights = weights / weights.sum()
    schedule = pd.DataFrame({"slice": range(1, num_slices + 1), "notional": total_notional * weights})
    return schedule


def twap_execution_plan(total_notional: float, num_slices: int) -> pd.DataFrame:
    """Placeholder TWAP schedule splitting orders evenly."""

    notional = total_notional / num_slices
    schedule = pd.DataFrame({"slice": range(1, num_slices + 1), "notional": [notional] * num_slices})
    return schedule


def pairs_spread_signal(asset_a: pd.Series, asset_b: pd.Series) -> float:
    spread = asset_a - asset_b
    z_score = (spread - spread.mean()) / spread.std(ddof=1)
    return float(z_score.iloc[-1])


def run_iteration() -> Dict[str, float]:
    df = load_dataset()
    features = feature_columns(df)

    records = []
    equity_curves = []

    for split_id, (train_idx, test_idx) in enumerate(walk_forward_splits(df, n_splits=3, train_min_period=252), start=1):
        train = df.iloc[train_idx].copy()
        test = df.iloc[test_idx].copy()
        if len(train) < WINDOW * 2:
            LOGGER.warning("Skipping split %s due to insufficient training data", split_id)
            continue

        meta_window = max(WINDOW + 5, int(0.2 * len(train)))
        inner_train = train.iloc[:-meta_window]
        meta_train = train.iloc[-meta_window:]

        scaler = StandardScaler()
        inner_train[features] = scaler.fit_transform(inner_train[features])
        meta_train[features] = scaler.transform(meta_train[features])
        test[features] = scaler.transform(test[features])

        # Level 1 models ------------------------------------------------------
        linear_model = LinearRegression()
        rf_model = GradientBoostingRegressor(random_state=SEED)

        linear_model.fit(inner_train[features], inner_train["next_day_return"])
        rf_model.fit(inner_train[features], inner_train["next_day_return"])

        X_meta_linear = linear_model.predict(meta_train[features])
        X_test_linear = linear_model.predict(test[features])

        X_meta_rf = rf_model.predict(meta_train[features])
        X_test_rf = rf_model.predict(test[features])

        # Sequence models
        lstm_model = None
        transformer_model = None

        X_seq_train, y_seq_train, _ = create_sequences_with_index(inner_train, features, "next_day_return", WINDOW)
        if X_seq_train.size > 0:
            lstm_model = build_lstm_regressor((WINDOW, len(features)))
            lstm_model.fit(
                X_seq_train,
                y_seq_train,
                epochs=15,
                batch_size=32,
                validation_split=0.2,
                callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)],
                verbose=0,
            )
            transformer_model = build_transformer_model((WINDOW, len(features)))
            transformer_model.fit(
                X_seq_train,
                y_seq_train,
                epochs=20,
                batch_size=32,
                validation_split=0.2,
                callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)],
                verbose=0,
            )

        X_seq_meta, _, meta_indices = create_sequences_with_index(meta_train, features, "next_day_return", WINDOW)
        X_seq_test, _, test_indices = create_sequences_with_index(test, features, "next_day_return", WINDOW)

        lstm_meta = predictions_to_df(meta_indices, lstm_model.predict(X_seq_meta, verbose=0).flatten(), "pred_lstm") if lstm_model and X_seq_meta.size else pd.DataFrame()
        lstm_test = predictions_to_df(test_indices, lstm_model.predict(X_seq_test, verbose=0).flatten(), "pred_lstm") if lstm_model and X_seq_test.size else pd.DataFrame()

        transformer_meta = predictions_to_df(meta_indices, transformer_model.predict(X_seq_meta, verbose=0).flatten(), "pred_transformer") if transformer_model and X_seq_meta.size else pd.DataFrame()
        transformer_test = predictions_to_df(test_indices, transformer_model.predict(X_seq_test, verbose=0).flatten(), "pred_transformer") if transformer_model and X_seq_test.size else pd.DataFrame()

        meta_df = predictions_to_df(meta_train.index.tolist(), X_meta_linear, "pred_linear")
        meta_df = meta_df.merge(predictions_to_df(meta_train.index.tolist(), X_meta_rf, "pred_rf"), on="row_index", how="left")
        if not lstm_meta.empty:
            meta_df = meta_df.merge(lstm_meta, on="row_index", how="left")
        if not transformer_meta.empty:
            meta_df = meta_df.merge(transformer_meta, on="row_index", how="left")
        meta_df["target"] = meta_train.loc[meta_df["row_index"], "next_day_return"].values

        test_df = predictions_to_df(test.index.tolist(), X_test_linear, "pred_linear")
        test_df = test_df.merge(predictions_to_df(test.index.tolist(), X_test_rf, "pred_rf"), on="row_index", how="left")
        if not lstm_test.empty:
            test_df = test_df.merge(lstm_test, on="row_index", how="left")
        if not transformer_test.empty:
            test_df = test_df.merge(transformer_test, on="row_index", how="left")

        meta_df.dropna(inplace=True)
        test_df.dropna(inplace=True)
        if meta_df.empty or test_df.empty:
            LOGGER.warning("Meta dataset empty for split %s", split_id)
            continue

        feature_cols = [col for col in meta_df.columns if col.startswith("pred_")]
        meta_reg = Ridge(alpha=1.0)
        meta_reg.fit(meta_df[feature_cols], meta_df["target"])

        meta_clf = LogisticRegression(max_iter=500)
        meta_clf.fit(meta_df[feature_cols], (meta_df["target"] > 0).astype(int))

        meta_pred = meta_reg.predict(test_df[feature_cols])
        class_probs = meta_clf.predict_proba(test_df[feature_cols])[:, 1]
        class_pred = (class_probs > 0.5).astype(int)

        # Attach outcomes for evaluation
        test_df["pred_return"] = meta_pred
        test_df["pred_prob_up"] = class_probs
        test_df["actual_return"] = test.loc[test_df["row_index"], "next_day_return"].values
        test_df["volatility"] = test.loc[test_df["row_index"], "volatility_21"].values
        test_df["date"] = test.loc[test_df["row_index"], "date"].values

        position_size = np.clip(test_df["pred_return"] / (test_df["volatility"].replace(0, np.nan) + 1e-6), -MAX_LEVERAGE, MAX_LEVERAGE)
        strategy_returns = position_size * test_df["actual_return"]

        record = {
            "rmse_meta": rmse(test_df["actual_return"], test_df["pred_return"]),
            "mae_meta": mae(test_df["actual_return"], test_df["pred_return"]),
            "directional_accuracy_meta": directional_accuracy(test_df["actual_return"], test_df["pred_return"]),
            "classification_accuracy_meta": (class_pred == (test_df["actual_return"] > 0).astype(int)).mean(),
            "hit_rate": np.mean(np.sign(test_df["actual_return"]) == np.sign(test_df["pred_return"])),
            "sharpe": sharpe_ratio(strategy_returns),
            "max_drawdown": max_drawdown(strategy_returns),
        }
        records.append(record)

        equity_curve = (1 + strategy_returns).cumprod()
        equity_curves.append(pd.DataFrame({"date": test_df["date"], "equity": equity_curve}))

    summary = aggregate_metrics(records)
    save_metrics_report(summary, REPORT_PATH)

    if equity_curves:
        combined = pd.concat(equity_curves).sort_values("date")
        plot_equity_curve(combined["date"], combined["equity"], "Iteration 5 Equity Curve", "iteration5_equity_curve.png")

    LOGGER.info("Iteration 5 summary: %s", summary)
    return summary


def main() -> Dict[str, float]:
    return run_iteration()


if __name__ == "__main__":
    main()
