"""Iteration 5: Meta-ensemble with risk-aware position sizing."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import directional_accuracy, mae, max_drawdown, r2, rmse, sharpe_ratio
from src.evaluation.reporting import plot_equity_curve, save_metrics_report
from src.evaluation.walkforward import aggregate_metrics, walk_forward_splits
from src.models.iteration1_baseline import feature_columns, load_dataset
from src.models.iteration3_lstm import build_lstm_regressor
from src.models.iteration4_transformer import build_transformer_model
from src.utils import PROCESSED_DIR, REPORTS_DIR

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


def tune_meta_logistic(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 3,
) -> Dict[str, object]:
    """Tune L1-penalised logistic regression via stratified cross-validation.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (meta-features from level 1 models).
    y : np.ndarray
        Binary target (1 = positive return, 0 = non-positive).
    n_folds : int
        Number of stratified CV folds.

    Returns
    -------
    dict
        Best hyperparameters including C, l1_ratio, solver, and class_weight.
    """
    # C values on a log scale from 0.001 to 10
    c_values = [0.001, 0.01, 0.1, 1.0, 10.0]
    grid = ParameterGrid({"C": c_values})

    best_params: Dict[str, object] = {}
    best_score = -np.inf

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    for params in grid:
        fold_scores = []
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = LogisticRegression(
                solver="saga",
                l1_ratio=1.0,  # L1 penalty (l1_ratio=1 is pure L1)
                max_iter=1000,
                class_weight="balanced",
                random_state=SEED,
                **params,
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            # Use balanced accuracy for imbalanced class handling
            # Balanced accuracy = (recall_class0 + recall_class1) / 2
            tp = np.sum((y_pred == 1) & (y_val == 1))
            tn = np.sum((y_pred == 0) & (y_val == 0))
            p = np.sum(y_val == 1)
            n = np.sum(y_val == 0)
            recall_pos = tp / p if p > 0 else 0.0
            recall_neg = tn / n if n > 0 else 0.0
            balanced_acc = (recall_pos + recall_neg) / 2
            fold_scores.append(balanced_acc)

        mean_score = np.mean(fold_scores)
        if mean_score > best_score:
            best_score = mean_score
            best_params = {
                "C": params["C"],
                "l1_ratio": 1.0,
                "solver": "saga",
                "max_iter": 1000,
                "class_weight": "balanced",
                "random_state": SEED,
            }

    LOGGER.info("Best meta-logistic params: C=%.4f with CV balanced accuracy %.4f", best_params.get("C", 1.0), best_score)
    return best_params


def run_iteration(
    data: Optional[pd.DataFrame] = None,
    report_path: Optional[Path] = None,
    generate_reports: bool = True,
    ticker: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = load_dataset(data)
    features = feature_columns(df)

    records: List[Dict[str, float]] = []
    equity_curves = []
    strategy_return_frames = []
    prediction_frames: List[pd.DataFrame] = []

    for split_id, (train_idx, test_idx) in enumerate(walk_forward_splits(df, n_splits=3, train_min_period=252), start=1):
        train = df.iloc[train_idx].copy()
        test = df.iloc[test_idx].copy()
        if len(train) < WINDOW * 2:
            LOGGER.warning("Skipping split %s due to insufficient training data", split_id)
            continue

        meta_window = max(WINDOW + 5, int(0.2 * len(train)))
        inner_train = train.iloc[:-meta_window]
        meta_train = train.iloc[-meta_window:]

        # Preserve unscaled volatility for risk sizing before we standardise features.
        test_volatility_raw = test["volatility_21"].copy()

        # Cast boolean feature columns to float to avoid dtype conflicts during scaling
        bool_cols = [col for col in features if inner_train[col].dtype == bool]
        for col in bool_cols:
            inner_train[col] = inner_train[col].astype(float)
            meta_train[col] = meta_train[col].astype(float)
            test[col] = test[col].astype(float)

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

        y = (meta_df["target"] > 0).astype(int)
        n_classes = y.nunique()

        meta_clf = None
        constant_class = None

        if n_classes < 2:
            # Edge case: only one class present -> cannot train logistic regression
            constant_class = int(y.iloc[0])  # 0 or 1
        else:
            meta_clf = LogisticRegression(max_iter=2000, solver="lbfgs")
            meta_clf.fit(meta_df[feature_cols], y)

        meta_pred = meta_reg.predict(test_df[feature_cols])

        # Handle prediction with fallback for single-class case
        if meta_clf is None:
            # constant probability of "up"
            class_probs = np.full(len(test_df), 1.0 if constant_class == 1 else 0.0)
        else:
            class_probs = meta_clf.predict_proba(test_df[feature_cols])[:, 1]

        class_pred = (class_probs > 0.5).astype(int)

        # Attach outcomes for evaluation
        test_df["pred_return"] = meta_pred
        test_df["pred_prob_up"] = class_probs
        test_df["actual_return"] = test.loc[test_df["row_index"], "next_day_return"].values
        test_df["volatility"] = test.loc[test_df["row_index"], "volatility_21"].values
        test_df["date"] = test.loc[test_df["row_index"], "date"].values
        test_df["ticker"] = test.loc[test_df["row_index"], "ticker"].values

        position_size = np.clip(test_df["pred_return"] / (test_df["volatility"].replace(0, np.nan) + 1e-6), -MAX_LEVERAGE, MAX_LEVERAGE)
        strategy_returns = position_size * test_df["actual_return"]

        record = {
            "rmse_meta": rmse(test_df["actual_return"], test_df["pred_return"]),
            "mae_meta": mae(test_df["actual_return"], test_df["pred_return"]),
            "r2_meta": r2(test_df["actual_return"], test_df["pred_return"]),
            "directional_accuracy_meta": directional_accuracy(test_df["actual_return"], test_df["pred_return"]),
            "classification_accuracy_meta": (class_pred == (test_df["actual_return"] > 0).astype(int)).mean(),
            "hit_rate": np.mean(np.sign(test_df["actual_return"]) == np.sign(test_df["pred_return"])),
            "sharpe": sharpe_ratio(strategy_returns),
            "max_drawdown": max_drawdown(strategy_returns),
        }
        records.append(record)
        prediction_frames.append(
            test_df[["date", "ticker", "actual_return", "pred_return"]].rename(
                columns={"actual_return": "actual", "pred_return": "predicted"}
            )
        )

        equity_curve = (1 + strategy_returns).cumprod()
        equity_curves.append(pd.DataFrame({"date": test_df["date"], "equity": equity_curve}))
        strategy_return_frames.append(
            pd.DataFrame(
                {
                    "date": test_df["date"].values,
                    "strategy_return": strategy_returns,
                }
            )
        )

    metrics_df = pd.DataFrame(records)
    summary = aggregate_metrics(records)
    if generate_reports:
        save_metrics_report(summary, report_path or REPORT_PATH)

    if generate_reports and strategy_return_frames:
        combined_returns = pd.concat(strategy_return_frames).sort_values("date")
        combined_returns["equity_curve"] = (1 + combined_returns["strategy_return"]).cumprod()
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        output_returns = PROCESSED_DIR / "iteration5_strategy_returns.csv"
        combined_returns.to_csv(output_returns, index=False)
        LOGGER.info("Saved strategy returns to %s", output_returns)

    if generate_reports and equity_curves:
        combined = pd.concat(equity_curves).sort_values("date")
        plot_equity_curve(combined["date"], combined["equity"], "Iteration 5 Equity Curve", "iteration5_equity_curve.png")

    suffix = f" for {ticker}" if ticker else ""
    LOGGER.info("Iteration 5 summary%s: %s", suffix, summary)
    predictions_df = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    return metrics_df, predictions_df


def main() -> Tuple[pd.DataFrame, pd.DataFrame]:
    return run_iteration()


if __name__ == "__main__":
    main()
