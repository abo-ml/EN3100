"""Iteration 1.1: Linear regression vs Support Vector Regression baseline."""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from src.evaluation.metrics import directional_accuracy, mae, r2, rmse
from src.evaluation.reporting import plot_pred_vs_actual, save_metrics_report
from src.evaluation.walkforward import aggregate_metrics, walk_forward_splits
from src.models.iteration1_baseline import feature_columns, load_dataset
from src.utils import REPORTS_DIR

logging.basicConfig(level="INFO")
LOGGER = logging.getLogger(__name__)

SEED = 42
REPORT_PATH = REPORTS_DIR / "iteration_1_1_svr_results.md"
PRED_FIG = "iteration1_1_pred_vs_actual.png"


def split_train_validation(train_df: pd.DataFrame, val_fraction: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create a chronological train/validation split inside the training window."""

    val_size = max(1, int(len(train_df) * val_fraction))
    return train_df.iloc[:-val_size], train_df.iloc[-val_size:]


def tune_svr(train_df: pd.DataFrame, features: List[str]) -> Dict[str, float]:
    """Lightweight grid search for SVR hyperparameters using a validation tail."""

    train_part, val_part = split_train_validation(train_df)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_part[features])
    y_train = train_part["next_day_return"]
    X_val = scaler.transform(val_part[features])
    y_val = val_part["next_day_return"]

    grid = ParameterGrid(
        {
            "C": [1, 10, 100],
            "epsilon": [0.001, 0.01, 0.1],
            "gamma": ["scale", 0.01, 0.1],
        }
    )
    best_params: Dict[str, float] | None = None
    best_score = np.inf

    for params in grid:
        model = SVR(kernel="rbf", **params)
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        score = rmse(y_val, val_pred)
        if score < best_score:
            best_score = score
            best_params = params
    assert best_params is not None
    LOGGER.info("Best SVR params %s with validation RMSE %.4f", best_params, best_score)
    return best_params


def run_iteration() -> Dict[str, float]:
    df = load_dataset()
    features = feature_columns(df)

    records = []
    all_actual = []
    all_pred_svr = []
    all_dates = []

    for split_id, (train_idx, test_idx) in enumerate(walk_forward_splits(df, n_splits=5, train_min_period=252), start=1):
        train = df.iloc[train_idx].copy()
        test = df.iloc[test_idx].copy()

        if len(train) < 50 or len(test) == 0:
            LOGGER.warning("Skipping split %s due to insufficient data", split_id)
            continue

        # Tune SVR using a validation tail inside the training window
        tuned_params = tune_svr(train, features)

        # Fit scalers on full training data for final evaluation
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train[features])
        X_test = scaler.transform(test[features])
        y_train = train["next_day_return"]
        y_test = test["next_day_return"]

        linear_model = LinearRegression()
        linear_model.fit(X_train, y_train)
        lin_pred = linear_model.predict(X_test)

        svr_model = SVR(kernel="rbf", **tuned_params)
        svr_model.fit(X_train, y_train)
        svr_pred = svr_model.predict(X_test)

        record = {
            "rmse_linear": rmse(y_test, lin_pred),
            "mae_linear": mae(y_test, lin_pred),
            "r2_linear": r2(y_test, lin_pred),
            "directional_accuracy_linear": directional_accuracy(y_test, lin_pred),
            "rmse_svr": rmse(y_test, svr_pred),
            "mae_svr": mae(y_test, svr_pred),
            "r2_svr": r2(y_test, svr_pred),
            "directional_accuracy_svr": directional_accuracy(y_test, svr_pred),
        }
        records.append(record)

        all_actual.extend(y_test.values)
        all_pred_svr.extend(svr_pred)
        all_dates.extend(test["date"].values)

        LOGGER.info(
            "Split %s | Linear RMSE %.4f vs SVR RMSE %.4f",
            split_id,
            record["rmse_linear"],
            record["rmse_svr"],
        )

    summary = aggregate_metrics(records)
    save_metrics_report(summary, REPORT_PATH)

    if all_pred_svr:
        plot_pred_vs_actual(
            dates=pd.Series(all_dates),
            actual=pd.Series(all_actual),
            predicted=pd.Series(all_pred_svr),
            title="Iteration 1.1: SVR vs Actual",
            filename=PRED_FIG,
        )

    LOGGER.info("Iteration 1.1 summary: %s", summary)
    return summary


def main() -> Dict[str, float]:
    return run_iteration()


if __name__ == "__main__":
    main()
