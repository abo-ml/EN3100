"""Iteration 1.1: Linear regression vs Support Vector Regression baseline."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def run_iteration(
    data: Optional[pd.DataFrame] = None,
    report_path: Optional[Path] = None,
    generate_reports: bool = True,
    ticker: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dataset = load_dataset(data)
    features = feature_columns(dataset)

    records: List[Dict[str, float]] = []
    prediction_records: List[Dict[str, object]] = []

    for split_id, (train_idx, test_idx) in enumerate(
        walk_forward_splits(dataset, n_splits=5, train_min_period=252), start=1
    ):
        train = dataset.iloc[train_idx].copy()
        test = dataset.iloc[test_idx].copy()

        if len(train) < 50 or len(test) == 0:
            LOGGER.warning("Skipping split %s due to insufficient data", split_id)
            continue

        tuned_params = tune_svr(train, features)

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

        prediction_records.extend(
            {
                "date": date,
                "ticker": tick,
                "actual": actual,
                "predicted": pred,
            }
            for date, tick, actual, pred in zip(test["date"].values, test["ticker"].values, y_test.values, svr_pred)
        )

        LOGGER.info(
            "Split %s | Linear RMSE %.4f vs SVR RMSE %.4f",
            split_id,
            record["rmse_linear"],
            record["rmse_svr"],
        )

    metrics_df = pd.DataFrame(records)
    summary = aggregate_metrics(records)
    if generate_reports:
        save_metrics_report(summary, report_path or REPORT_PATH)

        if prediction_records:
            pred_df = pd.DataFrame(prediction_records)
            plot_pred_vs_actual(
                dates=pd.Series(pred_df["date"]),
                actual=pd.Series(pred_df["actual"]),
                predicted=pd.Series(pred_df["predicted"]),
                title="Iteration 1.1: SVR vs Actual",
                filename=PRED_FIG,
            )

    suffix = f" for {ticker}" if ticker else ""
    LOGGER.info("Iteration 1.1 summary%s: %s", suffix, summary)
    predictions_df = pd.DataFrame(prediction_records)
    return metrics_df, predictions_df


def main() -> Tuple[pd.DataFrame, pd.DataFrame]:
    return run_iteration()


if __name__ == "__main__":
    main()
