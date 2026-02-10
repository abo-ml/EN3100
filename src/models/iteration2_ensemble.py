"""Iteration 2: Tree-based and ensemble models."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.evaluation.metrics import directional_accuracy, mae, r2, rmse
from src.evaluation.reporting import save_metrics_report
from src.evaluation.walkforward import aggregate_metrics, walk_forward_splits
from src.models.iteration1_baseline import feature_columns, load_dataset
from src.utils import REPORTS_DIR

try:  # pragma: no cover - optional dependency
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover
    XGBRegressor = None

from src.models.xgboost_tuning import (
    XGBoostTuningConfig,
    fit_tuned_xgboost,
    get_reduced_grid_config,
)

REPORT_PATH = REPORTS_DIR / "iteration_2_results.md"
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level="INFO")


def tune_random_forest(X_train, y_train) -> RandomForestRegressor:
    grid = ParameterGrid({"n_estimators": [100, 200], "max_depth": [5, 10], "min_samples_leaf": [2, 4]})
    best_score = -np.inf
    best_model = None
    for params in grid:
        model = RandomForestRegressor(random_state=42, n_jobs=-1, **params)
        model.fit(X_train, y_train)
        score = model.score(X_train, y_train)
        if score > best_score:
            best_score = score
            best_model = model
    return best_model


def fit_xgb(X_train, y_train, tune: bool = True, tuning_method: str = "grid"):
    """Fit XGBoost model with optional hyperparameter tuning.

    This function addresses XGBoost underperformance by implementing per-window
    hyperparameter tuning. Research emphasises that optimal hyperparameters change
    across rebalancing windows, making this re-tuning essential for good performance.

    Parameters
    ----------
    X_train : array-like
        Training features (already scaled).
    y_train : array-like
        Training target values.
    tune : bool, optional
        If True, performs hyperparameter tuning. If False, uses default params.
        Defaults to True.
    tuning_method : str, optional
        Tuning method: "grid" for grid search, "bayesian" for Bayesian optimization.
        Defaults to "grid".

    Returns
    -------
    XGBRegressor or None
        Fitted model, or None if XGBoost is not installed.
    """
    if XGBRegressor is None:
        LOGGER.warning("XGBoost not installed; skipping")
        return None

    if tune:
        # Use reduced grid for efficiency in walk-forward validation
        config = get_reduced_grid_config()
        return fit_tuned_xgboost(X_train, y_train, method=tuning_method, config=config)

    # Fall back to default parameters (original behavior)
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def run_iteration(
    data: Optional[pd.DataFrame] = None,
    report_path: Optional[Path] = None,
    generate_reports: bool = True,
    ticker: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dataset = load_dataset(data)
    features = feature_columns(dataset)

    rf_importances = []
    records: List[Dict[str, float]] = []
    prediction_records: List[Dict[str, object]] = []

    for split_id, (train_idx, test_idx) in enumerate(
        walk_forward_splits(dataset, n_splits=5, train_min_period=252), start=1
    ):
        train = dataset.iloc[train_idx]
        test = dataset.iloc[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(train[features])
        X_test = scaler.transform(test[features])
        y_train = train["next_day_return"]
        y_test = test["next_day_return"]

        rf_model = tune_random_forest(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        rf_importances.append(dict(zip(features, rf_model.feature_importances_)))

        svm_clf = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True)
        svm_clf.fit(X_train, train["target_direction"])
        svm_probs = svm_clf.predict_proba(X_test)[:, 1]
        svm_pred = (svm_probs > 0.5).astype(int)

        xgb_model = fit_xgb(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test) if xgb_model is not None else np.zeros_like(rf_pred)

        record = {
            "rmse_rf": rmse(y_test, rf_pred),
            "mae_rf": mae(y_test, rf_pred),
            "r2_rf": r2(y_test, rf_pred),
            "directional_accuracy_rf": directional_accuracy(y_test, rf_pred),
            "svm_accuracy": accuracy_score(test["target_direction"], svm_pred),
        }
        if xgb_model is not None:
            record.update(
                {
                    "rmse_xgb": rmse(y_test, xgb_pred),
                    "mae_xgb": mae(y_test, xgb_pred),
                    "r2_xgb": r2(y_test, xgb_pred),
                    "directional_accuracy_xgb": directional_accuracy(y_test, xgb_pred),
                }
            )
        records.append(record)
        prediction_records.extend(
            {
                "date": date,
                "ticker": tick,
                "actual": actual,
                "predicted": pred,
                "model": "random_forest",
            }
            for date, tick, actual, pred in zip(test["date"].values, test["ticker"].values, y_test.values, rf_pred)
        )

    metrics_df = pd.DataFrame(records)
    summary = aggregate_metrics(records)
    if generate_reports:
        save_metrics_report(summary, report_path or REPORT_PATH)

        if rf_importances:
            mean_importance = pd.DataFrame(rf_importances).mean().sort_values(ascending=False)
            importance_path = REPORTS_DIR / "iteration2_feature_importance.csv"
            importance_path.parent.mkdir(parents=True, exist_ok=True)
            mean_importance.to_csv(importance_path)

    suffix = f" for {ticker}" if ticker else ""
    LOGGER.info("Iteration 2 summary%s: %s", suffix, summary)
    predictions_df = pd.DataFrame(prediction_records)
    return metrics_df, predictions_df


def main() -> Tuple[pd.DataFrame, pd.DataFrame]:
    return run_iteration()


if __name__ == "__main__":
    main()
