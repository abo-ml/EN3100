"""Iteration 2: Tree-based and ensemble models."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, ParameterGrid
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

REPORT_PATH = REPORTS_DIR / "iteration_2_results.md"
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level="INFO")


def tune_random_forest(X_train, y_train, cv: int = 3) -> RandomForestRegressor:
    """Tune RandomForestRegressor using grid search cross-validation.

    This expanded tuning plan explores a comprehensive hyperparameter space
    based on a study suggesting grid search cross-validation with:
    - max_depth: {2, 4, 6, 8, 10} for controlling tree complexity
    - n_estimators: {64, 128, 256} for ensemble size
    - max_features: {'sqrt', 'log2', 0.8} for feature selection at each split
    - min_samples_leaf: {1, 2, 3, 4, 5} for regularization

    Parameters
    ----------
    X_train : array-like
        Training feature matrix.
    y_train : array-like
        Training target values.
    cv : int, optional
        Number of cross-validation folds (default: 3).

    Returns
    -------
    RandomForestRegressor
        Best model found by grid search.
    """
    param_grid = {
        "n_estimators": [64, 128, 256],
        "max_depth": [2, 4, 6, 8, 10],
        "max_features": ["sqrt", "log2", 0.8],
        "min_samples_leaf": [1, 2, 3, 4, 5],
    }

    base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        refit=True,
    )
    grid_search.fit(X_train, y_train)

    LOGGER.info(
        "Best RandomForest params: %s with CV score: %.4f",
        grid_search.best_params_,
        -grid_search.best_score_,
    )

    return grid_search.best_estimator_


def fit_xgb(X_train, y_train):
    if XGBRegressor is None:
        LOGGER.warning("XGBoost not installed; skipping")
        return None
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
