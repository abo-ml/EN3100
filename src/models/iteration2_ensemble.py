"""Iteration 2: Tree-based and ensemble models."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

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

try:  # pragma: no cover - optional dependency
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover
    XGBRegressor = None

REPORT_PATH = Path("reports/iteration_2_results.md")
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


def run_iteration() -> Dict[str, float]:
    df = load_dataset()
    features = feature_columns(df)

    rf_importances = []
    records = []

    for split_id, (train_idx, test_idx) in enumerate(walk_forward_splits(df, n_splits=5, train_min_period=252), start=1):
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]

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
            record.update({
                "rmse_xgb": rmse(y_test, xgb_pred),
                "mae_xgb": mae(y_test, xgb_pred),
                "r2_xgb": r2(y_test, xgb_pred),
            })
        records.append(record)

    summary = aggregate_metrics(records)
    save_metrics_report(summary, REPORT_PATH)
    LOGGER.info("Iteration 2 summary: %s", summary)

    # Save average feature importances
    if rf_importances:
        mean_importance = pd.DataFrame(rf_importances).mean().sort_values(ascending=False)
        importance_path = Path("reports/iteration2_feature_importance.csv")
        importance_path.parent.mkdir(parents=True, exist_ok=True)
        mean_importance.to_csv(importance_path)

    return summary


def main() -> Dict[str, float]:
    return run_iteration()


if __name__ == "__main__":
    main()
