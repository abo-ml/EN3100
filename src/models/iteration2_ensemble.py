"""Iteration 2: Tree-based / ensemble models with walk-forward validation.

This script trains nonlinear regressors and a direction classifier on the
engineered dataset. It mirrors the Iteration 1 structure but upgrades the
model class to capture interactions while still respecting time order.

Outputs
-------
- reports/iteration_2_results.md: human-readable metrics summary.
- reports/figures/iteration2_pred_vs_actual.png: predicted vs. actual returns.
- reports/figures/iteration2_feature_importances.png: average RF importances.

Usage
-----
python -m src.models.iteration2_ensemble
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.evaluation.metrics import directional_accuracy, mae, r2, rmse
from src.evaluation.reporting import (
    plot_pred_vs_actual,
    save_metrics_report,
)
from src.evaluation.walkforward import aggregate_metrics, walk_forward_splits
from src.models.iteration1_baseline import feature_columns, load_dataset
from src.utils import REPORTS_DIR

try:  # pragma: no cover - optional dependency
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover
    XGBRegressor = None


SEED = 42
REPORT_PATH = REPORTS_DIR / "iteration_2_results.md"
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level="INFO")


def tune_random_forest(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestRegressor:
    """Lightweight grid search for a RandomForestRegressor.

    Uses a small parameter grid to keep runtime reasonable in Colab while still
    exploring depth and leaf size trade-offs. Scoring is based on R^2 on the
    training fold only; walk-forward validation captures generalisation.
    """

    grid = ParameterGrid(
        {
            "n_estimators": [150, 300],
            "max_depth": [5, 10, None],
            "min_samples_leaf": [1, 3],
        }
    )
    best_score = -np.inf
    best_model: RandomForestRegressor | None = None
    for params in grid:
        model = RandomForestRegressor(random_state=SEED, n_jobs=-1, **params)
        model.fit(X_train, y_train)
        score = model.score(X_train, y_train)
        if score > best_score:
            best_score = score
            best_model = model
    assert best_model is not None
    return best_model


def fit_xgb(X_train: np.ndarray, y_train: np.ndarray):
    """Fit an optional XGBoost regressor if the dependency is available."""

    if XGBRegressor is None:
        LOGGER.warning("XGBoost not installed; skipping XGBRegressor")
        return None
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=SEED,
    )
    model.fit(X_train, y_train)
    return model


def _build_classifier() -> SVC:
    """Return a simple SVM classifier for direction classification."""

    return SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=SEED)


def run_iteration() -> Dict[str, float]:
    """Execute walk-forward training/evaluation for ensemble models."""

    df = load_dataset()
    features = feature_columns(df)

    rf_importances: List[Dict[str, float]] = []
    predictions_accum: List[float] = []
    actuals_accum: List[float] = []
    dates_accum: List[pd.Timestamp] = []
    records: List[Dict[str, float]] = []

    for split_id, (train_idx, test_idx) in enumerate(
        walk_forward_splits(df, n_splits=5, train_min_period=252), start=1
    ):
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(train[features])
        X_test = scaler.transform(test[features])
        y_train = train["next_day_return"].values
        y_test = test["next_day_return"].values

        rf_model = tune_random_forest(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        rf_importances.append(dict(zip(features, rf_model.feature_importances_)))

        classifier = _build_classifier()
        classifier.fit(X_train, train["target_direction"].values)
        class_probs = classifier.predict_proba(X_test)[:, 1]
        class_pred = (class_probs > 0.5).astype(int)

        xgb_model = fit_xgb(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test) if xgb_model is not None else None

        record: Dict[str, float] = {
            "rmse_rf": rmse(y_test, rf_pred),
            "mae_rf": mae(y_test, rf_pred),
            "r2_rf": r2(y_test, rf_pred),
            "directional_accuracy_rf": directional_accuracy(y_test, rf_pred),
            "svm_accuracy": accuracy_score(test["target_direction"], class_pred),
        }
        if xgb_pred is not None:
            record.update(
                {
                    "rmse_xgb": rmse(y_test, xgb_pred),
                    "mae_xgb": mae(y_test, xgb_pred),
                    "r2_xgb": r2(y_test, xgb_pred),
                }
            )
        records.append(record)

        predictions_accum.extend(rf_pred)
        actuals_accum.extend(y_test)
        dates_accum.extend(test["date"].values)

    summary = aggregate_metrics(records)

    if predictions_accum:
        plot_pred_vs_actual(
            dates=pd.Series(dates_accum),
            actual=pd.Series(actuals_accum),
            predicted=pd.Series(predictions_accum),
            title="Iteration 2: RF Predicted vs Actual Next-Day Returns",
            filename="iteration2_pred_vs_actual.png",
        )

    if rf_importances:
        mean_importance = pd.DataFrame(rf_importances).mean().sort_values(ascending=False)
        fig_path = REPORTS_DIR / "figures" / "iteration2_feature_importances.png"
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        ax = mean_importance.head(20).plot(kind="bar", title="Avg RF Feature Importances", figsize=(10, 4))
        ax.set_ylabel("Importance")
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig(fig_path, bbox_inches="tight")
        fig.savefig("iteration2_feature_importances.png", bbox_inches="tight")
        fig.clf()

    save_metrics_report(summary, REPORT_PATH)
    REPORT_PATH.write_text(
        "\n".join(
            [
                "# Iteration 2 Results",
                "",
                "Ensemble models (Random Forest + optional XGBoost) evaluated via walk-forward splits.",
                "",
            ]
            + [f"- **{k}**: {v:.4f}" for k, v in summary.items()]
        )
    )

    LOGGER.info("Iteration 2 summary: %s", summary)
    return summary


def main() -> Dict[str, float]:
    return run_iteration()


if __name__ == "__main__":
    main()
