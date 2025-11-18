"""Iteration 1: Linear baselines for market forecasting."""
from __future__ import annotations

import logging
from typing import Dict, List

import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import directional_accuracy, mae, r2, rmse
from src.evaluation.reporting import plot_pred_vs_actual, save_metrics_report
from src.evaluation.walkforward import aggregate_metrics, walk_forward_splits
from src.utils import PROCESSED_DIR, REPORTS_DIR

SEED = 42
REPORT_PATH = REPORTS_DIR / "iteration_1_results.md"


logging.basicConfig(level="INFO")
LOGGER = logging.getLogger(__name__)


def load_dataset() -> pd.DataFrame:
    path = PROCESSED_DIR / "model_features.parquet"
    if not path.exists():
        raise FileNotFoundError("Run feature engineering before training models")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(["date", "ticker"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def feature_columns(df: pd.DataFrame) -> List[str]:
    """Return numeric feature columns, excluding targets and categorical labels."""
    exclude = {
        "date",
        "ticker",
        "next_day_return",
        "target_direction",
        "next_day_close",
        "realised_vol_bucket",  # categorical; avoid scaling errors
    }
    return [col for col in df.columns if col not in exclude]


def run_iteration() -> Dict[str, float]:
    df = load_dataset()
    features = feature_columns(df)

    linear_model = LinearRegression()
    classifier = LogisticRegression(max_iter=1000, random_state=SEED)

    records = []
    predictions_accum = []
    actuals_accum = []
    dates_accum = []

    for split_id, (train_idx, test_idx) in enumerate(walk_forward_splits(df, n_splits=5, train_min_period=252), start=1):
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(train[features])
        X_test = scaler.transform(test[features])
        y_train = train["next_day_return"]
        y_test = test["next_day_return"]

        linear_model.fit(X_train, y_train)
        y_pred = linear_model.predict(X_test)

        classifier.fit(X_train, train["target_direction"])
        class_probs = classifier.predict_proba(X_test)[:, 1]
        class_pred = (class_probs > 0.5).astype(int)

        persistence_pred = test["return_1d"].values

        record = {
            "rmse_linear": rmse(y_test, y_pred),
            "mae_linear": mae(y_test, y_pred),
            "r2_linear": r2(y_test, y_pred),
            "directional_accuracy_linear": directional_accuracy(y_test, y_pred),
            "directional_accuracy_classifier": (class_pred == test["target_direction"].values).mean(),
            "directional_accuracy_persistence": directional_accuracy(y_test, persistence_pred),
        }
        records.append(record)

        predictions_accum.extend(y_pred)
        actuals_accum.extend(y_test.values)
        dates_accum.extend(test["date"].values)

    summary = aggregate_metrics(records)
    save_metrics_report(summary, REPORT_PATH)

    if predictions_accum:
        plot_pred_vs_actual(
            dates=pd.Series(dates_accum),
            actual=pd.Series(actuals_accum),
            predicted=pd.Series(predictions_accum),
            title="Iteration 1: Actual vs Predicted",
            filename="iteration1_pred_vs_actual.png",
        )

    LOGGER.info("Iteration 1 summary: %s", summary)
    return summary


def main() -> Dict[str, float]:
    return run_iteration()


if __name__ == "__main__":
    main()
