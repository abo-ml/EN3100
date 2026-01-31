"""Iteration 1: Linear baselines for market forecasting."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import directional_accuracy, mae, r2, rmse
from src.evaluation.reporting import plot_pred_vs_actual, save_metrics_report
from src.evaluation.walkforward import aggregate_metrics, walk_forward_splits
from src.utils import PROCESSED_DIR, REPORTS_DIR

SEED = 42
REPORT_PATH = REPORTS_DIR / "iteration_1_results.md"
PRED_FIG = "iteration1_pred_vs_actual.png"

logging.basicConfig(level="INFO")
LOGGER = logging.getLogger(__name__)


def load_dataset(df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Return model features from disk or a provided dataframe."""

    if df is None:
        path = PROCESSED_DIR / "model_features.parquet"
        if not path.exists():
            raise FileNotFoundError("Run feature engineering before training models")
        df = pd.read_parquet(path)
    else:
        df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(["date", "ticker"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def feature_columns(df: pd.DataFrame) -> List[str]:
    exclude = {"date", "ticker", "next_day_return", "target_direction", "next_day_close"}
    return [col for col in df.columns if col not in exclude]


def run_iteration(
    data: Optional[pd.DataFrame] = None,
    report_path: Optional[Path] = None,
    generate_reports: bool = True,
    ticker: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dataset = load_dataset(data)
    features = feature_columns(dataset)

    linear_model = LinearRegression()
    classifier = LogisticRegression(max_iter=1000, random_state=SEED)

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
        prediction_records.extend(
            {
                "date": date,
                "ticker": tick,
                "actual": actual,
                "predicted": pred,
            }
            for date, tick, actual, pred in zip(test["date"].values, test["ticker"].values, y_test.values, y_pred)
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
                title="Iteration 1: Actual vs Predicted",
                filename=PRED_FIG,
            )

    suffix = f" for {ticker}" if ticker else ""
    LOGGER.info("Iteration 1 summary%s: %s", suffix, summary)
    predictions_df = pd.DataFrame(prediction_records)
    return metrics_df, predictions_df


def main() -> Tuple[pd.DataFrame, pd.DataFrame]:
    return run_iteration()


if __name__ == "__main__":
    main()
