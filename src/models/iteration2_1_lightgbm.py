"""Iteration 2.1: LightGBM regression upgrade."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import directional_accuracy, mae, r2, rmse
from src.evaluation.reporting import plot_pred_vs_actual
from src.evaluation.walkforward import aggregate_metrics, walk_forward_splits
from src.models.iteration1_baseline import feature_columns, load_dataset
from src.utils import FIGURES_DIR, REPORTS_DIR

logging.basicConfig(level="INFO")
LOGGER = logging.getLogger(__name__)

SEED = 42
REPORT_PATH = REPORTS_DIR / "iteration_2_1_lightgbm_results.md"
PRED_FIG = "iteration2_1_pred_vs_actual.png"
IMPORTANCE_FIG = "iteration2_1_feature_importances.png"


def split_train_validation(train_df: pd.DataFrame, val_fraction: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    val_size = max(1, int(len(train_df) * val_fraction))
    return train_df.iloc[:-val_size], train_df.iloc[-val_size:]


def tune_lightgbm(train_df: pd.DataFrame, features: List[str]) -> Dict:
    base_params = dict(
        objective="regression",
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
    )
    train_part, val_part = split_train_validation(train_df)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_part[features])
    y_train = train_part["next_day_return"]
    X_val = scaler.transform(val_part[features])
    y_val = val_part["next_day_return"]

    grid = ParameterGrid(
        {
            "num_leaves": [15, 31, 63],
            "max_depth": [-1, 6, 10],
            "learning_rate": [0.01, 0.05, 0.1],
            "n_estimators": [200, 500, 800],
        }
    )

    best_params = base_params.copy()
    best_score = np.inf
    for params in grid:
        model = lgb.LGBMRegressor(**base_params, **params)
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        score = rmse(y_val, val_pred)
        if score < best_score:
            best_score = score
            best_params = {**base_params, **params}

    LOGGER.info("Best LightGBM params %s with validation RMSE %.4f", best_params, best_score)
    return best_params


def load_iteration2_metrics() -> Dict[str, float]:
    path = REPORTS_DIR / "iteration_2_results.md"
    if not path.exists():
        return {}
    metrics = {}
    for line in path.read_text().splitlines():
        if line.startswith("- **") and "**:" in line:
            try:
                key = line.split("**")[1]
                value = float(line.split(":")[-1].strip())
                metrics[key] = value
            except Exception:  # noqa: BLE001
                continue
    return metrics


def comparison_table(lightgbm_summary: Dict[str, float], baseline_metrics: Dict[str, float]) -> str:
    headers = "| Metric | Iteration 2 (RF) | Iteration 2.1 (LightGBM) |"
    divider = "|---|---|---|"
    lines = [headers, divider]
    for metric, label in [
        ("rmse", "RMSE"),
        ("mae", "MAE"),
        ("r2", "R2"),
        ("directional_accuracy", "Directional Accuracy"),
    ]:
        lgb_val = lightgbm_summary.get(f"{metric}_lgbm_mean")
        base_val = baseline_metrics.get(f"{metric}_rf_mean")

        def fmt(val):
            return f"{val:.4f}" if isinstance(val, (int, float, np.floating)) and not np.isnan(val) else "N/A"

        lines.append(f"| {label} | {fmt(base_val)} | {fmt(lgb_val)} |")
    return "\n".join(lines)


def run_iteration(
    data: Optional[pd.DataFrame] = None,
    report_path: Optional[Path] = None,
    generate_reports: bool = True,
    ticker: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dataset = load_dataset(data)
    features = feature_columns(dataset)

    records: List[Dict[str, float]] = []
    importances: List[np.ndarray] = []
    prediction_records: List[Dict[str, object]] = []

    for split_id, (train_idx, test_idx) in enumerate(
        walk_forward_splits(dataset, n_splits=5, train_min_period=252), start=1
    ):
        train = dataset.iloc[train_idx].copy()
        test = dataset.iloc[test_idx].copy()
        if len(test) == 0:
            continue

        tuned_params = tune_lightgbm(train, features)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(train[features])
        X_test = scaler.transform(test[features])
        y_train = train["next_day_return"]
        y_test = test["next_day_return"]

        model = lgb.LGBMRegressor(**tuned_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        importances.append(model.feature_importances_)
        records.append(
            {
                "rmse_lgbm": rmse(y_test, y_pred),
                "mae_lgbm": mae(y_test, y_pred),
                "r2_lgbm": r2(y_test, y_pred),
                "directional_accuracy_lgbm": directional_accuracy(y_test, y_pred),
            }
        )

        prediction_records.extend(
            {
                "date": date,
                "ticker": tick,
                "actual": actual,
                "predicted": pred,
            }
            for date, tick, actual, pred in zip(test["date"].values, test["ticker"].values, y_test.values, y_pred)
        )

        LOGGER.info(
            "Split %s | LightGBM RMSE %.4f, MAE %.4f, R2 %.4f",
            split_id,
            records[-1]["rmse_lgbm"],
            records[-1]["mae_lgbm"],
            records[-1]["r2_lgbm"],
        )

    metrics_df = pd.DataFrame(records)
    summary = aggregate_metrics(records)

    if generate_reports:
        baseline_metrics = load_iteration2_metrics()
        comparison_md = comparison_table(summary, baseline_metrics)

        lines = [
            "# Iteration 2.1: LightGBM Results",
            "",
            "## Metrics",
        ]
        for key, val in summary.items():
            lines.append(f"- **{key}**: {val:.4f}")
        lines.extend(["", "## Comparison vs Iteration 2", "", comparison_md])
        output_path = report_path or REPORT_PATH
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines))

        if prediction_records:
            pred_df = pd.DataFrame(prediction_records)
            plot_pred_vs_actual(
                dates=pd.Series(pred_df["date"]),
                actual=pd.Series(pred_df["actual"]),
                predicted=pd.Series(pred_df["predicted"]),
                title="Iteration 2.1: LightGBM Actual vs Predicted",
                filename=PRED_FIG,
            )

        if importances:
            mean_importance = np.mean(importances, axis=0)
            order = np.argsort(mean_importance)[::-1][:20]
            FIGURES_DIR.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(np.array(features)[order][::-1], mean_importance[order][::-1])
            ax.set_title("LightGBM Feature Importances (Top 20)")
            fig.tight_layout()
            fig.savefig(FIGURES_DIR / IMPORTANCE_FIG, bbox_inches="tight")
            plt.close(fig)

    suffix = f" for {ticker}" if ticker else ""
    LOGGER.info("Iteration 2.1 summary%s: %s", suffix, summary)
    predictions_df = pd.DataFrame(prediction_records)
    return metrics_df, predictions_df


def main() -> Tuple[pd.DataFrame, pd.DataFrame]:
    return run_iteration()


if __name__ == "__main__":
    main()
