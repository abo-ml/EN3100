"""LightGBM hyperparameter tuning module with time-series cross-validation.

This module provides a comprehensive hyperparameter tuning plan for LightGBM
following best practices from stock forecasting literature. Key features:

- Time-series cross-validation using walk-forward splits
- Parameter space following literature recommendations:
  - Small learning rates paired with larger num_leaves and max_depth
  - Constraint: num_leaves < 2^max_depth
- Extensive search over:
  - learning_rate: 0.01-0.1
  - num_leaves: 31-255
  - max_depth: 3-10
  - min_data_in_leaf: 20-100
  - feature_fraction: 0.6-1.0

References:
    - Literature suggests small learning rates with larger tree capacity
      work best for stock forecasting tasks.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import directional_accuracy, mae, r2, rmse
from src.evaluation.walkforward import aggregate_metrics, walk_forward_splits
from src.models.iteration1_baseline import feature_columns, load_dataset
from src.utils import REPORTS_DIR

logging.basicConfig(level="INFO")
LOGGER = logging.getLogger(__name__)

SEED = 42
REPORT_PATH = REPORTS_DIR / "lightgbm_tuning_results.md"


@dataclass
class TuningConfig:
    """Configuration for LightGBM hyperparameter tuning.

    Attributes
    ----------
    learning_rate_range : tuple
        Min and max learning rate values.
    num_leaves_range : tuple
        Min and max num_leaves values.
    max_depth_range : tuple
        Min and max max_depth values.
    min_data_in_leaf_range : tuple
        Min and max min_data_in_leaf values.
    feature_fraction_range : tuple
        Min and max feature_fraction values.
    n_cv_splits : int
        Number of time-series cross-validation splits.
    train_min_period : int
        Minimum training period for walk-forward validation.
    n_random_samples : int
        Number of random parameter combinations to sample.
        If None, uses full grid search (can be slow).
    """

    learning_rate_range: Tuple[float, float] = (0.01, 0.1)
    num_leaves_range: Tuple[int, int] = (31, 255)
    max_depth_range: Tuple[int, int] = (3, 10)
    min_data_in_leaf_range: Tuple[int, int] = (20, 100)
    feature_fraction_range: Tuple[float, float] = (0.6, 1.0)
    n_cv_splits: int = 5
    train_min_period: int = 252
    n_random_samples: Optional[int] = 50


def validate_num_leaves_constraint(num_leaves: int, max_depth: int) -> bool:
    """Validate that num_leaves < 2^max_depth per literature recommendations.

    Parameters
    ----------
    num_leaves : int
        Number of leaves in the tree.
    max_depth : int
        Maximum depth of the tree.

    Returns
    -------
    bool
        True if constraint is satisfied, False otherwise.
    """
    max_allowed = 2**max_depth
    return num_leaves < max_allowed


def generate_parameter_grid(config: TuningConfig) -> List[Dict[str, Any]]:
    """Generate parameter combinations respecting the num_leaves constraint.

    Parameters
    ----------
    config : TuningConfig
        Tuning configuration with parameter ranges.

    Returns
    -------
    List[Dict[str, Any]]
        List of valid parameter dictionaries.
    """
    # Define discrete values within ranges
    learning_rates = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1]
    num_leaves_values = [31, 50, 63, 100, 127, 150, 200, 255]
    max_depth_values = list(range(config.max_depth_range[0], config.max_depth_range[1] + 1))
    min_data_in_leaf_values = [20, 30, 50, 70, 100]
    feature_fraction_values = [0.6, 0.7, 0.8, 0.9, 1.0]

    # Filter values to be within configured ranges
    learning_rates = [lr for lr in learning_rates
                      if config.learning_rate_range[0] <= lr <= config.learning_rate_range[1]]
    num_leaves_values = [nl for nl in num_leaves_values
                         if config.num_leaves_range[0] <= nl <= config.num_leaves_range[1]]

    grid = ParameterGrid({
        "learning_rate": learning_rates,
        "num_leaves": num_leaves_values,
        "max_depth": max_depth_values,
        "min_data_in_leaf": min_data_in_leaf_values,
        "feature_fraction": feature_fraction_values,
    })

    # Filter combinations that satisfy the num_leaves < 2^max_depth constraint
    valid_params = []
    for params in grid:
        if validate_num_leaves_constraint(params["num_leaves"], params["max_depth"]):
            valid_params.append(params)

    LOGGER.info(
        "Generated %d valid parameter combinations (from %d total) after applying "
        "num_leaves < 2^max_depth constraint",
        len(valid_params),
        len(list(grid)),
    )

    return valid_params


def sample_parameters(
    params_list: List[Dict[str, Any]],
    n_samples: Optional[int],
    random_state: int = SEED,
) -> List[Dict[str, Any]]:
    """Randomly sample parameter combinations if n_samples is specified.

    Parameters
    ----------
    params_list : List[Dict[str, Any]]
        Full list of parameter combinations.
    n_samples : Optional[int]
        Number of samples to take. If None, returns all combinations.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    List[Dict[str, Any]]
        Sampled or full list of parameter combinations.
    """
    if n_samples is None or n_samples >= len(params_list):
        return params_list

    rng = np.random.default_rng(random_state)
    indices = rng.choice(len(params_list), size=n_samples, replace=False)
    return [params_list[i] for i in indices]


def cross_validate_params(
    df: pd.DataFrame,
    features: List[str],
    params: Dict[str, Any],
    n_splits: int = 5,
    train_min_period: int = 252,
) -> Dict[str, float]:
    """Evaluate a parameter set using time-series cross-validation.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with features and target.
    features : List[str]
        List of feature column names.
    params : Dict[str, Any]
        LightGBM parameters to evaluate.
    n_splits : int
        Number of walk-forward splits.
    train_min_period : int
        Minimum training period.

    Returns
    -------
    Dict[str, float]
        Cross-validation metrics (mean and std).
    """
    base_params = {
        "objective": "regression",
        "subsample": 0.8,
        "random_state": SEED,
        "verbosity": -1,
    }
    full_params = {**base_params, **params}

    # LightGBM natively supports feature_fraction, no mapping needed

    fold_metrics: List[Dict[str, float]] = []

    for train_idx, test_idx in walk_forward_splits(df, n_splits=n_splits, train_min_period=train_min_period):
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]

        if len(test) == 0:
            continue

        scaler = StandardScaler()
        X_train = scaler.fit_transform(train[features])
        X_test = scaler.transform(test[features])
        y_train = train["next_day_return"].values
        y_test = test["next_day_return"].values

        model = lgb.LGBMRegressor(**full_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        fold_metrics.append({
            "rmse": rmse(y_test, y_pred),
            "mae": mae(y_test, y_pred),
            "r2": r2(y_test, y_pred),
            "directional_accuracy": directional_accuracy(y_test, y_pred),
        })

    if not fold_metrics:
        return {
            "rmse_mean": float("inf"),
            "rmse_std": float("nan"),
            "mae_mean": float("inf"),
            "mae_std": float("nan"),
            "r2_mean": float("nan"),
            "r2_std": float("nan"),
            "directional_accuracy_mean": float("nan"),
            "directional_accuracy_std": float("nan"),
        }

    return aggregate_metrics(fold_metrics)


def run_tuning(
    data: Optional[pd.DataFrame] = None,
    config: Optional[TuningConfig] = None,
    report_path: Optional[Path] = None,
    generate_reports: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run LightGBM hyperparameter tuning with time-series cross-validation.

    Parameters
    ----------
    data : Optional[pd.DataFrame]
        Input dataset. If None, loads default dataset.
    config : Optional[TuningConfig]
        Tuning configuration. If None, uses defaults.
    report_path : Optional[Path]
        Path for saving the report.
    generate_reports : bool
        Whether to generate markdown reports.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        DataFrame with all results and the best parameters dict.
    """
    if config is None:
        config = TuningConfig()

    dataset = load_dataset(data)
    features = feature_columns(dataset)

    LOGGER.info("Starting LightGBM hyperparameter tuning")
    LOGGER.info("Dataset shape: %s", dataset.shape)
    LOGGER.info("Number of features: %d", len(features))
    LOGGER.info("Configuration: %s", config)

    # Generate and sample parameter combinations
    all_params = generate_parameter_grid(config)
    params_to_evaluate = sample_parameters(all_params, config.n_random_samples)

    LOGGER.info("Evaluating %d parameter combinations", len(params_to_evaluate))

    results: List[Dict[str, Any]] = []

    for idx, params in enumerate(params_to_evaluate, start=1):
        LOGGER.info("Evaluating combination %d/%d: %s", idx, len(params_to_evaluate), params)

        cv_metrics = cross_validate_params(
            dataset,
            features,
            params,
            n_splits=config.n_cv_splits,
            train_min_period=config.train_min_period,
        )

        result = {**params, **cv_metrics}
        results.append(result)

        LOGGER.info(
            "  CV RMSE: %.4f (±%.4f), MAE: %.4f (±%.4f)",
            cv_metrics.get("rmse_mean", float("nan")),
            cv_metrics.get("rmse_std", float("nan")),
            cv_metrics.get("mae_mean", float("nan")),
            cv_metrics.get("mae_std", float("nan")),
        )

    results_df = pd.DataFrame(results)

    # Find best parameters based on CV RMSE
    # Filter for finite RMSE values first
    valid_results = results_df[np.isfinite(results_df["rmse_mean"])]
    if valid_results.empty:
        LOGGER.error("No valid results obtained. All parameter combinations failed cross-validation.")
        raise ValueError("Hyperparameter tuning failed: no valid results obtained from cross-validation.")

    best_idx = valid_results["rmse_mean"].idxmin()
    best_result = results_df.loc[best_idx]
    best_params = {
        k: best_result[k]
        for k in ["learning_rate", "num_leaves", "max_depth", "min_data_in_leaf", "feature_fraction"]
    }

    LOGGER.info("=" * 60)
    LOGGER.info("TUNING COMPLETE")
    LOGGER.info("Best parameters: %s", best_params)
    LOGGER.info(
        "Best CV RMSE: %.4f (±%.4f)",
        best_result["rmse_mean"],
        best_result["rmse_std"],
    )
    LOGGER.info(
        "Best CV MAE: %.4f (±%.4f)",
        best_result["mae_mean"],
        best_result["mae_std"],
    )
    LOGGER.info("=" * 60)

    if generate_reports:
        _generate_report(results_df, best_params, best_result, config, report_path)

    return results_df, best_params


def _generate_report(
    results_df: pd.DataFrame,
    best_params: Dict[str, Any],
    best_result: pd.Series,
    config: TuningConfig,
    report_path: Optional[Path] = None,
) -> None:
    """Generate a markdown report of tuning results."""
    output_path = report_path or REPORT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Sort results by RMSE
    sorted_results = results_df.sort_values("rmse_mean").head(10)

    lines = [
        "# LightGBM Hyperparameter Tuning Results",
        "",
        "## Overview",
        "",
        "This report summarizes the results of hyperparameter tuning for LightGBM",
        "using time-series cross-validation with walk-forward splits.",
        "",
        "### Search Space",
        "",
        f"- **Learning Rate**: {config.learning_rate_range[0]} - {config.learning_rate_range[1]}",
        f"- **Num Leaves**: {config.num_leaves_range[0]} - {config.num_leaves_range[1]}",
        f"- **Max Depth**: {config.max_depth_range[0]} - {config.max_depth_range[1]}",
        f"- **Min Data in Leaf**: {config.min_data_in_leaf_range[0]} - {config.min_data_in_leaf_range[1]}",
        f"- **Feature Fraction**: {config.feature_fraction_range[0]} - {config.feature_fraction_range[1]}",
        "",
        "### Constraint",
        "",
        "Following literature recommendations: `num_leaves < 2^max_depth`",
        "",
        f"### Cross-Validation Configuration",
        "",
        f"- **Number of CV Splits**: {config.n_cv_splits}",
        f"- **Minimum Training Period**: {config.train_min_period} days",
        f"- **Parameter Combinations Evaluated**: {len(results_df)}",
        "",
        "## Best Parameters",
        "",
    ]

    for key, val in best_params.items():
        lines.append(f"- **{key}**: {val}")

    lines.extend([
        "",
        "## Best Metrics (Cross-Validated)",
        "",
        f"- **RMSE**: {best_result['rmse_mean']:.4f} (±{best_result['rmse_std']:.4f})",
        f"- **MAE**: {best_result['mae_mean']:.4f} (±{best_result['mae_std']:.4f})",
        f"- **R²**: {best_result['r2_mean']:.4f} (±{best_result['r2_std']:.4f})",
        f"- **Directional Accuracy**: {best_result['directional_accuracy_mean']:.4f} "
        f"(±{best_result['directional_accuracy_std']:.4f})",
        "",
        "## Top 10 Parameter Combinations",
        "",
        "| Rank | LR | Leaves | Depth | Min Leaf | Feat Frac | RMSE | MAE |",
        "|------|-----|--------|-------|----------|-----------|------|-----|",
    ])

    for rank, (_, row) in enumerate(sorted_results.iterrows(), start=1):
        lines.append(
            f"| {rank} | {row['learning_rate']:.3f} | {int(row['num_leaves'])} | "
            f"{int(row['max_depth'])} | {int(row['min_data_in_leaf'])} | "
            f"{row['feature_fraction']:.1f} | {row['rmse_mean']:.4f} | {row['mae_mean']:.4f} |"
        )

    lines.extend([
        "",
        "## Literature Notes",
        "",
        "According to stock forecasting literature:",
        "",
        "1. Small learning rates (0.01-0.05) paired with larger num_leaves and max_depth",
        "   often provide better generalization for financial time series.",
        "2. The constraint `num_leaves < 2^max_depth` prevents overfitting by ensuring",
        "   the tree structure remains balanced.",
        "3. Higher min_data_in_leaf values help prevent overfitting to noise in",
        "   financial data.",
        "4. Feature fraction (colsample_bytree) acts as regularization by using",
        "   only a subset of features for each tree.",
        "",
    ])

    output_path.write_text("\n".join(lines))
    LOGGER.info("Report saved to %s", output_path)


def main() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Entry point for running LightGBM tuning from command line."""
    return run_tuning()


if __name__ == "__main__":
    main()
