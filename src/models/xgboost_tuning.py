"""XGBoost hyperparameter tuning with grid search and optional Bayesian optimization.

This module implements a tuning procedure for XGBoost that addresses the underperformance
observed in the base model (RMSE 0.0131, R² < 0). Research emphasizes that optimal
hyperparameters change across rebalancing windows, requiring per-window re-tuning.

The tuning covers:
- learning_rate: 0.01 - 0.1
- max_depth: 3 - 8
- subsample: 0.5 - 1.0
- colsample_bytree: 0.5 - 1.0
- Regularization: gamma (0-5), reg_lambda (0.1-10), reg_alpha (0-1)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

from src.evaluation.metrics import rmse

try:  # pragma: no cover - optional dependency
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover
    XGBRegressor = None

# Optional: scikit-optimize for Bayesian optimization
try:  # pragma: no cover - optional dependency
    from skopt import BayesSearchCV
    from skopt.space import Integer, Real

    SKOPT_AVAILABLE = True
except ImportError:  # pragma: no cover
    SKOPT_AVAILABLE = False
    BayesSearchCV = None

LOGGER = logging.getLogger(__name__)


@dataclass
class XGBoostTuningConfig:
    """Configuration for XGBoost hyperparameter tuning."""

    # Learning rate range
    learning_rate_min: float = 0.01
    learning_rate_max: float = 0.1
    learning_rate_values: List[float] = field(default_factory=lambda: [0.01, 0.03, 0.05, 0.1])

    # Tree depth range
    max_depth_min: int = 3
    max_depth_max: int = 8
    max_depth_values: List[int] = field(default_factory=lambda: [3, 5, 6, 8])

    # Subsampling range
    subsample_min: float = 0.5
    subsample_max: float = 1.0
    subsample_values: List[float] = field(default_factory=lambda: [0.5, 0.7, 0.8, 1.0])

    # Column sampling range
    colsample_bytree_min: float = 0.5
    colsample_bytree_max: float = 1.0
    colsample_bytree_values: List[float] = field(default_factory=lambda: [0.5, 0.7, 0.8, 1.0])

    # Regularization ranges
    gamma_min: float = 0.0
    gamma_max: float = 5.0
    gamma_values: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.5, 1.0])

    reg_lambda_min: float = 0.1
    reg_lambda_max: float = 10.0
    reg_lambda_values: List[float] = field(default_factory=lambda: [0.1, 1.0, 5.0, 10.0])

    reg_alpha_min: float = 0.0
    reg_alpha_max: float = 1.0
    reg_alpha_values: List[float] = field(default_factory=lambda: [0.0, 0.01, 0.1, 0.5])

    # Number of estimators
    n_estimators: int = 300

    # Validation fraction for internal validation split
    validation_fraction: float = 0.2

    # Random state for reproducibility
    random_state: int = 42

    # Bayesian optimization iterations (if using skopt)
    n_bayes_iter: int = 30


def split_train_validation(
    X_train: np.ndarray, y_train: np.ndarray, val_fraction: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split training data into train and validation sets (chronologically).

    Uses the last `val_fraction` of data for validation to avoid lookahead bias.
    """
    val_size = max(1, int(len(X_train) * val_fraction))
    X_tr = X_train[:-val_size]
    X_val = X_train[-val_size:]
    y_tr = y_train[:-val_size]
    y_val = y_train[-val_size:]
    return X_tr, X_val, y_tr, y_val


def tune_xgboost_grid(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: Optional[XGBoostTuningConfig] = None,
) -> Dict[str, Any]:
    """Tune XGBoost hyperparameters using grid search.

    This function performs a grid search over the specified hyperparameter ranges,
    using a held-out validation set (last portion of training data) to evaluate
    performance. This is appropriate for walk-forward validation where the model
    is re-tuned at each rebalancing window.

    Parameters
    ----------
    X_train : np.ndarray
        Training features (already scaled).
    y_train : np.ndarray
        Training target values.
    config : XGBoostTuningConfig, optional
        Configuration for tuning ranges. Uses defaults if not provided.

    Returns
    -------
    Dict[str, Any]
        Best hyperparameters found.
    """
    if XGBRegressor is None:
        LOGGER.warning("XGBoost not installed; returning default params")
        return {}

    if config is None:
        config = XGBoostTuningConfig()

    # Split into train and validation
    X_tr, X_val, y_tr, y_val = split_train_validation(
        X_train, y_train, config.validation_fraction
    )

    base_params = {
        "objective": "reg:squarederror",
        "n_estimators": config.n_estimators,
        "random_state": config.random_state,
        "n_jobs": -1,
    }

    # Create parameter grid
    param_grid = ParameterGrid(
        {
            "learning_rate": config.learning_rate_values,
            "max_depth": config.max_depth_values,
            "subsample": config.subsample_values,
            "colsample_bytree": config.colsample_bytree_values,
            "gamma": config.gamma_values,
            "reg_lambda": config.reg_lambda_values,
            "reg_alpha": config.reg_alpha_values,
        }
    )

    best_params = base_params.copy()
    best_score = np.inf
    total_combos = len(param_grid)

    LOGGER.info("Starting XGBoost grid search with %d parameter combinations", total_combos)

    for i, params in enumerate(param_grid):
        try:
            model = XGBRegressor(**base_params, **params)
            model.fit(X_tr, y_tr)
            val_pred = model.predict(X_val)
            score = rmse(y_val, val_pred)

            if score < best_score:
                best_score = score
                best_params = {**base_params, **params}

            if (i + 1) % 100 == 0:
                LOGGER.debug("Grid search progress: %d/%d", i + 1, total_combos)

        except Exception as e:  # noqa: BLE001
            LOGGER.warning("Parameter combination failed: %s - %s", params, e)
            continue

    LOGGER.info(
        "Best XGBoost params (grid search) with validation RMSE %.6f: %s",
        best_score,
        {k: v for k, v in best_params.items() if k not in base_params},
    )

    return best_params


def tune_xgboost_bayesian(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: Optional[XGBoostTuningConfig] = None,
) -> Dict[str, Any]:
    """Tune XGBoost hyperparameters using Bayesian optimization (requires scikit-optimize).

    This method is more efficient than grid search for large parameter spaces,
    using Gaussian process-based optimization to find promising regions.

    Parameters
    ----------
    X_train : np.ndarray
        Training features (already scaled).
    y_train : np.ndarray
        Training target values.
    config : XGBoostTuningConfig, optional
        Configuration for tuning ranges. Uses defaults if not provided.

    Returns
    -------
    Dict[str, Any]
        Best hyperparameters found.

    Raises
    ------
    ImportError
        If scikit-optimize is not installed.
    """
    if XGBRegressor is None:
        LOGGER.warning("XGBoost not installed; returning default params")
        return {}

    if not SKOPT_AVAILABLE:
        raise ImportError(
            "scikit-optimize (skopt) is required for Bayesian optimization. "
            "Install with: pip install scikit-optimize"
        )

    if config is None:
        config = XGBoostTuningConfig()

    # Split into train and validation
    X_tr, X_val, y_tr, y_val = split_train_validation(
        X_train, y_train, config.validation_fraction
    )

    base_params = {
        "objective": "reg:squarederror",
        "n_estimators": config.n_estimators,
        "random_state": config.random_state,
        "n_jobs": -1,
    }

    # Define search space for Bayesian optimization
    search_space = {
        "learning_rate": Real(config.learning_rate_min, config.learning_rate_max, prior="log-uniform"),
        "max_depth": Integer(config.max_depth_min, config.max_depth_max),
        "subsample": Real(config.subsample_min, config.subsample_max),
        "colsample_bytree": Real(config.colsample_bytree_min, config.colsample_bytree_max),
        "gamma": Real(config.gamma_min, config.gamma_max),
        "reg_lambda": Real(config.reg_lambda_min, config.reg_lambda_max, prior="log-uniform"),
        "reg_alpha": Real(config.reg_alpha_min, config.reg_alpha_max),
    }

    LOGGER.info("Starting XGBoost Bayesian optimization with %d iterations", config.n_bayes_iter)

    model = XGBRegressor(**base_params)

    # Use BayesSearchCV with 2-fold time series split (simple for efficiency)
    opt = BayesSearchCV(
        model,
        search_space,
        n_iter=config.n_bayes_iter,
        cv=2,  # Simple 2-fold for speed; real validation uses held-out set
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        random_state=config.random_state,
    )

    opt.fit(X_tr, y_tr)

    best_params = {**base_params, **opt.best_params_}

    # Validate on held-out validation set
    final_model = XGBRegressor(**best_params)
    final_model.fit(X_tr, y_tr)
    val_pred = final_model.predict(X_val)
    val_score = rmse(y_val, val_pred)

    LOGGER.info(
        "Best XGBoost params (Bayesian) with validation RMSE %.6f: %s",
        val_score,
        {k: v for k, v in best_params.items() if k not in base_params},
    )

    return best_params


def tune_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    method: str = "grid",
    config: Optional[XGBoostTuningConfig] = None,
) -> Dict[str, Any]:
    """Tune XGBoost hyperparameters using specified method.

    Parameters
    ----------
    X_train : np.ndarray
        Training features (already scaled).
    y_train : np.ndarray
        Training target values.
    method : str, optional
        Tuning method: "grid" for grid search, "bayesian" for Bayesian optimization.
        Defaults to "grid".
    config : XGBoostTuningConfig, optional
        Configuration for tuning ranges. Uses defaults if not provided.

    Returns
    -------
    Dict[str, Any]
        Best hyperparameters found.
    """
    if method == "bayesian":
        if SKOPT_AVAILABLE:
            return tune_xgboost_bayesian(X_train, y_train, config)
        LOGGER.warning(
            "scikit-optimize not available; falling back to grid search. "
            "Install with: pip install scikit-optimize"
        )
        return tune_xgboost_grid(X_train, y_train, config)
    if method == "grid":
        return tune_xgboost_grid(X_train, y_train, config)
    raise ValueError(f"Unknown tuning method: {method}. Use 'grid' or 'bayesian'.")


def fit_tuned_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    method: str = "grid",
    config: Optional[XGBoostTuningConfig] = None,
) -> Optional[Any]:
    """Tune and fit an XGBoost model.

    This is a convenience function that combines tuning and fitting in one call.
    Use this in walk-forward validation to re-tune at each rebalancing window.

    Parameters
    ----------
    X_train : np.ndarray
        Training features (already scaled).
    y_train : np.ndarray
        Training target values.
    method : str, optional
        Tuning method: "grid" or "bayesian". Defaults to "grid".
    config : XGBoostTuningConfig, optional
        Configuration for tuning ranges. Uses defaults if not provided.

    Returns
    -------
    XGBRegressor or None
        Fitted XGBRegressor with tuned hyperparameters, or None if XGBoost is not installed.
    """
    if XGBRegressor is None:
        LOGGER.warning("XGBoost not installed; skipping")
        return None

    best_params = tune_xgboost(X_train, y_train, method=method, config=config)

    if not best_params:
        # Fall back to default params if tuning failed
        best_params = {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "reg:squarederror",
            "random_state": 42,
        }

    model = XGBRegressor(**best_params)
    model.fit(X_train, y_train)
    return model


def get_reduced_grid_config() -> XGBoostTuningConfig:
    """Get a reduced grid configuration for faster tuning.

    Use this when computational resources are limited or for quick experiments.
    """
    return XGBoostTuningConfig(
        learning_rate_values=[0.01, 0.05, 0.1],
        max_depth_values=[3, 5, 8],
        subsample_values=[0.7, 1.0],
        colsample_bytree_values=[0.7, 1.0],
        gamma_values=[0.0, 0.5],
        reg_lambda_values=[1.0, 5.0],
        reg_alpha_values=[0.0, 0.1],
    )
