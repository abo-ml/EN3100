"""Tests for XGBoost hyperparameter tuning module."""
import numpy as np
import pytest


def test_xgboost_tuning_config_defaults():
    """Test that XGBoostTuningConfig has proper default values."""
    from src.models.xgboost_tuning import XGBoostTuningConfig

    config = XGBoostTuningConfig()

    # Check learning rate range
    assert config.learning_rate_min == 0.01
    assert config.learning_rate_max == 0.1
    assert 0.01 in config.learning_rate_values
    assert 0.1 in config.learning_rate_values

    # Check max_depth range
    assert config.max_depth_min == 3
    assert config.max_depth_max == 8
    assert 3 in config.max_depth_values
    assert 8 in config.max_depth_values

    # Check subsample range
    assert config.subsample_min == 0.5
    assert config.subsample_max == 1.0

    # Check colsample_bytree range
    assert config.colsample_bytree_min == 0.5
    assert config.colsample_bytree_max == 1.0

    # Check regularization ranges
    assert config.gamma_min == 0.0
    assert config.gamma_max == 5.0
    assert config.reg_lambda_min == 0.1
    assert config.reg_lambda_max == 10.0
    assert config.reg_alpha_min == 0.0
    assert config.reg_alpha_max == 1.0


def test_split_train_validation():
    """Test the train/validation split function."""
    from src.models.xgboost_tuning import split_train_validation

    X = np.random.randn(100, 5)
    y = np.random.randn(100)

    X_tr, X_val, y_tr, y_val = split_train_validation(X, y, val_fraction=0.2)

    assert len(X_tr) == 80
    assert len(X_val) == 20
    assert len(y_tr) == 80
    assert len(y_val) == 20

    # Verify the split is chronological (last 20% is validation)
    np.testing.assert_array_equal(X_tr, X[:80])
    np.testing.assert_array_equal(X_val, X[80:])


def test_get_reduced_grid_config():
    """Test the reduced grid configuration for faster tuning."""
    from src.models.xgboost_tuning import get_reduced_grid_config

    config = get_reduced_grid_config()

    # Reduced grid should have fewer options
    assert len(config.learning_rate_values) == 3
    assert len(config.max_depth_values) == 3
    assert len(config.subsample_values) == 2
    assert len(config.colsample_bytree_values) == 2
    assert len(config.gamma_values) == 2
    assert len(config.reg_lambda_values) == 2
    assert len(config.reg_alpha_values) == 2


@pytest.mark.skipif(
    False,  # Enable test when XGBoost is installed
    reason="XGBoost may not be installed; run locally with XGBoost"
)
def test_tune_xgboost_grid_returns_params():
    """Test that grid search tuning returns valid parameters."""
    from src.models.xgboost_tuning import tune_xgboost_grid, get_reduced_grid_config

    np.random.seed(42)
    X = np.random.randn(200, 5)
    y = X[:, 0] * 0.5 + np.random.randn(200) * 0.1  # Simple linear relationship

    config = get_reduced_grid_config()
    params = tune_xgboost_grid(X, y, config=config)

    # Check that tuning returns the required parameters
    assert "learning_rate" in params
    assert "max_depth" in params
    assert "subsample" in params
    assert "colsample_bytree" in params
    assert "gamma" in params
    assert "reg_lambda" in params
    assert "reg_alpha" in params

    # Check that parameters are within expected ranges
    assert 0.01 <= params["learning_rate"] <= 0.1
    assert 3 <= params["max_depth"] <= 8
    assert 0.5 <= params["subsample"] <= 1.0
    assert 0.5 <= params["colsample_bytree"] <= 1.0


def test_tune_xgboost_invalid_method():
    """Test that invalid tuning method raises ValueError."""
    from src.models.xgboost_tuning import tune_xgboost

    X = np.random.randn(50, 5)
    y = np.random.randn(50)

    with pytest.raises(ValueError, match="Unknown tuning method"):
        tune_xgboost(X, y, method="invalid_method")


def test_tuning_config_custom_values():
    """Test that XGBoostTuningConfig accepts custom values."""
    from src.models.xgboost_tuning import XGBoostTuningConfig

    config = XGBoostTuningConfig(
        learning_rate_values=[0.02, 0.08],
        max_depth_values=[4, 6],
        n_estimators=100,
        validation_fraction=0.3,
    )

    assert config.learning_rate_values == [0.02, 0.08]
    assert config.max_depth_values == [4, 6]
    assert config.n_estimators == 100
    assert config.validation_fraction == 0.3
