"""Tests for deep learning hyperparameter tuning module.

This module tests the LSTM and Transformer tuning functionality including
the tunable model builders, configuration loading, and objective functions.
"""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch


class TestTuningConfiguration:
    """Test tuning configuration loading."""

    def test_load_tuning_config_returns_dict(self, tmp_path):
        """Test that load_tuning_config returns a valid dictionary."""
        from src.experiments.deep_learning_tuning import load_tuning_config

        config_file = tmp_path / "test_tuning.yaml"
        config_file.write_text("""
tuning:
  n_trials: 10
lstm:
  n_layers:
    min: 1
    max: 2
""")

        config = load_tuning_config(config_file)

        assert isinstance(config, dict)
        assert "tuning" in config
        assert config["tuning"]["n_trials"] == 10
        assert config["lstm"]["n_layers"]["min"] == 1

    def test_load_tuning_config_missing_file(self, tmp_path):
        """Test that missing config file returns empty dict."""
        from src.experiments.deep_learning_tuning import load_tuning_config

        config = load_tuning_config(tmp_path / "nonexistent.yaml")

        assert config == {}


class TestSequenceCreation:
    """Test sequence creation for time-series models."""

    def test_create_sequences_shape(self):
        """Test that create_sequences returns correct shapes."""
        from src.experiments.deep_learning_tuning import create_sequences

        df = pd.DataFrame({
            "ticker": ["AAPL"] * 100,
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "target": np.random.randn(100),
        })

        X, y = create_sequences(df, ["feature1", "feature2"], "target", window=30)

        assert X.shape == (70, 30, 2)
        assert y.shape == (70,)

    def test_create_sequences_empty_data(self):
        """Test create_sequences with insufficient data."""
        from src.experiments.deep_learning_tuning import create_sequences

        df = pd.DataFrame({
            "ticker": ["AAPL"] * 10,
            "feature1": np.random.randn(10),
            "target": np.random.randn(10),
        })

        X, y = create_sequences(df, ["feature1"], "target", window=30)

        assert X.shape[0] == 0
        assert y.shape[0] == 0

    def test_create_sequences_multiple_tickers(self):
        """Test create_sequences with multiple tickers."""
        from src.experiments.deep_learning_tuning import create_sequences

        df = pd.DataFrame({
            "ticker": ["AAPL"] * 50 + ["MSFT"] * 50,
            "feature1": np.random.randn(100),
            "target": np.random.randn(100),
        })

        X, y = create_sequences(df, ["feature1"], "target", window=20)

        assert X.shape[0] == 60  # 30 from each ticker


class TestTunableLSTM:
    """Test tunable LSTM model builder."""

    def test_build_tunable_lstm_default_params(self):
        """Test build_tunable_lstm with default parameters."""
        pytest.importorskip("tensorflow")

        from src.experiments.deep_learning_tuning import build_tunable_lstm

        model = build_tunable_lstm((60, 10))

        assert model is not None
        assert model.input_shape == (None, 60, 10)
        assert model.output_shape == (None, 1)

    def test_build_tunable_lstm_custom_layers(self):
        """Test build_tunable_lstm with custom layer configuration."""
        pytest.importorskip("tensorflow")

        from src.experiments.deep_learning_tuning import build_tunable_lstm

        model = build_tunable_lstm(
            (60, 10),
            n_layers=3,
            units_per_layer=[128, 96, 64],
            dropout_rate=0.4,
            learning_rate=0.0005,
        )

        assert model is not None

    def test_build_tunable_lstm_with_l2_reg(self):
        """Test build_tunable_lstm with L2 regularization."""
        pytest.importorskip("tensorflow")

        from src.experiments.deep_learning_tuning import build_tunable_lstm

        model = build_tunable_lstm(
            (60, 10),
            n_layers=2,
            units_per_layer=[96, 64],
            l2_reg=0.001,
        )

        assert model is not None

    def test_build_tunable_lstm_tanh_activation(self):
        """Test build_tunable_lstm with tanh activation."""
        pytest.importorskip("tensorflow")

        from src.experiments.deep_learning_tuning import build_tunable_lstm

        model = build_tunable_lstm(
            (30, 5),
            n_layers=1,
            units_per_layer=[64],
            activation="tanh",
        )

        assert model is not None


class TestTunableTransformer:
    """Test tunable Transformer model builder."""

    def test_build_tunable_transformer_default_params(self):
        """Test build_tunable_transformer with default parameters."""
        pytest.importorskip("tensorflow")

        from src.experiments.deep_learning_tuning import build_tunable_transformer

        model = build_tunable_transformer((60, 10))

        assert model is not None
        assert model.input_shape == (None, 60, 10)
        assert model.output_shape == (None, 1)

    def test_build_tunable_transformer_custom_params(self):
        """Test build_tunable_transformer with custom parameters."""
        pytest.importorskip("tensorflow")

        from src.experiments.deep_learning_tuning import build_tunable_transformer

        model = build_tunable_transformer(
            (90, 15),
            num_layers=3,
            num_heads=8,
            d_model=128,
            ff_dim=256,
            dropout_rate=0.3,
            learning_rate=0.0005,
        )

        assert model is not None

    def test_build_tunable_transformer_small_config(self):
        """Test build_tunable_transformer with minimal configuration."""
        pytest.importorskip("tensorflow")

        from src.experiments.deep_learning_tuning import build_tunable_transformer

        model = build_tunable_transformer(
            (30, 5),
            num_layers=1,
            num_heads=2,
            d_model=32,
            ff_dim=64,
        )

        assert model is not None


class TestModelEvaluation:
    """Test model evaluation utilities."""

    def test_evaluate_model_returns_metrics(self):
        """Test that evaluate_model returns expected metrics."""
        pytest.importorskip("tensorflow")

        from src.experiments.deep_learning_tuning import (
            build_tunable_lstm,
            evaluate_model,
        )

        model = build_tunable_lstm((10, 3), n_layers=1, units_per_layer=[16])

        X_test = np.random.randn(20, 10, 3).astype(np.float32)
        y_test = np.random.randn(20).astype(np.float32)

        metrics = evaluate_model(model, X_test, y_test)

        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert "directional_accuracy" in metrics
        assert isinstance(metrics["rmse"], float)


class TestOptunaIntegration:
    """Test Optuna integration (requires optuna package)."""

    def test_optuna_available(self):
        """Test that optuna can be imported."""
        optuna = pytest.importorskip("optuna")
        assert optuna is not None

    def test_tuning_requires_optuna(self):
        """Test that tuning functions check for optuna availability."""
        pytest.importorskip("tensorflow")
        pytest.importorskip("optuna")
        
        from src.experiments.deep_learning_tuning import run_lstm_tuning, run_transformer_tuning
        
        # Functions should exist and be callable (optuna is available)
        assert callable(run_lstm_tuning)
        assert callable(run_transformer_tuning)


class TestHyperparameterRanges:
    """Test that hyperparameter ranges match literature recommendations."""

    def test_lstm_config_ranges(self, tmp_path):
        """Test that LSTM config has correct literature-based ranges."""
        from src.experiments.deep_learning_tuning import load_tuning_config
        from pathlib import Path

        config = load_tuning_config(Path("configs/tuning.yaml"))

        if config and "lstm" in config:
            lstm = config["lstm"]
            
            # Check layer count range
            assert lstm.get("n_layers", {}).get("min", 1) >= 1
            assert lstm.get("n_layers", {}).get("max", 3) <= 3
            
            # Check units range (32-150 as per literature)
            assert lstm.get("units_per_layer", {}).get("min", 32) >= 32
            assert lstm.get("units_per_layer", {}).get("max", 150) <= 150
            
            # Check dropout range (0.0-0.5)
            assert lstm.get("dropout_rate", {}).get("min", 0.0) >= 0.0
            assert lstm.get("dropout_rate", {}).get("max", 0.5) <= 0.5
            
            # Check sequence length range (30-90 days)
            assert lstm.get("sequence_length", {}).get("min", 30) >= 30
            assert lstm.get("sequence_length", {}).get("max", 90) <= 90

    def test_transformer_config_ranges(self, tmp_path):
        """Test that Transformer config has correct ranges."""
        from src.experiments.deep_learning_tuning import load_tuning_config
        from pathlib import Path

        config = load_tuning_config(Path("configs/tuning.yaml"))

        if config and "transformer" in config:
            transformer = config["transformer"]
            
            # Check layer count range
            assert transformer.get("num_layers", {}).get("min", 1) >= 1
            assert transformer.get("num_layers", {}).get("max", 3) <= 3
            
            # Check dropout range
            assert transformer.get("dropout_rate", {}).get("min", 0.0) >= 0.0
            assert transformer.get("dropout_rate", {}).get("max", 0.5) <= 0.5
