"""Deep Learning Hyperparameter Tuning for LSTM and Transformer Models.

This module implements Bayesian optimization using Optuna to search over
hyperparameter spaces for LSTM and Transformer architectures. The tuning
follows literature-based recommendations for stock forecasting models.

Reference hyperparameter ranges:
- Number of layers: 1-3
- Units per layer: 32-150 (optimal: 96 or 128 neurons)
- Dropout rates: 0.0-0.5 (optimal: 0.3-0.4)
- Sequence lengths: 30-90 days
- Learning rates: 0.0005-0.01
- Batch sizes: 32-128
- Regularization: dropout, early stopping, L2 regularization
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from sklearn.preprocessing import StandardScaler

try:
    import optuna
    from optuna.trial import Trial
except ImportError:
    optuna = None
    Trial = None

from src.evaluation.metrics import directional_accuracy, mae, r2, rmse
from src.evaluation.walkforward import aggregate_metrics, walk_forward_splits
from src.models.iteration1_baseline import feature_columns, load_dataset
from src.utils import REPORTS_DIR

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level="INFO")

CONFIG_PATH = Path("configs/tuning.yaml")
TUNING_REPORTS_DIR = REPORTS_DIR / "tuning"


def load_tuning_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load tuning configuration from YAML file."""
    path = config_path or CONFIG_PATH
    if not path.exists():
        LOGGER.warning("Config file %s not found, using defaults", path)
        return {}
    with open(path) as f:
        return yaml.safe_load(f)


def create_sequences(
    df: pd.DataFrame, features: List[str], target_col: str, window: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences for time-series models."""
    sequences = []
    targets = []
    for _, group in df.groupby("ticker"):
        feature_array = group[features].values
        target_array = group[target_col].values
        if len(group) <= window:
            continue
        for i in range(window, len(group)):
            sequences.append(feature_array[i - window : i])
            targets.append(target_array[i])
    if not sequences:
        return np.empty((0, window, len(features))), np.empty((0,))
    return np.array(sequences), np.array(targets)


def build_tunable_lstm(
    input_shape: Tuple[int, int],
    n_layers: int = 2,
    units_per_layer: Optional[List[int]] = None,
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001,
    l2_reg: float = 0.0,
    activation: str = "tanh",
) -> tf.keras.Model:
    """Build an LSTM model with tunable hyperparameters.
    
    Parameters
    ----------
    input_shape : Tuple[int, int]
        Shape of input (seq_len, num_features).
    n_layers : int
        Number of LSTM layers (1-3 as per literature).
    units_per_layer : List[int], optional
        Number of units for each LSTM layer.
    dropout_rate : float
        Dropout rate for regularization (0.0-0.5).
    learning_rate : float
        Adam optimizer learning rate (0.0005-0.01).
    l2_reg : float
        L2 regularization strength.
    activation : str
        Recurrent activation function (tanh recommended for LSTMs).
    
    Returns
    -------
    tf.keras.Model
        Compiled LSTM model.
    """
    if units_per_layer is None:
        units_per_layer = [96] * n_layers
    
    regularizer = tf.keras.regularizers.l2(l2_reg) if l2_reg > 0 else None
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape))
    
    for i, units in enumerate(units_per_layer):
        return_sequences = i < n_layers - 1
        model.add(
            tf.keras.layers.LSTM(
                units,
                return_sequences=return_sequences,
                recurrent_activation=activation,
                kernel_regularizer=regularizer,
                recurrent_regularizer=regularizer,
            )
        )
        if dropout_rate > 0:
            model.add(tf.keras.layers.Dropout(dropout_rate))
    
    model.add(tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=regularizer))
    model.add(tf.keras.layers.Dense(1))
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mse")
    return model


def build_tunable_transformer(
    input_shape: Tuple[int, int],
    num_layers: int = 2,
    num_heads: int = 4,
    d_model: int = 64,
    ff_dim: int = 128,
    dropout_rate: float = 0.1,
    learning_rate: float = 0.001,
) -> tf.keras.Model:
    """Build a Transformer encoder model with tunable hyperparameters.
    
    Parameters
    ----------
    input_shape : Tuple[int, int]
        Shape of input (seq_len, num_features).
    num_layers : int
        Number of transformer encoder layers (1-3).
    num_heads : int
        Number of attention heads.
    d_model : int
        Dimension of the model.
    ff_dim : int
        Dimension of the feed-forward network.
    dropout_rate : float
        Dropout rate for regularization (0.0-0.5).
    learning_rate : float
        Adam optimizer learning rate (0.0005-0.01).
    
    Returns
    -------
    tf.keras.Model
        Compiled Transformer model.
    """
    seq_len, num_features = input_shape
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    x = tf.keras.layers.Dense(d_model)(inputs) if num_features != d_model else inputs
    
    positions = np.arange(seq_len)[:, np.newaxis]
    depths = np.arange(d_model // 2)[np.newaxis, :] / (d_model // 2)
    angle_rates = 1 / (10000 ** depths)
    angle_rads = positions * angle_rates
    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)
    pos_encoding = tf.cast(pos_encoding[:, :d_model], dtype=tf.float32)
    x = x + pos_encoding
    
    for _ in range(num_layers):
        attn_out = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model
        )(x, x)
        attn_out = tf.keras.layers.Dropout(dropout_rate)(attn_out)
        x = tf.keras.layers.Add()([x, attn_out])
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        
        ffn_out = tf.keras.layers.Dense(ff_dim, activation="relu")(x)
        ffn_out = tf.keras.layers.Dense(d_model)(ffn_out)
        ffn_out = tf.keras.layers.Dropout(dropout_rate)(ffn_out)
        x = tf.keras.layers.Add()([x, ffn_out])
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1)(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mse")
    return model


def evaluate_model(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """Evaluate model performance on test data."""
    y_pred = model.predict(X_test, verbose=0).flatten()
    return {
        "rmse": rmse(y_test, y_pred),
        "mae": mae(y_test, y_pred),
        "r2": r2(y_test, y_pred),
        "directional_accuracy": directional_accuracy(y_test, y_pred),
    }


def lstm_objective(trial: Trial, df: pd.DataFrame, config: Dict[str, Any]) -> float:
    """Optuna objective function for LSTM hyperparameter optimization."""
    lstm_config = config.get("lstm", {})
    tuning_config = config.get("tuning", {})
    
    n_layers = trial.suggest_int(
        "n_layers",
        lstm_config.get("n_layers", {}).get("min", 1),
        lstm_config.get("n_layers", {}).get("max", 3),
    )
    
    units_per_layer = [
        trial.suggest_int(
            f"units_layer_{i}",
            lstm_config.get("units_per_layer", {}).get("min", 32),
            lstm_config.get("units_per_layer", {}).get("max", 150),
        )
        for i in range(n_layers)
    ]
    
    seq_len_config = lstm_config.get("sequence_length", {})
    sequence_length = trial.suggest_int(
        "sequence_length",
        seq_len_config.get("min", 30),
        seq_len_config.get("max", 90),
        step=seq_len_config.get("step", 10),
    )
    
    dropout_rate = trial.suggest_float(
        "dropout_rate",
        lstm_config.get("dropout_rate", {}).get("min", 0.0),
        lstm_config.get("dropout_rate", {}).get("max", 0.5),
    )
    
    lr_config = lstm_config.get("learning_rate", {})
    learning_rate = trial.suggest_float(
        "learning_rate",
        lr_config.get("min", 0.0005),
        lr_config.get("max", 0.01),
        log=lr_config.get("log", True),
    )
    
    batch_choices = lstm_config.get("batch_size", {}).get("choices", [32, 64, 128])
    batch_size = trial.suggest_categorical("batch_size", batch_choices)
    
    l2_config = lstm_config.get("l2_regularization", {})
    l2_reg = trial.suggest_float(
        "l2_reg",
        l2_config.get("min", 0.0),
        l2_config.get("max", 0.01),
    )
    
    features = feature_columns(df)
    n_splits = tuning_config.get("n_splits", 3)
    train_min_period = tuning_config.get("train_min_period", 252)
    early_stopping_config = lstm_config.get("early_stopping", {})
    
    rmse_scores = []
    
    for split_id, (train_idx, test_idx) in enumerate(
        walk_forward_splits(df, n_splits=n_splits, train_min_period=train_min_period),
        start=1,
    ):
        train = df.iloc[train_idx].copy()
        test = df.iloc[test_idx].copy()
        
        scaler = StandardScaler()
        train[features] = scaler.fit_transform(train[features])
        test[features] = scaler.transform(test[features])
        
        X_train, y_train = create_sequences(train, features, "next_day_return", sequence_length)
        X_test, y_test = create_sequences(test, features, "next_day_return", sequence_length)
        
        if X_train.size == 0 or X_test.size == 0:
            LOGGER.warning("Insufficient sequence data for split %s", split_id)
            continue
        
        model = build_tunable_lstm(
            input_shape=(sequence_length, len(features)),
            n_layers=n_layers,
            units_per_layer=units_per_layer,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            l2_reg=l2_reg,
            activation=lstm_config.get("activation", "tanh"),
        )
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=early_stopping_config.get("patience", 10),
                min_delta=early_stopping_config.get("min_delta", 0.0001),
                restore_best_weights=early_stopping_config.get("restore_best_weights", True),
            )
        ]
        
        model.fit(
            X_train,
            y_train,
            epochs=lstm_config.get("epochs", {}).get("max", 100),
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=0,
        )
        
        metrics = evaluate_model(model, X_test, y_test)
        rmse_scores.append(metrics["rmse"])
        
        trial.report(np.mean(rmse_scores), split_id)
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return np.mean(rmse_scores) if rmse_scores else float("inf")


def transformer_objective(trial: Trial, df: pd.DataFrame, config: Dict[str, Any]) -> float:
    """Optuna objective function for Transformer hyperparameter optimization."""
    transformer_config = config.get("transformer", {})
    tuning_config = config.get("tuning", {})
    
    num_layers = trial.suggest_int(
        "num_layers",
        transformer_config.get("num_layers", {}).get("min", 1),
        transformer_config.get("num_layers", {}).get("max", 3),
    )
    
    num_heads_choices = transformer_config.get("num_heads", {}).get("choices", [2, 4, 8])
    num_heads = trial.suggest_categorical("num_heads", num_heads_choices)
    
    d_model_choices = transformer_config.get("d_model", {}).get("choices", [32, 64, 128])
    d_model = trial.suggest_categorical("d_model", d_model_choices)
    
    ff_dim_choices = transformer_config.get("ff_dim", {}).get("choices", [64, 128, 256])
    ff_dim = trial.suggest_categorical("ff_dim", ff_dim_choices)
    
    seq_len_config = transformer_config.get("sequence_length", {})
    sequence_length = trial.suggest_int(
        "sequence_length",
        seq_len_config.get("min", 30),
        seq_len_config.get("max", 90),
        step=seq_len_config.get("step", 10),
    )
    
    dropout_rate = trial.suggest_float(
        "dropout_rate",
        transformer_config.get("dropout_rate", {}).get("min", 0.0),
        transformer_config.get("dropout_rate", {}).get("max", 0.5),
    )
    
    lr_config = transformer_config.get("learning_rate", {})
    learning_rate = trial.suggest_float(
        "learning_rate",
        lr_config.get("min", 0.0005),
        lr_config.get("max", 0.01),
        log=lr_config.get("log", True),
    )
    
    batch_choices = transformer_config.get("batch_size", {}).get("choices", [32, 64, 128])
    batch_size = trial.suggest_categorical("batch_size", batch_choices)
    
    features = feature_columns(df)
    n_splits = tuning_config.get("n_splits", 3)
    train_min_period = tuning_config.get("train_min_period", 252)
    early_stopping_config = transformer_config.get("early_stopping", {})
    
    rmse_scores = []
    
    for split_id, (train_idx, test_idx) in enumerate(
        walk_forward_splits(df, n_splits=n_splits, train_min_period=train_min_period),
        start=1,
    ):
        train = df.iloc[train_idx].copy()
        test = df.iloc[test_idx].copy()
        
        scaler = StandardScaler()
        train[features] = scaler.fit_transform(train[features])
        test[features] = scaler.transform(test[features])
        
        X_train, y_train = create_sequences(train, features, "next_day_return", sequence_length)
        X_test, y_test = create_sequences(test, features, "next_day_return", sequence_length)
        
        if X_train.size == 0 or X_test.size == 0:
            LOGGER.warning("Insufficient sequence data for split %s", split_id)
            continue
        
        model = build_tunable_transformer(
            input_shape=(sequence_length, len(features)),
            num_layers=num_layers,
            num_heads=num_heads,
            d_model=d_model,
            ff_dim=ff_dim,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
        )
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=early_stopping_config.get("patience", 10),
                min_delta=early_stopping_config.get("min_delta", 0.0001),
                restore_best_weights=early_stopping_config.get("restore_best_weights", True),
            )
        ]
        
        model.fit(
            X_train,
            y_train,
            epochs=transformer_config.get("epochs", {}).get("max", 100),
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=0,
        )
        
        metrics = evaluate_model(model, X_test, y_test)
        rmse_scores.append(metrics["rmse"])
        
        trial.report(np.mean(rmse_scores), split_id)
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return np.mean(rmse_scores) if rmse_scores else float("inf")


def run_lstm_tuning(
    data: Optional[pd.DataFrame] = None,
    config: Optional[Dict[str, Any]] = None,
    n_trials: Optional[int] = None,
) -> Tuple[Dict[str, Any], optuna.Study]:
    """Run LSTM hyperparameter optimization.
    
    Parameters
    ----------
    data : pd.DataFrame, optional
        Dataset for training. If None, loads from disk.
    config : Dict[str, Any], optional
        Tuning configuration. If None, loads from configs/tuning.yaml.
    n_trials : int, optional
        Number of optimization trials. Overrides config if provided.
    
    Returns
    -------
    Tuple[Dict[str, Any], optuna.Study]
        Best hyperparameters and the Optuna study object.
    """
    if optuna is None:
        raise ImportError("optuna is required for hyperparameter tuning. Install with: pip install optuna")
    
    df = load_dataset(data)
    config = config or load_tuning_config()
    tuning_config = config.get("tuning", {})
    
    n_trials = n_trials or tuning_config.get("n_trials", 50)
    
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        study_name="lstm_tuning",
    )
    
    LOGGER.info("Starting LSTM hyperparameter tuning with %d trials", n_trials)
    study.optimize(lambda trial: lstm_objective(trial, df, config), n_trials=n_trials)
    
    LOGGER.info("Best LSTM trial:")
    LOGGER.info("  Value (RMSE): %.6f", study.best_trial.value)
    LOGGER.info("  Params: %s", study.best_trial.params)
    
    return study.best_trial.params, study


def run_transformer_tuning(
    data: Optional[pd.DataFrame] = None,
    config: Optional[Dict[str, Any]] = None,
    n_trials: Optional[int] = None,
) -> Tuple[Dict[str, Any], optuna.Study]:
    """Run Transformer hyperparameter optimization.
    
    Parameters
    ----------
    data : pd.DataFrame, optional
        Dataset for training. If None, loads from disk.
    config : Dict[str, Any], optional
        Tuning configuration. If None, loads from configs/tuning.yaml.
    n_trials : int, optional
        Number of optimization trials. Overrides config if provided.
    
    Returns
    -------
    Tuple[Dict[str, Any], optuna.Study]
        Best hyperparameters and the Optuna study object.
    """
    if optuna is None:
        raise ImportError("optuna is required for hyperparameter tuning. Install with: pip install optuna")
    
    df = load_dataset(data)
    config = config or load_tuning_config()
    tuning_config = config.get("tuning", {})
    
    n_trials = n_trials or tuning_config.get("n_trials", 50)
    
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        study_name="transformer_tuning",
    )
    
    LOGGER.info("Starting Transformer hyperparameter tuning with %d trials", n_trials)
    study.optimize(lambda trial: transformer_objective(trial, df, config), n_trials=n_trials)
    
    LOGGER.info("Best Transformer trial:")
    LOGGER.info("  Value (RMSE): %.6f", study.best_trial.value)
    LOGGER.info("  Params: %s", study.best_trial.params)
    
    return study.best_trial.params, study


def save_tuning_results(
    lstm_params: Dict[str, Any],
    transformer_params: Dict[str, Any],
    output_dir: Optional[Path] = None,
) -> None:
    """Save best hyperparameters to YAML file."""
    output_dir = output_dir or TUNING_REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "lstm": {
            "best_params": lstm_params,
            "description": "Best LSTM hyperparameters from Bayesian optimization",
        },
        "transformer": {
            "best_params": transformer_params,
            "description": "Best Transformer hyperparameters from Bayesian optimization",
        },
    }
    
    output_path = output_dir / "best_hyperparameters.yaml"
    with open(output_path, "w") as f:
        yaml.dump(results, f, default_flow_style=False)
    
    LOGGER.info("Saved best hyperparameters to %s", output_path)


def generate_tuning_report(
    lstm_study: optuna.Study,
    transformer_study: optuna.Study,
    output_dir: Optional[Path] = None,
) -> None:
    """Generate a markdown report of tuning results."""
    output_dir = output_dir or TUNING_REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_lines = [
        "# Deep Learning Hyperparameter Tuning Report",
        "",
        "## Summary",
        "",
        "This report summarizes the results of Bayesian optimization for LSTM and Transformer models.",
        "",
        "### Literature-Based Search Space",
        "",
        "| Parameter | Range | Literature Reference |",
        "|-----------|-------|---------------------|",
        "| Number of layers | 1-3 | Optimal architectures use 1-3 LSTM layers |",
        "| Units per layer | 32-150 | Optimal: 96 or 128 neurons |",
        "| Dropout rate | 0.0-0.5 | Optimal: 0.3-0.4 |",
        "| Sequence length | 30-90 days | Captures short to medium-term patterns |",
        "| Learning rate | 0.0005-0.01 | Log-uniform sampling |",
        "| Batch size | 32, 64, 128 | Standard choices for time-series |",
        "",
        "## LSTM Results",
        "",
        f"**Best RMSE:** {lstm_study.best_trial.value:.6f}",
        "",
        "**Best Parameters:**",
        "",
    ]
    
    for key, value in lstm_study.best_trial.params.items():
        report_lines.append(f"- `{key}`: {value}")
    
    report_lines.extend([
        "",
        f"**Total Trials:** {len(lstm_study.trials)}",
        f"**Completed Trials:** {len([t for t in lstm_study.trials if t.state == optuna.trial.TrialState.COMPLETE])}",
        "",
        "## Transformer Results",
        "",
        f"**Best RMSE:** {transformer_study.best_trial.value:.6f}",
        "",
        "**Best Parameters:**",
        "",
    ])
    
    for key, value in transformer_study.best_trial.params.items():
        report_lines.append(f"- `{key}`: {value}")
    
    report_lines.extend([
        "",
        f"**Total Trials:** {len(transformer_study.trials)}",
        f"**Completed Trials:** {len([t for t in transformer_study.trials if t.state == optuna.trial.TrialState.COMPLETE])}",
        "",
        "## Regularization Methods Applied",
        "",
        "1. **Dropout:** Applied between LSTM layers and in Transformer encoder",
        "2. **Early Stopping:** Patience-based with best weights restoration",
        "3. **L2 Regularization:** Applied to LSTM kernel and recurrent weights",
        "",
        "## Next Steps",
        "",
        "1. Apply best hyperparameters to full training pipeline",
        "2. Run extended validation across equity universe",
        "3. Compare tuned models against baseline iterations",
        "",
    ])
    
    report_path = output_dir / "tuning_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    
    LOGGER.info("Generated tuning report at %s", report_path)


def main(n_trials: int = 50) -> None:
    """Run full hyperparameter tuning for LSTM and Transformer models."""
    LOGGER.info("=" * 60)
    LOGGER.info("Deep Learning Hyperparameter Tuning")
    LOGGER.info("=" * 60)
    
    config = load_tuning_config()
    
    LOGGER.info("Running LSTM tuning...")
    lstm_params, lstm_study = run_lstm_tuning(config=config, n_trials=n_trials)
    
    LOGGER.info("Running Transformer tuning...")
    transformer_params, transformer_study = run_transformer_tuning(config=config, n_trials=n_trials)
    
    save_tuning_results(lstm_params, transformer_params)
    generate_tuning_report(lstm_study, transformer_study)
    
    LOGGER.info("=" * 60)
    LOGGER.info("Tuning complete!")
    LOGGER.info("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Deep Learning Hyperparameter Tuning")
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of optimization trials per model (default: 50)",
    )
    parser.add_argument(
        "--model",
        choices=["lstm", "transformer", "both"],
        default="both",
        help="Model to tune (default: both)",
    )
    args = parser.parse_args()
    
    if args.model == "both":
        main(n_trials=args.n_trials)
    elif args.model == "lstm":
        run_lstm_tuning(n_trials=args.n_trials)
    else:
        run_transformer_tuning(n_trials=args.n_trials)
