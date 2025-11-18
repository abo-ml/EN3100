"""Iteration 3: LSTM-based sequence models for next-day return forecasting.

This module builds sliding windows of engineered features, trains an LSTM
regressor with validation-based early stopping, and reports regression metrics
plus directional accuracy on a held-out test set. Plots are produced for
training/validation loss and predicted vs. actual returns.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import directional_accuracy, mae, r2, rmse
from src.evaluation.reporting import plot_pred_vs_actual, save_metrics_report
from src.models.iteration1_baseline import feature_columns, load_dataset
from src.utils import FIGURES_DIR, REPORTS_DIR

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

REPORT_PATH = REPORTS_DIR / "iteration_3_results.md"
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level="INFO")

WINDOW = 60
TRAIN_FRACTION = 0.7
VAL_FRACTION = 0.15  # test fraction derived as remainder


def build_lstm_regressor(input_shape: Tuple[int, int]) -> tf.keras.Model:
    """Construct a simple two-layer LSTM regressor with dropout."""

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
    )
    return model


def create_sequences(
    df: pd.DataFrame, features: List[str], target_col: str, window: int
) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp], List[int]]:
    """Create rolling sequences per ticker with corresponding targets and indices."""

    sequences: List[np.ndarray] = []
    targets: List[float] = []
    dates: List[pd.Timestamp] = []
    indices: List[int] = []

    for _, group in df.groupby("ticker"):
        feature_array = group[features].values
        target_array = group[target_col].values
        date_array = group["date"].values
        index_array = group["orig_index"].values
        if len(group) <= window:
            continue
        for i in range(window, len(group)):
            sequences.append(feature_array[i - window : i])
            targets.append(target_array[i])
            dates.append(pd.to_datetime(date_array[i]))
            indices.append(int(index_array[i]))

    if not sequences:
        return (
            np.empty((0, window, len(features))),
            np.empty((0,)),
            [],
            [],
        )
    return np.array(sequences), np.array(targets), dates, indices


def train_val_test_split_by_index(
    seq_indices: List[int], train_frac: float, val_frac: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return boolean masks for train/val/test based on chronological indices."""

    n = max(seq_indices) + 1 if seq_indices else 0
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    seq_indices_arr = np.array(seq_indices)
    train_mask = seq_indices_arr < train_end
    val_mask = (seq_indices_arr >= train_end) & (seq_indices_arr < val_end)
    test_mask = seq_indices_arr >= val_end
    return train_mask, val_mask, test_mask


def plot_loss_curves(history: tf.keras.callbacks.History, filename: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIGURES_DIR / filename
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history.history.get("loss", []), label="Train Loss")
    ax.plot(history.history.get("val_loss", []), label="Val Loss")
    ax.set_title("Iteration 3: LSTM Training Curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    # Duplicate at root for Colab convenience
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)


def run_iteration(window: int = WINDOW) -> Dict[str, float]:
    """Train and evaluate the LSTM regressor using sequential splits."""

    df = load_dataset()
    df["orig_index"] = np.arange(len(df))
    features = feature_columns(df)

    n_total = len(df)
    train_end = int(n_total * TRAIN_FRACTION)
    val_end = int(n_total * (TRAIN_FRACTION + VAL_FRACTION))

    scaler = StandardScaler()
    df.loc[:train_end - 1, features] = scaler.fit_transform(df.loc[:train_end - 1, features])
    df.loc[train_end:, features] = scaler.transform(df.loc[train_end:, features])

    X_reg, y_reg, seq_dates, seq_indices = create_sequences(df, features, "next_day_return", window)
    if X_reg.size == 0:
        raise ValueError("Insufficient data to form sequences; reduce window or extend history.")

    train_mask, val_mask, test_mask = train_val_test_split_by_index(
        seq_indices, TRAIN_FRACTION, VAL_FRACTION
    )

    if not train_mask.any() or not val_mask.any() or not test_mask.any():
        raise ValueError(
            "Train/val/test split produced an empty partition; adjust fractions or window size."
        )

    X_train, y_train = X_reg[train_mask], y_reg[train_mask]
    X_val, y_val = X_reg[val_mask], y_reg[val_mask]
    X_test, y_test = X_reg[test_mask], y_reg[test_mask]
    test_dates = pd.Series(np.array(seq_dates)[test_mask])

    model = build_lstm_regressor((window, len(features)))
    callbacks = [tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss")]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=0,
    )

    plot_loss_curves(history, "iteration3_loss_curves.png")

    y_pred = model.predict(X_test, verbose=0).flatten()

    metrics = {
        "rmse_lstm": rmse(y_test, y_pred),
        "mae_lstm": mae(y_test, y_pred),
        "r2_lstm": r2(y_test, y_pred),
        "directional_accuracy_lstm": directional_accuracy(y_test, y_pred),
    }

    plot_pred_vs_actual(
        dates=test_dates,
        actual=pd.Series(y_test),
        predicted=pd.Series(y_pred),
        title="Iteration 3: LSTM Predicted vs Actual Next-Day Returns",
        filename="iteration3_pred_vs_actual.png",
    )

    save_metrics_report(metrics, REPORT_PATH)
    REPORT_PATH.write_text(
        "\n".join(
            [
                "# Iteration 3 Results",
                "",
                "LSTM sequence model trained on rolling windows with validation early stopping.",
                "Key metrics on the held-out test split:",
                "",
            ]
            + [f"- **{k}**: {v:.4f}" for k, v in metrics.items()]
            + [
                "",
                "Interpretation: The LSTM captures temporal dependencies via sliding windows;",
                "directional accuracy reflects how often the predicted sign matches the realised",
                "next-day return. Use these metrics to compare against the linear baseline and",
                "tree ensembles.",
            ]
        )
    )

    LOGGER.info("Iteration 3 metrics: %s", metrics)
    return metrics


def main() -> Dict[str, float]:
    return run_iteration()


if __name__ == "__main__":
    main()
