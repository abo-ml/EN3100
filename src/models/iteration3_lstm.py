"""Iteration 3: LSTM-based sequence models."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import directional_accuracy, mae, r2, rmse
from src.evaluation.reporting import save_metrics_report
from src.evaluation.walkforward import aggregate_metrics, walk_forward_splits
from src.models.iteration1_baseline import feature_columns, load_dataset

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

REPORT_PATH = Path("reports/iteration_3_results.md")
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level="INFO")

WINDOW = 60


def build_lstm_regressor(input_shape: Tuple[int, int]) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.LSTM(64, return_sequences=False),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model


def build_lstm_classifier(input_shape: Tuple[int, int]) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.LSTM(64, return_sequences=False),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def create_sequences(df: pd.DataFrame, features: List[str], target_col: str, window: int) -> Tuple[np.ndarray, np.ndarray]:
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


def run_iteration() -> Dict[str, float]:
    df = load_dataset()
    features = feature_columns(df)

    records = []

    for split_id, (train_idx, test_idx) in enumerate(walk_forward_splits(df, n_splits=3, train_min_period=252), start=1):
        train = df.iloc[train_idx].copy()
        test = df.iloc[test_idx].copy()

        scaler = StandardScaler()
        train[features] = scaler.fit_transform(train[features])
        test[features] = scaler.transform(test[features])

        X_train, y_train = create_sequences(train, features, "next_day_return", WINDOW)
        X_test, y_test = create_sequences(test, features, "next_day_return", WINDOW)
        _, y_train_cls = create_sequences(train, features, "target_direction", WINDOW)
        _, y_test_cls = create_sequences(test, features, "target_direction", WINDOW)

        if X_train.size == 0 or X_test.size == 0:
            LOGGER.warning("Insufficient sequence data for split %s", split_id)
            continue

        reg_model = build_lstm_regressor((WINDOW, len(features)))
        reg_model.fit(
            X_train,
            y_train,
            epochs=20,
            batch_size=32,
            validation_split=0.2,
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
            verbose=0,
        )
        y_pred = reg_model.predict(X_test, verbose=0).flatten()

        clf_model = build_lstm_classifier((WINDOW, len(features)))
        clf_model.fit(
            X_train,
            y_train_cls,
            epochs=20,
            batch_size=32,
            validation_split=0.2,
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
            verbose=0,
        )
        class_probs = clf_model.predict(X_test, verbose=0).flatten()
        class_pred = (class_probs > 0.5).astype(int)

        record = {
            "rmse_lstm": rmse(y_test, y_pred),
            "mae_lstm": mae(y_test, y_pred),
            "r2_lstm": r2(y_test, y_pred),
            "directional_accuracy_lstm": directional_accuracy(y_test, y_pred),
            "classification_accuracy": (class_pred == y_test_cls).mean() if y_test_cls.size else np.nan,
        }
        records.append(record)

    summary = aggregate_metrics(records)
    save_metrics_report(summary, REPORT_PATH)
    LOGGER.info("Iteration 3 summary: %s", summary)
    return summary


def main() -> Dict[str, float]:
    return run_iteration()


if __name__ == "__main__":
    main()
