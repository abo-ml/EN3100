"""Iteration 4: Transformer encoder for time-series forecasting."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import directional_accuracy, mae, r2, rmse
from src.evaluation.reporting import save_metrics_report
from src.evaluation.walkforward import aggregate_metrics, walk_forward_splits
from src.models.iteration1_baseline import feature_columns, load_dataset
from src.models.iteration3_lstm import create_sequences
from src.utils import REPORTS_DIR

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

REPORT_PATH = REPORTS_DIR / "iteration_4_results.md"
REPORT_PATH = Path("reports/iteration_4_results.md")
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level="INFO")

WINDOW = 60


def positional_encoding(length: int, depth: int) -> tf.Tensor:
    depth = depth / 2
    positions = np.arange(length)[:, np.newaxis]
    depths = np.arange(depth)[np.newaxis, :] / depth
    angle_rates = 1 / (10000 ** depths)
    angle_rads = positions * angle_rates
    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)
    return tf.cast(pos_encoding, dtype=tf.float32)


def transformer_encoder(inputs: tf.Tensor, num_heads: int, ff_dim: int, dropout: float) -> tf.Tensor:
    attention_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1])(inputs, inputs)
    attention_output = tf.keras.layers.Dropout(dropout)(attention_output)
    out1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attention_output)

    ffn = tf.keras.Sequential([
        tf.keras.layers.Dense(ff_dim, activation="relu"),
        tf.keras.layers.Dense(inputs.shape[-1]),
    ])
    ffn_output = ffn(out1)
    ffn_output = tf.keras.layers.Dropout(dropout)(ffn_output)
    return tf.keras.layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)


def build_transformer_model(input_shape: Tuple[int, int]) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=input_shape)
    pos_encoding = positional_encoding(input_shape[0], input_shape[1])
    x = inputs + pos_encoding
    x = transformer_encoder(x, num_heads=4, ff_dim=128, dropout=0.2)
    x = transformer_encoder(x, num_heads=4, ff_dim=128, dropout=0.2)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return model


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

        if X_train.size == 0 or X_test.size == 0:
            LOGGER.warning("Insufficient sequence data for split %s", split_id)
            continue

        model = build_transformer_model((WINDOW, len(features)))
        model.fit(
            X_train,
            y_train,
            epochs=25,
            batch_size=32,
            validation_split=0.2,
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
            verbose=0,
        )
        y_pred = model.predict(X_test, verbose=0).flatten()

        record = {
            "rmse_transformer": rmse(y_test, y_pred),
            "mae_transformer": mae(y_test, y_pred),
            "r2_transformer": r2(y_test, y_pred),
            "directional_accuracy_transformer": directional_accuracy(y_test, y_pred),
        }
        records.append(record)

    summary = aggregate_metrics(records)
    save_metrics_report(summary, REPORT_PATH)
    LOGGER.info("Iteration 4 summary: %s", summary)
    return summary


def main() -> Dict[str, float]:
    return run_iteration()


if __name__ == "__main__":
    main()
