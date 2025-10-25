diff --git a/market_forecasting.py b/market_forecasting.py
new file mode 100644
index 0000000000000000000000000000000000000000..439bf6da66409fd3e7ff370d88cd185383f6ec36
--- /dev/null
+++ b/market_forecasting.py
@@ -0,0 +1,752 @@
+"""Market forecasting module implementing multiple machine learning iterations.
+
+This module demonstrates a progressive build-up of models for financial time-series
+forecasting ranging from linear regression baselines to deep learning ensembles.
+It includes feature engineering utilities, walk-forward validation helpers, and
+placeholder stubs for future enhancements such as order-flow imbalance, pattern
+recognition, execution algorithms, sentiment, and reinforcement learning.
+
+References:
+    - LSTM modelling inspired by Abhishek-k-git/Stock-Price-Prediction-LSTM.
+    - Technical indicator definitions adapted from quantifiedstrategies.com and
+      alphaarchitect.com.
+    - TODO placeholders highlight future integrations with order-book data,
+      sentiment feeds, and advanced execution APIs (Alpha Vantage, Tradier, IBKR).
+"""
+
+from __future__ import annotations
+
+import logging
+import math
+import os
+from typing import Dict, Iterable, List, Optional, Sequence, Tuple
+
+import numpy as np
+import pandas as pd
+import yfinance as yf
+from sklearn.linear_model import LinearRegression
+from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
+from sklearn.model_selection import ParameterGrid
+from sklearn.preprocessing import StandardScaler
+from sklearn.ensemble import RandomForestRegressor
+from tensorflow import keras
+
+# Reduce TensorFlow logging noise for clarity when running the module as a script.
+os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
+
+logger = logging.getLogger(__name__)
+logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
+
+
+# ---------------------------------------------------------------------------
+# Data acquisition utilities
+# ---------------------------------------------------------------------------
+
+def fetch_data(tickers: Sequence[str], start: str, end: str) -> pd.DataFrame:
+    """Download historical OHLCV data for tickers between start and end dates.
+
+    Parameters
+    ----------
+    tickers:
+        Iterable of ticker symbols understood by Yahoo! Finance.
+    start, end:
+        Date strings parsable by ``pandas.Timestamp`` (e.g. "2015-01-01").
+
+    Returns
+    -------
+    pandas.DataFrame
+        DataFrame indexed by date. Columns are flattened to the pattern
+        ``{ticker}_{field}`` (e.g. ``AAPL_Close``) for consistency across
+        single- and multi-ticker downloads.
+    """
+
+    logger.info("Fetching data for tickers: %s", ", ".join(tickers))
+    data = yf.download(
+        tickers=list(tickers),
+        start=start,
+        end=end,
+        auto_adjust=False,
+        progress=False,
+        actions=False,
+    )
+
+    if data.empty:
+        raise ValueError("No data returned from Yahoo Finance. Check tickers and date range.")
+
+    if isinstance(data.columns, pd.MultiIndex):
+        # Swap levels so tickers come first, flatten, and normalise column names.
+        data = data.swaplevel(0, 1, axis=1).sort_index(axis=1, level=0)
+        flattened_cols = []
+        for ticker, field in data.columns:
+            sanitized_field = field.replace(" ", "")
+            flattened_cols.append(f"{ticker}_{sanitized_field}")
+        data.columns = flattened_cols
+    else:
+        data.columns = [f"{tickers[0]}_{col.replace(' ', '')}" for col in data.columns]
+
+    data.index = pd.to_datetime(data.index)
+    logger.info("Fetched %d rows of data.", len(data))
+    return data
+
+
+# ---------------------------------------------------------------------------
+# Technical indicator calculations
+# ---------------------------------------------------------------------------
+
+def compute_macd(price: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
+    """Compute the Moving Average Convergence Divergence (MACD) indicator.
+
+    Returns a DataFrame with ``macd``, ``signal`` and ``hist`` columns."""
+
+    ema_fast = price.ewm(span=fast, adjust=False).mean()
+    ema_slow = price.ewm(span=slow, adjust=False).mean()
+    macd_line = ema_fast - ema_slow
+    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
+    hist = macd_line - signal_line
+    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "hist": hist})
+
+
+def compute_rsi(price: pd.Series, period: int = 14) -> pd.Series:
+    """Compute the Relative Strength Index (RSI)."""
+
+    delta = price.diff()
+    gain = delta.clip(lower=0)
+    loss = -delta.clip(upper=0)
+    avg_gain = gain.rolling(window=period, min_periods=period).mean()
+    avg_loss = loss.rolling(window=period, min_periods=period).mean()
+    rs = avg_gain / avg_loss
+    rsi = 100 - (100 / (1 + rs))
+    return rsi
+
+
+def compute_tsmom(price: pd.Series, window: int = 252) -> pd.Series:
+    """Compute time-series momentum (trailing ``window`` day return)."""
+
+    return price.pct_change(periods=window)
+
+
+def compute_ofi(df: pd.DataFrame) -> pd.Series:
+    """Placeholder for order-flow imbalance calculations.
+
+    TODO: integrate order-book level data and compute true OFI metrics.
+    Currently returns a zero series matching the input length."""
+
+    return pd.Series(0.0, index=df.index, name="ofi")
+
+
+def detect_head_and_shoulders(df: pd.DataFrame) -> pd.Series:
+    """Placeholder for head-and-shoulders pattern recognition.
+
+    TODO: implement pattern matching using peak/trough analysis or ML models.
+    Currently returns NaNs to denote unavailable data."""
+
+    return pd.Series(np.nan, index=df.index, name="head_shoulders")
+
+
+def detect_double_top(df: pd.DataFrame) -> pd.Series:
+    """Placeholder for double-top/double-bottom detection.
+
+    TODO: implement structural pattern detection or integrate third-party APIs.
+    Currently returns NaNs."""
+
+    return pd.Series(np.nan, index=df.index, name="double_top")
+
+
+# ---------------------------------------------------------------------------
+# Feature engineering
+# ---------------------------------------------------------------------------
+
+def _infer_tickers(df: pd.DataFrame) -> List[str]:
+    prefixes = sorted({col.split("_")[0] for col in df.columns})
+    return prefixes
+
+
+def build_features(df: pd.DataFrame, scaler: Optional[StandardScaler] = None) -> Tuple[pd.DataFrame, List[str], StandardScaler]:
+    """Construct engineered features for each ticker in ``df``.
+
+    Parameters
+    ----------
+    df:
+        DataFrame of OHLCV columns flattened as ``{ticker}_{field}``.
+    scaler:
+        Optional pre-fitted ``StandardScaler``. If provided, reuse it to transform
+        features. Otherwise, fit a new scaler on the numeric features.
+
+    Returns
+    -------
+    Tuple[pandas.DataFrame, List[str], StandardScaler]
+        * Feature-enhanced DataFrame with scaled numeric predictors.
+        * List of target column names (next-day returns per ticker).
+        * The ``StandardScaler`` used (fitted).
+    """
+
+    features = df.copy()
+    tickers = _infer_tickers(features)
+
+    target_cols: List[str] = []
+
+    for ticker in tickers:
+        close_col = f"{ticker}_AdjClose" if f"{ticker}_AdjClose" in features.columns else f"{ticker}_Close"
+        volume_col = f"{ticker}_Volume"
+
+        if close_col not in features.columns:
+            logger.warning("Ticker %s is missing a close price column; skipping indicator computation.", ticker)
+            continue
+
+        price_series = features[close_col]
+        returns_1d = price_series.pct_change()
+        features[f"{ticker}_return_1d"] = returns_1d
+        features[f"{ticker}_return_5d"] = price_series.pct_change(periods=5)
+
+        macd = compute_macd(price_series)
+        features[f"{ticker}_macd"] = macd["macd"]
+        features[f"{ticker}_macd_signal"] = macd["signal"]
+        features[f"{ticker}_macd_hist"] = macd["hist"]
+
+        features[f"{ticker}_rsi"] = compute_rsi(price_series)
+        features[f"{ticker}_tsmom"] = compute_tsmom(price_series)
+
+        ofi = compute_ofi(features[[col for col in features.columns if col.startswith(ticker)]])
+        features[f"{ticker}_ofi"] = ofi
+
+        features[f"{ticker}_head_shoulders"] = detect_head_and_shoulders(features[[close_col]])
+        features[f"{ticker}_double_top"] = detect_double_top(features[[close_col]])
+
+        if volume_col in features.columns:
+            features[f"{ticker}_volume_change"] = features[volume_col].pct_change()
+
+        target_col = f"{ticker}_target_return"
+        features[target_col] = returns_1d.shift(-1)
+        target_cols.append(target_col)
+
+    # Drop rows with NaNs originating from indicator/return calculations.
+    features = features.dropna()
+
+    # Identify feature columns (exclude targets).
+    feature_cols = [col for col in features.columns if col not in target_cols]
+
+    if not feature_cols:
+        raise ValueError("No feature columns detected after engineering.")
+
+    scaler = scaler or StandardScaler()
+    features[feature_cols] = scaler.fit_transform(features[feature_cols].values)
+
+    return features, target_cols, scaler
+
+
+# ---------------------------------------------------------------------------
+# Walk-forward validation helpers
+# ---------------------------------------------------------------------------
+
+def train_test_splits(data: pd.DataFrame, n_splits: int = 5, min_train_size: Optional[int] = None) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
+    """Generate walk-forward validation splits.
+
+    Parameters
+    ----------
+    data:
+        Feature-engineered DataFrame sorted by date index.
+    n_splits:
+        Number of walk-forward splits.
+    min_train_size:
+        Optional minimum number of observations for the initial training window.
+
+    Returns
+    -------
+    List[Tuple[pandas.DataFrame, pandas.DataFrame]]
+        List of (train_df, test_df) tuples.
+    """
+
+    if data.index.is_monotonic_increasing is False:
+        data = data.sort_index()
+
+    n_samples = len(data)
+    if n_samples < n_splits + 1:
+        raise ValueError("Not enough samples to create walk-forward splits.")
+
+    min_train_size = min_train_size or max(60, n_samples // (n_splits + 1))
+    split_points = np.linspace(min_train_size, n_samples, num=n_splits + 1, dtype=int)
+
+    splits: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
+    start_idx = 0
+    for i in range(n_splits):
+        train_end = split_points[i]
+        test_end = split_points[i + 1]
+        train_df = data.iloc[start_idx:train_end]
+        test_df = data.iloc[train_end:test_end]
+        if len(test_df) == 0:
+            continue
+        splits.append((train_df, test_df))
+
+    return splits
+
+
+# ---------------------------------------------------------------------------
+# Evaluation helpers
+# ---------------------------------------------------------------------------
+
+def compute_sharpe_ratio(returns: np.ndarray, trading_days: int = 252) -> float:
+    """Compute the annualised Sharpe ratio with zero risk-free rate."""
+
+    returns = np.asarray(returns)
+    if returns.size == 0 or np.std(returns) == 0:
+        return float("nan")
+    return math.sqrt(trading_days) * (np.mean(returns) / np.std(returns))
+
+
+def compute_sortino_ratio(returns: np.ndarray, trading_days: int = 252) -> float:
+    """Compute the annualised Sortino ratio for a return series."""
+
+    returns = np.asarray(returns)
+    downside = returns[returns < 0]
+    if returns.size == 0 or downside.size == 0:
+        return float("nan")
+    downside_std = downside.std(ddof=0)
+    if downside_std == 0:
+        return float("nan")
+    return math.sqrt(trading_days) * (np.mean(returns) / downside_std)
+
+
+def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
+    """Return standard regression metrics and the Sharpe ratio of predictions."""
+
+    metrics = {
+        "rmse": math.sqrt(mean_squared_error(y_true, y_pred)),
+        "mae": mean_absolute_error(y_true, y_pred),
+        "r2": r2_score(y_true, y_pred),
+        "sharpe": compute_sharpe_ratio(y_pred),
+    }
+    return metrics
+
+
+def print_iteration_results(iteration_name: str, results: Dict[str, List[Dict[str, float]]]) -> None:
+    """Pretty-print metrics per split and aggregate averages for each target."""
+
+    print(f"\n=== {iteration_name} ===")
+    for target, metrics_list in results.items():
+        if not metrics_list:
+            print(f"Target: {target} -> No results (insufficient data).")
+            continue
+        print(f"Target: {target}")
+        for idx, metrics in enumerate(metrics_list, start=1):
+            metric_str = ", ".join(f"{key.upper()}: {value:.4f}" for key, value in metrics.items())
+            print(f"  Split {idx}: {metric_str}")
+        averages = {key: np.nanmean([m[key] for m in metrics_list]) for key in metrics_list[0]}
+        avg_str = ", ".join(f"{key.upper()}: {value:.4f}" for key, value in averages.items())
+        print(f"  -> Average: {avg_str}")
+
+
+# ---------------------------------------------------------------------------
+# Iteration 1 – Linear Regression Baseline
+# ---------------------------------------------------------------------------
+
+def run_linear_regression_iteration(
+    data: pd.DataFrame,
+    feature_cols: Sequence[str],
+    target_cols: Sequence[str],
+    splits: Sequence[Tuple[pd.DataFrame, pd.DataFrame]],
+) -> Dict[str, List[Dict[str, float]]]:
+    """Evaluate a linear regression baseline across walk-forward splits."""
+
+    results: Dict[str, List[Dict[str, float]]] = {target: [] for target in target_cols}
+
+    for split_idx, (train_df, test_df) in enumerate(splits, start=1):
+        model = LinearRegression()
+        X_train = train_df[feature_cols]
+        X_test = test_df[feature_cols]
+        y_train = train_df[target_cols]
+        y_test = test_df[target_cols]
+
+        model.fit(X_train, y_train)
+        y_pred = model.predict(X_test)
+
+        for i, target in enumerate(target_cols):
+            metrics = evaluate_predictions(y_test.iloc[:, i].values, y_pred[:, i])
+            results[target].append(metrics)
+        logger.info("Linear Regression split %d complete.", split_idx)
+
+    print_iteration_results("Iteration 1 – Linear Regression", results)
+    return results
+
+
+# ---------------------------------------------------------------------------
+# Iteration 2 – Random Forest Regression
+# ---------------------------------------------------------------------------
+
+def run_random_forest_iteration(
+    data: pd.DataFrame,
+    feature_cols: Sequence[str],
+    target_cols: Sequence[str],
+    splits: Sequence[Tuple[pd.DataFrame, pd.DataFrame]],
+    param_grid: Optional[Dict[str, Iterable]] = None,
+) -> Dict[str, List[Dict[str, float]]]:
+    """Evaluate a RandomForestRegressor with a small grid search."""
+
+    param_grid = param_grid or {"n_estimators": [100], "max_depth": [5, 10, None]}
+
+    results: Dict[str, List[Dict[str, float]]] = {target: [] for target in target_cols}
+
+    for split_idx, (train_df, test_df) in enumerate(splits, start=1):
+        best_model: Optional[RandomForestRegressor] = None
+        best_score = float("inf")
+
+        X_train = train_df[feature_cols]
+        X_test = test_df[feature_cols]
+        y_train = train_df[target_cols]
+        y_test = test_df[target_cols]
+
+        for params in ParameterGrid(param_grid):
+            model = RandomForestRegressor(random_state=42, n_jobs=-1, **params)
+            model.fit(X_train, y_train)
+            preds = model.predict(X_test)
+            score = mean_squared_error(y_test.values, preds)
+            if score < best_score:
+                best_score = score
+                best_model = model
+
+        assert best_model is not None
+        best_preds = best_model.predict(X_test)
+
+        for i, target in enumerate(target_cols):
+            metrics = evaluate_predictions(y_test.iloc[:, i].values, best_preds[:, i])
+            results[target].append(metrics)
+        logger.info("Random Forest split %d complete with params %s.", split_idx, best_model.get_params())
+
+    print_iteration_results("Iteration 2 – Random Forest", results)
+    return results
+
+
+# ---------------------------------------------------------------------------
+# Sequence utilities for deep learning iterations
+# ---------------------------------------------------------------------------
+
+def create_sequences(
+    data: pd.DataFrame,
+    feature_cols: Sequence[str],
+    target_col: str,
+    seq_len: int = 60,
+) -> Tuple[np.ndarray, np.ndarray]:
+    """Transform tabular data into sequences for sequence models."""
+
+    X, y = [], []
+    for idx in range(len(data) - seq_len):
+        window = data.iloc[idx : idx + seq_len]
+        target_value = data.iloc[idx + seq_len][target_col]
+        X.append(window[feature_cols].values)
+        y.append(target_value)
+    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)
+
+
+def prepare_lstm_data(
+    train_df: pd.DataFrame,
+    test_df: pd.DataFrame,
+    feature_cols: Sequence[str],
+    target_col: str,
+    seq_len: int = 60,
+) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
+    """Create train/test sequences with lookback ``seq_len``."""
+
+    X_train, y_train = create_sequences(train_df, feature_cols, target_col, seq_len)
+    combined = pd.concat([train_df.tail(seq_len), test_df])
+    X_test, y_test = create_sequences(combined, feature_cols, target_col, seq_len)
+    return X_train, y_train, X_test, y_test
+
+
+def build_lstm(input_shape: Tuple[int, int]) -> keras.Model:
+    """Create an LSTM-based regression model."""
+
+    model = keras.Sequential(
+        [
+            keras.layers.LSTM(64, return_sequences=True, input_shape=input_shape),
+            keras.layers.Dropout(0.2),
+            keras.layers.LSTM(32),
+            keras.layers.Dense(1),
+        ]
+    )
+    model.compile(optimizer="adam", loss="mse")
+    return model
+
+
+def build_transformer(
+    num_features: int,
+    seq_len: int,
+    d_model: int = 64,
+    num_heads: int = 4,
+    ff_dim: int = 128,
+) -> keras.Model:
+    """Build a simple transformer encoder model."""
+
+    inputs = keras.Input(shape=(seq_len, num_features))
+    attn_out = keras.layers.MultiHeadAttention(num_heads, key_dim=d_model)(inputs, inputs)
+    attn_out = keras.layers.Add()([inputs, attn_out])
+    attn_out = keras.layers.LayerNormalization(epsilon=1e-6)(attn_out)
+
+    ff = keras.Sequential([
+        keras.layers.Dense(ff_dim, activation="relu"),
+        keras.layers.Dense(d_model),
+    ])
+    ff_out = ff(attn_out)
+    seq_out = keras.layers.Add()([attn_out, ff_out])
+    seq_out = keras.layers.LayerNormalization(epsilon=1e-6)(seq_out)
+    flat = keras.layers.Flatten()(seq_out)
+    outputs = keras.layers.Dense(1)(flat)
+    model = keras.Model(inputs=inputs, outputs=outputs)
+    model.compile(optimizer="adam", loss="mse")
+    return model
+
+
+# ---------------------------------------------------------------------------
+# Iteration 3 – LSTM
+# ---------------------------------------------------------------------------
+
+def run_lstm_iteration(
+    feature_cols: Sequence[str],
+    target_cols: Sequence[str],
+    splits: Sequence[Tuple[pd.DataFrame, pd.DataFrame]],
+    seq_len: int = 60,
+    epochs: int = 10,
+    batch_size: int = 32,
+) -> Dict[str, List[Dict[str, float]]]:
+    """Train and evaluate an LSTM per target across walk-forward splits."""
+
+    results: Dict[str, List[Dict[str, float]]] = {target: [] for target in target_cols}
+    callbacks = [keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
+
+    for target in target_cols:
+        for split_idx, (train_df, test_df) in enumerate(splits, start=1):
+            X_train, y_train, X_test, y_test = prepare_lstm_data(
+                train_df, test_df, feature_cols, target, seq_len
+            )
+            if len(X_train) == 0 or len(X_test) == 0:
+                logger.warning("Not enough data to create sequences for target %s split %d.", target, split_idx)
+                continue
+
+            model = build_lstm((seq_len, len(feature_cols)))
+            model.fit(
+                X_train,
+                y_train,
+                epochs=epochs,
+                batch_size=batch_size,
+                verbose=0,
+                callbacks=callbacks,
+            )
+            preds = model.predict(X_test, verbose=0).flatten()
+            metrics = evaluate_predictions(y_test, preds)
+            results[target].append(metrics)
+            logger.info("LSTM target %s split %d complete.", target, split_idx)
+
+    print_iteration_results("Iteration 3 – LSTM", results)
+    return results
+
+
+# ---------------------------------------------------------------------------
+# Iteration 4 – Transformer Encoder
+# ---------------------------------------------------------------------------
+
+def run_transformer_iteration(
+    feature_cols: Sequence[str],
+    target_cols: Sequence[str],
+    splits: Sequence[Tuple[pd.DataFrame, pd.DataFrame]],
+    seq_len: int = 60,
+    epochs: int = 10,
+    batch_size: int = 32,
+) -> Dict[str, List[Dict[str, float]]]:
+    """Train and evaluate a transformer encoder per target."""
+
+    results: Dict[str, List[Dict[str, float]]] = {target: [] for target in target_cols}
+    callbacks = [keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
+
+    for target in target_cols:
+        for split_idx, (train_df, test_df) in enumerate(splits, start=1):
+            X_train, y_train, X_test, y_test = prepare_lstm_data(
+                train_df, test_df, feature_cols, target, seq_len
+            )
+            if len(X_train) == 0 or len(X_test) == 0:
+                logger.warning(
+                    "Not enough data to create transformer sequences for target %s split %d.",
+                    target,
+                    split_idx,
+                )
+                continue
+
+            model = build_transformer(num_features=len(feature_cols), seq_len=seq_len)
+            model.fit(
+                X_train,
+                y_train,
+                epochs=epochs,
+                batch_size=batch_size,
+                verbose=0,
+                callbacks=callbacks,
+            )
+            preds = model.predict(X_test, verbose=0).flatten()
+            metrics = evaluate_predictions(y_test, preds)
+            results[target].append(metrics)
+            logger.info("Transformer target %s split %d complete.", target, split_idx)
+
+    print_iteration_results("Iteration 4 – Transformer", results)
+    return results
+
+
+# ---------------------------------------------------------------------------
+# Iteration 5 – Ensemble & Dynamic Position Sizing
+# ---------------------------------------------------------------------------
+
+def ensemble_predictions(preds_a: np.ndarray, preds_b: np.ndarray, weight: float = 0.5) -> np.ndarray:
+    """Weighted average ensemble of two prediction arrays."""
+
+    return weight * preds_a + (1 - weight) * preds_b
+
+
+def dynamic_position_sizing(predictions: pd.Series, window: int = 20) -> pd.Series:
+    """Scale position sizes by prediction confidence.
+
+    Positions are computed as ``sign(prediction) * min(|prediction| / rolling_std, 1)``.
+    """
+
+    rolling_std = predictions.rolling(window=window, min_periods=1).std().replace(0, np.nan)
+    confidence = (predictions.abs() / rolling_std).clip(upper=1.0).fillna(0.0)
+    positions = predictions.apply(np.sign) * confidence
+    return positions.clip(-1.0, 1.0)
+
+
+def run_ensemble_iteration(
+    feature_cols: Sequence[str],
+    target_cols: Sequence[str],
+    splits: Sequence[Tuple[pd.DataFrame, pd.DataFrame]],
+    seq_len: int = 60,
+    epochs: int = 10,
+    batch_size: int = 32,
+    weight: float = 0.5,
+) -> Dict[str, List[Dict[str, float]]]:
+    """Combine LSTM and Transformer forecasts, evaluate trading metrics."""
+
+    results: Dict[str, List[Dict[str, float]]] = {target: [] for target in target_cols}
+    callbacks = [keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
+
+    for target in target_cols:
+        for split_idx, (train_df, test_df) in enumerate(splits, start=1):
+            X_train, y_train, X_test, y_test = prepare_lstm_data(
+                train_df, test_df, feature_cols, target, seq_len
+            )
+            if len(X_train) == 0 or len(X_test) == 0:
+                logger.warning("Not enough data for ensemble target %s split %d.", target, split_idx)
+                continue
+
+            lstm_model = build_lstm((seq_len, len(feature_cols)))
+            transformer_model = build_transformer(num_features=len(feature_cols), seq_len=seq_len)
+
+            lstm_model.fit(
+                X_train,
+                y_train,
+                epochs=epochs,
+                batch_size=batch_size,
+                verbose=0,
+                callbacks=callbacks,
+            )
+            transformer_model.fit(
+                X_train,
+                y_train,
+                epochs=epochs,
+                batch_size=batch_size,
+                verbose=0,
+                callbacks=callbacks,
+            )
+
+            lstm_preds = lstm_model.predict(X_test, verbose=0).flatten()
+            transformer_preds = transformer_model.predict(X_test, verbose=0).flatten()
+            blended_preds = ensemble_predictions(lstm_preds, transformer_preds, weight)
+
+            metrics = evaluate_predictions(y_test, blended_preds)
+
+            # Dynamic position sizing using blended predictions.
+            prediction_series = pd.Series(blended_preds, index=test_df.index[-len(blended_preds):])
+            positions = dynamic_position_sizing(prediction_series)
+            actual_returns = pd.Series(y_test, index=prediction_series.index)
+            portfolio_returns = positions * actual_returns
+            metrics["strategy_sharpe"] = compute_sharpe_ratio(portfolio_returns.values)
+            metrics["strategy_sortino"] = compute_sortino_ratio(portfolio_returns.values)
+
+            results[target].append(metrics)
+            logger.info("Ensemble target %s split %d complete.", target, split_idx)
+
+    print_iteration_results("Iteration 5 – Ensemble", results)
+    return results
+
+
+# ---------------------------------------------------------------------------
+# Extension placeholders
+# ---------------------------------------------------------------------------
+
+def plan_vwap_execution(order_volume: float, intervals: int = 6) -> pd.DataFrame:
+    """Placeholder for VWAP execution schedule."""
+
+    volume_per_interval = order_volume / intervals
+    schedule = pd.DataFrame(
+        {
+            "interval": range(1, intervals + 1),
+            "scheduled_volume": [volume_per_interval] * intervals,
+        }
+    )
+    # TODO: Replace with volume-weighted allocations using intraday profiles.
+    return schedule
+
+
+def plan_twap_execution(order_volume: float, intervals: int = 6) -> pd.DataFrame:
+    """Placeholder for TWAP execution schedule (equal slices)."""
+
+    return plan_vwap_execution(order_volume, intervals)
+
+
+def find_cointegrated_pairs(price_data: pd.DataFrame) -> pd.DataFrame:
+    """Placeholder for pairs-trading cointegration analysis."""
+
+    # TODO: Integrate statsmodels.tsa.stattools.coint for real pair identification.
+    return pd.DataFrame(columns=["asset_a", "asset_b", "pvalue"])
+
+
+def scalping_hft_placeholder() -> None:
+    """Placeholder for HFT integration."""
+
+    # TODO: Integrate high-frequency data sources and event-driven backtesting.
+    pass
+
+
+def sentiment_analysis_placeholder(news_df: pd.DataFrame) -> pd.Series:
+    """Placeholder sentiment scoring function."""
+
+    # TODO: Replace with NLP model scoring of text headlines or tweets.
+    return pd.Series(np.nan, index=news_df.index if not news_df.empty else [])
+
+
+def reinforcement_learning_placeholder() -> None:
+    """Skeleton for deep reinforcement learning integration."""
+
+    # TODO: Integrate stable_baselines3 for policy optimisation over trading environment.
+    pass
+
+
+# ---------------------------------------------------------------------------
+# Example usage
+# ---------------------------------------------------------------------------
+
+def main() -> None:
+    tickers = ["AAPL", "TSLA"]
+    start_date = "2015-01-01"
+    end_date = "2023-12-31"
+
+    raw_data = fetch_data(tickers, start=start_date, end=end_date)
+    features, target_cols, scaler = build_features(raw_data)
+    feature_cols = [col for col in features.columns if col not in target_cols]
+    splits = train_test_splits(features, n_splits=3)
+
+    # Iteration 1 & 2 operate directly on tabular data.
+    run_linear_regression_iteration(features, feature_cols, target_cols, splits)
+    run_random_forest_iteration(features, feature_cols, target_cols, splits)
+
+    # Iteration 3 onwards (deep learning) can be compute-intensive; limit epochs in example.
+    run_lstm_iteration(feature_cols, target_cols, splits, seq_len=60, epochs=5, batch_size=32)
+    run_transformer_iteration(feature_cols, target_cols, splits, seq_len=60, epochs=5, batch_size=32)
+    run_ensemble_iteration(feature_cols, target_cols, splits, seq_len=60, epochs=5, batch_size=32)
+
+
+if __name__ == "__main__":
+    main()
