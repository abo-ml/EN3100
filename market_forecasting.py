"""Market forecasting module implementing multiple machine learning iterations.

This module demonstrates a progressive build-up of models for financial time-series
forecasting ranging from linear regression baselines to deep learning ensembles.
It includes feature engineering utilities, walk-forward validation helpers, and
placeholder stubs for future enhancements such as order-flow imbalance, pattern
recognition, execution algorithms, sentiment, and reinforcement learning.

References:
    - LSTM modelling inspired by Abhishek-k-git/Stock-Price-Prediction-LSTM.
    - Technical indicator definitions adapted from quantifiedstrategies.com and
      alphaarchitect.com.
    - TODO placeholders highlight future integrations with order-book data,
      sentiment feeds, and advanced execution APIs (Alpha Vantage, Tradier, IBKR).
"""

from __future__ import annotations

import argparse
import logging
import math
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from tensorflow import keras

# Reduce TensorFlow logging noise for clarity when running the module as a script.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Data acquisition utilities
# ---------------------------------------------------------------------------

def fetch_data(tickers: Sequence[str], start: str, end: str) -> pd.DataFrame:
    """Download historical OHLCV data for tickers between start and end dates.

    Parameters
    ----------
    tickers:
        Iterable of ticker symbols understood by Yahoo! Finance.
    start, end:
        Date strings parsable by ``pandas.Timestamp`` (e.g. "2015-01-01").

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by date. Columns are flattened to the pattern
        ``{ticker}_{field}`` (e.g. ``AAPL_Close``) for consistency across
        single- and multi-ticker downloads.
    """

    logger.info("Fetching data for tickers: %s", ", ".join(tickers))
    data = yf.download(
        tickers=list(tickers),
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        actions=False,
    )

    if data.empty:
        raise ValueError("No data returned from Yahoo Finance. Check tickers and date range.")

    if isinstance(data.columns, pd.MultiIndex):
        # Swap levels so tickers come first, flatten, and normalise column names.
        data = data.swaplevel(0, 1, axis=1).sort_index(axis=1, level=0)
        flattened_cols = []
        for ticker, field in data.columns:
            sanitized_field = field.replace(" ", "")
            flattened_cols.append(f"{ticker}_{sanitized_field}")
        data.columns = flattened_cols
    else:
        data.columns = [f"{tickers[0]}_{col.replace(' ', '')}" for col in data.columns]

    data.index = pd.to_datetime(data.index)
    logger.info("Fetched %d rows of data.", len(data))
    return data


# ---------------------------------------------------------------------------
# Technical indicator calculations
# ---------------------------------------------------------------------------

def compute_macd(price: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Compute the Moving Average Convergence Divergence (MACD) indicator.

    Returns a DataFrame with ``macd``, ``signal`` and ``hist`` columns."""

    ema_fast = price.ewm(span=fast, adjust=False).mean()
    ema_slow = price.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "hist": hist})


def compute_rsi(price: pd.Series, period: int = 14) -> pd.Series:
    """Compute the Relative Strength Index (RSI)."""

    delta = price.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_tsmom(price: pd.Series, window: int = 252) -> pd.Series:
    """Compute time-series momentum (trailing ``window`` day return)."""

    return price.pct_change(periods=window)


def compute_ofi(df: pd.DataFrame) -> pd.Series:
    """Compute order-flow imbalance (OFI) from available bid/ask volume data.

    OFI captures the imbalance between buying and selling pressure. When bid/ask
    volume columns are available, the formula is:
        OFI = (Δbid_volume - Δask_volume) * price_movement

    Where:
    - Δbid_volume = change in bid volume from previous period
    - Δask_volume = change in ask volume from previous period
    - price_movement = sign of close price change (1 for up, -1 for down, 0 for flat)

    If bid/ask volumes are not available, falls back to a volume-weighted return
    approximation using total volume and price changes.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with price and volume columns. Expected columns:
        - For full OFI: bid_volume, ask_volume, and close price
        - For fallback: volume and close price

    Returns
    -------
    pd.Series
        Order-flow imbalance series aligned to the input index.
        Positive values indicate buying pressure, negative values selling pressure.
    """
    result = pd.Series(0.0, index=df.index, name="ofi")

    # Find column names (handle ticker-prefixed columns)
    def find_col(patterns):
        for col in df.columns:
            col_lower = col.lower()
            for pattern in patterns:
                if pattern in col_lower:
                    return col
        return None

    bid_vol_col = find_col(["bid_volume", "bidvolume", "bid_vol"])
    ask_vol_col = find_col(["ask_volume", "askvolume", "ask_vol"])
    close_col = find_col(["adjclose", "close"])
    volume_col = find_col(["volume"])

    # Primary OFI calculation using bid/ask volumes
    if bid_vol_col is not None and ask_vol_col is not None:
        bid_vol = df[bid_vol_col].fillna(0).astype(float)
        ask_vol = df[ask_vol_col].fillna(0).astype(float)

        # Change in bid/ask volumes
        delta_bid = bid_vol.diff().fillna(0)
        delta_ask = ask_vol.diff().fillna(0)

        # Price direction
        if close_col is not None:
            price_change = df[close_col].diff().fillna(0)
            price_direction = np.sign(price_change)
        else:
            price_direction = 1

        # OFI = (Δbid - Δask) * price_direction
        ofi_raw = (delta_bid - delta_ask) * price_direction

        # Normalize by total volume change to make it comparable across assets.
        # Use small epsilon to avoid division by zero; this ensures stable results
        # when volume changes are zero (no activity).
        total_vol_change = delta_bid.abs() + delta_ask.abs()
        total_vol_change = total_vol_change.replace(0, 1e-10)  # Replace zeros with small epsilon
        ofi_normalized = ofi_raw / total_vol_change
        result = ofi_normalized.fillna(0)

    # Fallback: use volume-weighted return approximation
    elif volume_col is not None and close_col is not None:
        volume = df[volume_col].fillna(0).astype(float)
        close = df[close_col].astype(float)

        # Compute returns and volume change
        returns = close.pct_change().fillna(0)
        vol_change = volume.diff().fillna(0)

        # Approximate OFI: positive vol change + positive return = buying pressure
        # Negative vol change + negative return = selling pressure
        ofi_approx = returns * vol_change / volume.replace(0, np.nan)
        result = ofi_approx.fillna(0)

    result.name = "ofi"
    return result


def detect_head_and_shoulders(
    df: pd.DataFrame, window: int = 5, tolerance: float = 0.02
) -> pd.Series:
    """Detect head-and-shoulders pattern using peak/trough analysis.

    Algorithm:
    1. Identify local maxima (peaks) with a window of `window` bars.
    2. For each triplet of consecutive peaks, check if the middle peak (head)
       is higher than both shoulders.
    3. Require the head to be at least `tolerance` (2% default) higher than shoulders.
    4. Mark the bar at the right shoulder as a pattern signal (1 = detected, 0 = not).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with at least a close price column (AdjClose, Close, or close).
    window : int
        Number of bars on each side to identify local maxima. Default is 5.
    tolerance : float
        Minimum relative height difference required for head vs shoulders.
        Default is 0.02 (2%).

    Returns
    -------
    pd.Series
        Binary series (0 or 1) aligned to the input index, with 1 indicating
        a head-and-shoulders pattern was detected at that bar.
    """
    # Find the close price column
    close_col = None
    for col in df.columns:
        col_lower = col.lower()
        if "adjclose" in col_lower or col_lower == "close":
            close_col = col
            break
        if "close" in col_lower:
            close_col = col

    if close_col is None:
        # No close column found, return zeros
        return pd.Series(0, index=df.index, name="head_shoulders")

    close = df[close_col].values
    n = len(close)
    result = np.zeros(n, dtype=int)

    # Find local maxima (peaks): a point is a local max if it's higher than
    # `window` bars on each side.
    peaks = []
    for i in range(window, n - window):
        left_window = close[i - window : i]
        right_window = close[i + 1 : i + window + 1]
        if close[i] > left_window.max() and close[i] > right_window.max():
            peaks.append(i)

    # Check triplets of consecutive peaks for head-and-shoulders pattern
    for i in range(len(peaks) - 2):
        left_shoulder_idx = peaks[i]
        head_idx = peaks[i + 1]
        right_shoulder_idx = peaks[i + 2]

        left_shoulder = close[left_shoulder_idx]
        head = close[head_idx]
        right_shoulder = close[right_shoulder_idx]

        # Head must be higher than both shoulders by at least `tolerance`
        if (
            head > left_shoulder * (1 + tolerance)
            and head > right_shoulder * (1 + tolerance)
        ):
            # Shoulders should be at similar heights (within tolerance of each other)
            shoulder_diff = abs(left_shoulder - right_shoulder) / max(
                left_shoulder, right_shoulder
            )
            if shoulder_diff < tolerance * 2:
                # Mark the right shoulder as the pattern completion point
                result[right_shoulder_idx] = 1

    return pd.Series(result, index=df.index, name="head_shoulders")


def detect_double_top(
    df: pd.DataFrame, window: int = 5, tolerance: float = 0.02
) -> pd.Series:
    """Detect double-top/double-bottom patterns using peak analysis.

    Algorithm:
    1. Identify local maxima (peaks) with a window of `window` bars.
    2. For each pair of consecutive peaks, check if they are at similar heights
       (within `tolerance`).
    3. Require a trough between the peaks to confirm the pattern.
    4. Mark the bar at the second peak as a pattern signal (1 = double-top, -1 = double-bottom).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with at least a close price column (AdjClose, Close, or close).
    window : int
        Number of bars on each side to identify local maxima/minima. Default is 5.
    tolerance : float
        Maximum relative height difference between the two peaks/troughs.
        Default is 0.02 (2%).

    Returns
    -------
    pd.Series
        Series aligned to the input index with:
        - 1 for double-top pattern detected
        - -1 for double-bottom pattern detected
        - 0 for no pattern
    """
    # Find the close price column
    close_col = None
    for col in df.columns:
        col_lower = col.lower()
        if "adjclose" in col_lower or col_lower == "close":
            close_col = col
            break
        if "close" in col_lower:
            close_col = col

    if close_col is None:
        return pd.Series(0, index=df.index, name="double_top")

    close = df[close_col].values
    n = len(close)
    result = np.zeros(n, dtype=int)

    # Find local maxima (peaks) and minima (troughs)
    peaks = []
    troughs = []
    for i in range(window, n - window):
        left_window = close[i - window : i]
        right_window = close[i + 1 : i + window + 1]
        if close[i] > left_window.max() and close[i] > right_window.max():
            peaks.append(i)
        if close[i] < left_window.min() and close[i] < right_window.min():
            troughs.append(i)

    # Check pairs of consecutive peaks for double-top pattern
    for i in range(len(peaks) - 1):
        first_peak_idx = peaks[i]
        second_peak_idx = peaks[i + 1]

        first_peak = close[first_peak_idx]
        second_peak = close[second_peak_idx]

        # Peaks should be at similar heights (within tolerance)
        peak_diff = abs(first_peak - second_peak) / max(first_peak, second_peak)
        if peak_diff < tolerance:
            # Check for a trough between the peaks
            troughs_between = [
                t for t in troughs if first_peak_idx < t < second_peak_idx
            ]
            if troughs_between:
                trough_val = close[troughs_between[0]]
                # Trough should be notably lower than the peaks
                if trough_val < first_peak * (1 - tolerance):
                    result[second_peak_idx] = 1

    # Check pairs of consecutive troughs for double-bottom pattern
    for i in range(len(troughs) - 1):
        first_trough_idx = troughs[i]
        second_trough_idx = troughs[i + 1]

        first_trough = close[first_trough_idx]
        second_trough = close[second_trough_idx]

        # Troughs should be at similar heights (within tolerance)
        trough_diff = abs(first_trough - second_trough) / max(
            first_trough, second_trough
        )
        if trough_diff < tolerance:
            # Check for a peak between the troughs
            peaks_between = [
                p for p in peaks if first_trough_idx < p < second_trough_idx
            ]
            if peaks_between:
                peak_val = close[peaks_between[0]]
                # Peak should be notably higher than the troughs
                if peak_val > first_trough * (1 + tolerance):
                    result[second_trough_idx] = -1

    return pd.Series(result, index=df.index, name="double_top")


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _infer_tickers(df: pd.DataFrame) -> List[str]:
    prefixes = sorted({col.split("_")[0] for col in df.columns})
    return prefixes


def build_features(df: pd.DataFrame, scaler: Optional[StandardScaler] = None) -> Tuple[pd.DataFrame, List[str], StandardScaler]:
    """Construct engineered features for each ticker in ``df``.

    Parameters
    ----------
    df:
        DataFrame of OHLCV columns flattened as ``{ticker}_{field}``.
    scaler:
        Optional pre-fitted ``StandardScaler``. If provided, reuse it to transform
        features. Otherwise, fit a new scaler on the numeric features.

    Returns
    -------
    Tuple[pandas.DataFrame, List[str], StandardScaler]
        * Feature-enhanced DataFrame with scaled numeric predictors.
        * List of target column names (next-day returns per ticker).
        * The ``StandardScaler`` used (fitted).
    """

    features = df.copy()
    tickers = _infer_tickers(features)

    target_cols: List[str] = []

    for ticker in tickers:
        close_col = f"{ticker}_AdjClose" if f"{ticker}_AdjClose" in features.columns else f"{ticker}_Close"
        volume_col = f"{ticker}_Volume"

        if close_col not in features.columns:
            logger.warning("Ticker %s is missing a close price column; skipping indicator computation.", ticker)
            continue

        price_series = features[close_col]
        returns_1d = price_series.pct_change()
        features[f"{ticker}_return_1d"] = returns_1d
        features[f"{ticker}_return_5d"] = price_series.pct_change(periods=5)

        macd = compute_macd(price_series)
        features[f"{ticker}_macd"] = macd["macd"]
        features[f"{ticker}_macd_signal"] = macd["signal"]
        features[f"{ticker}_macd_hist"] = macd["hist"]

        features[f"{ticker}_rsi"] = compute_rsi(price_series)
        features[f"{ticker}_tsmom"] = compute_tsmom(price_series)

        ofi = compute_ofi(features[[col for col in features.columns if col.startswith(ticker)]])
        features[f"{ticker}_ofi"] = ofi

        features[f"{ticker}_head_shoulders"] = detect_head_and_shoulders(features[[close_col]])
        features[f"{ticker}_double_top"] = detect_double_top(features[[close_col]])

        if volume_col in features.columns:
            features[f"{ticker}_volume_change"] = features[volume_col].pct_change()

        target_col = f"{ticker}_target_return"
        features[target_col] = returns_1d.shift(-1)
        target_cols.append(target_col)

    # Drop rows with NaNs originating from indicator/return calculations.
    features = features.dropna()

    # Identify feature columns (exclude targets).
    feature_cols = [col for col in features.columns if col not in target_cols]

    if not feature_cols:
        raise ValueError("No feature columns detected after engineering.")

    scaler = scaler or StandardScaler()
    features[feature_cols] = scaler.fit_transform(features[feature_cols].values)

    return features, target_cols, scaler


# ---------------------------------------------------------------------------
# Walk-forward validation helpers
# ---------------------------------------------------------------------------

def train_test_splits(data: pd.DataFrame, n_splits: int = 5, min_train_size: Optional[int] = None) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Generate walk-forward validation splits.

    Parameters
    ----------
    data:
        Feature-engineered DataFrame sorted by date index.
    n_splits:
        Number of walk-forward splits.
    min_train_size:
        Optional minimum number of observations for the initial training window.

    Returns
    -------
    List[Tuple[pandas.DataFrame, pandas.DataFrame]]
        List of (train_df, test_df) tuples.
    """

    if data.index.is_monotonic_increasing is False:
        data = data.sort_index()

    n_samples = len(data)
    if n_samples < n_splits + 1:
        raise ValueError("Not enough samples to create walk-forward splits.")

    min_train_size = min_train_size or max(60, n_samples // (n_splits + 1))
    split_points = np.linspace(min_train_size, n_samples, num=n_splits + 1, dtype=int)

    splits: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
    start_idx = 0
    for i in range(n_splits):
        train_end = split_points[i]
        test_end = split_points[i + 1]
        train_df = data.iloc[start_idx:train_end]
        test_df = data.iloc[train_end:test_end]
        if len(test_df) == 0:
            continue
        splits.append((train_df, test_df))

    return splits


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def compute_sharpe_ratio(returns: np.ndarray, trading_days: int = 252) -> float:
    """Compute the annualised Sharpe ratio with zero risk-free rate."""

    returns = np.asarray(returns)
    if returns.size == 0 or np.std(returns) == 0:
        return float("nan")
    return math.sqrt(trading_days) * (np.mean(returns) / np.std(returns))


def compute_sortino_ratio(returns: np.ndarray, trading_days: int = 252) -> float:
    """Compute the annualised Sortino ratio for a return series."""

    returns = np.asarray(returns)
    downside = returns[returns < 0]
    if returns.size == 0 or downside.size == 0:
        return float("nan")
    downside_std = downside.std(ddof=0)
    if downside_std == 0:
        return float("nan")
    return math.sqrt(trading_days) * (np.mean(returns) / downside_std)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Return standard regression metrics and the Sharpe ratio of predictions."""

    metrics = {
        "rmse": math.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "sharpe": compute_sharpe_ratio(y_pred),
    }
    return metrics


def print_iteration_results(iteration_name: str, results: Dict[str, List[Dict[str, float]]]) -> None:
    """Pretty-print metrics per split and aggregate averages for each target."""

    print(f"\n=== {iteration_name} ===")
    for target, metrics_list in results.items():
        if not metrics_list:
            print(f"Target: {target} -> No results (insufficient data).")
            continue
        print(f"Target: {target}")
        for idx, metrics in enumerate(metrics_list, start=1):
            metric_str = ", ".join(f"{key.upper()}: {value:.4f}" for key, value in metrics.items())
            print(f"  Split {idx}: {metric_str}")
        averages = {key: np.nanmean([m[key] for m in metrics_list]) for key in metrics_list[0]}
        avg_str = ", ".join(f"{key.upper()}: {value:.4f}" for key, value in averages.items())
        print(f"  -> Average: {avg_str}")


# ---------------------------------------------------------------------------
# Iteration 1 – Linear Regression Baseline
# ---------------------------------------------------------------------------

def run_linear_regression_iteration(
    data: pd.DataFrame,
    feature_cols: Sequence[str],
    target_cols: Sequence[str],
    splits: Sequence[Tuple[pd.DataFrame, pd.DataFrame]],
) -> Dict[str, List[Dict[str, float]]]:
    """Evaluate a linear regression baseline across walk-forward splits."""

    results: Dict[str, List[Dict[str, float]]] = {target: [] for target in target_cols}

    for split_idx, (train_df, test_df) in enumerate(splits, start=1):
        model = LinearRegression()
        X_train = train_df[feature_cols]
        X_test = test_df[feature_cols]
        y_train = train_df[target_cols]
        y_test = test_df[target_cols]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        for i, target in enumerate(target_cols):
            metrics = evaluate_predictions(y_test.iloc[:, i].values, y_pred[:, i])
            results[target].append(metrics)
        logger.info("Linear Regression split %d complete.", split_idx)

    print_iteration_results("Iteration 1 – Linear Regression", results)
    return results


# ---------------------------------------------------------------------------
# Iteration 2 – Random Forest Regression
# ---------------------------------------------------------------------------

def run_random_forest_iteration(
    data: pd.DataFrame,
    feature_cols: Sequence[str],
    target_cols: Sequence[str],
    splits: Sequence[Tuple[pd.DataFrame, pd.DataFrame]],
    param_grid: Optional[Dict[str, Iterable]] = None,
    cv: int = 3,
) -> Dict[str, List[Dict[str, float]]]:
    """Evaluate a RandomForestRegressor using grid search cross-validation.

    This expanded tuning plan explores a comprehensive hyperparameter space
    based on a study suggesting grid search cross-validation with:
    - max_depth: {2, 4, 6, 8, 10} for controlling tree complexity
    - n_estimators: {64, 128, 256} for ensemble size
    - max_features: {'sqrt', 'log2', 0.8} for feature selection at each split
    - min_samples_leaf: {1, 2, 3, 4, 5} for regularization

    Parameters
    ----------
    data:
        DataFrame with features and targets.
    feature_cols:
        Column names for features.
    target_cols:
        Column names for targets.
    splits:
        List of (train_df, test_df) tuples from walk-forward validation.
    param_grid:
        Optional custom parameter grid. If None, uses the expanded default grid.
    cv:
        Number of cross-validation folds (default: 3).

    Returns
    -------
    dict:
        Mapping of target column names to lists of metrics per split.
    """

    if param_grid is None:
        param_grid = {
            "n_estimators": [64, 128, 256],
            "max_depth": [2, 4, 6, 8, 10],
            "max_features": ["sqrt", "log2", 0.8],
            "min_samples_leaf": [1, 2, 3, 4, 5],
        }

    results: Dict[str, List[Dict[str, float]]] = {target: [] for target in target_cols}

    for split_idx, (train_df, test_df) in enumerate(splits, start=1):
        X_train = train_df[feature_cols]
        X_test = test_df[feature_cols]
        y_train = train_df[target_cols]
        y_test = test_df[target_cols]

        base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            refit=True,
        )
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_preds = best_model.predict(X_test)

        for i, target in enumerate(target_cols):
            metrics = evaluate_predictions(y_test.iloc[:, i].values, best_preds[:, i])
            results[target].append(metrics)

        logger.info(
            "Random Forest split %d complete with best params: %s (CV score: %.4f)",
            split_idx,
            grid_search.best_params_,
            -grid_search.best_score_,
        )

    print_iteration_results("Iteration 2 – Random Forest", results)
    return results


# ---------------------------------------------------------------------------
# Sequence utilities for deep learning iterations
# ---------------------------------------------------------------------------

def create_sequences(
    data: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    seq_len: int = 60,
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform tabular data into sequences for sequence models."""

    X, y = [], []
    for idx in range(len(data) - seq_len):
        window = data.iloc[idx : idx + seq_len]
        target_value = data.iloc[idx + seq_len][target_col]
        X.append(window[feature_cols].values)
        y.append(target_value)
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


def prepare_lstm_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    seq_len: int = 60,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create train/test sequences with lookback ``seq_len``."""

    X_train, y_train = create_sequences(train_df, feature_cols, target_col, seq_len)
    combined = pd.concat([train_df.tail(seq_len), test_df])
    X_test, y_test = create_sequences(combined, feature_cols, target_col, seq_len)
    return X_train, y_train, X_test, y_test


def build_lstm(
    input_shape: Tuple[int, int],
    n_layers: int = 2,
    units_per_layer: Optional[List[int]] = None,
    classification: bool = False,
    dropout_rate: float = 0.2,
) -> keras.Model:
    """Create an LSTM-based model for regression or classification.

    Parameters
    ----------
    input_shape : Tuple[int, int]
        Shape of input (seq_len, num_features).
    n_layers : int
        Number of LSTM layers. Default is 2.
    units_per_layer : List[int], optional
        Number of units for each LSTM layer. If None, defaults to [64, 32]
        for 2 layers, or geometrically decreasing units for more layers.
    classification : bool
        If True, compile for binary classification with sigmoid output.
        If False, compile for regression with linear output. Default is False.
    dropout_rate : float
        Dropout rate between LSTM layers. Default is 0.2.

    Returns
    -------
    keras.Model
        Compiled LSTM model.
    """
    if units_per_layer is None:
        # Default units: start at 64 and halve for each subsequent layer
        units_per_layer = [max(8, 64 // (2 ** i)) for i in range(n_layers)]

    if len(units_per_layer) != n_layers:
        raise ValueError(
            f"units_per_layer length ({len(units_per_layer)}) must match n_layers ({n_layers})"
        )

    model = keras.Sequential()

    for i, units in enumerate(units_per_layer):
        return_sequences = i < n_layers - 1  # All but last layer return sequences
        if i == 0:
            model.add(
                keras.layers.LSTM(
                    units, return_sequences=return_sequences, input_shape=input_shape
                )
            )
        else:
            model.add(keras.layers.LSTM(units, return_sequences=return_sequences))

        if i < n_layers - 1:  # Add dropout between LSTM layers
            model.add(keras.layers.Dropout(dropout_rate))

    # Output layer
    if classification:
        model.add(keras.layers.Dense(1, activation="sigmoid"))
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    else:
        model.add(keras.layers.Dense(1))
        model.compile(optimizer="adam", loss="mse")

    return model


def build_transformer(
    num_features: int,
    seq_len: int,
    d_model: int = 64,
    num_heads: int = 4,
    ff_dim: int = 128,
    num_layers: int = 1,
    classification: bool = False,
    dropout_rate: float = 0.1,
) -> keras.Model:
    """Build a transformer encoder model for regression or classification.

    Parameters
    ----------
    num_features : int
        Number of input features.
    seq_len : int
        Sequence length.
    d_model : int
        Dimension of the model (key dimension for attention). Default is 64.
    num_heads : int
        Number of attention heads. Default is 4.
    ff_dim : int
        Dimension of the feed-forward network. Default is 128.
    num_layers : int
        Number of transformer encoder layers. Default is 1.
    classification : bool
        If True, compile for binary classification with sigmoid output.
        If False, compile for regression with linear output. Default is False.
    dropout_rate : float
        Dropout rate for regularization. Default is 0.1.

    Returns
    -------
    keras.Model
        Compiled transformer model.
    """
    inputs = keras.Input(shape=(seq_len, num_features))
    x = inputs

    # Optional: project to d_model if num_features != d_model
    if num_features != d_model:
        x = keras.layers.Dense(d_model)(x)

    # Stack transformer encoder layers
    for _ in range(num_layers):
        # Multi-head self-attention
        attn_out = keras.layers.MultiHeadAttention(num_heads, key_dim=d_model)(x, x)
        attn_out = keras.layers.Dropout(dropout_rate)(attn_out)
        x = keras.layers.Add()([x, attn_out])
        x = keras.layers.LayerNormalization(epsilon=1e-6)(x)

        # Feed-forward network
        ff_out = keras.layers.Dense(ff_dim, activation="relu")(x)
        ff_out = keras.layers.Dense(d_model)(ff_out)
        ff_out = keras.layers.Dropout(dropout_rate)(ff_out)
        x = keras.layers.Add()([x, ff_out])
        x = keras.layers.LayerNormalization(epsilon=1e-6)(x)

    # Output processing
    flat = keras.layers.Flatten()(x)

    if classification:
        outputs = keras.layers.Dense(1, activation="sigmoid")(flat)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    else:
        outputs = keras.layers.Dense(1)(flat)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="adam", loss="mse")

    return model


# ---------------------------------------------------------------------------
# Iteration 3 – LSTM
# ---------------------------------------------------------------------------

def run_lstm_iteration(
    feature_cols: Sequence[str],
    target_cols: Sequence[str],
    splits: Sequence[Tuple[pd.DataFrame, pd.DataFrame]],
    seq_len: int = 60,
    epochs: int = 10,
    batch_size: int = 32,
) -> Dict[str, List[Dict[str, float]]]:
    """Train and evaluate an LSTM per target across walk-forward splits."""

    results: Dict[str, List[Dict[str, float]]] = {target: [] for target in target_cols}
    callbacks = [keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]

    for target in target_cols:
        for split_idx, (train_df, test_df) in enumerate(splits, start=1):
            X_train, y_train, X_test, y_test = prepare_lstm_data(
                train_df, test_df, feature_cols, target, seq_len
            )
            if len(X_train) == 0 or len(X_test) == 0:
                logger.warning("Not enough data to create sequences for target %s split %d.", target, split_idx)
                continue

            model = build_lstm((seq_len, len(feature_cols)))
            model.fit(
                X_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
                callbacks=callbacks,
            )
            preds = model.predict(X_test, verbose=0).flatten()
            metrics = evaluate_predictions(y_test, preds)
            results[target].append(metrics)
            logger.info("LSTM target %s split %d complete.", target, split_idx)

    print_iteration_results("Iteration 3 – LSTM", results)
    return results


# ---------------------------------------------------------------------------
# Iteration 4 – Transformer Encoder
# ---------------------------------------------------------------------------

def run_transformer_iteration(
    feature_cols: Sequence[str],
    target_cols: Sequence[str],
    splits: Sequence[Tuple[pd.DataFrame, pd.DataFrame]],
    seq_len: int = 60,
    epochs: int = 10,
    batch_size: int = 32,
) -> Dict[str, List[Dict[str, float]]]:
    """Train and evaluate a transformer encoder per target."""

    results: Dict[str, List[Dict[str, float]]] = {target: [] for target in target_cols}
    callbacks = [keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]

    for target in target_cols:
        for split_idx, (train_df, test_df) in enumerate(splits, start=1):
            X_train, y_train, X_test, y_test = prepare_lstm_data(
                train_df, test_df, feature_cols, target, seq_len
            )
            if len(X_train) == 0 or len(X_test) == 0:
                logger.warning(
                    "Not enough data to create transformer sequences for target %s split %d.",
                    target,
                    split_idx,
                )
                continue

            model = build_transformer(num_features=len(feature_cols), seq_len=seq_len)
            model.fit(
                X_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
                callbacks=callbacks,
            )
            preds = model.predict(X_test, verbose=0).flatten()
            metrics = evaluate_predictions(y_test, preds)
            results[target].append(metrics)
            logger.info("Transformer target %s split %d complete.", target, split_idx)

    print_iteration_results("Iteration 4 – Transformer", results)
    return results


# ---------------------------------------------------------------------------
# Iteration 5 – Ensemble & Dynamic Position Sizing
# ---------------------------------------------------------------------------

def ensemble_predictions(preds_a: np.ndarray, preds_b: np.ndarray, weight: float = 0.5) -> np.ndarray:
    """Weighted average ensemble of two prediction arrays."""

    return weight * preds_a + (1 - weight) * preds_b


def dynamic_position_sizing(predictions: pd.Series, window: int = 20) -> pd.Series:
    """Scale position sizes by prediction confidence.

    Positions are computed as ``sign(prediction) * min(|prediction| / rolling_std, 1)``.
    """

    rolling_std = predictions.rolling(window=window, min_periods=1).std().replace(0, np.nan)
    confidence = (predictions.abs() / rolling_std).clip(upper=1.0).fillna(0.0)
    positions = predictions.apply(np.sign) * confidence
    return positions.clip(-1.0, 1.0)


def run_ensemble_iteration(
    feature_cols: Sequence[str],
    target_cols: Sequence[str],
    splits: Sequence[Tuple[pd.DataFrame, pd.DataFrame]],
    seq_len: int = 60,
    epochs: int = 10,
    batch_size: int = 32,
    weight: float = 0.5,
) -> Dict[str, List[Dict[str, float]]]:
    """Combine LSTM and Transformer forecasts, evaluate trading metrics."""

    results: Dict[str, List[Dict[str, float]]] = {target: [] for target in target_cols}
    callbacks = [keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]

    for target in target_cols:
        for split_idx, (train_df, test_df) in enumerate(splits, start=1):
            X_train, y_train, X_test, y_test = prepare_lstm_data(
                train_df, test_df, feature_cols, target, seq_len
            )
            if len(X_train) == 0 or len(X_test) == 0:
                logger.warning("Not enough data for ensemble target %s split %d.", target, split_idx)
                continue

            lstm_model = build_lstm((seq_len, len(feature_cols)))
            transformer_model = build_transformer(num_features=len(feature_cols), seq_len=seq_len)

            lstm_model.fit(
                X_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
                callbacks=callbacks,
            )
            transformer_model.fit(
                X_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
                callbacks=callbacks,
            )

            lstm_preds = lstm_model.predict(X_test, verbose=0).flatten()
            transformer_preds = transformer_model.predict(X_test, verbose=0).flatten()
            blended_preds = ensemble_predictions(lstm_preds, transformer_preds, weight)

            metrics = evaluate_predictions(y_test, blended_preds)

            # Dynamic position sizing using blended predictions.
            prediction_series = pd.Series(blended_preds, index=test_df.index[-len(blended_preds):])
            positions = dynamic_position_sizing(prediction_series)
            actual_returns = pd.Series(y_test, index=prediction_series.index)
            portfolio_returns = positions * actual_returns
            metrics["strategy_sharpe"] = compute_sharpe_ratio(portfolio_returns.values)
            metrics["strategy_sortino"] = compute_sortino_ratio(portfolio_returns.values)

            results[target].append(metrics)
            logger.info("Ensemble target %s split %d complete.", target, split_idx)

    print_iteration_results("Iteration 5 – Ensemble", results)
    return results


# ---------------------------------------------------------------------------
# Extension placeholders
# ---------------------------------------------------------------------------

def plan_vwap_execution(
    order_volume: float,
    intervals: int = 6,
    volume_profile: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """Create a VWAP (Volume-Weighted Average Price) execution schedule.

    Distributes the total order volume across intervals proportionally to a
    given volume profile. If no profile is provided, defaults to equal weights
    (equivalent to TWAP).

    Parameters
    ----------
    order_volume : float
        Total volume to be executed across all intervals.
    intervals : int
        Number of execution intervals. Default is 6.
    volume_profile : Sequence[float], optional
        Relative volume weights for each interval (e.g., intraday volume curve).
        Length must match `intervals`. Values are normalized to sum to 1.
        If None, uses equal weights (TWAP behavior).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - interval: 1-indexed interval number
        - allocated_volume: volume allocated to each interval
        - weight: normalized weight for each interval
    """
    if volume_profile is None:
        # Equal weights (TWAP behavior)
        weights = np.ones(intervals) / intervals
    else:
        profile = np.asarray(volume_profile, dtype=float)
        if len(profile) != intervals:
            raise ValueError(
                f"volume_profile length ({len(profile)}) must match intervals ({intervals})"
            )
        if profile.sum() == 0:
            # Fallback to equal weights if all zeros
            weights = np.ones(intervals) / intervals
        else:
            # Normalize to sum to 1
            weights = profile / profile.sum()

    allocated_volumes = weights * order_volume

    schedule = pd.DataFrame(
        {
            "interval": range(1, intervals + 1),
            "allocated_volume": allocated_volumes,
            "weight": weights,
        }
    )
    return schedule


def plan_twap_execution(order_volume: float, intervals: int = 6) -> pd.DataFrame:
    """Create a TWAP (Time-Weighted Average Price) execution schedule.

    Distributes the total order volume equally across all intervals.
    This is equivalent to calling plan_vwap_execution with no volume profile.

    Parameters
    ----------
    order_volume : float
        Total volume to be executed across all intervals.
    intervals : int
        Number of execution intervals. Default is 6.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - interval: 1-indexed interval number
        - allocated_volume: volume allocated to each interval (equal for TWAP)
        - weight: normalized weight for each interval (1/intervals for TWAP)
    """
    return plan_vwap_execution(order_volume, intervals, volume_profile=None)


def find_cointegrated_pairs(price_data: pd.DataFrame) -> pd.DataFrame:
    """Placeholder for pairs-trading cointegration analysis."""

    # TODO: Integrate statsmodels.tsa.stattools.coint for real pair identification.
    return pd.DataFrame(columns=["asset_a", "asset_b", "pvalue"])


def scalping_hft_placeholder() -> None:
    """Placeholder for HFT integration."""

    # TODO: Integrate high-frequency data sources and event-driven backtesting.
    pass


def sentiment_analysis_placeholder(news_df: pd.DataFrame) -> pd.Series:
    """Placeholder sentiment scoring function."""

    # TODO: Replace with NLP model scoring of text headlines or tweets.
    return pd.Series(np.nan, index=news_df.index if not news_df.empty else [])


def reinforcement_learning_placeholder() -> None:
    """Skeleton for deep reinforcement learning integration."""

    # TODO: Integrate stable_baselines3 for policy optimisation over trading environment.
    pass


# ---------------------------------------------------------------------------
# Reinforcement Learning CLI
# ---------------------------------------------------------------------------

def train_rl_trading_agent(
    tickers: Sequence[str] = ("AAPL",),
    start: str = "2015-01-01",
    end: str = "2023-12-31",
    algorithm: str = "PPO",
    total_timesteps: int = 10000,
    window_size: int = 60,
    transaction_cost: float = 0.0001,
    output_path: Optional[str] = None,
) -> None:
    """Train an RL trading agent on historical data.

    This function downloads data, builds features, creates a trading environment,
    and trains a reinforcement learning agent using PPO or A2C algorithms.

    Parameters
    ----------
    tickers : Sequence[str], default=("AAPL",)
        Ticker symbols to train on.
    start : str, default="2015-01-01"
        Start date for training data.
    end : str, default="2023-12-31"
        End date for training data.
    algorithm : str, default="PPO"
        RL algorithm to use: "PPO" or "A2C".
    total_timesteps : int, default=10000
        Total timesteps for training.
    window_size : int, default=60
        Observation window size for the environment.
    transaction_cost : float, default=0.0001
        Transaction cost per trade (as fraction).
    output_path : str, optional
        Path to save the trained model. If None, model is not saved.

    Example
    -------
    >>> train_rl_trading_agent(
    ...     tickers=["AAPL"],
    ...     start="2018-01-01",
    ...     end="2023-12-31",
    ...     algorithm="PPO",
    ...     total_timesteps=50000,
    ... )
    """
    try:
        from src.advanced.reinforcement_learning import (
            TradingEnv,
            TradingEnvConfig,
            train_rl_agent,
            evaluate_agent,
        )
    except ImportError as e:
        logger.error(
            "Failed to import RL modules. Ensure stable-baselines3 and gymnasium "
            f"are installed: {e}"
        )
        return

    logger.info(f"Downloading data for {tickers} from {start} to {end}...")
    raw_data = fetch_data(list(tickers), start=start, end=end)

    logger.info("Building features...")
    features_df, target_cols, scaler = build_features(raw_data)

    # Prepare data for RL environment
    # Use close prices as the price series
    close_cols = [col for col in features_df.columns if "close" in col.lower()]
    if not close_cols:
        logger.error("No close price column found in features")
        return

    prices = features_df[close_cols[0]].dropna().values

    # Use other features as observation features
    feature_cols = [
        col for col in features_df.columns
        if col not in target_cols and "close" not in col.lower()
    ]
    obs_features = features_df[feature_cols].fillna(0).values

    # Align lengths
    min_len = min(len(prices), len(obs_features))
    prices = prices[-min_len:]
    obs_features = obs_features[-min_len:]

    logger.info(f"Data prepared: {len(prices)} samples, {obs_features.shape[1]} features")

    # Create environment
    config = TradingEnvConfig(
        window_size=window_size,
        transaction_cost=transaction_cost,
    )
    env = TradingEnv(prices=prices, features=obs_features, config=config)

    logger.info(f"Training {algorithm} agent for {total_timesteps} timesteps...")
    model = train_rl_agent(
        env,
        algorithm=algorithm,
        total_timesteps=total_timesteps,
        verbose=1,
    )

    # Evaluate the trained model
    logger.info("Evaluating trained agent...")
    metrics = evaluate_agent(model, env, n_episodes=5)

    logger.info("=" * 50)
    logger.info("Training Results:")
    logger.info(f"  Mean Return: {metrics['mean_return']:.4f}")
    logger.info(f"  Std Return:  {metrics['std_return']:.4f}")
    logger.info(f"  Sharpe Ratio: {metrics['mean_sharpe']:.4f}")
    logger.info(f"  Max Drawdown: {metrics['max_drawdown']:.4f}")
    logger.info("=" * 50)

    # Save model if path provided
    if output_path:
        model.save(output_path)
        logger.info(f"Model saved to {output_path}")


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

def main() -> None:
    tickers = ["AAPL", "TSLA"]
    start_date = "2015-01-01"
    end_date = "2023-12-31"

    raw_data = fetch_data(tickers, start=start_date, end=end_date)
    features, target_cols, scaler = build_features(raw_data)
    feature_cols = [col for col in features.columns if col not in target_cols]
    splits = train_test_splits(features, n_splits=3)

    # Iteration 1 & 2 operate directly on tabular data.
    run_linear_regression_iteration(features, feature_cols, target_cols, splits)
    run_random_forest_iteration(features, feature_cols, target_cols, splits)

    # Iteration 3 onwards (deep learning) can be compute-intensive; limit epochs in example.
    run_lstm_iteration(feature_cols, target_cols, splits, seq_len=60, epochs=5, batch_size=32)
    run_transformer_iteration(feature_cols, target_cols, splits, seq_len=60, epochs=5, batch_size=32)
    run_ensemble_iteration(feature_cols, target_cols, splits, seq_len=60, epochs=5, batch_size=32)


def cli_main() -> None:
    """CLI entry point with subcommands for different operations."""
    parser = argparse.ArgumentParser(
        description="Market Forecasting Module - ML models and RL trading agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run traditional ML iterations
  python market_forecasting.py run

  # Train RL trading agent
  python market_forecasting.py train-rl --tickers AAPL TSLA --timesteps 50000

  # Train RL agent with A2C algorithm
  python market_forecasting.py train-rl --algorithm A2C --timesteps 100000
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command (default behavior)
    run_parser = subparsers.add_parser("run", help="Run traditional ML iterations")
    run_parser.add_argument(
        "--tickers", nargs="+", default=["AAPL", "TSLA"],
        help="Ticker symbols to analyze"
    )

    # Train RL command
    rl_parser = subparsers.add_parser("train-rl", help="Train RL trading agent")
    rl_parser.add_argument(
        "--tickers", nargs="+", default=["AAPL"],
        help="Ticker symbols to train on (default: AAPL)"
    )
    rl_parser.add_argument(
        "--start", default="2015-01-01",
        help="Start date (default: 2015-01-01)"
    )
    rl_parser.add_argument(
        "--end", default="2023-12-31",
        help="End date (default: 2023-12-31)"
    )
    rl_parser.add_argument(
        "--algorithm", choices=["PPO", "A2C"], default="PPO",
        help="RL algorithm to use (default: PPO)"
    )
    rl_parser.add_argument(
        "--timesteps", type=int, default=10000,
        help="Total timesteps for training (default: 10000)"
    )
    rl_parser.add_argument(
        "--window-size", type=int, default=60,
        help="Observation window size (default: 60)"
    )
    rl_parser.add_argument(
        "--transaction-cost", type=float, default=0.0001,
        help="Transaction cost per trade (default: 0.0001)"
    )
    rl_parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save trained model"
    )

    args = parser.parse_args()

    if args.command == "train-rl":
        train_rl_trading_agent(
            tickers=args.tickers,
            start=args.start,
            end=args.end,
            algorithm=args.algorithm,
            total_timesteps=args.timesteps,
            window_size=args.window_size,
            transaction_cost=args.transaction_cost,
            output_path=args.output,
        )
    elif args.command == "run" or args.command is None:
        main()
    else:
        parser.print_help()


if __name__ == "__main__":
    cli_main()
