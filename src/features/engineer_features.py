"""Feature engineering for market forecasting models.

Generates technical indicators, regime labels, and placeholder microstructure
features required by subsequent model iterations. The functions operate on the
aligned dataset produced by :mod:`src.data.align_data` and persist a modelling
ready dataframe containing feature columns and supervised learning targets.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from .target import compute_targets
from ..advanced.pattern_recognition import (
    moving_average_crossovers,
    swing_high_low_flags,
    flag_liquidity_grab,
    detect_fvg,
    asia_session_range_breakout,
)
from ..utils import PROCESSED_DIR

PROCESSED_DIR = Path("data/processed")
LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Technical indicator helpers
# ---------------------------------------------------------------------------

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_series = 100 - (100 / (1 + rs))
    return rsi_series


def rolling_volatility(returns: pd.Series, window: int = 21) -> pd.Series:
    return returns.rolling(window).std()


def rolling_volume_zscore(volume: pd.Series, window: int = 63) -> pd.Series:
    mean = volume.rolling(window).mean()
    std = volume.rolling(window).std()
    return (volume - mean) / std.replace(0, np.nan)


def time_series_momentum(close: pd.Series, lookback: int = 252) -> pd.Series:
    return close.pct_change(periods=lookback)


def swing_points(high: pd.Series, low: pd.Series, window: int = 3) -> Tuple[pd.Series, pd.Series]:
    swing_high = high[(high.shift(window) < high) & (high.shift(-window) < high)].astype(float)
    swing_low = low[(low.shift(window) > low) & (low.shift(-window) > low)].astype(float)
    return swing_high, swing_low


def detect_chart_patterns_stub() -> dict:
    """Return zeros as placeholders for chart pattern detection.

    .. deprecated::
        This stub is maintained for backward compatibility. Use
        :func:`detect_chart_patterns` instead for actual pattern detection.
    """

    return {
        "pattern_head_shoulders": 0,
        "pattern_double_top": 0,
        "pattern_double_bottom": 0,
    }


def detect_chart_patterns(
    prices: pd.DataFrame, window: int = 5, tolerance: float = 0.02
) -> dict:
    """Detect chart patterns and return binary flags for each pattern type.

    Uses rule-based algorithms to detect head-and-shoulders and double-top/
    double-bottom patterns in price data.

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame with at least a 'close' column and index aligned to dates.
    window : int
        Number of bars on each side to identify local maxima/minima. Default is 5.
    tolerance : float
        Height threshold for pattern detection. Default is 0.02 (2%).

    Returns
    -------
    dict
        Dictionary mapping pattern names to pd.Series aligned to the input index:
        - pattern_head_shoulders: 1 if detected, 0 otherwise
        - pattern_double_top: 1 if double-top detected, 0 otherwise
        - pattern_double_bottom: 1 if double-bottom detected, 0 otherwise
    """
    from ..advanced.pattern_recognition import detect_head_and_shoulders, detect_double_top

    head_shoulders = detect_head_and_shoulders(prices, window=window, tolerance=tolerance)
    double_pattern = detect_double_top(prices, window=window, tolerance=tolerance)

    # Convert double_top series: 1 = double-top, -1 = double-bottom
    double_top = (double_pattern == 1).astype(int)
    double_bottom = (double_pattern == -1).astype(int)

    return {
        "pattern_head_shoulders": head_shoulders,
        "pattern_double_top": double_top,
        "pattern_double_bottom": double_bottom,
    }


def ict_smt_asia_window_feature() -> float:
    """Placeholder feature for ICT/SMT Asia session concepts."""

    return 0.0


# ---------------------------------------------------------------------------
# Microstructure features
# ---------------------------------------------------------------------------

def order_flow_imbalance(bid_volume: pd.Series, ask_volume: pd.Series) -> pd.Series:
    denom = bid_volume + ask_volume
    with np.errstate(divide="ignore", invalid="ignore"):
        ofi = (bid_volume - ask_volume) / denom
    return ofi.replace([np.inf, -np.inf], np.nan)


def depth_ratio(bid_volume: pd.Series, ask_volume: pd.Series) -> pd.Series:
    denom = bid_volume + ask_volume
    return bid_volume / denom.replace(0, np.nan)


def bid_ask_spread_proxy(bid_price: pd.Series, ask_price: pd.Series, close: pd.Series) -> pd.Series:
    spread = ask_price - bid_price
    spread = spread.fillna(close * 0.0001)
    return spread


# ---------------------------------------------------------------------------
# Regime features
# ---------------------------------------------------------------------------

def realised_volatility_bucket(realised_vol: pd.Series) -> pd.Series:
    quantiles = realised_vol.quantile([0.33, 0.66]).values
    bins = [-np.inf, quantiles[0], quantiles[1], np.inf]
    labels = ["low", "medium", "high"]
    return pd.Categorical(pd.cut(realised_vol, bins=bins, labels=labels), categories=labels)


def rolling_drawdown(close: pd.Series) -> pd.Series:
    cumulative_max = close.cummax()
    drawdown = (close / cumulative_max) - 1.0
    return drawdown


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate features for each ticker and concatenate results."""

    feature_frames = []
    for ticker, group in df.groupby("ticker", as_index=False):
        group = group.sort_values("date").copy()
        group["return_1d"] = group["close"].pct_change()
        macd_line, signal_line = macd(group["close"])
        group["macd_line"] = macd_line
        group["macd_signal"] = signal_line
        group["macd_hist"] = group["macd_line"] - group["macd_signal"]
        group["rsi_14"] = rsi(group["close"])
        group["volatility_21"] = rolling_volatility(group["return_1d"], 21)
        # Handle NaN volume for FX pairs by filling with 0 before zscore calculation
        volume_series = group["volume"].fillna(0)
        group["volume_zscore_63"] = rolling_volume_zscore(volume_series, 63).fillna(0.0)
        group["tsmom_252"] = time_series_momentum(group["close"], 252)
        swing_high, swing_low = swing_points(group["high"], group["low"])
        # Reindex to group index to align sparse series, then fill NaNs.
        # We use ffill first (forward propagate last known swing), then bfill for
        # any remaining gaps at the start of the series, and finally fillna with
        # the actual high/low if still missing.
        swing_high = swing_high.reindex(group.index)
        swing_low = swing_low.reindex(group.index)
        group["swing_high"] = swing_high.ffill().bfill().fillna(group["high"])
        group["swing_low"] = swing_low.ffill().bfill().fillna(group["low"])

        bid_vol = group.get("bid_volume", pd.Series(0.0, index=group.index)).astype(float)
        ask_vol = group.get("ask_volume", pd.Series(0.0, index=group.index)).astype(float)
        group["ofi"] = order_flow_imbalance(bid_vol, ask_vol)
        group["depth_ratio"] = depth_ratio(bid_vol, ask_vol)
        group["bid_ask_spread"] = bid_ask_spread_proxy(group.get("bid_price"), group.get("ask_price"), group["close"])
        # Order-book placeholders often contain zeros only, which would otherwise
        # yield NaNs (division by zero) and break downstream scaling. Replace
        # missing microstructure fields with neutral zeros so the feature matrix
        # remains numeric until real depth data is integrated.
        group["ofi"] = group["ofi"].fillna(0.0)
        group["depth_ratio"] = group["depth_ratio"].fillna(0.0)
        group["bid_ask_spread"] = group["bid_ask_spread"].fillna(0.0)

        crossover_flags = moving_average_crossovers(group)
        for name, series in crossover_flags.items():
            group[name] = series

        swing_flags = swing_high_low_flags(group)
        for name, series in swing_flags.items():
            group[name] = series

        # Use the real pattern detection when enough data is available
        # (minimum 2 * window bars required for pattern detection)
        if len(group) > 20:
            pattern_flags = detect_chart_patterns(group, window=5, tolerance=0.02)
        else:
            # Fall back to stub for very small groups
            pattern_flags = detect_chart_patterns_stub()
        for name, value in pattern_flags.items():
            if isinstance(value, pd.Series):
                # Ensure Series is aligned to group index
                if len(value) == len(group):
                    group[name] = value.values
                else:
                    # Create zeros with correct length if Series length mismatch
                    group[name] = np.zeros(len(group), dtype=int)
            else:
                group[name] = value

        group["ict_smt_asia"] = ict_smt_asia_window_feature()
        group["sentiment_score"] = group.get("sentiment_score", 0.0)

        # Advanced pattern recognition features (wrapped for graceful degradation)
        try:
            group["liquidity_grab"] = flag_liquidity_grab(group)
        except (KeyError, ValueError, TypeError, AttributeError):
            group["liquidity_grab"] = pd.Series(0, index=group.index)

        try:
            group["fvg"] = detect_fvg(group)
        except (KeyError, ValueError, TypeError, AttributeError):
            group["fvg"] = pd.Series(0, index=group.index)

        try:
            group["asia_breakout"] = asia_session_range_breakout(group)
        except (KeyError, ValueError, TypeError, AttributeError):
            group["asia_breakout"] = pd.Series(0, index=group.index)

        group["realised_vol_bucket"] = realised_volatility_bucket(group["volatility_21"].bfill())
        group["drawdown"] = rolling_drawdown(group["close"])

        feature_frames.append(group)

    features = pd.concat(feature_frames, ignore_index=True)
    regime_dummies = pd.get_dummies(features["realised_vol_bucket"], prefix="regime")
    features = pd.concat([features, regime_dummies], axis=1)
    features = features.drop(columns=["realised_vol_bucket"])

    # Drop placeholder columns that are not useful as features (all NaN or constant)
    placeholder_cols_to_drop = ["bid_price", "ask_price", "bid_volume", "ask_volume"]
    features = features.drop(columns=[c for c in placeholder_cols_to_drop if c in features.columns], errors="ignore")

    # Fill remaining NaN values in numeric columns to ensure models can train
    # Volume is NaN for FX pairs - fill with 0
    features["volume"] = features["volume"].fillna(0)

    features.sort_values(["ticker", "date"], inplace=True)
    # Filter to rows after the warm-up period per ticker
    filtered_frames = []
    for _ticker, grp in features.groupby("ticker"):
        if len(grp) > 252:
            filtered_frames.append(grp.iloc[252:])
    features = pd.concat(filtered_frames, ignore_index=True) if filtered_frames else features

    targets = compute_targets(features)
    full_df = features.merge(targets, on=["ticker", "date"], how="inner")

    # Cast numeric columns to float32 to reduce memory footprint
    float_cols = full_df.select_dtypes(include=["float64"]).columns
    full_df[float_cols] = full_df[float_cols].astype("float32")

    return full_df


def save_features(df: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    LOGGER.info("Saved engineered features to %s", output_path)
    return output_path


def parse_args(cmd_args: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Engineer features for market forecasting.")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_DIR / "model_features.parquet",
        help="Path to output parquet file (default: data/processed/model_features.parquet)",
    )
    return parser.parse_args(cmd_args)


def main(cmd_args: Optional[Iterable[str]] = None) -> Path:
    logging.basicConfig(level="INFO")
    ns = parse_args(cmd_args)
    input_path = PROCESSED_DIR / "combined_features.csv"
    if not input_path.exists():
        raise FileNotFoundError("Run src.data.align_data before engineering features")
    df = pd.read_csv(input_path, parse_dates=["date"])
    engineered = engineer_features(df)
    return save_features(engineered, ns.output)


if __name__ == "__main__":
    main()
