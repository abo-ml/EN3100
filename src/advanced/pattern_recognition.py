"""Pattern recognition and ICT/SMT feature hooks.

This module provides rule-based detectors for common trading patterns:
- Liquidity grabs: Large volume spikes with price reversals
- Fair value gaps (FVG): Gaps between consecutive candles
- Asia session breakouts: Price breaking out of overnight range during London session
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def flag_liquidity_grab(
    prices: pd.DataFrame,
    volume_threshold: float = 2.0,
    reversal_threshold: float = 0.005,
    lookback: int = 5,
) -> pd.Series:
    """Detect liquidity grabs based on volume spikes and price reversals.

    A liquidity grab is identified when:
    1. Volume spikes above a threshold (relative to rolling average)
    2. Price shows a reversal pattern within a small window

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame with 'high', 'low', 'close', and 'volume' columns.
    volume_threshold : float, default=2.0
        Multiplier for volume spike detection. A value of 2.0 means
        volume must be 2x the rolling average to qualify as a spike.
    reversal_threshold : float, default=0.005
        Minimum price reversal as a percentage (0.005 = 0.5%).
        The price must reverse by at least this amount.
    lookback : int, default=5
        Rolling window size for calculating average volume and
        detecting price reversals.

    Returns
    -------
    pd.Series
        Series with 1 where liquidity grab is detected, 0 otherwise.
        Aligned to the input index.

    Notes
    -----
    Hyperparameters should be tuned based on asset volatility:
    - For volatile assets (crypto), increase reversal_threshold to 0.01-0.02
    - For low-volume assets, decrease volume_threshold to 1.5
    """
    result = pd.Series(0, index=prices.index, dtype=int, name="liquidity_grab")

    # Find column names (handle case-insensitive and ticker-prefixed columns)
    def find_col(patterns):
        for col in prices.columns:
            col_lower = col.lower()
            for pattern in patterns:
                if pattern in col_lower:
                    return col
        return None

    volume_col = find_col(["volume"])
    high_col = find_col(["high"])
    low_col = find_col(["low"])
    close_col = find_col(["adjclose", "close"])

    if not all([volume_col, high_col, low_col, close_col]):
        return result

    volume = prices[volume_col].astype(float)
    high = prices[high_col].astype(float)
    low = prices[low_col].astype(float)
    close = prices[close_col].astype(float)

    # Calculate rolling average volume
    avg_volume = volume.rolling(window=lookback, min_periods=1).mean()

    # Detect volume spikes
    volume_spike = volume > (avg_volume * volume_threshold)

    # Calculate price range and reversal indicators
    price_range = high - low
    close_position = (close - low) / price_range.replace(0, np.nan)

    # Detect bullish reversal: price touched low but closed near high (wick below)
    bullish_reversal = close_position > 0.7

    # Detect bearish reversal: price touched high but closed near low (wick above)
    bearish_reversal = close_position < 0.3

    # Calculate actual reversal magnitude
    prev_close = close.shift(1)
    price_change = (close - prev_close).abs() / prev_close.replace(0, np.nan)
    significant_move = price_change >= reversal_threshold

    # Combine conditions: volume spike + reversal pattern + significant move
    liquidity_grab = (
        volume_spike & (bullish_reversal | bearish_reversal) & significant_move
    )

    result = liquidity_grab.astype(int).fillna(0).astype(int)
    result.name = "liquidity_grab"
    return result


def detect_fvg(
    prices: pd.DataFrame,
    min_gap_percent: float = 0.001,
    fill_lookforward: int = 5,
) -> pd.Series:
    """Detect Fair Value Gaps (FVG) between consecutive candles.

    A bullish FVG occurs when: high of candle[i-2] < low of candle[i]
    A bearish FVG occurs when: low of candle[i-2] > high of candle[i]

    The gap is marked until it's filled by subsequent price action.

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame with 'high' and 'low' columns.
    min_gap_percent : float, default=0.001
        Minimum gap size as a percentage of price (0.001 = 0.1%).
        Helps filter out noise from tiny gaps.
    fill_lookforward : int, default=5
        Number of bars to look forward to check if gap is filled.
        The gap remains marked as 1 until price enters the gap zone.

    Returns
    -------
    pd.Series
        Series with values:
        - 1: Bullish FVG detected (gap up)
        - -1: Bearish FVG detected (gap down)
        - 0: No FVG
        Aligned to the input index.

    Notes
    -----
    FVGs are significant because they represent imbalances that price often
    returns to fill. Traders use unfilled FVGs as potential support/resistance.
    """
    result = pd.Series(0, index=prices.index, dtype=int, name="fvg")

    def find_col(patterns):
        for col in prices.columns:
            col_lower = col.lower()
            for pattern in patterns:
                if pattern in col_lower:
                    return col
        return None

    high_col = find_col(["high"])
    low_col = find_col(["low"])

    if not all([high_col, low_col]):
        return result

    high = prices[high_col].astype(float)
    low = prices[low_col].astype(float)

    n = len(prices)
    fvg_values = np.zeros(n, dtype=int)

    for i in range(2, n):
        # Bullish FVG: high of candle i-2 < low of candle i
        # This creates a gap where the middle candle's range doesn't overlap
        prev_high = high.iloc[i - 2]
        curr_low = low.iloc[i]
        # Use the midpoint of the gap range for percentage calculation
        gap_midpoint = (prev_high + curr_low) / 2

        gap_size = curr_low - prev_high
        if gap_size > 0 and gap_size / gap_midpoint >= min_gap_percent:
            # Check if gap is filled in the lookforward window
            filled = False
            for j in range(i + 1, min(i + 1 + fill_lookforward, n)):
                if low.iloc[j] <= prev_high:
                    filled = True
                    break
            if not filled:
                fvg_values[i] = 1

        # Bearish FVG: low of candle i-2 > high of candle i
        prev_low = low.iloc[i - 2]
        curr_high = high.iloc[i]
        # Use the midpoint of the gap range for percentage calculation
        gap_midpoint_bearish = (prev_low + curr_high) / 2

        gap_size = prev_low - curr_high
        if gap_size > 0 and gap_size / gap_midpoint_bearish >= min_gap_percent:
            # Check if gap is filled in the lookforward window
            filled = False
            for j in range(i + 1, min(i + 1 + fill_lookforward, n)):
                if high.iloc[j] >= prev_low:
                    filled = True
                    break
            if not filled:
                fvg_values[i] = -1

    result = pd.Series(fvg_values, index=prices.index, name="fvg")
    return result


def asia_session_range_breakout(
    prices: pd.DataFrame,
    asia_start: int = 0,
    asia_end: int = 6,
    london_start: int = 8,
    london_end: int = 12,
    timezone: Optional[str] = None,
) -> pd.Series:
    """Detect breakouts from Asia session range during London session.

    For forex pairs, this identifies when price breaks out of the overnight
    (00:00-06:00 UTC) high/low range during the London session (08:00-12:00 UTC).

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame with datetime index and 'high', 'low', 'close' columns.
        Index should be timezone-aware or in UTC.
    asia_start : int, default=0
        Start hour of Asia session (UTC). Default is midnight.
    asia_end : int, default=6
        End hour of Asia session (UTC). Default is 6 AM.
    london_start : int, default=8
        Start hour of London session (UTC). Default is 8 AM.
    london_end : int, default=12
        End hour of London session (UTC). Default is noon.
    timezone : str, optional
        Timezone to convert index to before analysis. If None, assumes UTC.

    Returns
    -------
    pd.Series
        Series with values:
        - 1: Bullish breakout (price breaks above Asia high)
        - -1: Bearish breakout (price breaks below Asia low)
        - 0: No breakout
        Aligned to the input index.

    Notes
    -----
    This pattern is particularly relevant for forex pairs like EURUSD, GBPUSD.
    Breakouts during London session often lead to significant directional moves.
    Requires intraday (hourly or minute) data for meaningful results.
    """
    result = pd.Series(0, index=prices.index, dtype=int, name="asia_breakout")

    def find_col(patterns):
        for col in prices.columns:
            col_lower = col.lower()
            for pattern in patterns:
                if pattern in col_lower:
                    return col
        return None

    high_col = find_col(["high"])
    low_col = find_col(["low"])
    close_col = find_col(["adjclose", "close"])

    if not all([high_col, low_col, close_col]):
        return result

    # Convert index to datetime if not already
    try:
        idx = pd.to_datetime(prices.index)
    except (ValueError, TypeError):
        # Index cannot be converted to datetime, return zeros
        return result

    # Check if we have intraday data (multiple points per day)
    dates = idx.normalize()
    if dates.nunique() == len(idx):
        # Daily data - can't compute intraday sessions
        # For daily data, use previous day as proxy for Asia session
        high = prices[high_col].astype(float)
        low = prices[low_col].astype(float)
        close = prices[close_col].astype(float)

        # Use previous day's high/low as Asia range proxy
        prev_high = high.shift(1)
        prev_low = low.shift(1)

        # Detect breakout
        bullish = close > prev_high
        bearish = close < prev_low

        result = pd.Series(0, index=prices.index, dtype=int, name="asia_breakout")
        result.loc[bullish] = 1
        result.loc[bearish] = -1
        return result

    # For intraday data, compute proper session ranges
    high = prices[high_col].astype(float)
    low = prices[low_col].astype(float)
    close = prices[close_col].astype(float)

    # Convert timezone if specified
    if timezone and idx.tz is None:
        idx = idx.tz_localize("UTC").tz_convert(timezone)
    elif timezone and idx.tz is not None:
        idx = idx.tz_convert(timezone)

    hours = idx.hour
    dates_only = idx.normalize()

    breakout_values = np.zeros(len(prices), dtype=int)
    unique_dates = dates_only.unique()

    for date in unique_dates:
        date_mask = dates_only == date

        # Get Asia session data for this date
        asia_mask = date_mask & (hours >= asia_start) & (hours < asia_end)
        if not asia_mask.any():
            continue

        asia_high = high.loc[asia_mask].max()
        asia_low = low.loc[asia_mask].min()

        # Get London session data for this date
        london_mask = date_mask & (hours >= london_start) & (hours < london_end)
        if not london_mask.any():
            continue

        london_indices = prices.index[london_mask]
        for idx_val in london_indices:
            pos = prices.index.get_loc(idx_val)
            if close.iloc[pos] > asia_high:
                breakout_values[pos] = 1
            elif close.iloc[pos] < asia_low:
                breakout_values[pos] = -1

    result = pd.Series(breakout_values, index=prices.index, name="asia_breakout")
    return result


def moving_average_crossovers(prices: pd.DataFrame, fast: int = 10, slow: int = 50) -> pd.DataFrame:
    """Flag simple MA crossovers to inject interpretable chart-pattern signals."""

    ma_fast = prices["close"].rolling(fast).mean()
    ma_slow = prices["close"].rolling(slow).mean()
    bullish = (ma_fast > ma_slow) & (ma_fast.shift(1) <= ma_slow.shift(1))
    bearish = (ma_fast < ma_slow) & (ma_fast.shift(1) >= ma_slow.shift(1))
    return pd.DataFrame(
        {
            "ma_bullish_crossover": bullish.astype(int),
            "ma_bearish_crossover": bearish.astype(int),
        },
        index=prices.index,
    )


def swing_high_low_flags(prices: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Identify local swing highs/lows using a rolling window."""

    highs = prices["high"]
    lows = prices["low"]
    swing_high = ((highs.shift(window) < highs) & (highs.shift(-window) < highs)).astype(int)
    swing_low = ((lows.shift(window) > lows) & (lows.shift(-window) > lows)).astype(int)
    return pd.DataFrame(
        {
            "swing_high_flag": swing_high,
            "swing_low_flag": swing_low,
        },
        index=prices.index,
    )


def detect_head_and_shoulders(
    prices: pd.DataFrame,
    window: int = 5,
    tolerance: float = 0.02,
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
    prices : pd.DataFrame
        DataFrame with at least a 'close' or 'Close' column.
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
    import numpy as np

    # Determine the close column name
    close_col = "close" if "close" in prices.columns else "Close"
    if close_col not in prices.columns:
        # No close column found, return zeros
        return pd.Series(0, index=prices.index, name="head_shoulders")

    close = prices[close_col].values
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
            # Guard against division by zero with max of shoulders and small epsilon
            shoulder_diff = abs(left_shoulder - right_shoulder) / max(
                left_shoulder, right_shoulder, 1e-10
            )
            if shoulder_diff < tolerance * 2:
                # Mark the right shoulder as the pattern completion point
                result[right_shoulder_idx] = 1

    return pd.Series(result, index=prices.index, name="head_shoulders")


def detect_double_top(
    prices: pd.DataFrame,
    window: int = 5,
    tolerance: float = 0.02,
) -> pd.Series:
    """Detect double-top pattern using peak analysis.

    Algorithm:
    1. Identify local maxima (peaks) with a window of `window` bars.
    2. For each pair of consecutive peaks, check if they are at similar heights
       (within `tolerance`).
    3. Require a trough between the peaks to confirm the pattern.
    4. Mark the bar at the second peak as a pattern signal (1 = detected, 0 = not).

    This also detects double-bottom patterns (inverted logic on local minima),
    returning -1 for double-bottom and 1 for double-top.

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame with at least a 'close' or 'Close' column.
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
    import numpy as np

    # Determine the close column name
    close_col = "close" if "close" in prices.columns else "Close"
    if close_col not in prices.columns:
        return pd.Series(0, index=prices.index, name="double_top")

    close = prices[close_col].values
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

    return pd.Series(result, index=prices.index, name="double_top")
