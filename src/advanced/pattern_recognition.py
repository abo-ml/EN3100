"""Placeholder pattern recognition and ICT/SMT feature hooks."""
from __future__ import annotations

import pandas as pd


def flag_liquidity_grab(prices: pd.DataFrame) -> pd.Series:
    """Return zeros; future implementation should detect liquidity grabs."""

    # TODO: Implement liquidity grab detection using intraday highs/lows.
    return pd.Series(0, index=prices.index)


def detect_fvg(prices: pd.DataFrame) -> pd.Series:
    """Return zeros; placeholder for fair value gap detection."""

    # TODO: Analyse candle gaps and mark potential FVG structures.
    return pd.Series(0, index=prices.index)


def asia_session_range_breakout(prices: pd.DataFrame) -> pd.Series:
    """Return zeros; placeholder for Asia session breakout logic."""

    # TODO: Restrict to Asia session (e.g. 00:00-06:00 UTC) once intraday data is available.
    return pd.Series(0, index=prices.index)


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
