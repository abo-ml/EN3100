"""Walk-forward validation utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Iterable, Tuple

import numpy as np
import pandas as pd

from .metrics import max_drawdown, sharpe_ratio


@dataclass
class WalkForwardConfig:
    n_splits: int
    train_min_period: int


def walk_forward_splits(df: pd.DataFrame, n_splits: int, train_min_period: int) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """Yield train/test indices for walk-forward validation."""

    total = len(df)
    split_size = max(1, (total - train_min_period) // n_splits)
    for i in range(n_splits):
        train_end = train_min_period + split_size * i
        test_end = train_end + split_size
        if i == n_splits - 1:
            test_end = total
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(train_end, test_end)
        if len(test_idx) == 0 or len(train_idx) < train_min_period:
            continue
        yield train_idx, test_idx


def aggregate_metrics(records: Iterable[dict]) -> dict:
    """Aggregate metrics across walk-forward splits."""

    records = list(records)
    if not records:
        return {}
    keys = records[0].keys()
    summary = {}
    for key in keys:
        values = np.array([rec[key] for rec in records if key in rec and not pd.isna(rec[key])])
        if values.size == 0:
            summary[key] = float("nan")
        else:
            summary[f"{key}_mean"] = float(values.mean())
            summary[f"{key}_std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
    return summary


def strategy_metrics(daily_returns: Iterable[float]) -> dict:
    returns = np.asarray(list(daily_returns))
    return {
        "sharpe": sharpe_ratio(returns),
        "max_drawdown": max_drawdown(returns),
        "mean_return": float(np.mean(returns)) if returns.size else float("nan"),
    }
