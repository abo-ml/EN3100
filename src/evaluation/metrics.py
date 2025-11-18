"""Evaluation metrics used across model iterations."""
from __future__ import annotations

import numpy as np
from sklearn import metrics as sk_metrics


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(sk_metrics.mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    return float(sk_metrics.mean_absolute_error(y_true, y_pred))


def r2(y_true, y_pred) -> float:
    return float(sk_metrics.r2_score(y_true, y_pred))


def directional_accuracy(y_true_returns, y_pred_returns) -> float:
    y_true_sign = np.sign(y_true_returns)
    y_pred_sign = np.sign(y_pred_returns)
    hits = (y_true_sign == y_pred_sign) & (y_true_sign != 0)
    if hits.size == 0:
        return float("nan")
    return float(hits.sum() / hits.size)


def sharpe_ratio(returns) -> float:
    returns = np.asarray(returns)
    if returns.std(ddof=1) == 0:
        return float("nan")
    return float(returns.mean() / returns.std(ddof=1) * np.sqrt(252))


def max_drawdown(returns) -> float:
    returns = np.asarray(returns)
    cumulative = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cumulative)
    drawdown = cumulative / peak - 1
    return float(drawdown.min())
