"""Monte Carlo simulation of Iteration 5 strategy returns."""
from __future__ import annotations

import argparse
import logging
from typing import Dict, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.evaluation.metrics import max_drawdown, sharpe_ratio
from src.utils import FIGURES_DIR, PROCESSED_DIR, REPORTS_DIR

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level="INFO")


def block_bootstrap(returns: np.ndarray, length: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Sample returns using block bootstrap to preserve short-term dependence."""

    if length == 0:
        return np.array([])
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    starts = np.arange(0, max(1, len(returns) - block_size + 1))
    sampled: list[np.ndarray] = []
    while sum(len(x) for x in sampled) < length:
        start = rng.choice(starts)
        sampled.append(returns[start : start + block_size])
    concatenated = np.concatenate(sampled)
    return concatenated[:length]


def simulate_equity_curves(
    returns: Iterable[float],
    n_paths: int = 10_000,
    block_size: int = 20,
    method: str = "block",
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Bootstrap daily strategy returns to estimate equity curve distributions."""

    base = np.asarray(list(returns), dtype=float)
    n = len(base)
    rng = np.random.default_rng(seed)

    final_equity = np.zeros(n_paths)
    max_dd = np.zeros(n_paths)
    sharpe_vals = np.zeros(n_paths)
    sample_equities: list[np.ndarray] = []

    for path in range(n_paths):
        if method == "iid":
            sampled = rng.choice(base, size=n, replace=True)
        else:
            sampled = block_bootstrap(base, n, block_size, rng)

        equity = np.cumprod(1 + sampled)
        final_equity[path] = equity[-1]
        max_dd[path] = max_drawdown(sampled)
        sharpe_vals[path] = sharpe_ratio(sampled)
        if path < 100:  # keep a small subset for fan chart
            sample_equities.append(equity)

    return {
        "final_equity": final_equity,
        "max_drawdown": max_dd,
        "sharpe": sharpe_vals,
        "sample_equities": np.array(sample_equities),
    }


def summarise_results(sim_results: Dict[str, np.ndarray]) -> Dict[str, float]:
    final_eq = sim_results["final_equity"]
    max_dd = sim_results["max_drawdown"]
    summary = {
        "final_equity_mean": float(np.mean(final_eq)),
        "final_equity_median": float(np.median(final_eq)),
        "final_equity_p5": float(np.percentile(final_eq, 5)),
        "final_equity_p95": float(np.percentile(final_eq, 95)),
        "max_dd_mean": float(np.mean(max_dd)),
        "max_dd_median": float(np.median(max_dd)),
        "max_dd_p95": float(np.percentile(max_dd, 95)),
        "prob_dd_gt_20pct": float(np.mean(max_dd < -0.2)),
        "sharpe_mean": float(np.mean(sim_results["sharpe"])),
    }
    return summary


def save_report(summary: Dict[str, float], output_path):
    lines = ["# Monte Carlo Risk Analysis (Iteration 5)", ""]
    for key, val in summary.items():
        lines.append(f"- **{key}**: {val:.4f}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


def save_raw_results(sim_results: Dict[str, np.ndarray], output_path) -> None:
    """Save raw simulation arrays (final_equity, max_drawdown, sharpe) to CSV.

    This allows downstream visualization scripts to plot true distribution
    histograms from the raw data rather than scalar summary statistics.
    """
    df = pd.DataFrame({
        "final_equity": sim_results["final_equity"],
        "max_drawdown": sim_results["max_drawdown"],
        "sharpe": sim_results["sharpe"],
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    LOGGER.info("Saved raw Monte Carlo results to %s", output_path)


def plot_histogram(values: np.ndarray, title: str, filename: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(values, bins=50, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Value")
    fig.savefig(FIGURES_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def plot_equity_fan(simulated: np.ndarray, historical: pd.Series, filename: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    for path in simulated:
        ax.plot(path, color="gray", alpha=0.1)
    ax.plot(historical.values, color="blue", label="Historical", linewidth=2)
    ax.set_title("Monte Carlo Equity Fan (first 100 paths)")
    ax.legend()
    fig.savefig(FIGURES_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def run_monte_carlo(method: str = "block", n_paths: int = 10_000, block_size: int = 20) -> Dict[str, float]:
    returns_path = PROCESSED_DIR / "iteration5_strategy_returns.csv"
    if not returns_path.exists():
        raise FileNotFoundError("Run iteration5_meta_ensemble to generate strategy returns first")

    df = pd.read_csv(returns_path, parse_dates=["date"])
    returns = df["strategy_return"]
    historical_equity = df["equity_curve"]

    LOGGER.info("Running Monte Carlo with %s method | paths=%s | block_size=%s", method, n_paths, block_size)
    results = simulate_equity_curves(returns, n_paths=n_paths, block_size=block_size, method=method)
    summary = summarise_results(results)

    save_report(summary, REPORTS_DIR / "iteration_5_monte_carlo.md")
    save_raw_results(results, PROCESSED_DIR / "iteration5_monte_carlo_results.csv")
    plot_histogram(results["final_equity"], "Final Equity Distribution", "iteration5_mc_final_equity_hist.png")
    plot_histogram(results["max_drawdown"], "Max Drawdown Distribution", "iteration5_mc_drawdown_hist.png")
    plot_equity_fan(results["sample_equities"], historical_equity, "iteration5_mc_equity_fan.png")

    LOGGER.info("Monte Carlo summary: %s", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monte Carlo simulation for Iteration 5 strategy returns")
    parser.add_argument("--method", choices=["block", "iid"], default="block")
    parser.add_argument("--paths", type=int, default=10_000, help="Number of simulated paths")
    parser.add_argument("--block-size", type=int, default=20, help="Block length for block bootstrap")
    return parser.parse_args()


def main():
    args = parse_args()
    return run_monte_carlo(method=args.method, n_paths=args.paths, block_size=args.block_size)


if __name__ == "__main__":
    main()
