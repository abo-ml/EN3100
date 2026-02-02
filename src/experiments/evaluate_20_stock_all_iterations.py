"""Run per-ticker evaluation across iterations 1, 1.1, 2, 2.1, 3, and 5."""
from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
from importlib import metadata
from pathlib import Path
from typing import Callable, Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.evaluation.metrics import directional_accuracy
from src.evaluation.walkforward import aggregate_metrics, walk_forward_splits
from src.models import iteration1_1_svr, iteration1_baseline, iteration2_1_lightgbm, iteration2_ensemble, iteration3_lstm, iteration5_meta_ensemble
from src.models.iteration1_1_svr import run_iteration as run_iteration_1_1
from src.models.iteration1_baseline import run_iteration as run_iteration_1
from src.models.iteration2_1_lightgbm import run_iteration as run_iteration_2_1
from src.models.iteration2_ensemble import run_iteration as run_iteration_2
from src.models.iteration3_lstm import run_iteration as run_iteration_3
from src.models.iteration5_meta_ensemble import run_iteration as run_iteration_5
from src.utils import FIGURES_DIR, PROCESSED_DIR, PROJECT_ROOT, REFERENCE_DIR, ensure_directories

LOGGER = logging.getLogger(__name__)

IterationRunner = Callable[..., tuple[pd.DataFrame, pd.DataFrame]]

ITERATION_CONFIGS = [
    (
        "1",
        "Iteration 1",
        run_iteration_1,
        {
            "rmse": "rmse_linear_mean",
            "mae": "mae_linear_mean",
            "r2": "r2_linear_mean",
            "directional_accuracy": "directional_accuracy_linear_mean",
        },
    ),
    (
        "1.1",
        "Iteration 1.1",
        run_iteration_1_1,
        {
            "rmse": "rmse_svr_mean",
            "mae": "mae_svr_mean",
            "r2": "r2_svr_mean",
            "directional_accuracy": "directional_accuracy_svr_mean",
        },
    ),
    (
        "2",
        "Iteration 2",
        run_iteration_2,
        {
            "rmse": "rmse_rf_mean",
            "mae": "mae_rf_mean",
            "r2": "r2_rf_mean",
            "directional_accuracy": "directional_accuracy_rf_mean",
        },
    ),
    (
        "2.1",
        "Iteration 2.1",
        run_iteration_2_1,
        {
            "rmse": "rmse_lgbm_mean",
            "mae": "mae_lgbm_mean",
            "r2": "r2_lgbm_mean",
            "directional_accuracy": "directional_accuracy_lgbm_mean",
        },
    ),
    (
        "3",
        "Iteration 3",
        run_iteration_3,
        {
            "rmse": "rmse_lstm_mean",
            "mae": "mae_lstm_mean",
            "r2": "r2_lstm_mean",
            "directional_accuracy": "directional_accuracy_lstm_mean",
        },
    ),
    (
        "5",
        "Iteration 5",
        run_iteration_5,
        {
            "rmse": "rmse_meta_mean",
            "mae": "mae_meta_mean",
            "r2": "r2_meta_mean",
            "directional_accuracy": "directional_accuracy_meta_mean",
        },
    ),
]


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate 20-stock universe across all iterations.")
    parser.add_argument(
        "--features-path",
        type=Path,
        default=PROCESSED_DIR / "model_features.parquet",
        help="Path to model features parquet file.",
    )
    parser.add_argument(
        "--universe-file",
        type=Path,
        default=None,
        help="Path to equity_universe_20.txt. Defaults to equity_universe_20.txt or the newest match.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "20_stock",
        help="Directory for CSV + plots.",
    )
    parser.add_argument(
        "--tickers",
        nargs="*",
        default=None,
        help="Optional list of tickers to override the universe file.",
    )
    return parser.parse_args(args)


def load_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Model features not found at {path}. Run feature engineering first.")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def resolve_universe_file(candidate: Optional[Path]) -> Optional[Path]:
    if candidate and candidate.exists():
        return candidate
    default = REFERENCE_DIR / "equity_universe_20.txt"
    if default.exists():
        return default
    matches = sorted(REFERENCE_DIR.glob("equity_universe_*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def read_universe(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def select_tickers(feature_df: pd.DataFrame, universe_path: Optional[Path], override: Optional[List[str]]) -> List[str]:
    if override:
        return override
    if universe_path:
        tickers = read_universe(universe_path)
    else:
        tickers = sorted(feature_df["ticker"].unique().tolist())
    if len(tickers) > 20:
        tickers = tickers[:20]
    return tickers


def extract_metrics(summary: Dict[str, float], mapping: Dict[str, str]) -> Dict[str, Optional[float]]:
    return {metric: summary.get(key) for metric, key in mapping.items()}


def slugify_iteration(iteration_id: str) -> str:
    return iteration_id.replace(".", "_")


def compute_persistence_accuracy(df: pd.DataFrame) -> float:
    if df.empty:
        return float("nan")
    return directional_accuracy(df["next_day_return"].values, df["return_1d"].values)


def plot_walk_forward_schematic(output_path: Path, n_splits: int = 5, train_min_period: int = 252, total_periods: int = 1000) -> None:
    if total_periods <= train_min_period:
        total_periods = train_min_period + n_splits * 10
    dummy = pd.DataFrame({"value": np.arange(total_periods)})

    fig, ax = plt.subplots(figsize=(12, 4))
    y_height = 8
    y_gap = 4
    for split_id, (train_idx, test_idx) in enumerate(
        walk_forward_splits(dummy, n_splits=n_splits, train_min_period=train_min_period), start=1
    ):
        y_base = split_id * (y_height + y_gap)
        train_start = train_idx[0]
        train_width = train_idx[-1] - train_idx[0] + 1
        test_start = test_idx[0]
        test_width = test_idx[-1] - test_idx[0] + 1
        ax.broken_barh([(train_start, train_width)], (y_base, y_height), facecolors="#4C72B0", label="Train" if split_id == 1 else "")
        ax.broken_barh([(test_start, test_width)], (y_base, y_height), facecolors="#DD8452", label="Test" if split_id == 1 else "")

    ax.set_xlabel("Time index")
    ax.set_ylabel("Split")
    ax.set_yticks(
        [(split_id * (y_height + y_gap)) + y_height / 2 for split_id in range(1, n_splits + 1)],
        labels=[f"Split {split_id}" for split_id in range(1, n_splits + 1)],
    )
    ax.set_title("Walk-Forward Split Schematic")
    ax.legend(loc="upper right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_improvement_tables(df: pd.DataFrame) -> str:
    lines = ["# Directional Accuracy Improvement vs Persistence", ""]
    for iteration in sorted(df["iteration"].unique(), key=lambda val: float(val)):
        subset = df[df["iteration"] == iteration].copy()
        subset["accuracy_delta"] = subset["directional_accuracy"] - subset["persistence_accuracy"]
        subset = subset.sort_values("accuracy_delta", ascending=False)

        top5 = subset.head(5)
        bottom5 = subset.tail(5).sort_values("accuracy_delta")

        lines.append(f"## Iteration {iteration}")
        lines.append("")
        lines.append("### Top 5 Improvements")
        lines.append("")
        lines.append(top5[["ticker", "directional_accuracy", "persistence_accuracy", "accuracy_delta"]].to_markdown(index=False))
        lines.append("")
        lines.append("### Bottom 5 Improvements")
        lines.append("")
        lines.append(bottom5[["ticker", "directional_accuracy", "persistence_accuracy", "accuracy_delta"]].to_markdown(index=False))
        lines.append("")
    return "\n".join(lines)


def save_improvement_tables(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty or "directional_accuracy" not in df:
        LOGGER.warning("Skipping improvement tables due to missing directional accuracy data.")
        return
    content = build_improvement_tables(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content)
    for iteration in sorted(df["iteration"].unique(), key=lambda val: float(val)):
        subset = df[df["iteration"] == iteration].copy()
        subset["accuracy_delta"] = subset["directional_accuracy"] - subset["persistence_accuracy"]
        subset = subset.sort_values("accuracy_delta", ascending=False)
        top5 = subset.head(5)
        bottom5 = subset.tail(5).sort_values("accuracy_delta")
        LOGGER.info("Iteration %s top 5 improvement tickers: %s", iteration, ", ".join(top5["ticker"].tolist()))
        LOGGER.info("Iteration %s bottom 5 improvement tickers: %s", iteration, ", ".join(bottom5["ticker"].tolist()))


def plot_accuracy_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    pivot = df.pivot(index="ticker", columns="iteration", values="directional_accuracy")
    if pivot.empty:
        LOGGER.warning("No directional accuracy values available for heatmap.")
        return
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis")
    plt.title("Directional Accuracy by Ticker and Iteration")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_accuracy_boxplot(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        LOGGER.warning("No directional accuracy values available for boxplot.")
        return
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="iteration", y="directional_accuracy")
    plt.title("Directional Accuracy Distribution by Iteration")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_iteration_bars(df: pd.DataFrame, output_dir: Path) -> None:
    if df.empty:
        LOGGER.warning("No directional accuracy values available for bar charts.")
        return
    for iteration, group in df.groupby("iteration", sort=False):
        fig, ax = plt.subplots(figsize=(12, 6))
        group = group.sort_values("ticker")
        ax.bar(group["ticker"], group["directional_accuracy"], color="steelblue")
        ax.set_title(f"Directional Accuracy by Ticker - Iteration {iteration}")
        ax.set_ylabel("Directional Accuracy")
        ax.set_ylim(0, 1)
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        filename = output_dir / f"accuracy_by_ticker_iter_{slugify_iteration(iteration)}.png"
        fig.savefig(filename, bbox_inches="tight")
        plt.close(fig)


def compute_iteration_summary(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    records: List[Dict[str, object]] = []
    for iteration, group in df.groupby("iteration", sort=False):
        for metric in metrics:
            series = group[metric].dropna()
            count = int(series.count())
            mean = float(series.mean()) if count else float("nan")
            median = float(series.median()) if count else float("nan")
            std = float(series.std(ddof=1)) if count > 1 else float("nan")
            ci = 1.96 * std / np.sqrt(count) if count > 1 else float("nan")
            records.append(
                {
                    "iteration": iteration,
                    "metric": metric,
                    "count": count,
                    "mean": mean,
                    "median": median,
                    "std": std,
                    "ci95_half_width": ci,
                    "ci95_lower": mean - ci if count > 1 else float("nan"),
                    "ci95_upper": mean + ci if count > 1 else float("nan"),
                }
            )
    return pd.DataFrame(records)


def compute_iteration_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    summary = df.groupby("iteration").agg(
        mean_da=("directional_accuracy", "mean"),
        std_da=("directional_accuracy", "std"),
        mean_rmse=("rmse", "mean"),
        std_rmse=("rmse", "std"),
        mean_mae=("mae", "mean"),
        std_mae=("mae", "std"),
        mean_r2=("r2", "mean"),
        std_r2=("r2", "std"),
    )
    return summary.reset_index()


def plot_iteration_mean_ci(summary_df: pd.DataFrame, output_path: Path) -> None:
    if summary_df.empty:
        LOGGER.warning("Skipping iteration mean/CI plot due to empty summary.")
        return
    metrics = ["directional_accuracy", "rmse", "mae", "r2"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    iteration_order = sorted(summary_df["iteration"].unique().tolist(), key=lambda val: float(val))
    for ax, metric in zip(axes, metrics):
        subset = summary_df[summary_df["metric"] == metric].copy()
        subset = subset.set_index("iteration").reindex(iteration_order).reset_index()
        ax.errorbar(
            subset["iteration"],
            subset["mean"],
            yerr=subset["ci95_half_width"],
            fmt="o-",
            capsize=4,
        )
        ax.set_title(f"{metric} mean ± 95% CI")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(metric)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def export_pivot_tables(df: pd.DataFrame, output_dir: Path) -> None:
    if df.empty:
        LOGGER.warning("Skipping pivot tables due to empty metrics data.")
        return
    accuracy_pivot = df.pivot(index="ticker", columns="iteration", values="directional_accuracy")
    rmse_pivot = df.pivot(index="ticker", columns="iteration", values="rmse")
    accuracy_pivot.to_csv(output_dir / "directional_accuracy_pivot.csv")
    rmse_pivot.to_csv(output_dir / "rmse_pivot.csv")


def get_package_version(package_name: str) -> str:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return "not-installed"


def build_run_metadata(
    features_path: Path,
    tickers: List[str],
    feature_df: pd.DataFrame,
) -> Dict[str, object]:
    date_ranges = (
        feature_df[feature_df["ticker"].isin(tickers)]
        .groupby("ticker")["date"]
        .agg(["min", "max"])
        .reset_index()
    )
    overall_dates = feature_df[feature_df["ticker"].isin(tickers)]["date"]
    overall_start = overall_dates.min().date().isoformat() if not overall_dates.empty else None
    overall_end = overall_dates.max().date().isoformat() if not overall_dates.empty else None
    ticker_dates = {
        row["ticker"]: {"start_date": row["min"].date().isoformat(), "end_date": row["max"].date().isoformat()}
        for _, row in date_ranges.iterrows()
    }
    random_seeds = {
        "iteration_1": iteration1_baseline.SEED,
        "iteration_1_1": iteration1_1_svr.SEED,
        "iteration_2": 42,
        "iteration_2_1": iteration2_1_lightgbm.SEED,
        "iteration_3": iteration3_lstm.SEED,
        "iteration_5": iteration5_meta_ensemble.SEED,
    }
    library_versions = {
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
        "seaborn": sns.__version__,
        "scikit-learn": get_package_version("scikit-learn"),
        "lightgbm": get_package_version("lightgbm"),
        "tensorflow": get_package_version("tensorflow"),
        "xgboost": get_package_version("xgboost"),
        "platform": platform.platform(),
    }
    return {
        "features_path": str(features_path),
        "tickers": tickers,
        "date_range": {"start": overall_start, "end": overall_end},
        "ticker_date_ranges": ticker_dates,
        "library_versions": library_versions,
        "global_seed": 42,
        "random_seeds": random_seeds,
    }


def evaluate_ticker(ticker: str, df: pd.DataFrame) -> List[Dict[str, object]]:
    ticker_df = df[df["ticker"] == ticker].reset_index(drop=True)
    if ticker_df.empty:
        LOGGER.warning("No data available for %s; skipping.", ticker)
        return []
    if len(ticker_df) < 252:
        LOGGER.warning("Skipping %s due to insufficient history (%d rows).", ticker, len(ticker_df))
        return []

    persistence_accuracy = compute_persistence_accuracy(ticker_df)

    records: List[Dict[str, object]] = []
    for iteration_id, iteration_label, runner, mapping in ITERATION_CONFIGS:
        metrics_df, _ = runner(data=ticker_df, generate_reports=False, ticker=ticker)
        summary = aggregate_metrics(metrics_df.to_dict("records"))
        metrics = extract_metrics(summary, mapping)
        if all(value is None for value in metrics.values()):
            LOGGER.info("Skipping iteration %s for %s due to missing metrics.", iteration_id, ticker)
            continue
        records.append(
            {
                "ticker": ticker,
                "iteration": iteration_id,
                "iteration_label": iteration_label,
                "persistence_accuracy": persistence_accuracy,
                "accuracy_delta": (metrics.get("directional_accuracy") - persistence_accuracy)
                if metrics.get("directional_accuracy") is not None
                else None,
                **metrics,
            }
        )
    return records


def main(cmd_args: Optional[List[str]] = None) -> None:
    logging.basicConfig(level="INFO")
    ns = parse_args(cmd_args)

    feature_df = load_features(ns.features_path)
    universe_path = resolve_universe_file(ns.universe_file)
    tickers = select_tickers(feature_df, universe_path, ns.tickers)
    ensure_directories(ns.output_dir, FIGURES_DIR)

    records: List[Dict[str, object]] = []
    for ticker in tickers:
        records.extend(evaluate_ticker(ticker, feature_df))

    if not records:
        raise RuntimeError("No metrics computed; check input data and tickers.")

    metrics_df = pd.DataFrame(records)
    metrics_df = metrics_df.sort_values(["iteration", "ticker"]).reset_index(drop=True)

    output_csv = ns.output_dir / "ticker_iteration_metrics.csv"
    metrics_df.to_csv(output_csv, index=False)

    export_pivot_tables(metrics_df, ns.output_dir)

    summary_df = compute_iteration_summary_stats(metrics_df)
    summary_path = ns.output_dir / "iteration_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    summary_metrics = ["directional_accuracy", "rmse", "mae", "r2"]
    summary_long_df = compute_iteration_summary(metrics_df, summary_metrics)
    summary_long_path = ns.output_dir / "iteration_summary_long.csv"
    summary_long_df.to_csv(summary_long_path, index=False)

    heatmap_path = ns.output_dir / "accuracy_heatmap.png"
    boxplot_path = ns.output_dir / "accuracy_boxplot.png"
    plot_accuracy_heatmap(metrics_df, heatmap_path)
    plot_accuracy_boxplot(metrics_df, boxplot_path)
    plot_iteration_bars(metrics_df, ns.output_dir)
    plot_iteration_mean_ci(summary_long_df, ns.output_dir / "iteration_mean_ci.png")
    plot_walk_forward_schematic(FIGURES_DIR / "walk_forward_splits.png")

    save_improvement_tables(metrics_df, ns.output_dir / "directional_accuracy_delta_tables.md")

    metadata = build_run_metadata(ns.features_path, tickers, feature_df)
    metadata_path = ns.output_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    LOGGER.info("Saved metrics to %s", output_csv)


if __name__ == "__main__":
    main()
