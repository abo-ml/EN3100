"""Application-layer runner for configuring and executing experiment pipelines."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List, Optional

try:  # pragma: no cover - optional until requirements installed
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from src.data.download_data import DownloadConfig, download_ohlcv
from src.experiments.download_equity_universe import save_universe, select_equity_universe
from src.experiments.per_asset_evaluation import main as run_per_asset
from src.experiments.per_asset_equity_evaluation import main as run_per_equity
from src.utils import FIGURES_DIR, RAW_DATA_DIR, REFERENCE_DIR, ensure_directories
from src.data import align_data
from src.features import engineer_features

LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG = Path("configs/universe.yaml")
CORE4 = ["AAPL", "EURUSD=X", "XAUUSD=X", "^GSPC"]


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    if yaml is None:
        raise ImportError("PyYAML is required to load configuration files. Install with `pip install pyyaml`.")
    return yaml.safe_load(path.read_text())


def with_tag(path: Path, tag: Optional[str]) -> Path:
    if tag:
        return path.with_stem(f"{path.stem}_{tag}")
    return path


def run_align(tickers: Optional[Iterable[str]] = None, ticker_file: Optional[Path] = None, output: Optional[Path] = None) -> Path:
    args: List[str] = []
    if tickers:
        args.extend(["--tickers", *tickers])
    if ticker_file:
        args.extend(["--ticker-file", str(ticker_file)])
    if output:
        args.extend(["--output", str(output)])
    return align_data.main(args)


def run_engineer_features(output: Optional[Path] = None) -> Path:
    _ = output  # reserved for future extension; current engineer_features.main takes no args.
    return engineer_features.main()


def run_download(tickers: List[str], start: str, end: str, provider: str) -> None:
    ensure_directories(RAW_DATA_DIR)
    config = DownloadConfig(tickers=tickers, start=start, end=end, provider=provider, interval="1d")
    download_ohlcv(config)


def run_core4(cfg: dict, tag: Optional[str]) -> None:
    tickers = cfg.get("tickers") or CORE4
    run_download(tickers, cfg["start"], cfg["end"], cfg["provider"])
    run_align(tickers=tickers)
    run_engineer_features()
    if cfg.get("evaluation", {}).get("per_asset", True):
        run_per_asset(["--tag", tag] if tag else [])


def run_custom(cfg: dict, tag: Optional[str]) -> None:
    tickers = cfg.get("tickers") or CORE4
    run_download(tickers, cfg["start"], cfg["end"], cfg["provider"])
    run_align(tickers=tickers)
    run_engineer_features()
    if cfg.get("evaluation", {}).get("per_asset", True):
        run_per_asset(["--tag", tag] if tag else [])


def run_sp500(cfg: dict, tag: Optional[str]) -> None:
    sp500_cfg = cfg.get("sp500", {}) or {}
    tickers = select_equity_universe(
        n=sp500_cfg.get("n", 20),
        sector=sp500_cfg.get("sector"),
        price_min=sp500_cfg.get("price_min"),
        price_max=sp500_cfg.get("price_max"),
        random_state=42,
    )
    universe_path = save_universe(tickers, len(tickers) if sp500_cfg.get("n") is None else sp500_cfg.get("n", 20))
    run_download(tickers, cfg["start"], cfg["end"], cfg["provider"])
    run_align(ticker_file=universe_path)
    run_engineer_features()
    if cfg.get("evaluation", {}).get("per_stock_equity", True):
        per_equity_args = ["--tag", tag] if tag else []
        run_per_equity(per_equity_args)


def parse_args(cmd_args: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run configured experiment pipelines.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to YAML configuration.")
    parser.add_argument("--tag", type=str, default=None, help="Optional tag appended to output filenames.")
    return parser.parse_args(cmd_args)


def run_from_config(cfg: dict, tag: Optional[str] = None) -> None:
    mode = cfg.get("mode", "core4")
    if mode not in {"core4", "sp500_sample", "custom"}:
        raise ValueError(f"Unsupported mode: {mode}")
    ensure_directories(REFERENCE_DIR, RAW_DATA_DIR, FIGURES_DIR)
    if mode == "core4":
        run_core4(cfg, tag)
    elif mode == "custom":
        run_custom(cfg, tag)
    elif mode == "sp500_sample":
        run_sp500(cfg, tag)


def main(cmd_args: Optional[Iterable[str]] = None) -> None:
    logging.basicConfig(level="INFO")
    ns = parse_args(cmd_args)
    cfg = load_config(ns.config)
    run_from_config(cfg, ns.tag)


if __name__ == "__main__":
    main()
