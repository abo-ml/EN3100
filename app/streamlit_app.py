"""Streamlit UI for running EN3100 pipelines and visualising outputs."""
from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st

from src.experiments.run_pipeline import CORE4, load_config, run_from_config
from src.utils import FIGURES_DIR, REPORTS_DIR, ensure_directories

CONFIG_PATH = Path("configs/universe.yaml")


def comma_separated(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def display_table_if_exists(path: Path, title: str) -> None:
    if path.exists():
        st.subheader(title)
        st.dataframe(pd.read_csv(path))


def display_image_if_exists(path: Path, caption: str) -> None:
    if path.exists():
        st.image(str(path), caption=caption, use_column_width=True)


def main() -> None:
    st.set_page_config(page_title="EN3100 Pipeline Runner", layout="wide")
    st.title("EN3100 Experiment Runner")
    ensure_directories(REPORTS_DIR, FIGURES_DIR)

    cfg = load_config(CONFIG_PATH) if CONFIG_PATH.exists() else {}

    with st.sidebar:
        st.header("Configuration")
        mode = st.selectbox("Mode", ["core4", "sp500_sample", "custom"], index=["core4", "sp500_sample", "custom"].index(cfg.get("mode", "core4")))
        provider = st.selectbox("Provider", ["yfinance", "alpha_vantage"], index=["yfinance", "alpha_vantage"].index(cfg.get("provider", "yfinance")))
        start = st.text_input("Start date", cfg.get("start", "2013-01-01"))
        end = st.text_input("End date", cfg.get("end", "2023-12-31"))
        tag = st.text_input("Optional tag for outputs", "")

        custom_tickers_text = ", ".join(cfg.get("tickers", CORE4))
        custom_tickers = st.text_input("Custom tickers (comma separated)", custom_tickers_text)

        sp500_n = st.number_input("S&P 500 sample size", min_value=1, max_value=500, value=int(cfg.get("sp500", {}).get("n", 20)))
        sp500_sector = st.text_input("Sector filter (optional)", cfg.get("sp500", {}).get("sector") or "")
        sp500_price_min = st.number_input("Price min (optional)", value=float(cfg.get("sp500", {}).get("price_min") or 0.0), step=1.0)
        sp500_price_max = st.number_input("Price max (optional)", value=float(cfg.get("sp500", {}).get("price_max") or 0.0), step=1.0)

        run_button = st.button("Run pipeline", type="primary")

    if run_button:
        updated_cfg = {
            "mode": mode,
            "provider": provider,
            "start": start,
            "end": end,
            "tickers": comma_separated(custom_tickers) if custom_tickers else CORE4,
            "sp500": {
                "n": int(sp500_n),
                "sector": sp500_sector or None,
                "price_min": sp500_price_min or None,
                "price_max": sp500_price_max or None,
            },
            "evaluation": {"per_asset": True, "per_stock_equity": True},
        }
        st.info("Running pipeline...check logs below.")
        run_from_config(updated_cfg, tag=tag or None)
        st.success("Pipeline complete.")

    st.header("Latest Reports")
    display_table_if_exists(REPORTS_DIR / "per_asset_metrics.csv", "Per-Asset Metrics (Core)")
    display_table_if_exists(REPORTS_DIR / "per_asset_equity_metrics.csv", "Per-Stock Equity Metrics")

    st.header("Figures")
    display_image_if_exists(FIGURES_DIR / "per_asset_directional_accuracy.png", "Per-Asset Directional Accuracy")
    display_image_if_exists(FIGURES_DIR / "equity_avg_directional_accuracy.png", "Equity Average Directional Accuracy")
    display_image_if_exists(FIGURES_DIR / "equity_da_heatmap.png", "Equity Directional Accuracy Heatmap")


if __name__ == "__main__":
    main()
