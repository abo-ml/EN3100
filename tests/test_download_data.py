import pandas as pd
import src.data.download_data as dd


class DummyConfig:
    providers = ["alpha_vantage", "yfinance"]
    format = "parquet"


def _df():
    return pd.DataFrame(
        {"Open": [1, 2], "High": [1, 2], "Low": [1, 2], "Close": [1.0, 1.02], "Volume": [100, 120]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )


def test_download_single_ticker_falls_back(monkeypatch):
    monkeypatch.setattr(
        dd, "_download_alpha_vantage", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("AV down"))
    )
    monkeypatch.setattr(dd, "_download_yfinance", lambda *a, **k: _df())
    monkeypatch.setattr(dd, "_basic_clean_ohlcv", lambda df: df)
    df = dd.download_single_ticker("AAPL", DummyConfig())
    assert not df.empty


def test_cli_parser_description_unique():
    assert "public APIs" in dd.parser.description
