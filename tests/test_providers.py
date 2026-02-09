"""Tests for data provider fallback logic and configuration warnings."""
import pandas as pd
import pytest

import src.data.download_data as dd


class DummyConfig:
    """Dummy configuration for testing."""
    providers = ["alpha_vantage", "yfinance", "stooq"]
    format = "parquet"
    api_key = None
    start = "2024-01-01"
    end = "2024-01-31"
    interval = "1d"
    pause = 0.0  # No pause for tests


def _df_with_date():
    """DataFrame already processed with date column and lowercase columns."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "open": [1, 2],
            "high": [1, 2],
            "low": [1, 2],
            "close": [1.0, 1.02],
            "adj_close": [1.0, 1.02],
            "volume": [100, 120],
            "ticker": ["AAPL", "AAPL"],
        }
    )


def test_stooq_fallback_when_av_and_yfinance_fail(monkeypatch):
    """Test that Stooq is used as fallback when Alpha Vantage and yfinance fail.

    This test injects a premium-only ticker (AAPL) into download_single_ticker()
    with providers = ["alpha_vantage", "yfinance", "stooq"]. Alpha Vantage is
    skipped for premium tickers, yfinance is mocked to throw an exception, and
    Stooq is mocked to return valid data.
    """
    stooq_called = []

    def mock_yfinance(*args, **kwargs):
        raise RuntimeError("yfinance unavailable")

    def mock_stooq(*args, **kwargs):
        stooq_called.append(True)
        return _df_with_date()

    monkeypatch.setattr(dd, "_download_yfinance", mock_yfinance)
    monkeypatch.setattr(dd, "_download_stooq", mock_stooq)
    monkeypatch.setattr(dd, "_basic_clean_ohlcv", lambda df: df)
    # Ensure pandas_datareader is treated as available
    monkeypatch.setattr(dd, "HAS_PANDAS_DATAREADER", True)

    config = DummyConfig()
    # AAPL is in PREMIUM_ONLY_TICKERS, so Alpha Vantage is skipped automatically
    df = dd.download_single_ticker("AAPL", config)

    assert not df.empty
    assert len(stooq_called) == 1  # Stooq should have been called as the fallback


def test_all_providers_fallback_order(monkeypatch):
    """Test that providers are tried in order until one succeeds."""
    call_order = []

    def mock_av(*args, **kwargs):
        call_order.append("alpha_vantage")
        raise RuntimeError("AV down")

    def mock_yfinance(*args, **kwargs):
        call_order.append("yfinance")
        raise RuntimeError("yfinance down")

    def mock_stooq(*args, **kwargs):
        call_order.append("stooq")
        return _df_with_date()

    monkeypatch.setattr(dd, "_download_alpha_vantage", mock_av)
    monkeypatch.setattr(dd, "_download_yfinance", mock_yfinance)
    monkeypatch.setattr(dd, "_download_stooq", mock_stooq)
    monkeypatch.setattr(dd, "_basic_clean_ohlcv", lambda df: df)
    monkeypatch.setattr(dd, "HAS_PANDAS_DATAREADER", True)

    # Use a non-premium ticker to test full fallback chain
    config = DummyConfig()
    config.providers = ["alpha_vantage", "yfinance", "stooq"]
    df = dd.download_single_ticker("XYZ", config)

    assert not df.empty
    assert call_order == ["alpha_vantage", "yfinance", "stooq"]


def test_env_var_warning_when_api_key_missing(monkeypatch, caplog):
    """Test that a warning is raised when Alpha Vantage API key is missing."""
    import logging

    # Remove any existing API key environment variables
    monkeypatch.delenv("ALPHAVANTAGE_API_KEY", raising=False)
    monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)

    config = DummyConfig()
    config.api_key = None

    with caplog.at_level(logging.WARNING):
        result = dd._get_api_key(config, "alpha_vantage")

    assert result is None
    assert any("Alpha Vantage API key not configured" in rec.message for rec in caplog.records)


def test_env_var_warning_when_api_key_empty(monkeypatch, caplog):
    """Test that a warning is raised when Alpha Vantage API key is empty string."""
    import logging

    # Set an empty API key
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "   ")
    monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)

    config = DummyConfig()
    config.api_key = None

    with caplog.at_level(logging.WARNING):
        result = dd._get_api_key(config, "alpha_vantage")

    assert result is None
    assert any("Alpha Vantage API key is empty" in rec.message for rec in caplog.records)


def test_env_var_fallback_to_alternative_key(monkeypatch):
    """Test that ALPHA_VANTAGE_API_KEY is used when ALPHAVANTAGE_API_KEY is not set."""
    monkeypatch.delenv("ALPHAVANTAGE_API_KEY", raising=False)
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "test_key_123")

    config = DummyConfig()
    config.api_key = None

    result = dd._get_api_key(config, "alpha_vantage")
    assert result == "test_key_123"
