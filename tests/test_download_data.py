import logging

import pandas as pd
import src.data.download_data as dd


class DummyConfig:
    providers = ["alpha_vantage", "yfinance"]
    format = "parquet"
    api_key = None
    start = "2024-01-01"
    end = "2024-01-31"
    interval = "1d"


def _df():
    return pd.DataFrame(
        {"Open": [1, 2], "High": [1, 2], "Low": [1, 2], "Close": [1.0, 1.02], "Volume": [100, 120]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )


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


def test_download_single_ticker_falls_back(monkeypatch):
    monkeypatch.setattr(
        dd, "_download_alpha_vantage", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("AV down"))
    )
    monkeypatch.setattr(dd, "_download_yfinance", lambda *a, **k: _df_with_date())
    monkeypatch.setattr(dd, "_basic_clean_ohlcv", lambda df: df)
    df = dd.download_single_ticker("AAPL", DummyConfig())
    assert not df.empty


def test_cli_parser_description_unique():
    assert "public APIs" in dd.parser.description


def test_convert_ohlcv_columns():
    """Test that _convert_ohlcv_columns renames non-price columns to lowercase."""
    df = pd.DataFrame({
        "Open": [1, 2],
        "High": [3, 4],
        "Low": [0.5, 1.5],
        "Close": [2, 3],
        "Adj Close": [2.1, 3.1],
        "Volume": [100, 200],
    })
    result = dd._convert_ohlcv_columns(df)
    # Check renamed columns
    assert "open" in result.columns
    assert "high" in result.columns
    assert "low" in result.columns
    assert "volume" in result.columns
    # Close and Adj Close should NOT be renamed
    assert "Close" in result.columns
    assert "Adj Close" in result.columns
    assert "close" not in result.columns
    assert "adj_close" not in result.columns


def test_basic_clean_ohlcv_forward_fills():
    """Test that _basic_clean_ohlcv forward-fills and drops NaN close."""
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
        "close": [1.0, None, 1.5],
        "open": [1.0, None, 1.5],
    })
    result = dd._basic_clean_ohlcv(df)
    # Forward-fill should have filled the None
    assert result["close"].iloc[1] == 1.0
    assert len(result) == 3


def test_basic_clean_ohlcv_drops_nan_close():
    """Test that _basic_clean_ohlcv drops rows with NaN close after forward-fill."""
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
        "close": [None, 1.0, None, 1.5],  # First row has NaN, can't forward-fill
        "open": [1.0, 1.0, 1.5, 1.5],
    })
    result = dd._basic_clean_ohlcv(df)
    # First row should be dropped (NaN cannot be forward-filled from nothing)
    # But forward-fill will fill the third row from second
    assert len(result) == 3
    assert result["close"].iloc[0] == 1.0


def test_download_yfinance_returns_none_on_empty(monkeypatch):
    """Test that _download_yfinance returns None for empty data."""
    import yfinance as yf

    def mock_download(*args, **kwargs):
        return pd.DataFrame()  # Empty

    monkeypatch.setattr(yf, "download", mock_download)
    result = dd._download_yfinance("INVALID", DummyConfig())
    assert result is None


def test_download_yfinance_returns_lowercase_columns(monkeypatch):
    """Test that _download_yfinance returns DataFrame with lowercase date, close, adj_close."""
    import yfinance as yf

    def mock_download(*args, **kwargs):
        df = pd.DataFrame({
            "Open": [1, 2],
            "High": [1.5, 2.5],
            "Low": [0.8, 1.8],
            "Close": [1.2, 2.2],
            "Adj Close": [1.25, 2.25],
            "Volume": [1000, 2000],
        }, index=pd.to_datetime(["2024-01-02", "2024-01-03"]))
        df.index.name = "Date"
        return df

    monkeypatch.setattr(yf, "download", mock_download)
    result = dd._download_yfinance("AAPL", DummyConfig())

    assert result is not None
    assert "date" in result.columns
    assert "close" in result.columns
    assert "adj_close" in result.columns
    assert "open" in result.columns
    assert "high" in result.columns
    assert "low" in result.columns
    assert "volume" in result.columns
    assert "ticker" in result.columns


def test_download_single_ticker_skips_alpha_vantage_for_premium(monkeypatch):
    """Test that download_single_ticker skips Alpha Vantage for known premium tickers."""
    alpha_vantage_called = []

    def mock_av(*args, **kwargs):
        alpha_vantage_called.append(True)
        raise RuntimeError("Should not be called for AAPL")

    monkeypatch.setattr(dd, "_download_alpha_vantage", mock_av)
    monkeypatch.setattr(dd, "_download_yfinance", lambda *a, **k: _df_with_date())
    monkeypatch.setattr(dd, "_basic_clean_ohlcv", lambda df: df)

    # AAPL is in PREMIUM_ONLY_TICKERS, so alpha_vantage should be skipped
    df = dd.download_single_ticker("AAPL", DummyConfig())
    assert not df.empty
    assert len(alpha_vantage_called) == 0  # Alpha Vantage should not have been called


def test_premium_only_tickers_set():
    """Test that PREMIUM_ONLY_TICKERS contains expected symbols."""
    assert "AAPL" in dd.PREMIUM_ONLY_TICKERS
    assert "MSFT" in dd.PREMIUM_ONLY_TICKERS
    assert "GOOGL" in dd.PREMIUM_ONLY_TICKERS


def test_download_yfinance_multilevel_columns(monkeypatch):
    """Test that _download_yfinance handles multi-level columns (e.g., ('Close', 'AAPL'))."""
    import yfinance as yf

    def mock_download(*args, **kwargs):
        # Create a DataFrame with multi-level columns as yfinance sometimes returns
        arrays = [
            ["Open", "High", "Low", "Close", "Adj Close", "Volume"],
            ["AAPL", "AAPL", "AAPL", "AAPL", "AAPL", "AAPL"],
        ]
        tuples = list(zip(*arrays))
        columns = pd.MultiIndex.from_tuples(tuples)
        data = pd.DataFrame(
            [[1, 1.5, 0.8, 1.2, 1.25, 1000], [2, 2.5, 1.8, 2.2, 2.25, 2000]],
            index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
            columns=columns,
        )
        data.index.name = "Date"
        # Flatten the multi-level columns (this is what yfinance does internally sometimes)
        data.columns = data.columns.get_level_values(0)
        return data

    monkeypatch.setattr(yf, "download", mock_download)
    result = dd._download_yfinance("AAPL", DummyConfig())

    assert result is not None
    assert "date" in result.columns
    assert "close" in result.columns
    assert "adj_close" in result.columns
    assert "open" in result.columns
    assert "high" in result.columns
    assert "low" in result.columns
    assert "volume" in result.columns


def test_download_yfinance_true_multiindex_columns(monkeypatch):
    """Test that _download_yfinance correctly flattens MultiIndex columns."""
    import yfinance as yf

    def mock_download(*args, **kwargs):
        # Create a DataFrame with true MultiIndex columns (not pre-flattened)
        arrays = [
            ["Open", "High", "Low", "Close", "Adj Close", "Volume"],
            ["AAPL", "AAPL", "AAPL", "AAPL", "AAPL", "AAPL"],
        ]
        tuples = list(zip(*arrays))
        columns = pd.MultiIndex.from_tuples(tuples)
        data = pd.DataFrame(
            [[1, 1.5, 0.8, 1.2, 1.25, 1000], [2, 2.5, 1.8, 2.2, 2.25, 2000]],
            index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
            columns=columns,
        )
        data.index.name = "Date"
        # Don't flatten - let the function handle it
        return data

    monkeypatch.setattr(yf, "download", mock_download)
    result = dd._download_yfinance("AAPL", DummyConfig())

    assert result is not None
    assert "date" in result.columns
    assert "close" in result.columns
    assert "adj_close" in result.columns
    assert result["close"].iloc[0] == 1.25  # Should use Adj Close
    assert result["adj_close"].iloc[0] == 1.25


def test_download_yfinance_case_insensitive_date(monkeypatch):
    """Test that _download_yfinance handles DATE, Date, date columns case-insensitively."""
    import yfinance as yf

    def mock_download(*args, **kwargs):
        df = pd.DataFrame({
            "Open": [1, 2],
            "High": [1.5, 2.5],
            "Low": [0.8, 1.8],
            "Close": [1.2, 2.2],
            "Volume": [1000, 2000],
        }, index=pd.to_datetime(["2024-01-02", "2024-01-03"]))
        df.index.name = "DATE"  # Uppercase DATE
        return df

    monkeypatch.setattr(yf, "download", mock_download)
    result = dd._download_yfinance("AAPL", DummyConfig())

    assert result is not None
    assert "date" in result.columns
    assert pd.api.types.is_datetime64_any_dtype(result["date"])


def test_download_yfinance_returns_correct_column_order(monkeypatch):
    """Test that _download_yfinance returns columns in the correct order."""
    import yfinance as yf

    def mock_download(*args, **kwargs):
        df = pd.DataFrame({
            "Open": [1, 2],
            "High": [1.5, 2.5],
            "Low": [0.8, 1.8],
            "Close": [1.2, 2.2],
            "Adj Close": [1.25, 2.25],
            "Volume": [1000, 2000],
        }, index=pd.to_datetime(["2024-01-02", "2024-01-03"]))
        df.index.name = "Date"
        return df

    monkeypatch.setattr(yf, "download", mock_download)
    result = dd._download_yfinance("AAPL", DummyConfig())

    expected_columns = ["date", "open", "high", "low", "close", "adj_close", "volume", "ticker"]
    assert list(result.columns) == expected_columns


def test_download_yfinance_adj_close_priority(monkeypatch):
    """Test that _download_yfinance uses Adj Close for both close and adj_close when available."""
    import yfinance as yf

    def mock_download(*args, **kwargs):
        df = pd.DataFrame({
            "Open": [1, 2],
            "High": [1.5, 2.5],
            "Low": [0.8, 1.8],
            "Close": [1.0, 2.0],  # Different from Adj Close
            "Adj Close": [1.25, 2.25],
            "Volume": [1000, 2000],
        }, index=pd.to_datetime(["2024-01-02", "2024-01-03"]))
        df.index.name = "Date"
        return df

    monkeypatch.setattr(yf, "download", mock_download)
    result = dd._download_yfinance("AAPL", DummyConfig())

    # Both close and adj_close should use Adj Close values
    assert result["close"].iloc[0] == 1.25
    assert result["adj_close"].iloc[0] == 1.25
    assert result["close"].iloc[1] == 2.25
    assert result["adj_close"].iloc[1] == 2.25


def test_download_yfinance_only_close_no_adj_close(monkeypatch):
    """Test that _download_yfinance uses Close for both when Adj Close is missing."""
    import yfinance as yf

    def mock_download(*args, **kwargs):
        df = pd.DataFrame({
            "Open": [1, 2],
            "High": [1.5, 2.5],
            "Low": [0.8, 1.8],
            "Close": [1.2, 2.2],
            "Volume": [1000, 2000],
        }, index=pd.to_datetime(["2024-01-02", "2024-01-03"]))
        df.index.name = "Date"
        return df

    monkeypatch.setattr(yf, "download", mock_download)
    result = dd._download_yfinance("AAPL", DummyConfig())

    # Both close and adj_close should use Close values
    assert result["close"].iloc[0] == 1.2
    assert result["adj_close"].iloc[0] == 1.2


def test_download_yfinance_timezone_aware_date(monkeypatch):
    """Test that _download_yfinance converts timezone-aware dates to naive."""
    import yfinance as yf

    def mock_download(*args, **kwargs):
        df = pd.DataFrame({
            "Open": [1, 2],
            "High": [1.5, 2.5],
            "Low": [0.8, 1.8],
            "Close": [1.2, 2.2],
            "Volume": [1000, 2000],
        }, index=pd.to_datetime(["2024-01-02", "2024-01-03"]).tz_localize("UTC"))
        df.index.name = "Date"
        return df

    monkeypatch.setattr(yf, "download", mock_download)
    result = dd._download_yfinance("AAPL", DummyConfig())

    assert result is not None
    assert result["date"].dt.tz is None  # Should be timezone naive


# === NEW TESTS FOR ISSUE #6: Add tests for new logic ===


def test_download_yfinance_multilevel_columns_with_ticker_suffix(monkeypatch):
    """Test that _download_yfinance handles multi-level columns like ('Close', 'TEST').
    
    This tests the scenario where yfinance returns a DataFrame with MultiIndex columns
    where the first level is the column name (e.g., 'Close') and the second level is
    the ticker symbol (e.g., 'TEST'). The function should correctly extract the data
    and return a DataFrame with 'date', 'close', and 'adj_close' columns.
    """
    import yfinance as yf

    def mock_download(*args, **kwargs):
        # Create a DataFrame with MultiIndex columns: ('Column', 'Ticker')
        arrays = [
            ["Open", "High", "Low", "Close", "Adj Close", "Volume"],
            ["TEST", "TEST", "TEST", "TEST", "TEST", "TEST"],
        ]
        tuples = list(zip(*arrays))
        columns = pd.MultiIndex.from_tuples(tuples)
        data = pd.DataFrame(
            [[100.0, 105.0, 98.0, 102.5, 102.0, 10000],
             [103.0, 108.0, 101.0, 106.5, 106.0, 12000]],
            index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
            columns=columns,
        )
        data.index.name = "Date"
        # Return with MultiIndex intact - function should handle flattening
        return data

    monkeypatch.setattr(yf, "download", mock_download)
    result = dd._download_yfinance("TEST", DummyConfig())

    # Verify the function returns a valid DataFrame with required columns
    assert result is not None, "Function should return data for valid input"
    assert "date" in result.columns, "Missing 'date' column"
    assert "close" in result.columns, "Missing 'close' column"
    assert "adj_close" in result.columns, "Missing 'adj_close' column"
    
    # Verify data integrity
    assert len(result) == 2, "Should have 2 rows"
    assert result["ticker"].iloc[0] == "TEST", "Ticker should be 'TEST'"


def test_download_single_ticker_alpha_vantage_premium_error_fallback(monkeypatch):
    """Test that download_single_ticker() falls back when Alpha Vantage throws a premium error.
    
    When Alpha Vantage returns an error (e.g., premium subscription required),
    the function should catch the exception and fall back to yfinance or Stooq.
    """
    fallback_providers_called = []

    def mock_av_premium_error(*args, **kwargs):
        # Simulate Alpha Vantage premium subscription error
        raise RuntimeError("API rate limit reached or premium subscription required")

    def mock_yfinance(*args, **kwargs):
        fallback_providers_called.append("yfinance")
        return _df_with_date()

    # Ensure we're testing a non-premium ticker so Alpha Vantage is actually tried
    monkeypatch.setattr(dd, "_download_alpha_vantage", mock_av_premium_error)
    monkeypatch.setattr(dd, "_download_yfinance", mock_yfinance)
    monkeypatch.setattr(dd, "_basic_clean_ohlcv", lambda df: df)
    
    # Create config with providers list
    config = DummyConfig()
    config.providers = ["alpha_vantage", "yfinance"]
    
    # Use a non-premium ticker (not in PREMIUM_ONLY_TICKERS)
    result = dd.download_single_ticker("XYZ", config)

    assert not result.empty, "Should return data from fallback provider"
    assert "yfinance" in fallback_providers_called, "yfinance should have been called as fallback"


def test_download_single_ticker_yfinance_stooq_fail_uses_last_provider(monkeypatch, caplog):
    """Test fallback chain when multiple providers fail for a specific ticker.
    
    Mock yfinance and earlier providers to fail, but the last provider (stooq) succeeds.
    This confirms the fallback mechanism works correctly through the entire chain.
    """
    providers_called = []

    def mock_av_fail(*args, **kwargs):
        providers_called.append("alpha_vantage")
        raise RuntimeError("Alpha Vantage unavailable")

    def mock_yfinance_fail(*args, **kwargs):
        providers_called.append("yfinance")
        raise RuntimeError("yfinance unavailable for XAUUSD=X")

    def mock_stooq_success(*args, **kwargs):
        providers_called.append("stooq")
        # Return valid data from Stooq
        return pd.DataFrame({
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "open": [2050.0, 2055.0],
            "high": [2060.0, 2065.0],
            "low": [2045.0, 2050.0],
            "close": [2055.0, 2060.0],
            "adj_close": [2055.0, 2060.0],
            "volume": [100, 120],
            "ticker": ["XAUUSD=X", "XAUUSD=X"],
        })

    monkeypatch.setattr(dd, "_download_alpha_vantage", mock_av_fail)
    monkeypatch.setattr(dd, "_download_yfinance", mock_yfinance_fail)
    monkeypatch.setattr(dd, "_download_stooq", mock_stooq_success)
    monkeypatch.setattr(dd, "_basic_clean_ohlcv", lambda df: df)
    monkeypatch.setattr(dd, "HAS_PANDAS_DATAREADER", True)

    config = DummyConfig()
    config.providers = ["alpha_vantage", "yfinance", "stooq"]
    
    with caplog.at_level(logging.WARNING):
        result = dd.download_single_ticker("XAUUSD=X", config)

    assert not result.empty, "Should return data from stooq fallback"
    assert "stooq" in providers_called, "Stooq should have been called"
    # Verify providers were tried in order
    assert providers_called == ["alpha_vantage", "yfinance", "stooq"]


def test_missing_alphavantage_api_key_logs_warning_and_uses_fallback(monkeypatch, caplog):
    """Test that missing ALPHAVANTAGE_API_KEY logs a warning and data is fetched via fallback.
    
    When ALPHAVANTAGE_API_KEY environment variable is not set, the function should:
    1. Log a clear warning about the missing API key
    2. Fall back to yfinance or Stooq to fetch data
    3. Successfully return data from the fallback provider
    """
    # Ensure no Alpha Vantage API key is configured
    monkeypatch.delenv("ALPHAVANTAGE_API_KEY", raising=False)
    monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)

    providers_called = []

    def mock_yfinance_success(*args, **kwargs):
        providers_called.append("yfinance")
        return _df_with_date()

    monkeypatch.setattr(dd, "_download_yfinance", mock_yfinance_success)
    monkeypatch.setattr(dd, "_basic_clean_ohlcv", lambda df: df)

    # Create a config without API key
    config = DummyConfig()
    config.api_key = None
    config.providers = ["alpha_vantage", "yfinance"]

    with caplog.at_level(logging.WARNING):
        # Use a non-premium ticker to test Alpha Vantage API key check
        result = dd.download_single_ticker("XYZ", config)

    # Verify a warning was logged about missing API key
    warning_found = any(
        "Alpha Vantage API key" in record.message or "API key not configured" in record.message
        for record in caplog.records
    )
    assert warning_found, "Should log a warning about missing Alpha Vantage API key"
    
    # Verify data was fetched via fallback provider
    assert not result.empty, "Should return data from fallback provider"
    assert "yfinance" in providers_called, "yfinance should have been used as fallback"


def test_download_yfinance_generic_ticker_multiindex(monkeypatch):
    """Test _download_yfinance with a generic ticker having MultiIndex columns.
    
    This ensures the function handles MultiIndex columns correctly regardless
    of what ticker symbol is in the second level of the MultiIndex.
    """
    import yfinance as yf

    def mock_download(*args, **kwargs):
        # Create DataFrame with MultiIndex columns having a different ticker
        arrays = [
            ["Open", "High", "Low", "Close", "Adj Close", "Volume"],
            ["GENERIC", "GENERIC", "GENERIC", "GENERIC", "GENERIC", "GENERIC"],
        ]
        tuples = list(zip(*arrays))
        columns = pd.MultiIndex.from_tuples(tuples)
        data = pd.DataFrame(
            [[50.0, 52.0, 48.0, 51.0, 50.5, 5000]],
            index=pd.to_datetime(["2024-01-02"]),
            columns=columns,
        )
        data.index.name = "Date"
        return data

    monkeypatch.setattr(yf, "download", mock_download)
    result = dd._download_yfinance("GENERIC", DummyConfig())

    # Verify proper handling of MultiIndex and required output columns
    assert result is not None
    assert "date" in result.columns
    assert "close" in result.columns
    assert "adj_close" in result.columns
    assert "open" in result.columns
    assert "high" in result.columns
    assert "low" in result.columns
    assert "volume" in result.columns
    assert "ticker" in result.columns
    
    # Verify Adj Close is used for close (as per priority logic)
    assert result["close"].iloc[0] == 50.5  # Should use Adj Close value
    assert result["adj_close"].iloc[0] == 50.5
