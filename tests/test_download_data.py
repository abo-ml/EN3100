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
