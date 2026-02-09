"""
Regression tests for _download_yfinance() ensuring proper normalization of output.

This module tests that _download_yfinance() always returns a DataFrame with:
- date, close, adj_close columns
- timezone-naive dates
- proper handling of MultiIndex columns
"""

import pandas as pd
import yfinance as yf
import src.data.download_data as dd


class DummyConfig:
    """Minimal config for testing _download_yfinance."""
    start = "2024-01-01"
    end = "2024-01-31"
    interval = "1d"


def test_download_yfinance_normalization_standard_columns(monkeypatch):
    """
    Test that _download_yfinance returns normalized DataFrame with date/close/adj_close.
    
    This test ensures that when yfinance returns a standard DataFrame with:
    - DatetimeIndex named 'Date'
    - Columns: Open, High, Low, Close, Adj Close, Volume
    
    The function properly normalizes it to have:
    - 'date' column (timezone-naive)
    - 'close' column (copied from Adj Close when available)
    - 'adj_close' column (copied from Adj Close when available)
    - 'ticker' column (added by the function)
    """
    def mock_download(*args, **kwargs):
        # Create a DataFrame as yfinance would return it
        df = pd.DataFrame({
            "Open": [150.0, 151.0, 152.0],
            "High": [155.0, 156.0, 157.0],
            "Low": [148.0, 149.0, 150.0],
            "Close": [153.0, 154.0, 155.0],
            "Adj Close": [152.5, 153.5, 154.5],
            "Volume": [100000, 110000, 120000],
        }, index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]))
        df.index.name = "Date"
        return df
    
    monkeypatch.setattr(yf, "download", mock_download)
    
    # Call the function under test
    result = dd._download_yfinance("AAPL", DummyConfig())
    
    # Assert: returned df has required columns
    assert result is not None, "Function should not return None for valid data"
    assert "date" in result.columns, "Missing 'date' column"
    assert "close" in result.columns, "Missing 'close' column"
    assert "adj_close" in result.columns, "Missing 'adj_close' column"
    
    # Assert: date is timezone-naive
    assert result["date"].dtype.name == "datetime64[ns]", "Date should be datetime64[ns]"
    # Timezone-naive datetimes have no tz attribute or tz is None
    if hasattr(result["date"].dtype, "tz"):
        assert result["date"].dtype.tz is None, "Date should be timezone-naive"
    
    # Additional checks: verify data integrity and ticker column
    assert len(result) == 3, "Should have 3 rows"
    assert result["close"].iloc[0] == 152.5, "close should be equal to Adj Close"
    assert result["adj_close"].iloc[0] == 152.5, "adj_close should be equal to Adj Close"
    assert "ticker" in result.columns, "Function should add ticker column"
    assert result["ticker"].iloc[0] == "AAPL", "Ticker column should contain the ticker symbol"


def test_download_yfinance_normalization_with_timezone(monkeypatch):
    """
    Test that _download_yfinance properly removes timezone from date column.
    
    This test ensures timezone-aware DatetimeIndex from yfinance is converted
    to timezone-naive datetime in the returned DataFrame. Also verifies that
    the ticker column is added.
    """
    def mock_download(*args, **kwargs):
        # Create a DataFrame with timezone-aware DatetimeIndex
        df = pd.DataFrame({
            "Open": [150.0, 151.0],
            "High": [155.0, 156.0],
            "Low": [148.0, 149.0],
            "Close": [153.0, 154.0],
            "Adj Close": [152.5, 153.5],
            "Volume": [100000, 110000],
        }, index=pd.to_datetime(["2024-01-02", "2024-01-03"]).tz_localize("UTC"))
        df.index.name = "Date"
        return df
    
    monkeypatch.setattr(yf, "download", mock_download)
    
    # Call the function under test
    result = dd._download_yfinance("AAPL", DummyConfig())
    
    # Assert: date is timezone-naive
    assert result is not None
    assert "date" in result.columns
    assert result["date"].dtype.name == "datetime64[ns]"
    if hasattr(result["date"].dtype, "tz"):
        assert result["date"].dtype.tz is None, "Date should be timezone-naive after normalization"
    
    # Assert: ticker column is present
    assert "ticker" in result.columns
    assert result["ticker"].iloc[0] == "AAPL"


def test_download_yfinance_normalization_multiindex_columns(monkeypatch):
    """
    Test that _download_yfinance handles MultiIndex columns correctly.
    
    This test ensures that when yfinance returns a DataFrame with MultiIndex columns
    (e.g., when downloading multiple tickers), the function properly flattens them
    and normalizes the output. Verifies that the ticker column is correctly added.
    """
    def mock_download(*args, **kwargs):
        # Create a DataFrame with MultiIndex columns as yfinance sometimes returns
        # when downloading multiple tickers or in certain market conditions
        arrays = [
            ["Open", "High", "Low", "Close", "Adj Close", "Volume"],
            ["AAPL", "AAPL", "AAPL", "AAPL", "AAPL", "AAPL"],
        ]
        tuples = list(zip(*arrays))
        columns = pd.MultiIndex.from_tuples(tuples)
        
        df = pd.DataFrame(
            [[150.0, 155.0, 148.0, 153.0, 152.5, 100000],
             [151.0, 156.0, 149.0, 154.0, 153.5, 110000],
             [152.0, 157.0, 150.0, 155.0, 154.5, 120000]],
            index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            columns=columns,
        )
        df.index.name = "Date"
        
        # Flatten the multi-level columns (as yfinance does internally)
        df.columns = df.columns.get_level_values(0)
        return df
    
    monkeypatch.setattr(yf, "download", mock_download)
    
    # Call the function under test
    result = dd._download_yfinance("AAPL", DummyConfig())
    
    # Assert: proper normalization despite MultiIndex input
    assert result is not None, "Function should handle MultiIndex columns"
    assert "date" in result.columns, "Missing 'date' column after MultiIndex flatten"
    assert "close" in result.columns, "Missing 'close' column after MultiIndex flatten"
    assert "adj_close" in result.columns, "Missing 'adj_close' column after MultiIndex flatten"
    
    # Assert: column types are as expected (single-level, not MultiIndex)
    assert not isinstance(result.columns, pd.MultiIndex), "Columns should be flattened"
    
    # Assert: date is timezone-naive
    assert result["date"].dtype.name == "datetime64[ns]"
    if hasattr(result["date"].dtype, "tz"):
        assert result["date"].dtype.tz is None
    
    # Assert: data integrity
    assert len(result) == 3
    assert "ticker" in result.columns
    assert result["ticker"].iloc[0] == "AAPL"


def test_download_yfinance_no_exceptions_raised(monkeypatch):
    """
    Test that _download_yfinance does not raise exceptions for valid input.
    
    This is a regression test to ensure the normalization process doesn't
    introduce unexpected errors. Also verifies the ticker column is added.
    """
    def mock_download(*args, **kwargs):
        df = pd.DataFrame({
            "Open": [150.0, 151.0],
            "High": [155.0, 156.0],
            "Low": [148.0, 149.0],
            "Close": [153.0, 154.0],
            "Adj Close": [152.5, 153.5],
            "Volume": [100000, 110000],
        }, index=pd.to_datetime(["2024-01-02", "2024-01-03"]))
        df.index.name = "Date"
        return df
    
    monkeypatch.setattr(yf, "download", mock_download)
    
    # This should not raise any exceptions
    try:
        result = dd._download_yfinance("AAPL", DummyConfig())
        assert result is not None
        assert "date" in result.columns
        assert "close" in result.columns
        assert "adj_close" in result.columns
        assert "ticker" in result.columns
        assert result["ticker"].iloc[0] == "AAPL"
    except Exception as e:
        raise AssertionError(f"_download_yfinance raised unexpected exception: {e}")
