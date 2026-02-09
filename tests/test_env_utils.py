"""Tests for environment variable utilities."""
import logging
import os
import pytest
from src.utils.env import (
    MissingEnvironmentVariableError,
    check_env_var,
    get_api_key,
    get_env_var,
)


class TestGetEnvVar:
    """Tests for get_env_var utility function."""

    def test_get_env_var_returns_value_when_set(self, monkeypatch):
        """Test that get_env_var returns the value when the variable is set."""
        monkeypatch.setenv("TEST_VAR", "test_value")
        result = get_env_var("TEST_VAR")
        assert result == "test_value"

    def test_get_env_var_strips_whitespace(self, monkeypatch):
        """Test that get_env_var strips leading and trailing whitespace."""
        monkeypatch.setenv("TEST_VAR", "  test_value  ")
        result = get_env_var("TEST_VAR")
        assert result == "test_value"

    def test_get_env_var_raises_when_not_set_and_required(self, monkeypatch):
        """Test that get_env_var raises MissingEnvironmentVariableError when required and not set."""
        monkeypatch.delenv("TEST_VAR_NOT_SET", raising=False)
        with pytest.raises(MissingEnvironmentVariableError) as exc_info:
            get_env_var("TEST_VAR_NOT_SET")
        assert "TEST_VAR_NOT_SET" in str(exc_info.value)
        assert exc_info.value.var_name == "TEST_VAR_NOT_SET"

    def test_get_env_var_raises_when_empty_and_required(self, monkeypatch):
        """Test that get_env_var raises when the variable is set to empty string."""
        monkeypatch.setenv("TEST_VAR", "")
        with pytest.raises(MissingEnvironmentVariableError) as exc_info:
            get_env_var("TEST_VAR")
        assert "TEST_VAR" in str(exc_info.value)

    def test_get_env_var_raises_when_whitespace_only_and_required(self, monkeypatch):
        """Test that get_env_var raises when the variable is only whitespace."""
        monkeypatch.setenv("TEST_VAR", "   ")
        with pytest.raises(MissingEnvironmentVariableError):
            get_env_var("TEST_VAR")

    def test_get_env_var_returns_none_when_not_set_and_not_required(self, monkeypatch):
        """Test that get_env_var returns None when variable is not set and required=False."""
        monkeypatch.delenv("TEST_VAR_NOT_SET", raising=False)
        result = get_env_var("TEST_VAR_NOT_SET", required=False)
        assert result is None

    def test_get_env_var_returns_none_when_empty_and_not_required(self, monkeypatch):
        """Test that get_env_var returns None when variable is empty and required=False."""
        monkeypatch.setenv("TEST_VAR", "")
        result = get_env_var("TEST_VAR", required=False)
        assert result is None


class TestCheckEnvVar:
    """Tests for check_env_var utility function."""

    def test_check_env_var_returns_value_when_set(self, monkeypatch):
        """Test that check_env_var returns the value when set."""
        monkeypatch.setenv("TEST_VAR", "test_value")
        result = check_env_var("TEST_VAR")
        assert result == "test_value"

    def test_check_env_var_strips_whitespace(self, monkeypatch):
        """Test that check_env_var strips leading and trailing whitespace."""
        monkeypatch.setenv("TEST_VAR", "  test_value  ")
        result = check_env_var("TEST_VAR")
        assert result == "test_value"

    def test_check_env_var_returns_none_when_not_set(self, monkeypatch, caplog):
        """Test that check_env_var returns None and logs a warning when not set."""
        monkeypatch.delenv("TEST_VAR_NOT_SET", raising=False)
        with caplog.at_level(logging.WARNING):
            result = check_env_var("TEST_VAR_NOT_SET")
        assert result is None
        assert "TEST_VAR_NOT_SET" in caplog.text

    def test_check_env_var_returns_none_when_empty(self, monkeypatch, caplog):
        """Test that check_env_var returns None and logs a warning when empty."""
        monkeypatch.setenv("TEST_VAR", "")
        with caplog.at_level(logging.WARNING):
            result = check_env_var("TEST_VAR")
        assert result is None
        assert "TEST_VAR" in caplog.text

    def test_check_env_var_no_warning_when_disabled(self, monkeypatch, caplog):
        """Test that check_env_var does not log when warn_if_missing=False."""
        monkeypatch.delenv("TEST_VAR_NOT_SET", raising=False)
        with caplog.at_level(logging.WARNING):
            result = check_env_var("TEST_VAR_NOT_SET", warn_if_missing=False)
        assert result is None
        assert "TEST_VAR_NOT_SET" not in caplog.text


class TestMissingEnvironmentVariableError:
    """Tests for MissingEnvironmentVariableError exception."""

    def test_exception_contains_var_name(self):
        """Test that the exception message contains the variable name."""
        exc = MissingEnvironmentVariableError("MY_API_KEY")
        assert "MY_API_KEY" in str(exc)
        assert exc.var_name == "MY_API_KEY"

    def test_exception_with_custom_message(self):
        """Test that a custom message can be provided."""
        exc = MissingEnvironmentVariableError("MY_API_KEY", "Custom error message")
        assert str(exc) == "Custom error message"
        assert exc.var_name == "MY_API_KEY"


class TestAlphaVantageApiKeyCheck:
    """Tests for Alpha Vantage API key checking in download_data."""

    def test_download_alpha_vantage_returns_none_when_no_key(self, monkeypatch, caplog):
        """Test that _download_alpha_vantage returns None when API key is not set."""
        import src.data.download_data as dd

        # Ensure API key is not set
        monkeypatch.delenv("ALPHAVANTAGE_API_KEY", raising=False)
        monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)

        # Create a config without api_key
        config = dd.DownloadConfig(
            tickers=["AAPL"],
            start="2024-01-01",
            end="2024-01-31",
            api_key=None,
        )

        with caplog.at_level(logging.WARNING):
            result = dd._download_alpha_vantage("AAPL", config)

        assert result is None
        assert "API key" in caplog.text

    def test_download_alpha_vantage_uses_env_key(self, monkeypatch):
        """Test that _download_alpha_vantage uses ALPHAVANTAGE_API_KEY from env."""
        import src.data.download_data as dd

        # Set the API key in environment
        monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "test_api_key")

        config = dd.DownloadConfig(
            tickers=["AAPL"],
            start="2024-01-01",
            end="2024-01-31",
            api_key=None,
        )

        api_key = dd._get_api_key(config, "alpha_vantage")
        assert api_key == "test_api_key"

    def test_download_alpha_vantage_uses_alt_env_key(self, monkeypatch):
        """Test that _download_alpha_vantage uses ALPHA_VANTAGE_API_KEY (with underscore)."""
        import src.data.download_data as dd

        # Only set the alternate key name
        monkeypatch.delenv("ALPHAVANTAGE_API_KEY", raising=False)
        monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "alt_api_key")

        config = dd.DownloadConfig(
            tickers=["AAPL"],
            start="2024-01-01",
            end="2024-01-31",
            api_key=None,
        )

        api_key = dd._get_api_key(config, "alpha_vantage")
        assert api_key == "alt_api_key"

    def test_download_alpha_vantage_prefers_config_key(self, monkeypatch):
        """Test that config.api_key takes precedence over environment variables."""
        import src.data.download_data as dd

        # Set both environment and config API keys
        monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "env_key")

        config = dd.DownloadConfig(
            tickers=["AAPL"],
            start="2024-01-01",
            end="2024-01-31",
            api_key="config_key",
        )

        api_key = dd._get_api_key(config, "alpha_vantage")
        assert api_key == "config_key"


class TestGetApiKey:
    """Tests for get_api_key utility function."""

    def test_get_api_key_returns_value_when_set(self, monkeypatch):
        """Test that get_api_key returns the value when the variable is set."""
        monkeypatch.setenv("TEST_API_KEY", "test_value")
        result = get_api_key("TEST_API_KEY")
        assert result == "test_value"

    def test_get_api_key_strips_whitespace(self, monkeypatch):
        """Test that get_api_key strips leading and trailing whitespace."""
        monkeypatch.setenv("TEST_API_KEY", "  test_value  ")
        result = get_api_key("TEST_API_KEY")
        assert result == "test_value"

    def test_get_api_key_returns_empty_string_when_not_set(self, monkeypatch, caplog):
        """Test that get_api_key returns empty string and logs a warning when not set."""
        monkeypatch.delenv("TEST_API_KEY_NOT_SET", raising=False)
        with caplog.at_level(logging.WARNING):
            result = get_api_key("TEST_API_KEY_NOT_SET")
        assert result == ""
        assert "TEST_API_KEY_NOT_SET" in caplog.text
        assert "not set" in caplog.text

    def test_get_api_key_returns_empty_string_when_empty(self, monkeypatch, caplog):
        """Test that get_api_key returns empty string and logs a warning when empty."""
        monkeypatch.setenv("TEST_API_KEY", "")
        with caplog.at_level(logging.WARNING):
            result = get_api_key("TEST_API_KEY")
        assert result == ""
        assert "TEST_API_KEY" in caplog.text

    def test_get_api_key_returns_empty_string_when_whitespace_only(self, monkeypatch, caplog):
        """Test that get_api_key returns empty string when value is only whitespace."""
        monkeypatch.setenv("TEST_API_KEY", "   ")
        with caplog.at_level(logging.WARNING):
            result = get_api_key("TEST_API_KEY")
        assert result == ""
        assert "TEST_API_KEY" in caplog.text

    def test_get_api_key_warning_message_format(self, monkeypatch, caplog):
        """Test that get_api_key logs a clear warning message with README reference."""
        monkeypatch.delenv("ALPHAVANTAGE_API_KEY", raising=False)
        with caplog.at_level(logging.WARNING):
            get_api_key("ALPHAVANTAGE_API_KEY")
        assert "ALPHAVANTAGE_API_KEY not set" in caplog.text
        assert "README.md" in caplog.text
