"""Environment variable utilities for API configuration.

This module provides helpers for safely accessing environment variables
used for API authentication (Alpha Vantage, FRED, NewsAPI, Alpaca, etc.).
"""
from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


class MissingEnvironmentVariableError(Exception):
    """Raised when a required environment variable is missing or empty."""

    def __init__(self, var_name: str, message: Optional[str] = None):
        self.var_name = var_name
        if message is None:
            message = (
                f"Required environment variable '{var_name}' is not set or is empty. "
                f"Please set {var_name} in your environment or .env file."
            )
        super().__init__(message)


def get_env_var(var_name: str, required: bool = True) -> Optional[str]:
    """Get an environment variable with validation.

    This utility function wraps `os.environ.get` and provides consistent
    handling of missing or empty environment variables used for API keys.

    Parameters
    ----------
    var_name : str
        Name of the environment variable to retrieve.
    required : bool, default=True
        If True, raises MissingEnvironmentVariableError when the variable
        is not set or is empty. If False, returns None for missing variables.

    Returns
    -------
    str or None
        The value of the environment variable, or None if not set and
        required=False.

    Raises
    ------
    MissingEnvironmentVariableError
        If the variable is not set or is empty and required=True.

    Examples
    --------
    >>> # Required variable (raises if missing)
    >>> api_key = get_env_var("ALPHAVANTAGE_API_KEY")
    
    >>> # Optional variable (returns None if missing)
    >>> api_key = get_env_var("ALPHAVANTAGE_API_KEY", required=False)
    """
    value = os.environ.get(var_name)

    if value is None or value.strip() == "":
        if required:
            raise MissingEnvironmentVariableError(var_name)
        return None

    return value.strip()


def check_env_var(var_name: str, warn_if_missing: bool = True) -> Optional[str]:
    """Check an environment variable and optionally log a warning if missing.

    Unlike `get_env_var`, this function never raises an exception. It returns
    the value if set, or None if missing, optionally logging a warning.

    Parameters
    ----------
    var_name : str
        Name of the environment variable to check.
    warn_if_missing : bool, default=True
        If True, logs a warning when the variable is not set or empty.

    Returns
    -------
    str or None
        The value of the environment variable, or None if not set.

    Examples
    --------
    >>> api_key = check_env_var("FRED_API_KEY")
    >>> if api_key is None:
    ...     # Use fallback or skip functionality
    ...     pass
    """
    value = os.environ.get(var_name)

    if value is None or value.strip() == "":
        if warn_if_missing:
            logger.warning(
                "Environment variable '%s' is not set or is empty. "
                "Some functionality may be unavailable.",
                var_name,
            )
        return None

    return value.strip()


def get_api_key(name: str) -> str:
    """Get an API key from environment variables with clear warning if missing.

    This helper function uses os.getenv() to retrieve API keys and logs a clear
    warning message if the key is not set or is empty. It is designed for use
    with external service API keys like ALPHAVANTAGE_API_KEY, FRED_API_KEY,
    NEWSAPI_KEY, APCA_API_KEY_ID, and APCA_API_SECRET_KEY.

    Parameters
    ----------
    name : str
        Name of the environment variable containing the API key.
        Common values include:
        - ALPHAVANTAGE_API_KEY: Alpha Vantage market data
        - FRED_API_KEY: Federal Reserve Economic Data
        - NEWSAPI_KEY: News API for sentiment analysis
        - APCA_API_KEY_ID: Alpaca trading API key ID
        - APCA_API_SECRET_KEY: Alpaca trading API secret

    Returns
    -------
    str
        The API key value if set, or an empty string if not set or empty.
        Callers should check for empty string to determine if the key is valid.

    Examples
    --------
    >>> api_key = get_api_key("ALPHAVANTAGE_API_KEY")
    >>> if not api_key:
    ...     print("API key not set, skipping Alpha Vantage")
    ...     return None

    >>> fred_key = get_api_key("FRED_API_KEY")
    >>> if fred_key:
    ...     # Use FRED API
    ...     pass
    """
    value = os.getenv(name)

    if value is None or value.strip() == "":
        logger.warning(
            "%s not set. Please set this environment variable to use the "
            "associated API. See README.md for configuration instructions.",
            name,
        )
        return ""

    return value.strip()


__all__ = [
    "MissingEnvironmentVariableError",
    "get_env_var",
    "check_env_var",
    "get_api_key",
]
