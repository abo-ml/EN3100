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


__all__ = [
    "MissingEnvironmentVariableError",
    "get_env_var",
    "check_env_var",
]
