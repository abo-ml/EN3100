"""Utility helpers for shared project functionality."""

from .env import (
    MissingEnvironmentVariableError,
    check_env_var,
    get_env_var,
)
from .paths import (
    DATA_DIR,
    FIGURES_DIR,
    PROJECT_ROOT,
    PROCESSED_DIR,
    RAW_DATA_DIR,
    REFERENCE_DIR,
    REPORTS_DIR,
    ensure_directories,
)

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "RAW_DATA_DIR",
    "REFERENCE_DIR",
    "PROCESSED_DIR",
    "REPORTS_DIR",
    "FIGURES_DIR",
    "ensure_directories",
    "get_env_var",
    "check_env_var",
    "MissingEnvironmentVariableError",
]
