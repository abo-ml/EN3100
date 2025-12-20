"""Centralised filesystem paths for the EN3100 project."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
REFERENCE_DIR = DATA_DIR / "reference"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def ensure_directories(*paths: Iterable[Path]) -> None:
    """Create directories required at runtime."""

    if not paths:
        targets = [DATA_DIR, RAW_DATA_DIR, REFERENCE_DIR, PROCESSED_DIR, REPORTS_DIR, FIGURES_DIR]
    else:
        targets = []
        for path in paths:
            if isinstance(path, (list, tuple, set)):
                targets.extend(path)
            else:
                targets.append(path)
    for path in targets:
        Path(path).mkdir(parents=True, exist_ok=True)


# Ensure common runtime directories exist when utilities are imported.
ensure_directories()


__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "RAW_DATA_DIR",
    "REFERENCE_DIR",
    "PROCESSED_DIR",
    "REPORTS_DIR",
    "FIGURES_DIR",
    "ensure_directories",
]
