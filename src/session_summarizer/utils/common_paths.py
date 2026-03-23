from __future__ import annotations

from collections.abc import Iterator
from enum import StrEnum
from pathlib import Path

_DATA_DIR: Path = Path("data")
_FRAGMENTS_DIR: Path = Path("fragments")

def ensure_directory(dir: Path) -> None:
    dir.mkdir(parents=True, exist_ok=True)


def data_path() -> Path:
    """Return the path to the computed datasets directory under outputs."""
    return _DATA_DIR


def fragments_path() -> Path:
    """Return the shared fragments directory path."""
    return _FRAGMENTS_DIR


