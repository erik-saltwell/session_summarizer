from __future__ import annotations

from pathlib import Path

_DATA_DIR: Path = Path("data")
_FRAGMENTS_DIR: Path = Path("fragments")
_VOICE_SAMPLES: Path = Path("voice_samples")


def ensure_directory(dir: Path) -> None:
    dir.mkdir(parents=True, exist_ok=True)


def data_path() -> Path:
    """Return the path to the computed datasets directory under outputs."""
    return _DATA_DIR


def fragments_path() -> Path:
    """Return the shared fragments directory path."""
    return _FRAGMENTS_DIR


def voice_samples_path() -> Path:
    return data_path() / _VOICE_SAMPLES
