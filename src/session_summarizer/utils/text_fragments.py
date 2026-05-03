from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from .common_paths import fragments_dir


class FragmentID(StrEnum):
    """Identifiers for text fragment files stored in the fragments directory."""

    NONE = "none.md"
    SUMMARIZE_SESSION_SYSTEM_PROMPT = "summarize_session_prompt.md"
    TRANSCRIPT_CLEANER_PROMPT = "transcript_cleaner.md"
    INFER_PLAYERS_PROMPT = "infer_players_prompt.md"


def get_fragment_path(fragment_id: FragmentID) -> Path:
    """Return the directory path where the given fragment file is stored."""
    return fragments_dir()


def get_fragment(fragment_id: FragmentID) -> str:
    """Read and return the text content of the specified fragment file."""
    fragment_path = get_fragment_path(fragment_id) / Path(fragment_id.value)
    with open(fragment_path, encoding="utf-8") as f:
        return f.read()
