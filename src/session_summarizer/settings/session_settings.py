from __future__ import annotations

from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, Field, field_validator

_SETTINGS_FILE = "settings.yaml"


class SessionSettings(BaseModel, frozen=True):
    attendees: Annotated[
        list[str],
        Field(min_length=1, description="Names of all speakers present in the session"),
    ]

    @field_validator("attendees")
    @classmethod
    def _attendee_names_must_be_non_empty(cls, names: list[str]) -> list[str]:
        for i, name in enumerate(names):
            stripped = name.strip()
            if not stripped:
                raise ValueError(
                    f"attendees[{i}] is blank — every attendee name must be a non-empty string, got {name!r}"
                )
        return names

    @property
    def number_of_speakers(self) -> int:
        """Derived from the length of attendees; used by the diarizer."""
        return len(self.attendees)

    @classmethod
    def load(cls, path: Path) -> SessionSettings:
        with path.open("r", encoding="utf-8") as f:
            data: dict = yaml.safe_load(f) or {}
        return cls(**data)

    @classmethod
    def load_cascading(cls, session_id: str) -> SessionSettings:
        from session_summarizer.utils.common_paths import data_dir, session_dir

        base_file = data_dir() / _SETTINGS_FILE
        session_file = session_dir(session_id) / _SETTINGS_FILE

        if not base_file.exists() and not session_file.exists():
            raise FileNotFoundError(
                f"No settings file found — looked in:\n"
                f"  {base_file}\n"
                f"  {session_file}\n"
                f"Place a {_SETTINGS_FILE} in either location."
            )

        base: dict = {}
        if base_file.exists():
            with base_file.open("r", encoding="utf-8") as f:
                base = yaml.safe_load(f) or {}

        override: dict = {}
        if session_file.exists():
            with session_file.open("r", encoding="utf-8") as f:
                override = yaml.safe_load(f) or {}

        return cls(**{**base, **override})


# Backwards-compatible alias
session_settings = SessionSettings
