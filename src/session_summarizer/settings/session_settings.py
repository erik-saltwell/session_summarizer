from __future__ import annotations

from pathlib import Path

import yaml
from attr import dataclass

_SETTINGS_FILE = "settings.yaml"


@dataclass
class session_settings:
    attendees: list[str]  # names of all speakers present in the session; must be non-empty

    def __attrs_post_init__(self) -> None:
        if not self.attendees:
            raise ValueError("attendees must contain at least one name")
        for name in self.attendees:
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"each attendee name must be a non-empty string, got {name!r}")

    @property
    def number_of_speakers(self) -> int:
        """Derived from the length of attendees; used by the diarizer."""
        return len(self.attendees)

    @classmethod
    def load(cls, path: Path) -> session_settings:
        with path.open("r", encoding="utf-8") as f:
            data: dict = yaml.safe_load(f) or {}
        return cls(**data)

    @classmethod
    def load_cascading(cls, session_id: str) -> session_settings:
        from session_summarizer.utils.common_paths import data_path, session_path

        base_file = data_path() / _SETTINGS_FILE
        session_file = session_path(session_id) / _SETTINGS_FILE

        base: dict = {}
        if base_file.exists():
            with base_file.open("r", encoding="utf-8") as f:
                base = yaml.safe_load(f) or {}

        override: dict = {}
        if session_file.exists():
            with session_file.open("r", encoding="utf-8") as f:
                override = yaml.safe_load(f) or {}

        return cls(**{**base, **override})
