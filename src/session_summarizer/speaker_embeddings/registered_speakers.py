from __future__ import annotations

from pathlib import Path

import yaml


class RegisteredSpeakers(dict[str, list[float]]):
    """A mapping of speaker name to embedding vector, with load/save support."""

    @classmethod
    def load(cls, path: Path) -> RegisteredSpeakers:
        instance = cls()
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                raw: dict = yaml.safe_load(f) or {}
            for name, value in raw.items():
                instance[name] = value["embedding"]
        return instance

    def save(self, path: Path) -> None:
        raw = {name: {"embedding": embedding} for name, embedding in self.items()}
        with path.open("w", encoding="utf-8") as f:
            yaml.dump(raw, f, default_flow_style=False, allow_unicode=True)

    def integrate_embedding(self, name: str, embedding: list[float]) -> None:
        self[name] = embedding
