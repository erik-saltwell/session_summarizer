"""Integration test: register a single speaker and verify the embedding is saved."""

from __future__ import annotations

from pathlib import Path

import pytest
from rich.console import Console

from session_summarizer.commands.register_speakers import RegisterSpeakersCommand
from session_summarizer.logging import CompositeLogger, FileLogger, RichConsoleLogger
from session_summarizer.protocols.logging_protocol import LoggingProtocol
from session_summarizer.utils import common_paths

from .temp_session import TempSession

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LOG_FILE = PROJECT_ROOT / "logs" / "test_register_speaker.log"


@pytest.fixture()
def logger() -> LoggingProtocol:
    _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    return CompositeLogger([RichConsoleLogger(Console()), FileLogger(_LOG_FILE)])


def test_register_speaker_fee013(logger: LoggingProtocol) -> None:
    """Register speaker FEE013 and assert the embedding is written to YAML."""
    with TempSession():
        RegisterSpeakersCommand().execute(logger)

        yaml_path = common_paths.build_speakers_file_path()
        assert yaml_path.exists(), f"Speakers file was not created at {yaml_path}"

        import yaml

        data = yaml.safe_load(yaml_path.read_text())
        assert "FEE013" in data, "Speaker FEE013 not found in speakers YAML"

        entry = data["FEE013"]
        embedding = entry["embedding"] if isinstance(entry, dict) else entry
        assert isinstance(embedding, list), f"Embedding should be a list, got {type(embedding)}"
        assert len(embedding) == 192, f"Expected 192-dim embedding, got {len(embedding)}"
