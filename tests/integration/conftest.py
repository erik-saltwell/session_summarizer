from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from rich.console import Console

from session_summarizer.commands.register_speakers import RegisterSpeakersCommand
from session_summarizer.commands.transcribe_audio import TranscribeAudioCommand
from session_summarizer.logging import CompositeLogger, FileLogger, RichConsoleLogger
from session_summarizer.processing_results import TranscriptionResult, TranscriptionSegment
from session_summarizer.protocols.logging_protocol import LoggingProtocol
from session_summarizer.utils import common_paths

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SESSION_ID = "_integration_test_ground_truth"
_LOG_FILE = PROJECT_ROOT / "test_run.log"


@pytest.fixture(scope="module")
def transcription_result() -> TranscriptionResult:
    """
    Run the full pipeline once per module.

    Skips setup and transcription if transcript.json already exists (resume support).
    Returns the deserialized TranscriptionResult.
    """
    test_logger: LoggingProtocol = CompositeLogger([RichConsoleLogger(Console()), FileLogger(_LOG_FILE)])

    session_dir = common_paths.session_dir(SESSION_ID)
    transcript_path = session_dir / "transcript.json"

    if not transcript_path.exists():
        # Setup: copy audio
        common_paths.ensure_directory(session_dir)
        src = PROJECT_ROOT / "test_meeting" / "original.m4a"
        dst = session_dir / "original.m4a"
        if not dst.exists():
            shutil.copy2(src, dst)

        # Register speakers
        RegisterSpeakersCommand().execute(test_logger)

        # Run transcription
        TranscribeAudioCommand(session_id=SESSION_ID).execute(test_logger)

    raw = json.loads(transcript_path.read_text(encoding="utf-8"))
    segments = [TranscriptionSegment(**s) for s in raw["segments"]]
    return TranscriptionResult(segments=segments, full_text=raw.get("full_text", ""))
