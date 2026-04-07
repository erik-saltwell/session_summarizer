from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from ..vad.segment_splitter import SegmentSplitResultSet


from ..protocols.logging_protocol import LoggingProtocol
from .transcription_result import TranscriptionResult


class TranscriberProtocol(Protocol):
    def transcribe(
        self, audio_path: Path, segments: SegmentSplitResultSet, logger: LoggingProtocol
    ) -> TranscriptionResult: ...
