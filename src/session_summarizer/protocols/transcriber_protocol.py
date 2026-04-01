from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from ..vad.segment_splitter import SegmentSplitResultSet


from ..processing_results import TranscriptionResult
from .logging_protocol import LoggingProtocol


class TranscriberProtocol(Protocol):
    def transcribe(
        self, audio_path: Path, segments: SegmentSplitResultSet, logger: LoggingProtocol
    ) -> TranscriptionResult: ...
