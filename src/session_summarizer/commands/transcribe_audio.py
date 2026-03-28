from __future__ import annotations

from dataclasses import dataclass

import session_summarizer.utils.common_paths as common_paths

from ..protocols import LoggingProtocol, NullLogger
from ..protocols.transcriber_protocol import TranscriberProtocol


@dataclass
class TranscribeAudioCommand:
    """Transcribe normalized_audio.wav for a session and write transcript.json."""

    session_id: str
    transcriber: TranscriberProtocol
    logger: LoggingProtocol = NullLogger()

    def execute(self, logger: LoggingProtocol) -> None:
        self.logger = logger
        audio_path = common_paths.audio_file_from_step(
            common_paths.audio_processing_step.normalized_audio,
            self.session_id,
        )
        output_path = common_paths.audio_file_from_step(
            common_paths.audio_processing_step.transcript,
            self.session_id,
        )
