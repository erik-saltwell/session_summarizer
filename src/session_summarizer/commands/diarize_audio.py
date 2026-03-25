from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import session_summarizer.utils.common_paths as common_paths
from session_summarizer.utils.common_paths import audio_processing_step

from ..diarization.msdd_diarizer import DiarizationResult, MsddDiarizer
from ..protocols import LoggingProtocol, NullLogger


@dataclass
class DiarizeAudioCommand:
    """Diarize normalized_audio.wav, writing diarization.json with speaker segments."""

    session_id: str
    diarizer: MsddDiarizer
    logger: LoggingProtocol = NullLogger()

    def execute(self, logger: LoggingProtocol) -> None:
        self.logger = logger

        audio_path = common_paths.audio_file_from_step(audio_processing_step.normalized_audio, self.session_id)
        output_path = common_paths.audio_file_from_step(audio_processing_step.diarization, self.session_id)

        if not audio_path.exists():
            raise FileNotFoundError(
                f"Audio not found: {audio_path}. Run `clean-audio --session {self.session_id}` first."
            )

        result: DiarizationResult = self.diarizer.diarize(audio_path, logger)

        output_path.write_text(
            json.dumps(asdict(result), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self.logger.report_message(f"[green]Diarization written to {output_path}[/green]")
