from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import session_summarizer.utils.common_paths as common_paths
from session_summarizer.utils.common_paths import audio_processing_step

from ..alignment.parakeet_ctc_aligner import AlignmentResult, ParakeetCTCAligner
from ..protocols import LoggingProtocol, NullLogger


@dataclass
class AlignAudioCommand:
    """Align words in normalized_audio.wav against transcript.json, writing word_alignments.json."""

    session_id: str
    aligner: ParakeetCTCAligner
    logger: LoggingProtocol = NullLogger()

    def execute(self, logger: LoggingProtocol) -> None:
        self.logger = logger

        audio_path = common_paths.audio_file_from_step(audio_processing_step.normalized_audio, self.session_id)
        transcript_path = common_paths.audio_file_from_step(audio_processing_step.transcript, self.session_id)
        output_path = common_paths.audio_file_from_step(audio_processing_step.word_alignments, self.session_id)

        if not audio_path.exists():
            raise FileNotFoundError(
                f"Audio not found: {audio_path}. Run `clean-audio --session {self.session_id}` first."
            )
        if not transcript_path.exists():
            raise FileNotFoundError(
                f"Transcript not found: {transcript_path}. Run `transcribe --session {self.session_id}` first."
            )

        self.logger.report_message(f"[blue]Loading transcript from {transcript_path}...[/blue]")
        raw = json.loads(transcript_path.read_text(encoding="utf-8"))
        full_text: str = raw.get("full_text", "")

        result: AlignmentResult = self.aligner.align(audio_path, full_text, logger)

        output_path.write_text(
            json.dumps(asdict(result), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self.logger.report_message(f"[green]Word alignments written to {output_path}[/green]")
