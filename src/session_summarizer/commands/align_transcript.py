from __future__ import annotations

from dataclasses import dataclass

import session_summarizer.utils.common_paths as common_paths
from session_summarizer.protocols.session_settings import SessionSettings
from session_summarizer.transcription.parakeet_ctc_aligner import AlignmentResult

from ..helpers.audio_cleaner import clean_audio
from ..helpers.audio_transcriber import transcribe_from_cleaned_audio
from ..helpers.transcript_aligner import align_transcript
from ..protocols import LoggingProtocol
from ..protocols.transcriber_protocol import TranscriptionResult
from .session_processing_command import SessionProcessingCommand


def clean_text(text: str) -> str:
    ret: str = text
    ret = ret.replace("  ", " ")

    return ret


def _compare_alignment_to_transcript(
    alignment: AlignmentResult, transcription: TranscriptionResult, logger: LoggingProtocol
) -> None:
    aligned_text = " ".join(w.word for w in alignment.words)
    aligned_text = clean_text(aligned_text)
    transcript_text = transcription.full_text
    transcript_text = clean_text(transcript_text)
    if aligned_text == transcript_text:
        logger.report_message("[green]Aligned words match transcript text exactly.[/green]")
    else:
        aligned_words = aligned_text.split()
        transcript_words = transcript_text.split()
        logger.report_message(
            f"[yellow]Aligned words differ from transcript text. "
            f"Aligned: {len(aligned_words)} words, Transcript: {len(transcript_words)} words.[/yellow]"
        )


@dataclass
class AlignTranscriptCommand(SessionProcessingCommand):
    def name(self) -> str:
        return "Align Transcript"

    def process_session(self, settings: SessionSettings, session_dir: common_paths.Path) -> None:
        self.gpu_logging_enabled = True
        clean_audio(settings, session_dir, True, self, self.logger)
        result: TranscriptionResult = transcribe_from_cleaned_audio(settings, session_dir, True, self, self.logger)
        alignment: AlignmentResult = align_transcript(settings, session_dir, result, self, self.logger)
        alignment.save(session_dir / settings.aligned_transcript_path)
        # alignment: AlignmentResult = AlignmentResult.load(session_dir / settings.aligned_transcript_path)
        _compare_alignment_to_transcript(alignment, result, self.logger)
