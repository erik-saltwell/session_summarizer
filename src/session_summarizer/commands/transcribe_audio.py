from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import session_summarizer.utils.common_paths as common_paths

from ..audio import (
    convert_to_48k_wav,
    enhance_with_mossformer2,
    measure_loudness,
    normalize_and_export_16k_mono,
)
from ..protocols import LoggingProtocol, NullLogger
from ..protocols.transcriber_protocol import TranscriberProtocol, TranscriptionResult
from ..transcription.parakeet_ctc_aligner import AlignmentResult, ParakeetCTCAligner


def _map_segment_confidence(result: TranscriptionResult, alignment: AlignmentResult) -> None:
    """Mutate result segments in place: set confidence = min of overlapping aligned words."""
    for seg in result.segments:
        overlapping = [w.confidence for w in alignment.words if w.end > seg.start and w.start < seg.end]
        seg.confidence = min(overlapping) if overlapping else 0.0


@dataclass
class TranscribeAudioCommand:
    """Clean audio and transcribe a session, writing transcript.json."""

    session_id: str
    transcriber: TranscriberProtocol
    aligner: ParakeetCTCAligner | None = None
    logger: LoggingProtocol = NullLogger()

    def _report(self, message: str) -> None:
        self.logger.report_message(f"[blue]{message}[/blue]")

    def execute(self, logger: LoggingProtocol) -> None:
        self.logger = logger
        session = common_paths.session_path(self.session_id)

        original = session / "original.m4a"
        wav_48k = session / "wav_48k.wav"
        cleaned_audio = session / "cleaned_audio.wav"
        normalized_audio = session / "normalized_audio.wav"
        transcript_path = session / "transcript.json"

        self._report("converting to 48k wav...")
        convert_to_48k_wav(original, wav_48k)

        self._report("enhancing with MossFormer2...")
        enhance_with_mossformer2(wav_48k, cleaned_audio)

        self._report("measuring loudness...")
        stats = measure_loudness(cleaned_audio)

        self._report("normalizing to 16k mono...")
        normalize_and_export_16k_mono(cleaned_audio, normalized_audio, stats)

        result: TranscriptionResult = self.transcriber.transcribe(normalized_audio, logger)

        if self.aligner is not None:
            alignment: AlignmentResult = self.aligner.align(normalized_audio, result.full_text, logger)
            _map_segment_confidence(result, alignment)

        transcript_path.write_text(
            json.dumps(asdict(result), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.report_message(f"[green]Transcript written to {transcript_path}[/green]")
