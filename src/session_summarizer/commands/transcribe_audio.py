from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import torch

import session_summarizer.utils.common_paths as common_paths

from ..audio import (
    convert_to_48k_wav,
    enhance_with_mossformer2,
    measure_loudness,
    normalize_and_export_16k_mono,
)
from ..protocols import LoggingProtocol, NullLogger
from ..protocols.transcriber_protocol import TranscriberProtocol, TranscriptionResult, TranscriptionSegment
from ..transcription.parakeet_ctc_aligner import AlignmentResult, ParakeetCTCAligner


def _log_gpu_usage(logger: LoggingProtocol, label: str) -> None:
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    logger.report_message(
        f"[dim]GPU RAM ({label}): {allocated:.1f} GB allocated, {reserved:.1f} GB reserved, {total:.1f} GB total[/dim]"
    )


_PAUSE_THRESHOLD_S = 0.5  # gap between words that triggers a new segment
_MAX_SEGMENT_DURATION_S = 3.0  # hard cap on segment length
_SENTENCE_ENDERS = frozenset(".?!")


def _rebuild_segments_from_alignment(alignment: AlignmentResult) -> list[TranscriptionSegment]:
    """Create fine-grained segments from word-level alignment, splitting on pauses, sentences, and duration."""
    if not alignment.words:
        return []

    segments: list[TranscriptionSegment] = []
    current_words: list[str] = [alignment.words[0].word]
    seg_start = alignment.words[0].start
    seg_end = alignment.words[0].end
    min_conf = alignment.words[0].confidence

    def _flush() -> None:
        nonlocal current_words, seg_start, min_conf
        if current_words:
            segments.append(
                TranscriptionSegment(
                    text=" ".join(current_words),
                    start=seg_start,
                    end=seg_end,
                    confidence=min_conf,
                )
            )
            current_words = []

    for prev, word in zip(alignment.words, alignment.words[1:], strict=False):
        gap = word.start - prev.end
        ends_sentence = prev.word and prev.word[-1] in _SENTENCE_ENDERS
        duration = word.start - seg_start
        if gap >= _PAUSE_THRESHOLD_S or ends_sentence or duration >= _MAX_SEGMENT_DURATION_S:
            _flush()
            seg_start = word.start
            min_conf = word.confidence

        current_words.append(word.word)
        seg_end = word.end
        min_conf = min(min_conf, word.confidence)

    _flush()
    return segments


@dataclass
class TranscribeAudioCommand:
    """Clean audio and transcribe a session, writing transcript.json."""

    session_id: str
    transcriber: TranscriberProtocol
    aligner: ParakeetCTCAligner | None = None
    logger: LoggingProtocol = NullLogger()

    def name(self) -> str:
        return "Transcribe"

    def execute(self, logger: LoggingProtocol) -> None:
        self.logger = logger
        session = common_paths.session_dir(self.session_id)

        original = session / "original.m4a"
        wav_48k = session / "wav_48k.wav"
        cleaned_audio = session / "cleaned_audio.wav"
        normalized_audio = session / "normalized_audio.wav"
        transcript_path = session / "transcript.json"

        _log_gpu_usage(logger, "before processing")

        with self.logger.status("Converting to 48k WAV..."):
            convert_to_48k_wav(original, wav_48k)
        _log_gpu_usage(logger, "after 48k conversion")

        with self.logger.status("Enhancing with MossFormer2..."):
            enhance_with_mossformer2(wav_48k, cleaned_audio)
        _log_gpu_usage(logger, "after MossFormer2 enhancement")

        with self.logger.status("Measuring loudness..."):
            stats = measure_loudness(cleaned_audio)
        _log_gpu_usage(logger, "after loudness measurement")

        with self.logger.status("Normalizing to 16k mono..."):
            normalize_and_export_16k_mono(cleaned_audio, normalized_audio, stats)
        _log_gpu_usage(logger, "after 16k normalization")

        result: TranscriptionResult = self.transcriber.transcribe(normalized_audio, logger)
        _log_gpu_usage(logger, "after transcription")

        if self.aligner is not None:
            alignment: AlignmentResult = self.aligner.align(normalized_audio, result.full_text, logger)
            _log_gpu_usage(logger, "after alignment")
            result.segments = _rebuild_segments_from_alignment(alignment)

        transcript_path.write_text(
            json.dumps(asdict(result), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.report_message(f"[green]Transcript written to {transcript_path}[/green]")
