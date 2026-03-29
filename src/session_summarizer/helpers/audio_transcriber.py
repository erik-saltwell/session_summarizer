from __future__ import annotations

from pathlib import Path

from ..protocols import (
    GpuLogger,
    LoggingProtocol,
    SessionSettings,
    TranscriberProtocol,
    TranscriptionResult,
    TranscriptionSegment,
)
from ..transcription import AlignmentResult, CanaryQwenTranscriber, ParakeetCTCAligner

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


def transcribe_from_cleaned_audio(
    settings: SessionSettings,
    session_dir: Path,
    use_cache_if_present: bool,
    gpu_logger: GpuLogger,
    logger: LoggingProtocol,
) -> TranscriptionResult:
    original_path: Path = session_dir / settings.cleaned_audio_file
    final_path: Path = session_dir / settings.transcript_file

    logger.report_message(f"[blue]Transcribing from clean audio at {original_path}[/blue]")
    if final_path.exists():
        logger.report_message(f"[yellow]{final_path} already exists, returning cached instance.[/yellow]")
        return TranscriptionResult.load(final_path)

    if not original_path.exists():
        raise FileNotFoundError(original_path)

    gpu_logger.report_gpu_usage("before processing")

    transcriber: TranscriberProtocol
    aligner: ParakeetCTCAligner
    with logger.status("Creating transcriber and aligner."):
        transcriber = CanaryQwenTranscriber(device=settings.device)
        gpu_logger.report_gpu_usage("Created transcriber")

        aligner = ParakeetCTCAligner(device=settings.device)
        gpu_logger.report_gpu_usage("Created aligner")

    result: TranscriptionResult = transcriber.transcribe(original_path, logger)
    gpu_logger.report_gpu_usage("After transcription")

    alignment: AlignmentResult = aligner.align(original_path, result.full_text, logger)
    gpu_logger.report_gpu_usage("after alignment")

    with logger.status("Rebuilding segments with confidence."):
        result.segments = _rebuild_segments_from_alignment(alignment)

    logger.report_message("[blue]Transcription complete.[/blue]")
    return result
