from __future__ import annotations

from pathlib import Path

from ..processing_results import TranscriptionResult, TranscriptionSegment
from ..protocols import (
    GpuLogger,
    LoggingProtocol,
    SessionSettings,
)
from ..transcription import AlignmentResult, ParakeetCTCWordAligner
from ..vad import SegmentSplitResultSet

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


def align_transcript(
    settings: SessionSettings,
    session_dir: Path,
    transcription: TranscriptionResult,
    segments: SegmentSplitResultSet,
    use_cache_if_present: bool,
    gpu_logger: GpuLogger,
    logger: LoggingProtocol,
) -> AlignmentResult:
    logger.report_message("[blue]Word aligning transcription.[/blue]")
    final_path: Path = session_dir / settings.aligned_transcript_path
    if final_path.exists() and use_cache_if_present:
        logger.report_message(f"[yellow]{final_path} already exists, returning cached instance.[/yellow]")
        return AlignmentResult.load_from_json(final_path)

    gpu_logger.report_gpu_usage("before processing")

    aligner: ParakeetCTCWordAligner
    with logger.status("Creating aligner."):
        aligner = ParakeetCTCWordAligner(device=settings.device)
        gpu_logger.report_gpu_usage("Created aligner")

    alignment: AlignmentResult = aligner.align(
        session_dir / settings.cleaned_audio_file, transcription, segments, logger
    )
    gpu_logger.report_gpu_usage("after alignment")

    logger.report_message("[blue]Alignment complete.[/blue]")
    return alignment
