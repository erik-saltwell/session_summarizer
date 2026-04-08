from __future__ import annotations

from pathlib import Path

from ..processing_results import AlignmentResult, TranscriptionSegment
from ..protocols import (
    GpuLogger,
    LoggingProtocol,
    SessionSettings,
)
from ..transcription import ParakeetCTCConfidenceScorer
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
    seg_start = alignment.words[0].start_time
    seg_end = alignment.words[0].end_time
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
        gap = word.start_time - prev.end_time
        ends_sentence = prev.word and prev.word[-1] in _SENTENCE_ENDERS
        duration = word.start_time - seg_start
        if gap >= _PAUSE_THRESHOLD_S or ends_sentence or duration >= _MAX_SEGMENT_DURATION_S:
            _flush()
            seg_start = word.start_time
            min_conf = word.confidence

        current_words.append(word.word)
        seg_end = word.end_time
        min_conf = min(min_conf, word.confidence)

    _flush()
    return segments


def score_confidence(
    settings: SessionSettings,
    session_dir: Path,
    aligned_transcription: AlignmentResult,
    segments: SegmentSplitResultSet,
    gpu_logger: GpuLogger,
    logger: LoggingProtocol,
) -> AlignmentResult:
    logger.report_message("[blue]Creating confidence scores.[/blue]")
    audio_path: Path = session_dir / settings.cleaned_audio_file

    gpu_logger.report_gpu_usage("before processing")

    scorer: ParakeetCTCConfidenceScorer
    with logger.status("Creating scorer."):
        scorer = ParakeetCTCConfidenceScorer(device=settings.device)
        gpu_logger.report_gpu_usage("Created aligner")

    scored_alignment: AlignmentResult
    with logger.status("Scoring confidence."):
        scored_alignment = scorer.score(audio_path, aligned_transcription, segments, logger)

    gpu_logger.report_gpu_usage("after alignment")

    logger.report_message("[blue]Alignment complete.[/blue]")
    return scored_alignment
