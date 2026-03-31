from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Self

from .nemo_vad_detector import SpeechSegment, VadResult

logger = logging.getLogger(__name__)


@dataclass
class CutPoint:
    """A point where audio should be split, located in the center of a silence gap."""

    time: float  # seconds — the cut location
    silence_start: float  # start of the silence gap
    silence_end: float  # end of the silence gap
    silence_duration: float


@dataclass
class AudioSegment:
    """A chunk of audio bounded by cut points."""

    index: int
    start: float  # seconds
    end: float  # seconds
    duration: float  # seconds
    speech_ratio: float  # fraction of segment that is speech (0.0–1.0)


@dataclass
class SegmentSplitResult:
    """Complete output of VAD-based segmentation."""

    cut_points: list[CutPoint]
    segments: list[AudioSegment]
    audio_duration: float
    min_segment_length: float
    max_segment_length: float
    speech_segments: list[SpeechSegment]

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "audio_duration": self.audio_duration,
                    "min_segment_length": self.min_segment_length,
                    "max_segment_length": self.max_segment_length,
                    "segments": [asdict(s) for s in self.segments],
                    "cut_points": [asdict(c) for c in self.cut_points],
                    "speech_segments": [asdict(s) for s in self.speech_segments],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> Self:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            audio_duration=data["audio_duration"],
            min_segment_length=data["min_segment_length"],
            max_segment_length=data["max_segment_length"],
            segments=[AudioSegment(**s) for s in data["segments"]],
            cut_points=[CutPoint(**c) for c in data["cut_points"]],
            speech_segments=[SpeechSegment(**s) for s in data["speech_segments"]],
        )


def _compute_speech_ratio(
    start: float,
    end: float,
    speech_segments: list[SpeechSegment],
) -> float:
    """Compute the fraction of [start, end) that overlaps with speech."""
    duration = end - start
    if duration <= 0:
        return 0.0
    speech_time = 0.0
    for seg in speech_segments:
        if not seg.is_speech:
            continue
        overlap_start = max(start, seg.start)
        overlap_end = min(end, seg.end)
        if overlap_end > overlap_start:
            speech_time += overlap_end - overlap_start
    return min(speech_time / duration, 1.0)


def compute_segments(
    vad_result: VadResult,
    min_length: float,
    max_length: float,
) -> SegmentSplitResult:
    """Compute optimal cut points from VAD results respecting min/max segment length.

    The algorithm prefers cutting at the center of the longest silence gap
    within the allowed window. If no silence gap exists within the window,
    a hard cut is made at max_length.
    """
    total = vad_result.audio_duration

    # Short audio — single segment, no cuts
    if total <= max_length:
        seg = AudioSegment(
            index=0,
            start=0.0,
            end=total,
            duration=total,
            speech_ratio=_compute_speech_ratio(0.0, total, vad_result.segments),
        )
        return SegmentSplitResult(
            cut_points=[],
            segments=[seg],
            audio_duration=total,
            min_segment_length=min_length,
            max_segment_length=max_length,
            speech_segments=vad_result.segments,
        )

    # Build candidate cut points from silence gaps
    silence_gaps = vad_result.silence_segments()
    candidates: list[CutPoint] = []
    for gap in silence_gaps:
        center = (gap.start + gap.end) / 2.0
        candidates.append(
            CutPoint(
                time=center,
                silence_start=gap.start,
                silence_end=gap.end,
                silence_duration=gap.duration,
            )
        )
    candidates.sort(key=lambda c: c.time)

    # Greedy sweep: pick the widest silence gap within the [min, max] window
    committed_cuts: list[CutPoint] = []
    current_start = 0.0
    idx = 0  # pointer into candidates

    while current_start + max_length < total:
        best: CutPoint | None = None

        # Scan candidates in the [min_length, max_length] window from current_start
        scan = idx
        while scan < len(candidates):
            c = candidates[scan]
            distance = c.time - current_start

            if distance < min_length:
                scan += 1
                continue

            if distance > max_length:
                break

            # Prefer the widest silence gap
            if best is None or c.silence_duration > best.silence_duration:
                best = c

            scan += 1

        if best is not None:
            committed_cuts.append(best)
            current_start = best.time
            # Advance idx past the committed cut
            while idx < len(candidates) and candidates[idx].time <= current_start:
                idx += 1
        else:
            # No silence gap in window — force-cut at max_length
            force_time = current_start + max_length
            logger.warning(
                "No silence gap found between %.1fs and %.1fs; forcing hard cut at %.1fs",
                current_start + min_length,
                current_start + max_length,
                force_time,
            )
            committed_cuts.append(
                CutPoint(
                    time=force_time,
                    silence_start=force_time,
                    silence_end=force_time,
                    silence_duration=0.0,
                )
            )
            current_start = force_time
            # Advance idx past the force-cut
            while idx < len(candidates) and candidates[idx].time <= current_start:
                idx += 1

    # Build segments from cut points
    boundaries = [0.0] + [c.time for c in committed_cuts] + [total]
    segments: list[AudioSegment] = []
    for i in range(len(boundaries) - 1):
        seg_start = boundaries[i]
        seg_end = boundaries[i + 1]
        segments.append(
            AudioSegment(
                index=i,
                start=seg_start,
                end=seg_end,
                duration=seg_end - seg_start,
                speech_ratio=_compute_speech_ratio(seg_start, seg_end, vad_result.segments),
            )
        )

    return SegmentSplitResult(
        cut_points=committed_cuts,
        segments=segments,
        audio_duration=total,
        min_segment_length=min_length,
        max_segment_length=max_length,
        speech_segments=vad_result.segments,
    )


@dataclass
class SegmentSplitResultSet:
    short: SegmentSplitResult
    long: SegmentSplitResult

    def _result_to_dict(self, result: SegmentSplitResult) -> dict:
        return {
            "audio_duration": result.audio_duration,
            "min_segment_length": result.min_segment_length,
            "max_segment_length": result.max_segment_length,
            "segments": [asdict(s) for s in result.segments],
            "cut_points": [asdict(c) for c in result.cut_points],
            "speech_segments": [asdict(s) for s in result.speech_segments],
        }

    @staticmethod
    def _result_from_dict(data: dict) -> SegmentSplitResult:
        return SegmentSplitResult(
            audio_duration=data["audio_duration"],
            min_segment_length=data["min_segment_length"],
            max_segment_length=data["max_segment_length"],
            segments=[AudioSegment(**s) for s in data["segments"]],
            cut_points=[CutPoint(**c) for c in data["cut_points"]],
            speech_segments=[SpeechSegment(**s) for s in data["speech_segments"]],
        )

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "short": self._result_to_dict(self.short),
                    "long": self._result_to_dict(self.long),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> SegmentSplitResultSet:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            short=cls._result_from_dict(data["short"]),
            long=cls._result_from_dict(data["long"]),
        )
