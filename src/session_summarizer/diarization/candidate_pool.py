from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from heapq import heappop, heappush
from typing import NamedTuple

from ..processing_results import SpeechClip, SpeechClipSet, WordAlignment
from ..settings.diarization_stitching_settings import DiarizationStitchingSettings


class CandidateInfo(NamedTuple):
    end_time: float = 0.0
    clip_id: int = 0


@dataclass
class CandidatePool:
    _candidates: list[CandidateInfo] = field(default_factory=list)
    _next_clip_id: int = 0

    def update_pool(
        self,
        current_word: WordAlignment,
        all_clips: SpeechClipSet,
        settings: DiarizationStitchingSettings,
        epsilon: float,
    ) -> None:
        radius: float = settings.max_nearest_gap_seconds if settings.fill_nearest else 0.0
        radius += epsilon
        extended_start: float = current_word.start_time - radius
        extended_end: float = current_word.end_time + radius

        clip_count: int = len(all_clips)
        while self._next_clip_id < clip_count and all_clips[self._next_clip_id].start_time <= extended_end:
            heappush(self._candidates, CandidateInfo(all_clips[self._next_clip_id].end_time, self._next_clip_id))
            self._next_clip_id += 1

        while self._candidates and self._candidates[0].end_time < extended_start:
            heappop(self._candidates)

    def iterate_candidates(self, all_clips: SpeechClipSet) -> Iterator[SpeechClip]:
        for candidate in self._candidates:
            yield all_clips[candidate.clip_id]
