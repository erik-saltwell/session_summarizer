from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from heapq import heappop, heappush
from typing import NamedTuple

from session_summarizer.processing_results.alignment_result import WordAlignment
from session_summarizer.processing_results.speech_clip_set import SpeechClipSet

from ..processing_results import SpeechClip
from ..settings.diarization_stitching_settings import DiarizationStitchingSettings


class CandidateInfo(NamedTuple):
    end_time: float = 0.0
    clip_id: int = 0


@dataclass
class CandidatePool:
    candidates: list[CandidateInfo] = field(default_factory=list)
    next_clip_id: int = 0

    def update_pool(
        self, current_word: WordAlignment, all_clips: SpeechClipSet, settings: DiarizationStitchingSettings
    ) -> None:
        radius: float = settings.max_nearest_distance if settings.fill_nearest else 0.0
        radius += settings.epsilon
        extended_start: float = current_word.start_time - radius
        extended_end: float = current_word.end_time + radius

        clip_count: int = len(all_clips)
        while self.next_clip_id < clip_count and all_clips[self.next_clip_id].start_time <= extended_end:
            heappush(self.candidates, CandidateInfo(all_clips[self.next_clip_id].end_time, self.next_clip_id))
            self.next_clip_id += 1

        while self.candidates and self.candidates[0].end_time < extended_start:
            heappop(self.candidates)

    def iterate_candidates(self, all_clips: SpeechClipSet) -> Iterator[SpeechClip]:
        for candidate in self.candidates:
            yield all_clips[candidate.clip_id]
