from __future__ import annotations

from typing import NamedTuple

from ..processing_results import SpeechClip, WordAlignment
from ..settings import DiarizationStitchingSettings


class CandidateScore(NamedTuple):
    score: float


def score_candidate(
    candidate_clip: SpeechClip, word: WordAlignment, settings: DiarizationStitchingSettings
) -> CandidateScore:
    # Example scoring function based on overlap duration and confidence

    return CandidateScore(score=0.0)
