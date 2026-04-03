from __future__ import annotations

from typing import NamedTuple

from ..processing_results import SpeechClip, WordAlignment
from ..settings.diarization_stitching_settings import ScoringMode


class CandidateScore(NamedTuple):
    """Lexicographic score tuple where larger is better.

    Fields:
        primary:        Main score determined by scoring_mode.
        neg_mid_dist:   Negative midpoint distance (closer = higher).
        neg_gap:        Negative gap distance (in-range preferred, then closer).
        shorter_bonus:  Negative segment duration when prefer_shorter_on_tie is set.
        neg_start:      Negative segment start time for deterministic tie-breaking.
    """

    overlap_score: float
    neg_mid_dist: float
    neg_gap: float
    shorter_bonus: float
    neg_start: float


def score_candidate(
    candidate_clip: SpeechClip,
    word: WordAlignment,
    epsilon: float,
    scoring_mode: ScoringMode,
    prefer_shorter_on_tie: bool,
    ignore_overlap: bool = False,
) -> CandidateScore:
    minimum_meaningful_length = epsilon

    word_duration = max(word.duration, minimum_meaningful_length)
    segment_duration = max(candidate_clip.duration, minimum_meaningful_length)

    overlap = 0.0 if ignore_overlap else word.overlap(candidate_clip, epsilon)
    gap = word.gap_distance(candidate_clip, minimum_meaningful_length)
    midpoint_distance = abs(word.midpoint - candidate_clip.midpoint)
    iou_overlap_ratio = overlap / max(word_duration + segment_duration - overlap, minimum_meaningful_length)

    overlap_score: float
    if scoring_mode == ScoringMode.overlap_seconds_then_midpoint:
        overlap_score = overlap
    elif scoring_mode == ScoringMode.overlap_fraction_word_then_midpoint:
        overlap_score = overlap / word_duration
    else:  # settings.scoring_mode == ScoringMode.iou_then_midpoint:
        overlap_score = iou_overlap_ratio

    shorter_bonus = -segment_duration if prefer_shorter_on_tie else 0.0

    return CandidateScore(
        overlap_score=overlap_score,
        neg_mid_dist=-midpoint_distance,
        neg_gap=-gap,
        shorter_bonus=shorter_bonus,
        neg_start=-candidate_clip.start_time,
    )
