from enum import StrEnum

from pydantic import BaseModel, field_validator


class ScoringMode(StrEnum):
    """How to rank candidate diarization segments that overlap a word.

    Each mode first scores by its primary metric; ties are broken by
    midpoint distance (word midpoint vs. segment midpoint).
    """

    # Primary: raw overlap in seconds. Favors the segment that covers the
    # most absolute time of the word, regardless of word or segment length.
    overlap_seconds_then_midpoint = "overlap_seconds_then_midpoint"

    # Primary: overlap as a fraction of the word's duration. Useful when
    # words vary widely in length and you want "percentage covered" to drive
    # the decision rather than raw seconds.
    overlap_fraction_word_then_midpoint = "overlap_fraction_word_then_midpoint"

    # Primary: intersection-over-union of the word and segment intervals.
    # Penalizes very long segments that happen to contain the word, because
    # the union is large. Good when diarization segments vary in length.
    iou_then_midpoint = "iou_then_midpoint"


class DiarizationStitchingSettings(BaseModel, frozen=True):
    """Policy knobs for assigning ASR words (with timestamps) to diarized
    speaker segments (with timestamps and speaker labels).

    The stitching algorithm iterates words in time order and, for each word,
    scores candidate diarization segments by overlap.  When no segment
    overlaps acceptably, a fallback chain applies: nearest-segment assignment,
    then anonymous-segment creation.  After all words are assigned,
    optional post-processing merges and expands segments.

    See .research/speaker_segment_assignment.md for the full design rationale.
    """

    # ── Overlap acceptance thresholds ──────────────────────────────────
    # A candidate segment may pass *either* thresholds to count as an
    # "in-range" overlap.  Relaxed defaults accommodate the boundary jitter
    # inherent in both ASR word timestamps and diarization segment edges.

    # Minimum fraction of the word's duration that must be overlapped by
    # the candidate segment.  0.20 means at least 20% of the word must
    # fall inside the segment.  Avoids "barely touching" overlaps that
    # arise from boundary jitter.
    min_overlap_fraction_word: float

    # Absolute floor: ignore overlaps shorter than this (seconds).
    # 20 ms matches typical speech-processing frame sizes (25 ms); overlaps
    # below one frame are not acoustically meaningful.
    min_overlap_seconds: float

    # ── Fallback: nearest-segment assignment ───────────────────────────
    # When no candidate passes the overlap thresholds, the algorithm can
    # assign the word to the closest segment by midpoint distance, as long
    # as the gap between their intervals is within max_nearest_distance.

    # Enable nearest-segment fallback.
    fill_nearest: bool

    # Maximum gap (seconds) between a word and a non-overlapping segment
    # for nearest-assignment to apply. 250 ms is a common tolerance scale
    # in speech scoring; keeps the fallback conservative so it won't jump
    # speakers across long silences.
    max_nearest_distance: float

    # ── Fallback: anonymous segments ───────────────────────────────────
    # If nearest-assignment also fails (or is disabled), words can be
    # placed into auto-created "anonymous" segments so that every word is
    # covered.  Consecutive anonymous words close in time are merged into
    # a single anonymous span.

    # Maximum gap (seconds) between consecutive anonymous words that will
    # be merged into the same anonymous segment.
    anonymous_join_gap: float

    # ── Post-processing ────────────────────────────────────────────────

    # Merge adjacent segments that share the same speaker label when
    # separated by at most merge_gap_seconds.  Reduces fragmentation
    # caused by brief pauses or diarization over-segmentation.
    merge_gap_seconds: float

    # Widen each segment's time boundaries to fully contain its assigned
    # words.  Useful for UI rendering where words must not extend beyond
    # their parent segment, but reduces diarization boundary fidelity.
    expand_segments_to_fit_words: bool

    # Cap on how far a segment boundary may be expanded (seconds).
    expansion_limit_seconds: float

    # ── Candidate scoring ──────────────────────────────────────────────

    # How to rank candidate segments that overlap a word. See ScoringMode
    # for details on each strategy.
    scoring_mode: ScoringMode

    # When two candidates score identically, prefer the shorter segment.
    # This avoids bias toward long segments that span many words.
    prefer_shorter_on_tie: bool

    # ── Numeric tolerance ──────────────────────────────────────────────

    # Small value added/subtracted when comparing floating-point time
    # boundaries to avoid edge cases from imprecision and quantization.
    epsilon: float

    @field_validator(
        "min_overlap_seconds",
        "max_nearest_distance",
        "anonymous_join_gap",
        "merge_gap_seconds",
        "expansion_limit_seconds",
        "epsilon",
    )
    @classmethod
    def non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("must be non-negative")
        return v

    @field_validator("min_overlap_fraction_word")
    @classmethod
    def zero_to_one(cls, v: float) -> float:
        if v < 0 or v > 1:
            raise ValueError("must be between 0 and 1")
        return v
