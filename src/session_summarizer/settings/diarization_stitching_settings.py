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

    # Maximum gap (seconds) between an unfinished speech clip (not marked
    # as an end-of-turn) and a following clip with the same speaker, for
    # them to be merged. Helps preserve conversational flow by avoiding
    # artificial breaks in ongoing speech, while respecting turn boundaries.
    unfinished_clip_merge_max_length: float

    # Maximum gap (seconds) between two clips with the same identified
    # speaker that can still be merged into a single clip during identity
    # stitching.  Larger values allow merging across longer pauses; smaller
    # values keep clips separate when the same speaker resumes after silence.
    identity_stitching_max_gap: float

    # Minimum cosine similarity (0.0–1.0) between two clip embeddings for
    # them to be considered the same speaker during identity stitching.
    # Lower values accept weaker matches; higher values require stronger
    # acoustic similarity before merging.
    identity_similarity_threshold: float

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

    # ── Backchannel detection ──────────────────────────────────────────

    # Maximum duration (seconds) a clip may be to still qualify as a
    # backchannel utterance (e.g. "mm-hmm", "right", "yeah").  Clips longer
    # than this threshold are never treated as backchannels.
    max_backchannel_duration: float

    # Maximum gap (seconds) between a clip and its predecessor for the clip
    # to be considered a backchannel.  Backchannels typically occur during or
    # immediately after another speaker's speech; large prior-gaps indicate a
    # new independent utterance instead.
    max_backchannel_prior_gap: float

    # Maximum gap (seconds) between a clip and its successor for the clip to
    # be considered a backchannel.  If the following speech is far away, the
    # short utterance is more likely a standalone contribution than a
    # mid-stream backchannel.
    max_backchannel_next_gap: float

    # ── Identity-based backchannel detection ───────────────────────────
    # Same concept as backchannel detection above, but applied during
    # identity stitching where speaker labels come from embedding
    # similarity rather than diarization labels.

    # Maximum duration (seconds) a clip may be to still qualify as a
    # backchannel during identity stitching.
    max_identity_backchannel_duration: float

    # Maximum gap (seconds) between a clip and its predecessor for the
    # clip to be considered a backchannel during identity stitching.
    max_identity_backchannel_prior_gap: float

    # Maximum gap (seconds) between a clip and its successor for the
    # clip to be considered a backchannel during identity stitching.
    max_identity_backchannel_next_gap: float

    # ── Turn detection ─────────────────────────────────────────────────

    # Probability threshold for classifying a speech clip as the end of a
    # conversational turn.  A clip whose AI-model turn-end probability meets
    # or exceeds this value is flagged as a turn boundary.  Lower values
    # increase sensitivity (more boundaries detected); higher values require
    # stronger model confidence.
    turn_end_probability_threshold: float

    # ── Numeric tolerance ──────────────────────────────────────────────

    # Small value added/subtracted when comparing floating-point time
    # boundaries to avoid edge cases from imprecision and quantization.
    epsilon: float

    @field_validator(
        "min_overlap_seconds",
        "max_nearest_distance",
        "anonymous_join_gap",
        "merge_gap_seconds",
        "unfinished_clip_merge_max_length",
        "identity_stitching_max_gap",
        "expansion_limit_seconds",
        "max_backchannel_duration",
        "max_backchannel_prior_gap",
        "max_backchannel_next_gap",
        "max_identity_backchannel_duration",
        "max_identity_backchannel_prior_gap",
        "max_identity_backchannel_next_gap",
        "epsilon",
    )
    @classmethod
    def non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("must be non-negative")
        return v

    @field_validator("min_overlap_fraction_word", "turn_end_probability_threshold", "identity_similarity_threshold")
    @classmethod
    def zero_to_one(cls, v: float) -> float:
        if v < 0 or v > 1:
            raise ValueError("must be between 0 and 1")
        return v
