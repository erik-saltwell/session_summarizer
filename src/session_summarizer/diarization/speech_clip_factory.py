from __future__ import annotations

from session_summarizer.settings.diarization_stitching_settings import ScoringMode

from ..processing_results.speech_clip_set import SpeechClip, SpeechClipSet
from ..settings.session_settings import DiarizationStitchingSettings, SessionSettings
from ..transcription.parakeet_ctc_confidence_scorer import AlignmentResult, WordAlignment
from .anonymous_clips import AnonymousClips
from .candidate_pool import CandidatePool
from .candidate_score import CandidateScore, score_candidate
from .diarizen_diarizer import MergedDiarizationResult, MergedDiarizationSegment

# _CROSSTALK = "[crosstalk]"


# def _create_text_for_segment(words: list[WordAlignment], is_multispeaker: bool) -> str:
#     # if is_multispeaker:
#     #     return _CROSSTALK
#     return " ".join(w.word for w in words)


# def _create_confidence_avg_for_segment(words: list[WordAlignment]) -> float:
#     if not words:
#         return 0.0
#     return sum(w.confidence for w in words) / len(words)


def _merge_clips(initial_clips: SpeechClipSet, settings: DiarizationStitchingSettings) -> SpeechClipSet:
    if not settings.merge_adjacent_same_speaker:
        return initial_clips

    merged_clips: SpeechClipSet = SpeechClipSet()
    for clip in initial_clips:
        if len(merged_clips) == 0:
            merged_clips.add_clip(clip)
        else:
            last: SpeechClip = merged_clips[-1]
            same_speaker: bool = clip.speakers == last.speakers
            gap: float = last.gap_distance(clip, settings.epsilon)

            if same_speaker and gap <= (settings.merge_gap_seconds + settings.epsilon):
                last.merge(clip)
            else:
                merged_clips.add_clip(clip)
    return merged_clips


def _is_acceptable_overlap(word: WordAlignment, clip: SpeechClip, settings: DiarizationStitchingSettings) -> bool:
    overlap = word.overlap(clip, settings.epsilon)
    if word.duration <= settings.epsilon or clip.duration <= settings.epsilon:
        return False  # if either is too short, we don't consider it a meaningful overlap

    duration_within_meaningful_boundaries = word.duration_inside_meaningful_boundaries(settings.epsilon)
    if duration_within_meaningful_boundaries <= settings.epsilon:
        return False
    if overlap >= settings.min_overlap_seconds - settings.epsilon:
        return True
    if (overlap / duration_within_meaningful_boundaries) >= settings.min_overlap_fraction_word - settings.epsilon:
        return True
    return False


def _create_initial_clips(diarization_result: MergedDiarizationResult) -> SpeechClipSet:
    clip_set: SpeechClipSet = SpeechClipSet()
    segments: list[MergedDiarizationSegment] = sorted(
        diarization_result.segments, key=lambda s: (s.start_time, s.end_time)
    )
    for segment in segments:
        speech_clip = SpeechClip(
            start_time=segment.start_time,
            end_time=segment.end_time,
            speakers=set(segment.speakers),
            confidence_avg=0.0,
            text="",
        )
        clip_set.add_clip(speech_clip)
    return clip_set


def _find_best_candidate(
    pool: CandidatePool,
    word: WordAlignment,
    stitch_settings: DiarizationStitchingSettings,
    speech_clips: SpeechClipSet,
    epsilon: float,
    scoring_mode: ScoringMode,
    prefer_shorter_on_tie: bool,
    should_fill_nearest: bool,
    max_nearest_distance: float,
) -> tuple[SpeechClip | None, CandidateScore | None]:
    best_candidate: SpeechClip | None = None
    best_score: CandidateScore | None = None
    for candidate in pool.iterate_candidates(speech_clips):
        overlap = word.overlap(candidate, epsilon)
        gap = word.gap_distance(candidate, epsilon)
        if _is_acceptable_overlap(word, candidate, stitch_settings):
            score = score_candidate(candidate, word, epsilon, scoring_mode, prefer_shorter_on_tie)
            if best_score is None or score > best_score:
                best_score = score
                best_candidate = candidate
        elif should_fill_nearest and (overlap > 0.0 or gap <= max_nearest_distance):
            score = score_candidate(candidate, word, epsilon, scoring_mode, prefer_shorter_on_tie, ignore_overlap=True)
            if best_score is None or score > best_score:
                best_score = score
                best_candidate = candidate
    return (best_candidate, best_score)


def create_speech_clips(
    diarization_result: MergedDiarizationResult,
    alignment_result: AlignmentResult,
    settings: SessionSettings,
) -> SpeechClipSet:
    speech_clips = _create_initial_clips(diarization_result)
    speech_clips = _merge_clips(speech_clips, settings.diarization_stitching)
    pool: CandidatePool = CandidatePool()
    anonymous_clips: AnonymousClips = AnonymousClips([])

    # caching values for efficiency since they are used in inner loops
    epsilon: float = settings.diarization_stitching.epsilon
    stitch_settings = settings.diarization_stitching
    should_fill_nearest: bool = stitch_settings.fill_nearest
    max_nearest_distance: float = stitch_settings.max_nearest_distance + epsilon
    scoring_mode: ScoringMode = stitch_settings.scoring_mode
    prefer_shorter_on_tie: bool = stitch_settings.prefer_shorter_on_tie

    alignment_result.sort()

    for word in alignment_result.words:
        pool.update_pool(word, speech_clips, stitch_settings)

        best_candidate: SpeechClip | None
        best_score: CandidateScore | None
        best_candidate, best_score = _find_best_candidate(
            pool,
            word,
            stitch_settings,
            speech_clips,
            epsilon,
            scoring_mode,
            prefer_shorter_on_tie,
            should_fill_nearest,
            max_nearest_distance,
        )

        if best_candidate is not None:
            best_candidate.add_word(word)
            anonymous_clips.flush_current_clip()
        elif settings.diarization_stitching.create_anonymous_segments:
            anonymous_clips.add_anonymous_word(word, stitch_settings)
        else:
            nearest_candidate = pool.get_nearest_candidate_to_midpoint(word, speech_clips)
            nearest_candidate.add_word(word)
            anonymous_clips.flush_current_clip()

    speech_clips.extend_clips(anonymous_clips.get_clips())

    if settings.diarization_stitching.expand_segments_to_fit_words:
        for clip in speech_clips:
            clip.expand_bounds_to_include_words(epsilon, settings.diarization_stitching.expansion_limit_seconds)

    speech_clips = _merge_clips(speech_clips, settings.diarization_stitching)

    speech_clips.sort_clips()

    for clip in speech_clips:
        clip.finalize_words()

    return speech_clips
