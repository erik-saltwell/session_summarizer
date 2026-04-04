from __future__ import annotations

from attr import dataclass

from session_summarizer.diarization.clip_merger import MergeType

from ..processing_results.speech_clip_set import SpeechClip, SpeechClipSet
from ..protocols import LoggingProtocol
from ..settings.diarization_stitching_settings import ScoringMode
from ..settings.session_settings import DiarizationStitchingSettings, SessionSettings
from ..transcription.parakeet_ctc_confidence_scorer import AlignmentResult, WordAlignment
from .anonymous_clips import AnonymousClips
from .candidate_pool import CandidatePool
from .candidate_score import CandidateScore, score_candidate
from .clip_merger import MergeSelector, clips_are_close_enough, clips_are_same_speaker, merge_clips
from .diarizen_diarizer import MergedDiarizationResult, MergedDiarizationSegment


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


@dataclass
class SimpleMergeSelector(MergeSelector):
    exempt_anonymous = False

    def ShouldMerge(
        self,
        prior_clip: SpeechClip,
        current_clip: SpeechClip,
        next_clip: SpeechClip | None,
        settings: DiarizationStitchingSettings,
        logger: LoggingProtocol,
    ) -> MergeType:
        if not clips_are_close_enough(prior_clip, current_clip, settings.merge_gap_seconds, settings.epsilon, logger):
            return MergeType.NO_MERGE
        if not clips_are_same_speaker(prior_clip, current_clip, settings, self.exempt_anonymous, logger):
            return MergeType.NO_MERGE
        return MergeType.MERGE_WITH_PRIOR


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


def _remove_empty_clips(clips: SpeechClipSet) -> SpeechClipSet:
    return_clips: SpeechClipSet = SpeechClipSet()
    for clip in clips:
        if clip.words:
            return_clips.add_clip(clip)
    return_clips.sort_clips()
    return return_clips


def create_speech_clips(
    diarization_result: MergedDiarizationResult,
    alignment_result: AlignmentResult,
    settings: SessionSettings,
    logger: LoggingProtocol,
) -> SpeechClipSet:
    speech_clips = _create_initial_clips(diarization_result)
    speech_clips = merge_clips(speech_clips, SimpleMergeSelector(), settings.diarization_stitching, logger)
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
        else:
            anonymous_clips.add_anonymous_word(word, stitch_settings)

    speech_clips.extend_clips(anonymous_clips.get_clips())

    if settings.diarization_stitching.expand_segments_to_fit_words:
        for clip in speech_clips:
            clip.expand_bounds_to_include_words(epsilon, settings.diarization_stitching.expansion_limit_seconds)

    speech_clips = merge_clips(speech_clips, SimpleMergeSelector(), settings.diarization_stitching, logger)
    speech_clips = _remove_empty_clips(speech_clips)
    speech_clips.sort_clips()

    for clip in speech_clips:
        clip.compute_word_derived_values()

    return speech_clips
