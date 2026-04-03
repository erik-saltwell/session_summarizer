from __future__ import annotations

from session_summarizer.protocols.logging_protocol import LoggingProtocol
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


def _fmt_clip(clip: SpeechClip) -> str:
    speakers = ",".join(sorted(clip.speakers))
    return f"clip[{clip.start_time:.3f}-{clip.end_time:.3f} spk={speakers} words={clip.word_count}]"


def _fmt_word(word: WordAlignment) -> str:
    return f"'{word.word}'[{word.start_time:.3f}-{word.end_time:.3f}]"


def _merge_clips(
    initial_clips: SpeechClipSet,
    settings: DiarizationStitchingSettings,
    logger: LoggingProtocol,
    should_trace: bool,
) -> SpeechClipSet:
    merged_clips: SpeechClipSet = SpeechClipSet()
    for clip in initial_clips:
        if len(merged_clips) == 0:
            merged_clips.add_clip(clip)
        else:
            last: SpeechClip = merged_clips[-1]
            same_speaker: bool = clip.speakers == last.speakers
            gap: float = last.gap_distance(clip, settings.epsilon)

            if same_speaker and gap <= (settings.merge_gap_seconds + settings.epsilon):
                if should_trace:
                    logger.report_message(
                        f"  merge_clips: merging {_fmt_clip(last)} + {_fmt_clip(clip)}"
                        f" (gap={gap:.3f}s <= merge_gap={settings.merge_gap_seconds:.3f}s)"
                    )
                last.merge(clip)
            else:
                if should_trace:
                    reason = (
                        f"different speakers ({','.join(sorted(last.speakers))} vs {','.join(sorted(clip.speakers))})"
                        if not same_speaker
                        else f"gap={gap:.3f}s > merge_gap={settings.merge_gap_seconds:.3f}s"
                    )
                    logger.report_message(
                        f"  merge_clips: keeping separate {_fmt_clip(last)} | {_fmt_clip(clip)} ({reason})"
                    )
                merged_clips.add_clip(clip)
    return merged_clips


def _is_acceptable_overlap(
    word: WordAlignment,
    clip: SpeechClip,
    settings: DiarizationStitchingSettings,
    logger: LoggingProtocol,
    should_trace: bool,
) -> bool:
    overlap = word.overlap(clip, settings.epsilon)
    if word.duration <= settings.epsilon or clip.duration <= settings.epsilon:
        if should_trace:
            logger.report_message(
                f"    overlap_check: REJECT (too short) {_fmt_word(word)} vs {_fmt_clip(clip)}"
                f" word_dur={word.duration:.3f}s clip_dur={clip.duration:.3f}s eps={settings.epsilon:.3f}"
            )
        return False  # if either is too short, we don't consider it a meaningful overlap

    duration_within_meaningful_boundaries = word.duration_inside_meaningful_boundaries(settings.epsilon)
    if duration_within_meaningful_boundaries <= settings.epsilon:
        if should_trace:
            logger.report_message(
                f"    overlap_check: REJECT (word outside meaningful bounds) {_fmt_word(word)} vs {_fmt_clip(clip)}"
                f" dur_in_bounds={duration_within_meaningful_boundaries:.3f}s"
            )
        return False
    if overlap >= settings.min_overlap_seconds - settings.epsilon:
        if should_trace:
            logger.report_message(
                f"    overlap_check: ACCEPT (overlap_seconds) {_fmt_word(word)} vs {_fmt_clip(clip)}"
                f" overlap={overlap:.3f}s >= min_overlap={settings.min_overlap_seconds:.3f}s"
            )
        return True
    overlap_fraction = overlap / duration_within_meaningful_boundaries
    if overlap_fraction >= settings.min_overlap_fraction_word - settings.epsilon:
        if should_trace:
            logger.report_message(
                f"    overlap_check: ACCEPT (overlap_fraction) {_fmt_word(word)} vs {_fmt_clip(clip)}"
                f" overlap={overlap:.3f}s fraction={overlap_fraction:.3f}"
                f" >= min_frac={settings.min_overlap_fraction_word:.3f}"
            )
        return True
    if should_trace:
        logger.report_message(
            f"    overlap_check: REJECT {_fmt_word(word)} vs {_fmt_clip(clip)}"
            f" overlap={overlap:.3f}s fraction={overlap_fraction:.3f}"
            f" (needs >={settings.min_overlap_seconds:.3f}s or >={settings.min_overlap_fraction_word:.3f} frac)"
        )
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
    logger: LoggingProtocol,
    should_trace: bool,
) -> tuple[SpeechClip | None, CandidateScore | None]:
    best_candidate: SpeechClip | None = None
    best_score: CandidateScore | None = None

    if should_trace:
        candidates_in_pool = list(pool.iterate_candidates(speech_clips))
        logger.report_message(f"  find_best: {_fmt_word(word)} — {len(candidates_in_pool)} candidate(s) in pool:")
        for c in candidates_in_pool:
            logger.report_message(f"    candidate: {_fmt_clip(c)}")

    for candidate in pool.iterate_candidates(speech_clips):
        overlap = word.overlap(candidate, epsilon)
        gap = word.gap_distance(candidate, epsilon)
        if _is_acceptable_overlap(word, candidate, stitch_settings, logger, should_trace):
            score = score_candidate(candidate, word, epsilon, scoring_mode, prefer_shorter_on_tie)
            is_new_best = best_score is None or score > best_score
            if should_trace:
                logger.report_message(
                    f"    scored (overlap path): {_fmt_clip(candidate)}"
                    f" score=({score.overlap_score:.4f},{score.neg_mid_dist:.4f},{score.neg_gap:.4f})"
                    + (" <- NEW BEST" if is_new_best else "")
                )
            if is_new_best:
                best_score = score
                best_candidate = candidate
        elif should_fill_nearest and (overlap > 0.0 or gap <= max_nearest_distance):
            score = score_candidate(candidate, word, epsilon, scoring_mode, prefer_shorter_on_tie, ignore_overlap=True)
            is_new_best = best_score is None or score > best_score
            if should_trace:
                logger.report_message(
                    f"    scored (fill_nearest path): {_fmt_clip(candidate)}"
                    f" overlap={overlap:.3f}s gap={gap:.3f}s max_nearest={max_nearest_distance:.3f}s"
                    f" score=({score.overlap_score:.4f},{score.neg_mid_dist:.4f},{score.neg_gap:.4f})"
                    + (" <- NEW BEST" if is_new_best else "")
                )
            if is_new_best:
                best_score = score
                best_candidate = candidate
        elif should_trace:
            logger.report_message(
                f"    skipped: {_fmt_clip(candidate)}"
                f" overlap={overlap:.3f}s gap={gap:.3f}s"
                f" (no acceptable overlap, fill_nearest={should_fill_nearest})"
            )

    if should_trace:
        if best_candidate is not None:
            logger.report_message(f"  find_best: winner -> {_fmt_clip(best_candidate)}")
        else:
            logger.report_message("  find_best: no winner found")

    return (best_candidate, best_score)


def create_speech_clips(
    diarization_result: MergedDiarizationResult,
    alignment_result: AlignmentResult,
    settings: SessionSettings,
    logger: LoggingProtocol,
) -> SpeechClipSet:
    speech_clips = _create_initial_clips(diarization_result)
    speech_clips = _merge_clips(speech_clips, settings.diarization_stitching, logger, should_trace=False)
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
    should_trace: bool = False
    seen_count: int = 0
    for word in alignment_result.words:
        if word.word.lower() == "no," or word.word.lower() == "backside":
            seen_count += 1
            should_trace = seen_count <= 2
        else:
            should_trace = False

        if should_trace:
            logger.report_message(f"TRACE word: {_fmt_word(word)} dur={word.duration:.3f}s conf={word.confidence:.3f}")

        pool.update_pool(word, speech_clips, stitch_settings, logger, should_trace)

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
            logger,
            should_trace,
        )

        if best_candidate is not None:
            if should_trace:
                logger.report_message(f"  ASSIGN: {_fmt_word(word)} -> {_fmt_clip(best_candidate)}")
            best_candidate.add_word(word)
            anonymous_clips.flush_current_clip()
        else:
            if should_trace:
                logger.report_message(f"  ASSIGN: {_fmt_word(word)} -> anonymous segment")
            anonymous_clips.add_anonymous_word(word, stitch_settings)

    speech_clips.extend_clips(anonymous_clips.get_clips())

    if settings.diarization_stitching.expand_segments_to_fit_words:
        for clip in speech_clips:
            clip.expand_bounds_to_include_words(epsilon, settings.diarization_stitching.expansion_limit_seconds)

    speech_clips = _merge_clips(speech_clips, settings.diarization_stitching, logger, should_trace=False)

    speech_clips.sort_clips()

    for clip in speech_clips:
        clip.finalize_words()

    return speech_clips
