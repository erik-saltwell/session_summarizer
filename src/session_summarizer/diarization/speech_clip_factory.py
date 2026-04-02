from __future__ import annotations

from ..processing_results.speech_clip_set import SpeechClip, SpeechClipSet
from ..settings.session_settings import DiarizationStitchingSettings, SessionSettings
from ..transcription.parakeet_ctc_confidence_scorer import AlignmentResult, WordAlignment
from .candidate_pool import CandidatePool
from .diarizen_diarizer import MergedDiarizationResult, MergedDiarizationSegment

_CROSSTALK = "[crosstalk]"


def _create_text_for_segment(words: list[WordAlignment], is_multispeaker: bool) -> str:
    # if is_multispeaker:
    #     return _CROSSTALK
    return " ".join(w.word for w in words)


def _create_confidence_avg_for_segment(words: list[WordAlignment]) -> float:
    if not words:
        return 0.0
    return sum(w.confidence for w in words) / len(words)


def _merge_initial_clips(initial_clips: SpeechClipSet, settings: DiarizationStitchingSettings) -> SpeechClipSet:
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


def _create_initial_clips(dierization_result: MergedDiarizationResult) -> SpeechClipSet:
    clip_set: SpeechClipSet = SpeechClipSet()
    clip_id: int = 0
    segments: list[MergedDiarizationSegment] = sorted(
        dierization_result.segments, key=lambda s: (s.start_time, s.end_time)
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
        clip_id += 1
    return clip_set


def create_speech_clips(
    diarization_result: MergedDiarizationResult,
    alignment_result: AlignmentResult,
    settings: SessionSettings,
) -> SpeechClipSet:
    speech_clips = _create_initial_clips(diarization_result)
    speech_clips = _merge_initial_clips(speech_clips, settings.diarization_stitching)
    pool: CandidatePool = CandidatePool()
    for word in alignment_result.words:
        pool.update_pool(word, speech_clips, settings.diarization_stitching)
        for _clip in pool.iterate_candidates(speech_clips):
            pass

    for clip in speech_clips:
        clip.finalize_words()

    return speech_clips
