from __future__ import annotations

from ..diarization.diarizen_diarizer import MergedDiarizationResult
from ..processing_results.speech_clip_set import SpeechClip, SpeechClipSet
from ..transcription.parakeet_ctc_confidence_scorer import AlignmentResult, WordAlignment

_CROSSTALK = "[crosstalk]"


def create_text_for_segment(words: list[WordAlignment], is_multispeaker: bool) -> str:
    # if is_multispeaker:
    #     return _CROSSTALK
    return " ".join(w.word for w in words)


def create_confidence_avg_for_segment(words: list[WordAlignment]) -> float:
    if not words:
        return 0.0
    return sum(w.confidence for w in words) / len(words)


def create_speech_clips(
    diarization_result: MergedDiarizationResult,
    alignment_result: AlignmentResult,
) -> SpeechClipSet:
    speech_clips = SpeechClipSet()
    clip_id: int = 0
    for segment in diarization_result.segments:
        start_time = segment.start
        end_time = segment.end
        speakers = segment.speakers

        words: list[WordAlignment] = alignment_result.get_segments_for_time_range(start_time, end_time)
        text = create_text_for_segment(words, segment.is_multispeaker)
        if not text or len(text.strip()) == 0:
            continue
        confidence_avg = create_confidence_avg_for_segment(words)
        speech_clip = SpeechClip(
            clip_id=clip_id,
            parent=speech_clips,
            start_time=start_time,
            end_time=end_time,
            speakers=speakers,
            confidence_avg=confidence_avg,
            text=text,
        )
        speech_clips.append(speech_clip)
        clip_id += 1

    return speech_clips
