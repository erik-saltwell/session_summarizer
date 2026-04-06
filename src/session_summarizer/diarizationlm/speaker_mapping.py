from __future__ import annotations

from dataclasses import dataclass, field

from session_summarizer.processing_results.speech_clip_set import SpeechClip, SpeechClipSet

_MAX_SPEAKERS = 10


def _effective_speaker(clip: SpeechClip) -> str:
    """Return the best speaker label for a clip: identity if set, else first speaker alphabetically."""
    if clip.identity is not None:
        return clip.identity
    return sorted(clip.speakers)[0]


@dataclass
class SpeakerMapping:
    """Bidirectional mapping between string speaker labels and numeric 1..N IDs used by DiarizationLM."""

    _str_to_int: dict[str, int] = field(default_factory=dict)
    _int_to_str: dict[int, str] = field(default_factory=dict)

    @classmethod
    def build_from_clip_set(cls, clip_set: SpeechClipSet) -> SpeakerMapping:
        unique_speakers: set[str] = set()
        for clip in clip_set:
            unique_speakers.add(_effective_speaker(clip))

        if len(unique_speakers) > _MAX_SPEAKERS:
            raise ValueError(
                f"DiarizationLM supports at most {_MAX_SPEAKERS} speakers, "
                f"but found {len(unique_speakers)}: {sorted(unique_speakers)}"
            )

        sorted_speakers = sorted(unique_speakers)
        str_to_int = {spk: idx + 1 for idx, spk in enumerate(sorted_speakers)}
        int_to_str = {idx + 1: spk for idx, spk in enumerate(sorted_speakers)}
        return cls(_str_to_int=str_to_int, _int_to_str=int_to_str)

    def to_numeric(self, speaker: str) -> int:
        return self._str_to_int[speaker]

    def to_string(self, numeric_id: int) -> str:
        return self._int_to_str[numeric_id]

    @property
    def speaker_count(self) -> int:
        return len(self._str_to_int)
