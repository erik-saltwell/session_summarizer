from __future__ import annotations

from dataclasses import dataclass

from ..processing_results import SpeechClip, WordAlignment
from ..settings import DiarizationStitchingSettings


@dataclass
class AnonymousClips:
    _clips: list[SpeechClip]
    _current_clip: SpeechClip | None = None

    def add_anonymous_word(self, new_word: WordAlignment, settings: DiarizationStitchingSettings) -> None:
        radius: float = settings.max_nearest_distance if settings.fill_nearest else 0.0
        radius += settings.epsilon

        if self._current_clip is not None and new_word.start_time > self._current_clip.end_time + radius:
            self.flush_current_clip()  # we need to start a new clip since the new word is too far from the current clip

        if self._current_clip is None:
            self._current_clip = SpeechClip.create_from_word(
                new_word
            )  # add class function in SpeechClip called create_from_word
        else:
            self._current_clip.merge_with_word(new_word)  # add class function in SpeechClip called merge_with_word

    def flush_current_clip(self) -> None:
        if self._current_clip is not None:
            self._clips.append(self._current_clip)
            self._current_clip = None

    def get_clips(self) -> list[SpeechClip]:
        self.flush_current_clip()
        return self._clips
