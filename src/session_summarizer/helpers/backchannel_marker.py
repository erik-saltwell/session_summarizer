from __future__ import annotations

from ..evaluation.text_cleaner import clean_text_for_evaluation
from ..processing_results.speech_clip import SpeechClip

_backchannel_strings: set[str] = {
    "uh-huh",
    "um-hum",
    "mmm",
    "yeah",
    "yep",
    "yup",
    "right",
    "okay",
    "ok",
    "okey",
    "gotcha",
    "hmm",
    "mm-hm",
    "mhmm",
    "oh",
    "alright",
    "sure",
    "huh",
    "wow",
    "really",
    "yes",
    "indeed",
    "mhm",
    "exactly",
}


def is_backchannel(clip: SpeechClip) -> bool:
    if not clip.words or len(clip.words) > 1:
        return False
    text: str = clean_text_for_evaluation(clip.words[0].word)
    return text in _backchannel_strings
