from __future__ import annotations

from pathlib import Path

from ..evaluation.text_cleaner import clean_text_for_evaluation
from ..processing_results import SpeechClip, SpeechClipFlags, SpeechClipSet, WordAlignment
from ..protocols import GpuLogger, LoggingProtocol
from ..settings import SessionSettings
from ..utils import Tracer

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
    "um",
    "umm",
    "ummm",
    "hm",
    "hmmm",
    "ah",
    "nope",
    "no",
}


def is_backchannel_word(word: WordAlignment) -> bool:
    text = clean_text_for_evaluation(word.word, do_mathspell=False)
    return text in _backchannel_strings


def is_backchannel(clip: SpeechClip) -> bool:
    if not clip.words or len(clip.words) > 1:
        return False
    return is_backchannel_word(clip.words[0])


def mark_backchannels(
    settings: SessionSettings,
    session_dir: Path,
    clips: SpeechClipSet,
    gpu_logger: GpuLogger,
    logger: LoggingProtocol,
    tracer: Tracer,
) -> SpeechClipSet:
    backchannel_count = 0
    for clip in clips:
        clip.sort_words()
        is_clip_backchannel = is_backchannel(clip)
        clip.set_flag(SpeechClipFlags.IS_BACKCHANNEL, is_clip_backchannel)
        if is_clip_backchannel:
            backchannel_count += 1
    tracer.add_context("backchannel_count", backchannel_count)
    return clips
