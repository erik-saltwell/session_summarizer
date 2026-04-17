from __future__ import annotations

import re
from pathlib import Path

from ..processing_results import SpeechClip, SpeechClipSet
from ..processing_results.alignment_result import WordAlignment
from ..protocols import GpuLogger, LoggingProtocol
from ..settings import SessionSettings

# Edit this list to add/remove backchannel words (compared lowercase, punctuation ignored)
BACKCHANNEL_WORDS: set[str] = {
    "uh-huh",
    "um",
    "umm",
    "ummm",
    "hm",
    "hmm",
    "hmmm",
    "mmm",
    "yeah",
    "yes",
    "yep",
    "yup",
    "ok",
    "okay",
    "okey",
    "gotcha",
    "oh",
    "alright",
    "sure",
    "huh",
    "wow",
    "really",
    "indeed",
    "exactly",
    "ah",
    "nope",
    "no",
}

_PUNCTUATION_RE = re.compile(r"[^\w]")


def _strip_punctuation(word: str) -> str:
    return _PUNCTUATION_RE.sub("", word).lower()


def _is_backchannel(word: str) -> bool:
    return _strip_punctuation(word) in BACKCHANNEL_WORDS


def _find_period_split(
    words: list[WordAlignment],
) -> tuple[list[WordAlignment], list[WordAlignment]] | None:
    """
    Find the first word containing a period and return (words_before, words_after).
    words_before includes the period-containing word itself.
    Returns None if no qualifying split is found.
    """
    for i, w in enumerate(words):
        if "." not in w.word:
            continue
        words_before = words[: i + 1]
        words_after = words[i + 1 :]
        if 1 <= len(words_before) <= 2 and len(words_after) >= 5:
            return words_before, words_after
        # Only check the first period occurrence
        break
    return None


def fix_dangling_sentences(
    settings: SessionSettings,
    session_dir: Path,
    clips: SpeechClipSet,
    gpu_logger: GpuLogger,
    logger: LoggingProtocol,
) -> SpeechClipSet:
    clips.sort_clips()

    result: SpeechClipSet = SpeechClipSet()
    prior_clip: SpeechClip | None = None

    for clip in clips:
        if prior_clip is None or clip.words is None or len(clip.words) == 0:
            result.add_clip(clip)
            prior_clip = clip
            continue

        sorted_words = sorted(clip.words, key=lambda w: (w.start_time, w.end_time))
        split = _find_period_split(sorted_words)

        if split is None:
            result.add_clip(clip)
            prior_clip = clip
            continue

        words_before, words_after = split

        # Check that none of the pre-period words are backchannels
        if any(_is_backchannel(w.word) for w in words_before):
            result.add_clip(clip)
            prior_clip = clip
            continue

        # Move dangling words to the prior clip
        for w in words_before:
            prior_clip.merge_with_word(w)

        # Rebuild current clip with only the post-period words
        clip.words = words_after
        clip.compute_word_derived_values()
        if words_after:
            clip.start_time = min(w.start_time for w in words_after)

        result.add_clip(clip)
        prior_clip = clip

    result.sort_clips()
    return result
