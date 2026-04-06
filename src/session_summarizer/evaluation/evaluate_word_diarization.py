from __future__ import annotations

import math
from difflib import SequenceMatcher

from ..processing_results.speech_clip_set import SpeechClipSet
from .text_cleaner import clean_text_for_evaluation


def _ref_words(clips: SpeechClipSet) -> list[tuple[str, str]]:
    """Flatten reference clips into (cleaned_word, speaker) pairs.

    Reference clips have segment-level text only (no word-level alignment),
    so we split each clip's text on whitespace.
    """
    result: list[tuple[str, str]] = []
    for clip in clips:
        speaker = clip.identity or ""
        for raw in clip.text.split():
            cleaned = clean_text_for_evaluation(raw)
            if cleaned:
                result.append((cleaned, speaker))
    return result


def _hyp_words(clips: SpeechClipSet) -> list[tuple[str, str]]:
    """Flatten hypothesis clips into (cleaned_word, speaker) pairs.

    Hypothesis clips have word-level alignment data; each word inherits its
    parent clip's identity.  Falls back to splitting clip.text if words is None.
    """
    result: list[tuple[str, str]] = []
    for clip in clips:
        speaker = clip.identity or ""
        if clip.words is not None:
            for wa in clip.words:
                cleaned = clean_text_for_evaluation(wa.word)
                if cleaned:
                    result.append((cleaned, speaker))
        else:
            for raw in clip.text.split():
                cleaned = clean_text_for_evaluation(raw)
                if cleaned:
                    result.append((cleaned, speaker))
    return result


def compute_wder(hyp: SpeechClipSet, ref: SpeechClipSet) -> float:
    """Word Diarization Error Rate.

    Aligns hypothesis and reference word sequences using LCS (via
    difflib.SequenceMatcher), then counts the fraction of matched words
    where the speaker label differs.

    Returns nan if no words could be matched.
    """
    ref_word_list = _ref_words(ref)
    hyp_word_list = _hyp_words(hyp)

    ref_tokens = [w for w, _ in ref_word_list]
    hyp_tokens = [w for w, _ in hyp_word_list]

    matcher = SequenceMatcher(None, ref_tokens, hyp_tokens, autojunk=False)

    total_matched = 0
    wrong_speaker = 0
    for block in matcher.get_matching_blocks():
        for i in range(block.size):
            _, ref_spk = ref_word_list[block.a + i]
            _, hyp_spk = hyp_word_list[block.b + i]
            total_matched += 1
            if ref_spk != hyp_spk:
                wrong_speaker += 1

    if total_matched == 0:
        return math.nan

    return wrong_speaker / total_matched
