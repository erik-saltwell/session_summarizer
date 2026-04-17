from __future__ import annotations

import math
from difflib import SequenceMatcher

from ..helpers.unassigned_speaker import UNASSIGNED_SPEAKER_NAME
from ..processing_results.speech_clip import SpeechClip
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


def compute_matched_words_ratio(hyp: SpeechClipSet) -> float:
    total_words: float = 0.0
    matched_words: float = 0.0
    for clip in hyp:
        if not clip.words:
            continue
        for word in clip.words:
            total_words += 1.0
            if word.ground_truth is not None:
                matched_words += 1.0

    if total_words == 0.0 and matched_words == 0.0:
        return 0.0
    return matched_words / (total_words + matched_words)


def compute_wder(hyp: SpeechClipSet, ref: SpeechClipSet, epsilon: float) -> float:
    correct_count: float = 0.0
    incorrect_count: float = 0.0

    last_clip_idx: int = len(hyp) - 1

    for idx, clip in enumerate(hyp):
        if not clip.words:
            continue
        if clip.identity == UNASSIGNED_SPEAKER_NAME:
            continue
        if clip.is_backchannel:
            continue

        prior: SpeechClip | None = None
        next: SpeechClip | None = None
        if idx > 0:
            prior = hyp[idx - 1]
        if idx < last_clip_idx:
            next = hyp[idx + 1]

        identity: str = clip.compute_speaker(prior, next, epsilon)
        for word in clip.words:
            if word.ground_truth is None:
                continue
            if word.ground_truth == identity:
                correct_count += 1.0
            else:
                incorrect_count += 1.0
    if correct_count == 0.0 and incorrect_count == 0.0:
        return 0.0
    return incorrect_count / (correct_count + incorrect_count)


def compute_wder_old(hyp: SpeechClipSet, ref: SpeechClipSet) -> float:
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
