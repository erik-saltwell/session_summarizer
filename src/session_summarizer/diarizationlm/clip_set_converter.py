from __future__ import annotations

import re
from dataclasses import dataclass

from session_summarizer.processing_results.alignment_result import WordAlignment
from session_summarizer.processing_results.speech_clip_set import (
    SpeechClip,
    SpeechClipFlags,
    SpeechClipSet,
)

from .speaker_mapping import SpeakerMapping, _effective_speaker

# Fisher transcripts are lowercase with no punctuation (except apostrophes in
# contractions).  The DiarizationLM-8b-Fisher-v2 model was fine-tuned on this
# format, so we must normalise ASR words before building prompts.
_NON_WORD_RE = re.compile(r"[^\w'\-]", re.UNICODE)


def _normalize_for_fisher(word: str) -> str:
    """Lowercase and strip punctuation (keeping apostrophes/hyphens) for Fisher compatibility."""
    result = _NON_WORD_RE.sub("", word.lower())
    return result or word.lower()


@dataclass
class WordRecord:
    """Back-reference linking a word in the flat DiarizationLM representation to its source clip."""

    word: WordAlignment
    source_clip_index: int
    effective_speaker: str


@dataclass
class ConversionResult:
    """Result of converting a SpeechClipSet to DiarizationLM utterance format."""

    hyp_text: str
    hyp_spk: str
    word_records: list[WordRecord]
    passthrough_clip_indices: list[int]


def clip_set_to_utterance(
    clip_set: SpeechClipSet,
    mapping: SpeakerMapping,
    epsilon: float,
) -> ConversionResult:
    """Convert a SpeechClipSet to the flat hyp_text + hyp_spk format used by DiarizationLM."""
    words: list[str] = []
    spk_ids: list[str] = []
    word_records: list[WordRecord] = []
    passthrough_clip_indices: list[int] = []

    for clip_idx, clip in enumerate(clip_set):
        if clip.words is None or len(clip.words) == 0:
            passthrough_clip_indices.append(clip_idx)
            continue

        prior = clip_set[clip_idx - 1] if clip_idx > 0 else None
        next_clip = clip_set[clip_idx + 1] if clip_idx < len(clip_set) - 1 else None
        speaker = clip.compute_speaker(prior, next_clip, epsilon)
        numeric_id = str(mapping.to_numeric(speaker))

        for w in clip.words:
            words.append(_normalize_for_fisher(w.word))
            spk_ids.append(numeric_id)
            word_records.append(
                WordRecord(
                    word=w,
                    source_clip_index=clip_idx,
                    effective_speaker=speaker,
                )
            )

    return ConversionResult(
        hyp_text=" ".join(words),
        hyp_spk=" ".join(spk_ids),
        word_records=word_records,
        passthrough_clip_indices=passthrough_clip_indices,
    )


def _build_clip(
    original: SpeechClipSet,
    speaker: str,
    words: list[WordAlignment],
    source_clip_indices: list[int],
) -> SpeechClip:
    """Build a single SpeechClip from a group of consecutive same-speaker words."""
    start_time = min(w.start_time for w in words)
    end_time = max(w.end_time for w in words)

    # Carry forward metadata if all words came from the same original clip
    # and the speaker didn't change.
    unique_sources = set(source_clip_indices)
    identity: str | None = None
    embedding: list[float] | None = None
    flags = SpeechClipFlags.NONE
    end_of_turn_probability: float | None = None

    if len(unique_sources) == 1:
        src_clip = original[source_clip_indices[0]]
        # flags and end_of_turn_probability are audio/text properties — always carry forward.
        flags = src_clip.flags
        end_of_turn_probability = src_clip.end_of_turn_probability
        # identity and embedding are speaker-specific — only carry when the speaker didn't change.
        if _effective_speaker(src_clip) == speaker:
            identity = src_clip.identity
            embedding = src_clip.embedding

    clip = SpeechClip(
        start_time=start_time,
        end_time=end_time,
        speakers={speaker},
        confidence_avg=0.0,
        text="",
        identity=identity,
        embedding=embedding,
        flags=flags,
        end_of_turn_probability=end_of_turn_probability,
        words=list(words),
    )
    clip.compute_word_derived_values()
    return clip


def rebuild_clip_set(
    original: SpeechClipSet,
    word_records: list[WordRecord],
    corrected_speakers: list[str],
    passthrough_clip_indices: list[int],
) -> SpeechClipSet:
    """Build a new SpeechClipSet from corrected per-word speaker assignments.

    Args:
        original: The original SpeechClipSet (used for passthrough clips and metadata carry-forward).
        word_records: The WordRecord list from clip_set_to_utterance.
        corrected_speakers: Per-word corrected speaker strings (same length as word_records).
        passthrough_clip_indices: Indices of clips that were not processed (no words).
    """
    if len(word_records) != len(corrected_speakers):
        raise ValueError(
            f"word_records length ({len(word_records)}) != corrected_speakers length ({len(corrected_speakers)})"
        )

    # Build new clips by grouping consecutive words with the same corrected speaker.
    new_clips: list[SpeechClip] = []
    if word_records:
        current_speaker = corrected_speakers[0]
        current_words: list[WordAlignment] = [word_records[0].word]
        current_source_indices: list[int] = [word_records[0].source_clip_index]

        for i in range(1, len(word_records)):
            if corrected_speakers[i] == current_speaker:
                current_words.append(word_records[i].word)
                current_source_indices.append(word_records[i].source_clip_index)
            else:
                new_clips.append(_build_clip(original, current_speaker, current_words, current_source_indices))
                current_speaker = corrected_speakers[i]
                current_words = [word_records[i].word]
                current_source_indices = [word_records[i].source_clip_index]

        new_clips.append(_build_clip(original, current_speaker, current_words, current_source_indices))

    # Collect passthrough clips.
    passthrough_clips = [original[i] for i in passthrough_clip_indices]

    # Merge and sort all clips by time.
    result = SpeechClipSet()
    result.extend_clips(new_clips)
    result.extend_clips(passthrough_clips)
    result.sort_clips()
    return result
