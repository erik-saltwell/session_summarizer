from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Protocol, cast

from diarizationlm import utils as dlm_utils

from session_summarizer.processing_results.speech_clip_set import SpeechClipSet

from .clip_set_converter import clip_set_to_utterance, rebuild_clip_set
from .speaker_mapping import SpeakerMapping

logger = logging.getLogger(__name__)

_PROMPT_OPTIONS = dlm_utils.PromptOptions(
    prompt_suffix=" --> ",
    completion_suffix="",
)
_FALLBACK_CONTEXT_WINDOW_TOKENS = 8192
_CONTEXT_SAFETY_TOKENS = 128
_COMPLETION_TOKEN_MULTIPLIER = 1.25
_MAX_COMPLETION_TOKEN_MULTIPLIER = 2.0
_MIN_COMPLETION_TOKENS = 128


class _DiarizationLMModelProtocol(Protocol):
    @property
    def is_loaded(self) -> bool: ...

    @property
    def context_window(self) -> int | None: ...

    def count_tokens(self, text: str) -> int: ...

    def infer(self, prompt: str, max_new_tokens: int | None = None) -> str: ...


@dataclass(frozen=True)
class _PromptSegment:
    start_word: int
    end_word: int
    prompt: str
    prompt_tokens: int
    max_new_tokens: int

    @property
    def word_count(self) -> int:
        return self.end_word - self.start_word


def _build_prompt(words: list[str], speaker_ids: list[str], start: int, end: int) -> str:
    prompt = cast(str, _PROMPT_OPTIONS.prompt_prefix)
    previous_speaker = ""

    for idx in range(start, end):
        speaker_id = speaker_ids[idx]
        if speaker_id != previous_speaker:
            if previous_speaker:
                prompt += " "
            prompt += _PROMPT_OPTIONS.speaker_prefix + speaker_id + _PROMPT_OPTIONS.speaker_suffix
        prompt += " " + words[idx]
        previous_speaker = speaker_id

    prompt += cast(str, _PROMPT_OPTIONS.prompt_suffix)
    return prompt


def _build_completion(words: list[str], speaker_ids: list[str], start: int, end: int) -> str:
    return cast(
        str,
        dlm_utils.create_diarized_text(
            words[start:end],
            speaker_ids[start:end],
            po=_PROMPT_OPTIONS,
        ),
    )


def _desired_completion_tokens(prompt_tokens: int) -> int:
    return max(_MIN_COMPLETION_TOKENS, math.ceil(prompt_tokens * _COMPLETION_TOKEN_MULTIPLIER))


def _max_completion_tokens(prompt_tokens: int) -> int:
    return max(_MIN_COMPLETION_TOKENS, math.ceil(prompt_tokens * _MAX_COMPLETION_TOKEN_MULTIPLIER))


class DiarizationLMProcessor:
    """Post-processes a SpeechClipSet using DiarizationLM to improve speaker assignments."""

    def __init__(self, model: _DiarizationLMModelProtocol):
        self._model = model
        self._last_prompt_segment_count: int | None = None
        self._last_prompt_word_count: int | None = None

    @property
    def last_prompt_segment_count(self) -> int | None:
        return self._last_prompt_segment_count

    @property
    def last_prompt_word_count(self) -> int | None:
        return self._last_prompt_word_count

    def _context_window(self) -> int:
        context_window = self._model.context_window
        if context_window is None:
            logger.warning(
                "Could not determine DiarizationLM context window from model config; using fallback of %d tokens.",
                _FALLBACK_CONTEXT_WINDOW_TOKENS,
            )
            return _FALLBACK_CONTEXT_WINDOW_TOKENS
        return context_window

    def _build_prompt_segment(
        self,
        words: list[str],
        speaker_ids: list[str],
        start: int,
        end: int,
        context_window: int,
    ) -> _PromptSegment:
        prompt = _build_prompt(words, speaker_ids, start, end)
        prompt_tokens = self._model.count_tokens(prompt)
        available_completion_tokens = context_window - prompt_tokens - _CONTEXT_SAFETY_TOKENS
        max_new_tokens = min(_max_completion_tokens(prompt_tokens), available_completion_tokens)
        return _PromptSegment(
            start_word=start,
            end_word=end,
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
        )

    def _prompt_segment_fits(self, segment: _PromptSegment) -> bool:
        return segment.max_new_tokens >= _desired_completion_tokens(segment.prompt_tokens)

    def _build_prompt_segments(
        self,
        words: list[str],
        speaker_ids: list[str],
        context_window: int,
        start: int,
        end: int,
    ) -> list[_PromptSegment]:
        segment = self._build_prompt_segment(words, speaker_ids, start, end, context_window)
        if self._prompt_segment_fits(segment):
            return [segment]

        if end - start <= 1:
            raise RuntimeError(
                f"Single-word DiarizationLM prompt does not fit the model context window: "
                f"{segment.prompt_tokens} prompt tokens, {context_window} context tokens."
            )

        midpoint = (start + end) // 2
        return [
            *self._build_prompt_segments(words, speaker_ids, context_window, start, midpoint),
            *self._build_prompt_segments(words, speaker_ids, context_window, midpoint, end),
        ]

    def _postprocess_completions(
        self,
        words: list[str],
        speaker_ids: list[str],
        completions: list[str],
        start_word: int,
        end_word: int,
        utterance_id: str,
    ) -> list[str]:
        utt: dict[str, Any] = {
            "utterance_id": utterance_id,
            "hyp_text": " ".join(words[start_word:end_word]),
            "hyp_spk": " ".join(speaker_ids[start_word:end_word]),
            "completions": completions,
        }
        try:
            dlm_utils.postprocess_completions_for_utt(
                utt,
                llm_text_field="llm_text",
                llm_speaker_field="llm_spk",
                transfered_llm_speaker_field="hyp_spk_llm",
                hyp_text_field="hyp_text",
                hyp_spk_field="hyp_spk",
                po=_PROMPT_OPTIONS,
            )
        except Exception as exc:
            raise ValueError(f"TPST postprocessing failed for {utterance_id}: {exc}") from exc

        corrected_ids = cast(str, utt.get("hyp_spk_llm", utt["hyp_spk"])).split()
        word_count = end_word - start_word
        if len(corrected_ids) != word_count:
            raise ValueError(
                f"TPST output length ({len(corrected_ids)}) != input length ({word_count}) for {utterance_id}"
            )

        return corrected_ids

    def _process_segment(
        self,
        segment: _PromptSegment,
        words: list[str],
        speaker_ids: list[str],
        segment_index: int,
        segment_count: int,
    ) -> str:
        completion = self._model.infer(segment.prompt, max_new_tokens=segment.max_new_tokens)
        logger.info(
            "Segment %d/%d: words %d-%d, prompt %d tokens, max completion %d tokens, completion %d chars.",
            segment_index + 1,
            segment_count,
            segment.start_word,
            segment.end_word - 1,
            segment.prompt_tokens,
            segment.max_new_tokens,
            len(completion),
        )
        logger.debug("Segment %d prompt:     %s", segment_index + 1, segment.prompt[:300])
        logger.debug("Segment %d completion: %s", segment_index + 1, completion[:300])

        try:
            self._postprocess_completions(
                words,
                speaker_ids,
                [completion],
                segment.start_word,
                segment.end_word,
                f"session_seg{segment_index}",
            )
        except ValueError as exc:
            logger.warning(
                "Segment %d completion is malformed: %s. Falling back to original speakers for this segment.",
                segment_index + 1,
                exc,
            )
            return _build_completion(words, speaker_ids, segment.start_word, segment.end_word)

        return completion

    def process(self, clip_set: SpeechClipSet, epsilon: float) -> SpeechClipSet:
        self._last_prompt_segment_count = None
        self._last_prompt_word_count = None

        if not self._model.is_loaded:
            raise RuntimeError("Model not loaded. Call model.load() before processing.")

        # Step 1: Build speaker mapping.
        mapping = SpeakerMapping.build_from_clip_set(clip_set, epsilon)
        logger.info("Speaker mapping: %d speakers", mapping.speaker_count)

        # Step 2: Convert to DiarizationLM utterance format.
        conversion = clip_set_to_utterance(clip_set, mapping, epsilon)
        if not conversion.word_records:
            logger.warning("No words found in clip set — returning original unchanged.")
            self._last_prompt_segment_count = 0
            self._last_prompt_word_count = 0
            return clip_set

        if conversion.passthrough_clip_indices:
            logger.warning(
                "%d clips have no word alignments and will pass through unchanged.",
                len(conversion.passthrough_clip_indices),
            )

        # Step 3: Build token-aware prompt segments. Most short transcripts
        # stay as one segment; longer transcripts split only when the model's
        # actual context window would be exceeded.
        words = conversion.hyp_text.split()
        speaker_ids = conversion.hyp_spk.split()
        self._last_prompt_word_count = len(words)
        if len(words) != len(conversion.word_records) or len(speaker_ids) != len(conversion.word_records):
            logger.warning(
                "DiarizationLM conversion length mismatch: %d words, %d speakers, %d records. "
                "Returning original unchanged.",
                len(words),
                len(speaker_ids),
                len(conversion.word_records),
            )
            return clip_set

        context_window = self._context_window()
        segments = self._build_prompt_segments(words, speaker_ids, context_window, 0, len(words))
        self._last_prompt_segment_count = len(segments)
        logger.info(
            "Generated %d prompt segment(s) for %d words using a %d-token context window.",
            len(segments),
            len(words),
            context_window,
        )

        # Step 4: Run inference on each segment. If one segment produces
        # malformed output, replace that segment with its original diarized
        # text before the whole-session TPST pass.
        completions: list[str] = []
        for idx, segment in enumerate(segments):
            completions.append(self._process_segment(segment, words, speaker_ids, idx, len(segments)))

        # Step 5: Run TPST once across the full utterance so speaker IDs are
        # mapped globally rather than per chunk.
        try:
            corrected_numeric_ids = self._postprocess_completions(
                words,
                speaker_ids,
                completions,
                0,
                len(words),
                "session",
            )
        except ValueError as exc:
            logger.warning("Whole-session TPST postprocessing failed: %s. Falling back to original speakers.", exc)
            corrected_numeric_ids = speaker_ids

        # Step 6: Map numeric IDs back to speaker strings.
        corrected_speakers = []
        for word_idx, nid in enumerate(corrected_numeric_ids):
            try:
                corrected_speakers.append(mapping.to_string(int(nid)))
            except (ValueError, KeyError):
                logger.warning("Invalid speaker ID '%s' at word %d in TPST output, using original.", nid, word_idx)
                corrected_speakers.append(conversion.word_records[word_idx].effective_speaker)

        # Step 7: Rebuild the SpeechClipSet.
        return rebuild_clip_set(
            clip_set,
            conversion.word_records,
            corrected_speakers,
            conversion.passthrough_clip_indices,
        )
