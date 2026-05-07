from __future__ import annotations

import copy
import logging
import sys

from diarizationlm import utils as dlm_utils

from session_summarizer.processing_results.speech_clip_set import SpeechClipSet

from ..utils import Tracer
from .clip_set_converter import clip_set_to_utterance, rebuild_clip_set
from .llm_inference import DiarizationLMModelProtocol
from .speaker_mapping import SpeakerMapping

logger = logging.getLogger(__name__)

_PROMPT_OPTIONS = dlm_utils.PromptOptions(
    emit_input_length=896,
    emit_target_length=896,
    prompt_suffix=" --> ",
    completion_suffix="",
)


def _segment_prompts(utt: dict) -> list[str]:
    # Mirror dlm_utils.generate_prompts but skip its over-strict short-prompt guard.
    # The library's recursive-halving segmentation can produce a short tail, which
    # is valid inference input; raising emit_input_length to dodge the guard hurts
    # TPST quality, so we call the underlying reader directly instead.
    po_seg = copy.deepcopy(_PROMPT_OPTIONS)
    po_seg.emit_target_length = sys.maxsize
    reader = dlm_utils.JsonUtteranceReader(
        json_files="",
        text_field="hyp_text",
        input_speaker_field="hyp_spk",
        target_speaker_field="",
        po=po_seg,
        utt=utt,
    )
    prompts = [p for _, p, _ in reader.generate_data_tuple()]

    threshold = _PROMPT_OPTIONS.emit_input_length / 3
    if len(prompts) > 1 and len(prompts[-1]) < threshold:
        tail = prompts.pop()
        suffix = _PROMPT_OPTIONS.prompt_suffix
        prev = prompts[-1]
        if prev.endswith(suffix):
            prev = prev[: -len(suffix)]
        prompts[-1] = prev.rstrip() + " " + tail.lstrip()
        logger.info(
            "Merged short trailing segment (%d chars, threshold %d) into previous segment.",
            len(tail),
            int(threshold),
        )

    return prompts


class DiarizationLMProcessor:
    """Post-processes a SpeechClipSet using DiarizationLM to improve speaker assignments."""

    def __init__(self, model: DiarizationLMModelProtocol):
        self._model = model

    def process(self, clip_set: SpeechClipSet, epsilon: float, tracer: Tracer) -> SpeechClipSet:
        if not self._model.is_loaded:
            raise RuntimeError("Model not loaded. Call model.load() before processing.")

        # Step 1: Build speaker mapping.
        mapping = SpeakerMapping.build_from_clip_set(clip_set, epsilon)
        logger.info("Speaker mapping: %d speakers", mapping.speaker_count)

        # Step 2: Convert to DiarizationLM utterance format.
        conversion = clip_set_to_utterance(clip_set, mapping, epsilon)
        if not conversion.word_records:
            logger.warning("No words found in clip set — returning original unchanged.")
            return clip_set

        if conversion.passthrough_clip_indices:
            logger.warning(
                "%d clips have no word alignments and will pass through unchanged.",
                len(conversion.passthrough_clip_indices),
            )

        # Step 3: Build utterance dict for diarizationlm utils.
        utt = {
            "utterance_id": "session",
            "hyp_text": conversion.hyp_text,
            "hyp_spk": conversion.hyp_spk,
        }

        # Step 4: Generate segmented prompts.
        prompts = _segment_prompts(utt)
        tracer.add_context("input_segment_count", len(prompts))

        # Step 5: Run inference on each prompt.
        completions = []
        for i, prompt in enumerate(prompts):
            completion = self._model.infer(prompt)
            logger.info(
                "Segment %d/%d: prompt %d chars, completion %d chars.",
                i + 1,
                len(prompts),
                len(prompt),
                len(completion),
            )
            logger.debug("Segment %d prompt:     %s", i + 1, prompt[:300])
            logger.debug("Segment %d completion: %s", i + 1, completion[:300])
            completions.append(completion)

        # Step 6: Postprocess completions with TPST.
        utt_post: dict = dict(utt)
        utt_post["completions"] = completions
        dlm_utils.postprocess_completions_for_utt(
            utt_post,
            llm_text_field="llm_text",
            llm_speaker_field="llm_spk",
            transfered_llm_speaker_field="hyp_spk_llm",
            hyp_text_field="hyp_text",
            hyp_spk_field="hyp_spk",
            po=_PROMPT_OPTIONS,
        )

        corrected_spk_str = utt_post.get("hyp_spk_llm", conversion.hyp_spk)
        corrected_numeric_ids = corrected_spk_str.split()

        if len(corrected_numeric_ids) != len(conversion.word_records):
            logger.warning(
                "TPST output length (%d) != input length (%d). Falling back to original speakers.",
                len(corrected_numeric_ids),
                len(conversion.word_records),
            )
            return clip_set

        # Step 7: Map numeric IDs back to speaker strings.
        corrected_speakers = []
        for word_idx, nid in enumerate(corrected_numeric_ids):
            try:
                corrected_speakers.append(mapping.to_string(int(nid)))
            except (ValueError, KeyError):
                logger.warning("Invalid speaker ID '%s' at word %d in TPST output, using original.", nid, word_idx)
                corrected_speakers.append(conversion.word_records[word_idx].effective_speaker)

        # Step 8: Rebuild the SpeechClipSet.
        result: SpeechClipSet = rebuild_clip_set(
            clip_set,
            conversion.word_records,
            corrected_speakers,
            conversion.passthrough_clip_indices,
        )
        tracer.add_context("output_clip_count", len(result))
        return result
