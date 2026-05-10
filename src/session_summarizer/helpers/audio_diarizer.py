from __future__ import annotations

import os
import re
from pathlib import Path

from ..processing_results import SpeechClipSet
from ..processing_results.alignment_result import WordAlignment
from ..processing_results.speech_clip import SpeechClip
from ..protocols import (
    GpuLogger,
    LoggingProtocol,
    SessionSettings,
)
from ..utils import Tracer

_LANGUAGE_CODE = "eng"
_MODEL_ID = "scribe_v2"
_SPEAKER_ID_PATTERN = re.compile(r"^speaker_(\d+)$")


def _normalize_speaker_id(eleven_labs_speaker_id: str) -> str:
    """Convert an ElevenLabs ``speaker_<n>`` id into the pipeline-wide
    ``SPEAKER_<NN>`` Input Speaker Label format produced by DiariZen.

    Falls back to upper-casing when the id does not match the expected
    pattern, so unexpected formats still flow through rather than raising.
    """
    match = _SPEAKER_ID_PATTERN.match(eleven_labs_speaker_id)
    if match is None:
        return eleven_labs_speaker_id.upper()
    return f"SPEAKER_{int(match.group(1)):02d}"


def diarize_audio(
    settings: SessionSettings,
    session_dir: Path,
    gpu_logger: GpuLogger,
    logger: LoggingProtocol,
    tracer: Tracer,
) -> SpeechClipSet:
    audio_path = session_dir / settings.paths.cleaned_audio
    clip_gap_seconds = settings.eleven_labs.clip_gap_seconds

    gpu_logger.report_gpu_usage("before processing")

    try:
        from elevenlabs import ElevenLabs
    except ImportError as e:
        raise ImportError("elevenlabs is required: pip install elevenlabs") from e

    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY is not set in the environment.")

    client = ElevenLabs(api_key=api_key)

    logger.report_message("[blue]Submitting audio to ElevenLabs Scribe v2 with diarization...[/blue]")
    with logger.status("ElevenLabs diarizing..."):
        with audio_path.open("rb") as audio_file:
            response = client.speech_to_text.convert(
                file=audio_file,
                model_id=_MODEL_ID,
                language_code=_LANGUAGE_CODE,
                diarize=True,
                tag_audio_events=False,
                timestamps_granularity="word",
                num_speakers=len(settings.attendees),
                request_options={"timeout_in_seconds": 900},
            )

    raw_words = getattr(response, "words", None) or []

    with logger.status("Building speech clips from ElevenLabs words..."):
        result: SpeechClipSet = SpeechClipSet()
        current_clip: SpeechClip | None = None
        current_speaker: str | None = None
        last_word_end: float | None = None
        word_count = 0

        for w in raw_words:
            if getattr(w, "type", "word") != "word":
                continue

            text = (getattr(w, "text", "") or "").strip()
            if not text:
                continue

            start = float(getattr(w, "start", 0.0))
            end = float(getattr(w, "end", 0.0))
            speaker_id = _normalize_speaker_id(getattr(w, "speaker_id", "speaker_0"))

            word_alignment = WordAlignment(word=text, start_time=start, end_time=end)
            word_count += 1

            speaker_changed = current_speaker is not None and speaker_id != current_speaker
            gap_exceeded = last_word_end is not None and (start - last_word_end) > clip_gap_seconds

            if current_clip is None or speaker_changed or gap_exceeded:
                if current_clip is not None:
                    current_clip.compute_word_derived_values()
                    result.add_clip(current_clip)
                current_clip = SpeechClip(
                    start_time=start,
                    end_time=end,
                    speakers={speaker_id},
                    text="",
                    words=[word_alignment],
                )
                current_speaker = speaker_id
            else:
                current_clip.end_time = end
                current_clip.add_word(word_alignment)

            last_word_end = end

        if current_clip is not None:
            current_clip.compute_word_derived_values()
            result.add_clip(current_clip)

    tracer.add_context("eleven_labs_word_count", word_count)
    tracer.add_context("clip_count", len(result))
    logger.report_message(f"[green]ElevenLabs returned {word_count} words → {len(result)} speech clips[/green]")

    return result
