from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from ..processing_results import TranscriptionResult, TranscriptionSegment
from ..protocols import LoggingProtocol, TranscriberProtocol
from ..utils import Tracer
from ..vad.segment_splitter import SegmentSplitResultSet

_SEGMENT_GAP_SECONDS = 0.5
_LANGUAGE_CODE = "eng"
_MODEL_ID = "scribe_v2"


@dataclass
class ElevenLabsTranscriber(TranscriberProtocol):
    @staticmethod
    def _group_words_into_segments(words: list) -> list[TranscriptionSegment]:
        segments: list[TranscriptionSegment] = []
        current_parts: list[str] = []
        current_start: float | None = None
        current_end: float | None = None
        last_word_end: float | None = None

        def flush() -> None:
            if current_start is None or current_end is None:
                return
            text = "".join(current_parts).strip()
            if not text:
                return
            segments.append(
                TranscriptionSegment(
                    text=text,
                    start=current_start,
                    end=current_end,
                    confidence=0.0,
                )
            )

        for w in words:
            text = getattr(w, "text", "") or ""
            wtype = getattr(w, "type", "word")

            if wtype == "spacing":
                if current_parts:
                    current_parts.append(text)
                continue

            start = float(getattr(w, "start", 0.0))
            end = float(getattr(w, "end", 0.0))

            if last_word_end is not None and (start - last_word_end) > _SEGMENT_GAP_SECONDS:
                flush()
                current_parts = []
                current_start = None
                current_end = None

            if current_start is None:
                current_start = start
            current_end = end
            current_parts.append(text)
            last_word_end = end

        flush()
        return segments

    def transcribe(
        self, audio_path: Path, segments: SegmentSplitResultSet, logger: LoggingProtocol, tracer: Tracer
    ) -> TranscriptionResult:
        try:
            from elevenlabs import ElevenLabs
        except ImportError as e:
            raise ImportError("elevenlabs is required: pip install elevenlabs") from e

        api_key = os.environ.get("ELEVENLABS_API_KEY")
        if not api_key:
            raise RuntimeError("ELEVENLABS_API_KEY is not set in the environment.")

        client = ElevenLabs(api_key=api_key)

        logger.report_message("[blue]Submitting audio to ElevenLabs Scribe v2...[/blue]")
        with logger.status("ElevenLabs transcribing..."):
            with audio_path.open("rb") as audio_file:
                response = client.speech_to_text.convert(
                    file=audio_file,
                    model_id=_MODEL_ID,
                    language_code=_LANGUAGE_CODE,
                    diarize=False,
                    tag_audio_events=False,
                    timestamps_granularity="word",
                )

        words = getattr(response, "words", None) or []
        result_segments = self._group_words_into_segments(words)
        full_text = " ".join(seg.text for seg in result_segments).strip()

        tracer.add_context("eleven_labs_word_count", len(words))
        tracer.add_context("segment_count", len(result_segments))
        logger.report_message(
            f"[green]ElevenLabs returned {len(words)} words → {len(result_segments)} segments[/green]"
        )

        return TranscriptionResult(segments=result_segments, full_text=full_text)
