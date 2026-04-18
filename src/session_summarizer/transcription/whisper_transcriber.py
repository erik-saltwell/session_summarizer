from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..processing_results import TranscriptionResult, TranscriptionSegment
from ..protocols import LoggingProtocol
from ..utils import Tracer
from ..vad.segment_splitter import SegmentSplitResultSet


@dataclass
class WhisperTranscriber:
    """Transcriber using OpenAI Whisper (turbo model by default).

    Whisper turbo is the largest and latest model in the Whisper family.
    Produces segment-level timestamps with word-level timestamps when requested.

    VRAM discipline: model is loaded and fully unloaded within each transcribe() call.

    Requires: openai-whisper (``pip install openai-whisper``)
    """

    model_name: str = "turbo"
    device: str = "cuda"

    def transcribe(
        self, audio_path: Path, segments: SegmentSplitResultSet, logger: LoggingProtocol, tracer: Tracer
    ) -> TranscriptionResult:
        try:
            import whisper
        except ImportError as e:
            raise ImportError("openai-whisper is required: pip install openai-whisper") from e

        import torch

        with logger.status(f"Loading model {self.model_name}..."):
            logger.report_message(f"[blue]Loading Whisper model '{self.model_name}'...[/blue]")
            model = whisper.load_model(self.model_name, device=self.device)
            model.eval()
        gc.collect()
        torch.cuda.empty_cache()

        try:
            with torch.inference_mode():
                raw = model.transcribe(
                    str(audio_path),
                    language="en",
                    word_timestamps=True,
                )

            raw_segments: Any = raw.get("segments", [])
            whisper_segments: list[dict] = list(raw_segments)
            tracer.add_context("segment_count", len(whisper_segments))

            result_segments: list[TranscriptionSegment] = []
            texts: list[str] = []
            for seg in whisper_segments:
                text = seg.get("text", "").strip()
                if not text:
                    continue
                texts.append(text)
                result_segments.append(
                    TranscriptionSegment(
                        text=text,
                        start=float(seg["start"]),
                        end=float(seg["end"]),
                        confidence=0.0,  # overwritten after alignment
                    )
                )

            full_text = " ".join(texts)
            return TranscriptionResult(segments=result_segments, full_text=full_text)

        finally:
            del model
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
