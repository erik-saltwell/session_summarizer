from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..protocols import LoggingProtocol
from ..protocols.transcriber_protocol import TranscriptionResult, TranscriptionSegment


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

    def transcribe(self, audio_path: Path, logger: LoggingProtocol) -> TranscriptionResult:
        try:
            import whisper
        except ImportError as e:
            raise ImportError("openai-whisper is required: pip install openai-whisper") from e

        import torch

        logger.report_message(f"[blue]Loading Whisper model '{self.model_name}'...[/blue]")
        model = whisper.load_model(self.model_name, device=self.device)
        model.eval()
        gc.collect()
        torch.cuda.empty_cache()

        try:
            logger.report_message(f"[blue]Transcribing {audio_path.name}...[/blue]")
            with torch.inference_mode():
                raw = model.transcribe(
                    str(audio_path),
                    language="en",
                    word_timestamps=True,
                )

            raw_segments: Any = raw.get("segments", [])
            whisper_segments: list[dict] = list(raw_segments)
            logger.report_message(f"[blue]Whisper returned {len(whisper_segments)} segments.[/blue]")

            segments: list[TranscriptionSegment] = []
            texts: list[str] = []
            for seg in whisper_segments:
                text = seg.get("text", "").strip()
                if not text:
                    continue
                texts.append(text)
                segments.append(
                    TranscriptionSegment(
                        text=text,
                        start=float(seg["start"]),
                        end=float(seg["end"]),
                        confidence=0.0,  # overwritten after alignment
                    )
                )

            full_text = " ".join(texts)
            logger.report_message("[green]Transcription complete[/green]")
            return TranscriptionResult(segments=segments, full_text=full_text)

        finally:
            del model
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
