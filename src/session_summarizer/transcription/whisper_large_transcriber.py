from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path

from ..protocols import LoggingProtocol
from ..protocols.transcriber_protocol import TranscriptionResult, TranscriptionSegment


@dataclass
class WhisperLargeTranscriber:
    """Transcriber using OpenAI Whisper (openai-whisper package).

    Produces segment-level timestamps. For word-level timestamps use WhisperXTranscriber.
    Install: uv add --group whisper openai-whisper
    """

    model_size: str = "large"
    device: str = "cuda"

    def transcribe(self, audio_path: Path, logger: LoggingProtocol) -> TranscriptionResult:
        try:
            import whisper  # pyright: ignore[reportMissingImports]
        except ImportError as e:
            raise ImportError("openai-whisper is required: uv add --group whisper openai-whisper") from e

        logger.report_message(f"[blue]Loading Whisper {self.model_size} model on {self.device}...[/blue]")
        model = whisper.load_model(self.model_size, device=self.device)

        try:
            logger.report_message(f"[blue]Transcribing {audio_path.name}...[/blue]")
            result = model.transcribe(str(audio_path), fp16=(self.device == "cuda"), verbose=False)

            segments = [
                TranscriptionSegment(
                    text=seg["text"].strip(),
                    start=float(seg["start"]),
                    end=float(seg["end"]),
                )
                for seg in result["segments"]
            ]
            full_text = result["text"].strip()
            logger.report_message(f"[green]Transcription complete: {len(segments)} segments[/green]")
            return TranscriptionResult(segments=segments, full_text=full_text)
        finally:
            del model
            gc.collect()
            try:
                import torch

                torch.cuda.empty_cache()
            except ImportError:
                pass
