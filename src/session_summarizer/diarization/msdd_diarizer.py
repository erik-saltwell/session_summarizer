from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..protocols import LoggingProtocol


@dataclass
class DiarizationSegment:
    speaker: str
    start: float  # seconds
    end: float  # seconds


@dataclass
class DiarizationResult:
    segments: list[DiarizationSegment]


@dataclass
class MsddDiarizer:
    """Speaker diarizer using NeMo's Multi-Scale Diarization Decoder (MSDD).

    Uses oracle speaker count for clustering when num_speakers is provided,
    which significantly improves accuracy over automatic estimation.

    VRAM discipline: model is loaded and fully unloaded within each diarize() call.
    """

    model_name: str = "diar_msdd_telephonic"
    device: str = "cuda"
    num_speakers: int = 2

    def diarize(self, audio_path: Path, logger: LoggingProtocol) -> DiarizationResult:
        try:
            import torch
            from nemo.collections.asr.models.msdd_models import NeuralDiarizer
        except ImportError as e:
            raise ImportError("NeMo is required: pip install nemo_toolkit[asr]") from e

        logger.report_message(f"[blue]Loading MSDD diarization model {self.model_name}...[/blue]")
        diarizer = NeuralDiarizer.from_pretrained(
            model_name=self.model_name,
            map_location=self.device,
        )

        try:
            logger.report_message(
                f"[blue]Running diarization with {self.num_speakers} speakers (oracle mode)...[/blue]"
            )
            annotation: Any = diarizer(
                audio_filepath=str(audio_path.resolve()),
                num_speakers=self.num_speakers,
            )

            segments: list[DiarizationSegment] = []
            for segment, _track, speaker in annotation.itertracks(yield_label=True):
                segments.append(
                    DiarizationSegment(
                        speaker=str(speaker),
                        start=float(segment.start),
                        end=float(segment.end),
                    )
                )

            segments.sort(key=lambda s: s.start)
            logger.report_message(f"[green]Diarization complete: {len(segments)} segments.[/green]")
            return DiarizationResult(segments=segments)

        finally:
            del diarizer
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
