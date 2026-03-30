from __future__ import annotations

import gc
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Self

from ..protocols import LoggingProtocol

logger = logging.getLogger(__name__)


@dataclass
class SpeechSegment:
    """A contiguous region of speech or silence."""

    start: float  # seconds
    end: float  # seconds
    is_speech: bool

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class VadResult:
    """Complete VAD output for an audio file."""

    segments: list[SpeechSegment]
    audio_duration: float

    def speech_segments(self) -> list[SpeechSegment]:
        return [s for s in self.segments if s.is_speech]

    def silence_segments(self) -> list[SpeechSegment]:
        return [s for s in self.segments if not s.is_speech]

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "audio_duration": self.audio_duration,
                    "segments": [asdict(s) for s in self.segments],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> Self:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            audio_duration=data["audio_duration"],
            segments=[SpeechSegment(**s) for s in data["segments"]],
        )


def _frames_to_segments(
    probs: list[float],
    frame_duration: float,
    onset: float,
    offset: float,
    min_duration_on: float,
    min_duration_off: float,
    pad_onset: float,
    pad_offset: float,
    total_duration: float,
) -> list[SpeechSegment]:
    """Convert frame-level speech probabilities to speech/silence segments.

    Uses a hysteresis state machine: transition to speech when probability
    exceeds *onset*, back to silence when it drops below *offset*.
    """
    is_speech = False
    speech_start = 0.0
    raw_speech: list[tuple[float, float]] = []

    for i, p in enumerate(probs):
        t = i * frame_duration
        if not is_speech and p >= onset:
            is_speech = True
            speech_start = t
        elif is_speech and p < offset:
            is_speech = False
            raw_speech.append((speech_start, t))

    # Close any open speech region
    if is_speech:
        raw_speech.append((speech_start, len(probs) * frame_duration))

    # Filter by min_duration_on
    raw_speech = [(s, e) for s, e in raw_speech if (e - s) >= min_duration_on]

    # Bridge short silence gaps (min_duration_off)
    merged: list[tuple[float, float]] = []
    for start, end in raw_speech:
        if merged and (start - merged[-1][1]) < min_duration_off:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    # Apply padding
    padded: list[tuple[float, float]] = []
    for start, end in merged:
        s = max(0.0, start - pad_onset)
        e = min(total_duration, end + pad_offset)
        # Merge with previous if padding caused overlap
        if padded and s <= padded[-1][1]:
            padded[-1] = (padded[-1][0], e)
        else:
            padded.append((s, e))

    # Build alternating speech/silence segment list
    segments: list[SpeechSegment] = []
    prev_end = 0.0
    for start, end in padded:
        if start > prev_end:
            segments.append(SpeechSegment(start=prev_end, end=start, is_speech=False))
        segments.append(SpeechSegment(start=start, end=end, is_speech=True))
        prev_end = end

    if prev_end < total_duration:
        segments.append(SpeechSegment(start=prev_end, end=total_duration, is_speech=False))

    return segments


@dataclass
class NemoVadDetector:
    """Wraps a NeMo VAD model to produce speech/silence segments."""

    model_name: str = "vad_multilingual_frame_marblenet"
    device: str = "cuda"
    onset: float = 0.7
    offset: float = 0.4
    min_duration_on: float = 0.3
    min_duration_off: float = 0.3
    pad_onset: float = 0.1
    pad_offset: float = 0.1

    def detect(self, audio_path: Path, app_logger: LoggingProtocol) -> VadResult:
        """Run VAD on the given audio file and return speech/silence segments."""
        try:
            import soundfile as sf
            import torch
            from nemo.collections.asr.models import EncDecFrameClassificationModel
        except ImportError as e:
            raise ImportError("NeMo is required for VAD: pip install 'nemo_toolkit[asr]'") from e

        app_logger.report_message(f"[blue]Loading VAD model: {self.model_name}[/blue]")
        model = EncDecFrameClassificationModel.from_pretrained(model_name=self.model_name)
        model = model.to(self.device)
        model.eval()

        try:
            info = sf.info(str(audio_path))
            total_duration = info.duration
            sample_rate = info.samplerate

            app_logger.report_message(f"[blue]Running VAD on {total_duration:.1f}s audio ({audio_path.name})[/blue]")

            # Load audio as float32 tensor
            audio_data, sr = sf.read(str(audio_path), dtype="float32")
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)

            # Process in chunks to avoid segfaults on long audio
            chunk_seconds = 60
            chunk_samples = chunk_seconds * sample_rate
            total_samples = len(audio_data)
            all_probs: list[float] = []

            with torch.no_grad():
                for offset in range(0, total_samples, chunk_samples):
                    chunk = audio_data[offset : offset + chunk_samples]
                    audio_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(self.device)
                    audio_length = torch.tensor([audio_tensor.shape[1]], dtype=torch.int64).to(self.device)
                    log_probs = model(input_signal=audio_tensor, input_signal_length=audio_length)
                    chunk_probs = torch.softmax(log_probs, dim=-1)[0, :, 1].cpu().tolist()
                    all_probs.extend(chunk_probs)

            probs = all_probs

            # Compute frame duration from input length and output length
            frame_duration = total_duration / len(probs) if probs else 0.02

            app_logger.report_message(
                f"[blue]VAD produced {len(probs)} frames (frame duration: {frame_duration * 1000:.1f}ms)[/blue]"
            )

            segments = _frames_to_segments(
                probs=probs,
                frame_duration=frame_duration,
                onset=self.onset,
                offset=self.offset,
                min_duration_on=self.min_duration_on,
                min_duration_off=self.min_duration_off,
                pad_onset=self.pad_onset,
                pad_offset=self.pad_offset,
                total_duration=total_duration,
            )

            speech_total = sum(s.duration for s in segments if s.is_speech)
            app_logger.report_message(
                f"[blue]VAD detected {len([s for s in segments if s.is_speech])} speech regions "
                f"({speech_total:.1f}s speech / {total_duration:.1f}s total)[/blue]"
            )

            return VadResult(segments=segments, audio_duration=total_duration)

        finally:
            del model
            gc.collect()
            try:
                import torch as _torch

                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
            except Exception:
                pass
