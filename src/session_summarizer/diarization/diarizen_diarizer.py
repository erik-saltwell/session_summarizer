from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..protocols import LoggingProtocol


def _patch_torchaudio_for_pyannote() -> None:
    """Monkey-patch torchaudio and lightning_fabric APIs for compatibility with torchaudio 2.10+.

    torchaudio 2.10 dropped AudioMetaData, list_audio_backends(), and info() and
    made load() require torchcodec (which needs libnvrtc that is unavailable here).
    We shim them with soundfile so pyannote-audio can import and run normally.

    Also patches lightning_fabric's _load to use weights_only=False for local files
    since PyTorch 2.6+ defaults to True, which breaks HuggingFace checkpoints that
    contain TorchVersion and other non-tensor globals.

    This patch is idempotent.
    """
    import torchaudio

    if getattr(torchaudio, "_patched_for_pyannote", False):
        return

    # PyTorch 2.6+ changed torch.load to default weights_only=True, but diarizen's
    # HuggingFace checkpoints (pyannote, wespeaker, wavlm) contain non-tensor globals
    # (TorchVersion etc.) that fail with weights_only=True. Patch torch.load to default
    # to False so these trusted local checkpoints load correctly.
    import functools as _functools

    import torch as _torch

    _orig_torch_load = _torch.load

    @_functools.wraps(_orig_torch_load)
    def _patched_torch_load(*args: Any, **kwargs: Any) -> Any:
        # Default to False when not specified OR when explicitly passed as None,
        # since lightning_fabric passes weights_only=None which triggers the default True.
        if kwargs.get("weights_only") is None:
            kwargs["weights_only"] = False
        return _orig_torch_load(*args, **kwargs)

    _torch.load = _patched_torch_load

    import soundfile as sf
    import torch

    class AudioMetaData:
        def __init__(self, sample_rate: int, num_frames: int, num_channels: int, bits_per_sample: int, encoding: str):
            self.sample_rate = sample_rate
            self.num_frames = num_frames
            self.num_channels = num_channels
            self.bits_per_sample = bits_per_sample
            self.encoding = encoding

    torchaudio.AudioMetaData = AudioMetaData  # pyright: ignore[reportAttributeAccessIssue]

    def list_audio_backends() -> list[str]:
        return ["soundfile"]

    torchaudio.list_audio_backends = list_audio_backends  # pyright: ignore[reportAttributeAccessIssue]

    def info(path: str | Path, backend: str | None = None) -> AudioMetaData:
        meta = sf.info(str(path))
        return AudioMetaData(
            sample_rate=meta.samplerate,
            num_frames=meta.frames,
            num_channels=meta.channels,
            bits_per_sample=16,
            encoding="PCM_S",
        )

    torchaudio.info = info  # pyright: ignore[reportAttributeAccessIssue]

    def load(
        uri: str | Path,
        frame_offset: int = 0,
        num_frames: int = -1,
        normalize: bool = True,
        channels_first: bool = True,
        format: str | None = None,
        buffer_size: int = 4096,
        backend: str | None = None,
    ) -> tuple[torch.Tensor, int]:
        import numpy as np

        data, sr = sf.read(str(uri), dtype="float32", always_2d=True)
        if frame_offset > 0:
            data = data[frame_offset:]
        if num_frames > 0:
            data = data[:num_frames]
        waveform = torch.from_numpy(np.ascontiguousarray(data.T))  # [channels, time]
        if not channels_first:
            waveform = waveform.T
        return waveform, sr

    torchaudio.load = load  # pyright: ignore[reportAttributeAccessIssue]
    torchaudio._patched_for_pyannote = True  # pyright: ignore[reportAttributeAccessIssue]


@dataclass
class DiarizationSegment:
    speaker: str
    start: float  # seconds
    end: float  # seconds


@dataclass
class DiarizationResult:
    segments: list[DiarizationSegment]


@dataclass
class MergedDiarizationSegment:
    start: float  # seconds
    end: float  # seconds
    speakers: list[str]  # sorted for determinism

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def is_multispeaker(self) -> bool:
        return len(self.speakers) > 1


@dataclass
class MergedDiarizationResult:
    segments: list[MergedDiarizationSegment]

    def save(self, path: Path) -> None:
        data = [{"start": s.start, "end": s.end, "speakers": s.speakers} for s in self.segments]
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> MergedDiarizationResult:
        data = json.loads(path.read_text())
        segments = [MergedDiarizationSegment(start=s["start"], end=s["end"], speakers=s["speakers"]) for s in data]
        return cls(segments=segments)


def merge_overlapping_diarization(raw: DiarizationResult) -> MergedDiarizationResult:
    """Convert overlapping per-speaker segments into non-overlapping windows.

    Uses a sweep-line over all unique start/end timestamps. Each consecutive
    pair of breakpoints becomes one output window whose speakers list contains
    every raw segment active (overlapping) during that window.
    """
    if not raw.segments:
        return MergedDiarizationResult(segments=[])

    breakpoints = sorted({t for seg in raw.segments for t in (seg.start, seg.end)})

    merged: list[MergedDiarizationSegment] = []
    for i in range(len(breakpoints) - 1):
        t_start = breakpoints[i]
        t_end = breakpoints[i + 1]
        active = sorted({seg.speaker for seg in raw.segments if seg.start < t_end and seg.end > t_start})
        if active:
            merged.append(MergedDiarizationSegment(start=t_start, end=t_end, speakers=active))

    return MergedDiarizationResult(segments=merged)


@dataclass
class DiarizenDiarizer:
    """Speaker diarizer using BUT-FIT/diarizen-wavlm-large-s80-md-v2.

    Speaker count is inferred automatically (no num_speakers required).
    Supports up to 4 overlapping speakers.

    VRAM discipline: model is loaded and fully unloaded within each diarize() call.
    """

    model_name: str = "BUT-FIT/diarizen-wavlm-large-s80-md-v2"

    def diarize(self, audio_path: Path, logger: LoggingProtocol) -> MergedDiarizationResult:
        _patch_torchaudio_for_pyannote()

        try:
            import torch
            import torch.torch_version
            from diarizen.pipelines.inference import DiariZenPipeline
        except ImportError as e:
            raise ImportError("diarizen is required: pip install diarizen") from e

        # PyTorch 2.6+ defaults weights_only=True; the diarizen checkpoint stores
        # TorchVersion objects, so we need to add it to the safe-globals allowlist.
        torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])

        logger.report_message(f"[blue]Loading DiariZen diarization model {self.model_name}...[/blue]")
        pipeline = DiariZenPipeline.from_pretrained(self.model_name)

        try:
            logger.report_message("[blue]Running diarization (speaker count inferred automatically)...[/blue]")
            annotation = pipeline(str(audio_path.resolve()))

            segments: list[DiarizationSegment] = []
            for turn, _, speaker in annotation.itertracks(yield_label=True):  # pyright: ignore[reportAssignmentType]
                segments.append(
                    DiarizationSegment(
                        speaker=str(speaker),
                        start=float(turn.start),
                        end=float(turn.end),
                    )
                )

            segments.sort(key=lambda s: s.start)
            raw = DiarizationResult(segments=segments)
            result = merge_overlapping_diarization(raw)
            logger.report_message(f"[green]Diarization complete: {len(result.segments)} merged segments.[/green]")
            return result

        finally:
            del pipeline
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
