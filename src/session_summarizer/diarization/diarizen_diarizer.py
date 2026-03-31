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

    _torch.load = _patched_torch_load  # type: ignore[assignment]

    import soundfile as sf
    import torch

    class AudioMetaData:
        def __init__(self, sample_rate: int, num_frames: int, num_channels: int, bits_per_sample: int, encoding: str):
            self.sample_rate = sample_rate
            self.num_frames = num_frames
            self.num_channels = num_channels
            self.bits_per_sample = bits_per_sample
            self.encoding = encoding

    torchaudio.AudioMetaData = AudioMetaData  # type: ignore[attr-defined]

    def list_audio_backends() -> list[str]:
        return ["soundfile"]

    torchaudio.list_audio_backends = list_audio_backends  # type: ignore[attr-defined]

    def info(path: str | Path, backend: str | None = None) -> AudioMetaData:  # type: ignore[misc]
        meta = sf.info(str(path))
        return AudioMetaData(
            sample_rate=meta.samplerate,
            num_frames=meta.frames,
            num_channels=meta.channels,
            bits_per_sample=16,
            encoding="PCM_S",
        )

    torchaudio.info = info  # type: ignore[attr-defined]

    def load(  # type: ignore[misc]
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

    torchaudio.load = load  # type: ignore[attr-defined]
    torchaudio._patched_for_pyannote = True  # type: ignore[attr-defined]


@dataclass
class DiarizationSegment:
    speaker: str
    start: float  # seconds
    end: float  # seconds


@dataclass
class DiarizationResult:
    segments: list[DiarizationSegment]

    def save(self, path: Path) -> None:
        data = [{"speaker": s.speaker, "start": s.start, "end": s.end} for s in self.segments]
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> DiarizationResult:
        data = json.loads(path.read_text())
        segments = [DiarizationSegment(speaker=s["speaker"], start=s["start"], end=s["end"]) for s in data]
        return cls(segments=segments)


@dataclass
class DiarizenDiarizer:
    """Speaker diarizer using BUT-FIT/diarizen-wavlm-large-s80-md-v2.

    Speaker count is inferred automatically (no num_speakers required).
    Supports up to 4 overlapping speakers.

    VRAM discipline: model is loaded and fully unloaded within each diarize() call.
    """

    model_name: str = "BUT-FIT/diarizen-wavlm-large-s80-md-v2"

    def diarize(self, audio_path: Path, logger: LoggingProtocol) -> DiarizationResult:
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
            for turn, _, speaker in annotation.itertracks(yield_label=True):
                segments.append(
                    DiarizationSegment(
                        speaker=str(speaker),
                        start=float(turn.start),
                        end=float(turn.end),
                    )
                )

            segments.sort(key=lambda s: s.start)
            logger.report_message(f"[green]Diarization complete: {len(segments)} segments.[/green]")
            return DiarizationResult(segments=segments)

        finally:
            del pipeline
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
