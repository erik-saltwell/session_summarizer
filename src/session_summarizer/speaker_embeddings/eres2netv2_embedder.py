from __future__ import annotations

import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torchaudio

from ..protocols import EmbeddingFactory, LoggingProtocol


def _patch_torchaudio_sox_effects() -> None:
    """Shim torchaudio.sox_effects for torchaudio 2.10+ where it was removed.

    ModelScope's speaker verification pipeline calls
    torchaudio.sox_effects.apply_effects_tensor() to resample audio.
    We replace it with torchaudio.functional.resample.
    """
    if hasattr(torchaudio, "sox_effects"):
        return

    def apply_effects_tensor(
        tensor: torch.Tensor, sample_rate: int, effects: list[list[str]]
    ) -> tuple[torch.Tensor, int]:
        for effect in effects:
            if effect[0] == "rate":
                new_rate = int(effect[1])
                tensor = torchaudio.functional.resample(tensor, sample_rate, new_rate)
                sample_rate = new_rate
        return tensor, sample_rate

    sox = types.ModuleType("torchaudio.sox_effects")
    sox.apply_effects_tensor = apply_effects_tensor  # pyright: ignore[reportAttributeAccessIssue]  # ty: ignore[unresolved-attribute]
    torchaudio.sox_effects = sox  # pyright: ignore[reportAttributeAccessIssue]  # ty: ignore[unresolved-attribute]


_patch_torchaudio_sox_effects()


@dataclass
class ERes2NetV2EmbeddingFactory(EmbeddingFactory):
    """Speaker embedding extractor using ModelScope's ERes2NetV2 model.

    Produces a 192-dimensional L2-normalized speaker embedding from a WAV file.
    Model: iic/speech_eres2netv2_sv_zh-cn_16k-common (downloaded on first use).

    The pipeline is initialized once at construction and reused across extract() calls.
    """

    model_id: str = "iic/speech_eres2netv2_sv_zh-cn_16k-common"
    device: str = "cuda"
    _pipe: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks

        self._pipe = pipeline(task=Tasks.speaker_verification, model=self.model_id, device=self.device)

    def extract(self, audio_path: Path, logger: LoggingProtocol) -> list[float]:
        with torch.no_grad():
            result: Any = self._pipe([str(audio_path.resolve())], output_emb=True)
        # result['embs'] is a numpy array of shape [N, 192]; take the first (and only) row
        emb = result["embs"][0]
        embedding: list[float] = [float(x) for x in emb]
        return embedding
