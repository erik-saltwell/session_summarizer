from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

from ..protocols import EmbeddingFactory, LoggingProtocol


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
        self._pipe = pipeline(task=Tasks.speaker_verification, model=self.model_id, device=self.device)

    def extract(self, audio_path: Path, logger: LoggingProtocol) -> list[float]:
        with torch.no_grad():
            result: Any = self._pipe([str(audio_path.resolve())], output_emb=True)
        # result['embs'] is a numpy array of shape [N, 192]; take the first (and only) row
        emb = result["embs"][0]
        embedding: list[float] = [float(x) for x in emb]
        return embedding
