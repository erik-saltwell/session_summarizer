from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..protocols import LoggingProtocol


@dataclass
class ERes2NetV2Embedder:
    """Speaker embedding extractor using ModelScope's ERes2NetV2 model.

    Produces a 192-dimensional L2-normalized speaker embedding from a WAV file.
    Model: iic/speech_eres2netv2_sv_zh-cn_16k-common (downloaded on first use).

    VRAM discipline: pipeline is loaded and unloaded within each extract() call.
    """

    model_id: str = "iic/speech_eres2netv2_sv_zh-cn_16k-common"
    device: str = "cuda"

    def extract(self, audio_path: Path, logger: LoggingProtocol) -> list[float]:
        try:
            import torch
            from modelscope.pipelines import pipeline
            from modelscope.utils.constant import Tasks
        except ImportError as e:
            raise ImportError("modelscope is required: uv add modelscope") from e

        logger.report_message(f"[blue]Loading ERes2NetV2 model {self.model_id}...[/blue]")
        pipe: Any = pipeline(task=Tasks.speaker_verification, model=self.model_id, device=self.device)

        try:
            logger.report_message("[blue]Extracting speaker embedding...[/blue]")
            result: Any = pipe([str(audio_path.resolve())], output_emb=True)
            # result['embs'] is a numpy array of shape [N, 192]; take the first (and only) row
            emb = result["embs"][0]
            embedding: list[float] = [float(x) for x in emb]
            logger.report_message(f"[green]Embedding extracted: {len(embedding)}-dimensional vector.[/green]")
            return embedding

        finally:
            del pipe
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
