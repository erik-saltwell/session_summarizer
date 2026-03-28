from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..protocols import LoggingProtocol


@dataclass
class WordAlignment:
    word: str
    start: float  # seconds
    end: float  # seconds


@dataclass
class AlignmentResult:
    words: list[WordAlignment]


@dataclass
class ParakeetCTCAligner:
    """Word-level forced aligner using NeMo's parakeet-ctc-0.6b CTC model.

    Reads normalized_audio.wav and transcript text, and produces word-level
    timestamps via CTC forced alignment (Viterbi decoding).

    VRAM discipline: model is loaded and fully unloaded within each align() call.

    Install NeMo separately (not in pyproject.toml due to complex CUDA deps):
        pip install nemo_toolkit[asr]
    """

    model_name: str = "nvidia/parakeet-ctc-0.6b"
    device: str = "cuda"
    batch_size: int = 4

    def align(self, audio_path: Path, full_text: str, logger: LoggingProtocol) -> AlignmentResult:
        if not full_text.strip():
            logger.report_warning("[yellow]Empty transcript — skipping alignment.[/yellow]")
            return AlignmentResult(words=[])

        try:
            import torch
            from nemo.collections.asr.models import ASRModel
            from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
            from nemo.collections.asr.parts.utils.timestamp_utils import (
                get_forced_aligned_timestamps_with_external_model,
            )
        except ImportError as e:
            raise ImportError("NeMo is required: pip install nemo_toolkit[asr]") from e

        logger.report_message(f"[blue]Loading CTC model {self.model_name}...[/blue]")
        ctc_model: Any = ASRModel.from_pretrained(model_name=self.model_name)
        ctc_model.to(self.device)
        ctc_model.eval()

        try:
            logger.report_message("[blue]Running forced alignment...[/blue]")
            hypothesis = Hypothesis(score=0.0, y_sequence=[], text=full_text)

            aligned: Any = get_forced_aligned_timestamps_with_external_model(
                audio=[str(audio_path)],
                external_ctc_model=ctc_model,
                main_model_predictions=[hypothesis],
                batch_size=self.batch_size,
                viterbi_device=torch.device(self.device),
                timestamp_type="all",
            )

            words: list[WordAlignment] = []
            timestamp: Any = aligned[0].timestamp if aligned else None
            if timestamp and "word" in timestamp:
                for entry in timestamp["word"]:
                    words.append(
                        WordAlignment(
                            word=str(entry["word"]),
                            start=float(entry["start"]),
                            end=float(entry["end"]),
                        )
                    )

            logger.report_message(f"[green]Alignment complete: {len(words)} words aligned.[/green]")
            return AlignmentResult(words=words)

        finally:
            del ctc_model
            gc.collect()
            try:
                import torch

                torch.cuda.empty_cache()
            except ImportError:
                pass
