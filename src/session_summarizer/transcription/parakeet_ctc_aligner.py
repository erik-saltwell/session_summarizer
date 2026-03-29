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
    confidence: float  # acoustic confidence [0.0, 1.0]


@dataclass
class AlignmentResult:
    words: list[WordAlignment]


def _map_confidence_by_time(
    salm_words: list[WordAlignment],
    parakeet_timestamps: list[dict],
    parakeet_confidence: list[float],
    time_stride: float = 0.04,
) -> None:
    """Mutate salm_words in place, setting confidence from overlapping Parakeet words (min)."""

    def _time(pw: dict, key: str) -> float:
        """Extract time in seconds, handling both 'start'/'end' and offset-based keys."""
        if key in pw:
            return float(pw[key])
        offset_key = f"{key}_offset"
        return float(pw.get(offset_key, 0)) * time_stride

    for sw in salm_words:
        overlapping = [
            conf
            for pw, conf in zip(parakeet_timestamps, parakeet_confidence, strict=False)
            if _time(pw, "end") > sw.start and _time(pw, "start") < sw.end
        ]
        sw.confidence = min(overlapping) if overlapping else 0.0


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
            from nemo.collections.asr.parts.submodules.ctc_decoding import CTCBPEDecodingConfig
            from nemo.collections.asr.parts.utils.asr_confidence_utils import (
                ConfidenceConfig,
                ConfidenceMethodConfig,
            )
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
        ctc_model.bfloat16()

        confidence_cfg = ConfidenceConfig(
            preserve_word_confidence=True,
            aggregation="min",
            method_cfg=ConfidenceMethodConfig(name="max_prob"),
        )
        decoding_cfg = CTCBPEDecodingConfig(
            preserve_alignments=True,
            compute_timestamps=True,
            confidence_cfg=confidence_cfg,
        )
        ctc_model.change_decoding_strategy(decoding_cfg)

        try:
            with torch.inference_mode():
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
                                confidence=0.0,  # overwritten by _map_confidence_by_time below
                            )
                        )

                logger.report_message("[blue]Running free transcription pass for confidence scores...[/blue]")
                parakeet_hypotheses: Any = ctc_model.transcribe([str(audio_path)], return_hypotheses=True)
                parakeet_hyp: Any = parakeet_hypotheses[0]
                parakeet_word_timestamps: list[dict] = (
                    parakeet_hyp.timestamp.get("word", []) if parakeet_hyp.timestamp else []
                )
                parakeet_word_confidence: list[float] = parakeet_hyp.word_confidence or []

                # Compute frame stride for offset-to-seconds conversion
                window_stride = ctc_model.cfg.preprocessor.window_stride
                sub_factor = getattr(ctc_model.cfg.encoder, "subsampling_factor", 4)
                stride = window_stride * sub_factor
                _map_confidence_by_time(words, parakeet_word_timestamps, parakeet_word_confidence, time_stride=stride)

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
