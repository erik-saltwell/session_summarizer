from __future__ import annotations

import gc
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..protocols import LoggingProtocol
from ..protocols.transcriber_protocol import TranscriptionResult
from ..vad.segment_splitter import SegmentSplitResultSet
from .parakeet_ctc_confidence_scorer import AlignmentResult, WordAlignment


@dataclass
class ParakeetCTCWordAligner:
    """Word-level forced aligner using NeMo's parakeet-ctc-0.6b CTC model.

    Reads normalized_audio.wav and transcript text, and produces word-level
    timestamps via CTC forced alignment (Viterbi decoding).

    Audio is processed in chunks defined by the long VAD segments to avoid OOM
    on recordings longer than a few minutes.

    VRAM discipline: model is loaded and fully unloaded within each align() call.

    Install NeMo separately (not in pyproject.toml due to complex CUDA deps):
        pip install nemo_toolkit[asr]
    """

    model_name: str = "nvidia/parakeet-ctc-0.6b"
    device: str = "cuda"
    batch_size: int = 4

    def align(
        self,
        audio_path: Path,
        transcription: TranscriptionResult,
        segments: SegmentSplitResultSet,
        logger: LoggingProtocol,
    ) -> AlignmentResult:
        if not transcription.full_text.strip():
            logger.report_warning("[yellow]Empty transcript — skipping alignment.[/yellow]")
            return AlignmentResult(words=[])

        try:
            import soundfile as sf
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
        ctc_model.bfloat16()

        try:
            audio_data, sample_rate = sf.read(str(audio_path), dtype="float32")
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)

            audio_segments = segments.long.segments
            logger.report_message(f"[blue]Running forced alignment over {len(audio_segments)} VAD segments...[/blue]")

            all_words: list[WordAlignment] = []

            with torch.inference_mode():
                for seg in audio_segments:
                    chunk_text = " ".join(
                        ts.text for ts in transcription.segments if ts.end > seg.start and ts.start < seg.end
                    )
                    if not chunk_text.strip():
                        continue

                    start_sample = int(seg.start * sample_rate)
                    end_sample = int(seg.end * sample_rate)
                    chunk_audio = audio_data[start_sample:end_sample]

                    hypothesis = Hypothesis(score=0.0, y_sequence=[], text=chunk_text)

                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                        sf.write(tmp.name, chunk_audio, sample_rate)
                        aligned: Any = get_forced_aligned_timestamps_with_external_model(
                            audio=[tmp.name],
                            external_ctc_model=ctc_model,
                            main_model_predictions=[hypothesis],
                            batch_size=self.batch_size,
                            viterbi_device=torch.device(self.device),
                            timestamp_type="all",
                        )

                    timestamp: Any = aligned[0].timestamp if aligned else None
                    if timestamp and "word" in timestamp:
                        for entry in timestamp["word"]:
                            all_words.append(
                                WordAlignment(
                                    word=str(entry["word"]),
                                    start_time=float(entry["start"]) + seg.start,
                                    end_time=float(entry["end"]) + seg.start,
                                )
                            )

                    logger.report_message(
                        f"[blue]Segment {seg.start:.0f}s–{seg.end:.0f}s: {len(all_words)} words aligned so far.[/blue]"
                    )

            logger.report_message(f"[green]Alignment complete: {len(all_words)} words aligned.[/green]")
            return AlignmentResult(words=all_words)

        finally:
            del ctc_model
            gc.collect()
            try:
                import torch

                torch.cuda.empty_cache()
            except ImportError:
                pass
