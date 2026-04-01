from __future__ import annotations

import gc
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..processing_results import AlignmentResult, WordAlignment
from ..protocols import LoggingProtocol
from ..vad.segment_splitter import AudioSegment, SegmentSplitResultSet


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
class ParakeetCTCConfidenceScorer:
    """Scores word-level confidence using NeMo's parakeet-ctc-0.6b CTC model.

    Takes an existing AlignmentResult (with word timestamps from forced alignment)
    and adds acoustic confidence scores by running a free transcription pass and
    mapping per-word confidence onto the aligned words by time overlap.

    VRAM discipline: model is loaded and fully unloaded within each score() call.

    Install NeMo separately (not in pyproject.toml due to complex CUDA deps):
        pip install nemo_toolkit[asr]
    """

    model_name: str = "nvidia/parakeet-ctc-0.6b"
    device: str = "cuda"

    def _transcribe_segments(
        self,
        audio_path: Path,
        ctc_model: Any,
        stride: float,
        audio_segments: list[AudioSegment],
        logger: LoggingProtocol,
    ) -> tuple[list[dict], list[float]]:
        """Run free transcription over VAD segments, returning merged word timestamps and confidences."""
        import soundfile as sf
        import torch

        info = sf.info(str(audio_path))
        sample_rate = info.samplerate

        all_timestamps: list[dict] = []
        all_confidences: list[float] = []

        for seg in audio_segments:
            start_sample = int(seg.start * sample_rate)
            end_sample = int(seg.end * sample_rate)
            audio_data, sr = sf.read(str(audio_path), start=start_sample, stop=end_sample, dtype="float32")

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                sf.write(tmp.name, audio_data, sr)
                with torch.inference_mode():
                    hyps: Any = ctc_model.transcribe([tmp.name], return_hypotheses=True)

            hyp: Any = hyps[0]
            word_ts: list[dict] = hyp.timestamp.get("word", []) if hyp.timestamp else []
            word_conf: list[float] = hyp.word_confidence or []

            # Shift timestamps from segment-local to absolute time
            for wt in word_ts:
                if "start_offset" in wt:
                    wt["start_offset"] = wt["start_offset"] + seg.start / stride
                    wt["end_offset"] = wt["end_offset"] + seg.start / stride
                elif "start" in wt:
                    wt["start"] = float(wt["start"]) + seg.start
                    wt["end"] = float(wt["end"]) + seg.start

            all_timestamps.extend(word_ts)
            all_confidences.extend(word_conf)

            logger.report_message(f"[blue]Segment {seg.start:.0f}s–{seg.end:.0f}s: {len(word_ts)} words[/blue]")

        return all_timestamps, all_confidences

    def score(
        self, audio_path: Path, alignment: AlignmentResult, segments: SegmentSplitResultSet, logger: LoggingProtocol
    ) -> AlignmentResult:
        if not alignment.words:
            logger.report_warning("[yellow]Empty alignment — skipping confidence scoring.[/yellow]")
            return alignment

        try:
            import torch
            from nemo.collections.asr.models import ASRModel
            from nemo.collections.asr.parts.submodules.ctc_decoding import CTCBPEDecodingConfig
            from nemo.collections.asr.parts.utils.asr_confidence_utils import (
                ConfidenceConfig,
                ConfidenceMethodConfig,
            )
        except ImportError as e:
            raise ImportError("NeMo is required: pip install nemo_toolkit[asr]") from e

        logger.report_message(f"[blue]Loading CTC model {self.model_name} for confidence scoring...[/blue]")
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
            window_stride = ctc_model.cfg.preprocessor.window_stride
            sub_factor = getattr(ctc_model.cfg.encoder, "subsampling_factor", 4)
            stride = window_stride * sub_factor

            logger.report_message("[blue]Running free transcription pass for confidence scores...[/blue]")
            parakeet_word_timestamps, parakeet_word_confidence = self._transcribe_segments(
                audio_path, ctc_model, stride, segments.long.segments, logger
            )

            _map_confidence_by_time(
                alignment.words, parakeet_word_timestamps, parakeet_word_confidence, time_stride=stride
            )

            logger.report_message(f"[green]Confidence scoring complete: {len(alignment.words)} words scored.[/green]")
            return alignment

        finally:
            del ctc_model
            gc.collect()
            try:
                import torch

                torch.cuda.empty_cache()
            except ImportError:
                pass
