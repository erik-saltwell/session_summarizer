from __future__ import annotations

import gc
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ..processing_results import TranscriptionResult, TranscriptionSegment
from ..protocols import LoggingProtocol
from ..vad.segment_splitter import SegmentSplitResultSet

_MAX_NEW_TOKENS: int = 512


def _transcribe_chunk(
    model: object,
    chunk_audio: object,
    sample_rate: int,
) -> str:
    """Write chunk to a temp WAV, transcribe via SALM, return text."""
    import soundfile as sf

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    sf.write(tmp_path, chunk_audio, sample_rate, subtype="PCM_16")

    try:
        answer_ids = model.generate(  # type: ignore[attr-defined]
            prompts=[
                [
                    {
                        "role": "user",
                        "content": f"Transcribe the following: {model.audio_locator_tag}",  # type: ignore[attr-defined]
                        "audio": [tmp_path],
                    }
                ]
            ],
            max_new_tokens=_MAX_NEW_TOKENS,
        )
        text: str = model.tokenizer.ids_to_text(answer_ids[0].tolist())  # type: ignore[attr-defined]
    finally:
        import os

        os.unlink(tmp_path)

    return text.strip()


@dataclass
class CanaryQwenTranscriber:
    """Transcriber using NVIDIA Canary Qwen 2.5B via NeMo (English-only, ~8 GB VRAM).

    Best-in-class English ASR: 5.63% WER on Open ASR Leaderboard.
    Does NOT produce word-level timestamps — run NeMo Forced Aligner (Phase 3) afterward.

    Segments are provided by the caller (VAD short segments, cut at silence boundaries).
    VRAM discipline: model is loaded and fully unloaded within each transcribe() call.

    Install NeMo separately (not in pyproject.toml due to complex CUDA deps):
        pip install nemo_toolkit[asr]
    Requires: torch, torchaudio, soundfile
    """

    model_name: str = "nvidia/canary-qwen-2.5b"
    device: str = "cuda"

    def transcribe(
        self, audio_path: Path, segments: SegmentSplitResultSet, logger: LoggingProtocol
    ) -> TranscriptionResult:
        try:
            import soundfile as sf
            from nemo.collections.speechlm2 import SALM
        except ImportError as e:
            raise ImportError("NeMo and soundfile are required: pip install nemo_toolkit[asr] soundfile") from e

        import torch

        logger.report_message(f"[blue]Loading Canary Qwen model {self.model_name}...[/blue]")
        torch.backends.cudnn.benchmark = False
        model = SALM.from_pretrained(self.model_name)
        model.to(self.device)
        model.eval()
        gc.collect()
        torch.cuda.empty_cache()

        try:
            audio_data, sample_rate = sf.read(str(audio_path), dtype="float32")
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)

            audio_segments = segments.short.segments
            logger.report_message(f"[blue]Transcribing {len(audio_segments)} VAD segments...[/blue]")

            result_segments: list[TranscriptionSegment] = []
            texts: list[str] = []

            with (
                torch.inference_mode(),
                logger.progress("Transcribing segments", total=len(audio_segments)) as progress,
            ):
                for seg in audio_segments:
                    start_idx = int(seg.start * sample_rate)
                    end_idx = int(seg.end * sample_rate)
                    chunk = audio_data[start_idx:end_idx]
                    text = _transcribe_chunk(model, chunk, sample_rate)
                    if text:
                        texts.append(text)
                        result_segments.append(
                            TranscriptionSegment(
                                text=text,
                                start=seg.start,
                                end=seg.end,
                                confidence=0.0,  # overwritten after alignment
                            )
                        )
                    torch.cuda.empty_cache()
                    progress.advance(1)

            full_text = " ".join(texts)
            logger.report_message("[green]Transcription complete[/green]")
            return TranscriptionResult(segments=result_segments, full_text=full_text)

        finally:
            del model
            gc.collect()
            try:
                import torch

                torch.cuda.empty_cache()
            except ImportError:
                pass
