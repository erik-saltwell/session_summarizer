from __future__ import annotations

import gc
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ..protocols import LoggingProtocol
from ..protocols.transcriber_protocol import TranscriptionResult, TranscriptionSegment

# Canary Qwen 2.5B training max duration and chunking parameters (from architecture doc)
_CHUNK_DURATION_S: float = 40.0
_OVERLAP_S: float = 5.0
_DISCARD_S: float = 2.5  # discard first/last 2.5s of each chunk at stitching boundaries
_MAX_NEW_TOKENS: int = 512


def _build_chunks(
    audio_data: object,
    sample_rate: int,
    total_duration: float,
) -> list[tuple[object, float, bool, bool]]:
    """Slice audio into overlapping chunks.

    Returns list of (chunk_audio, chunk_start_seconds, is_first, is_last).
    """
    chunks: list[tuple[object, float, bool, bool]] = []
    stride = _CHUNK_DURATION_S - _OVERLAP_S
    start_s = 0.0
    while start_s < total_duration:
        end_s = min(start_s + _CHUNK_DURATION_S, total_duration)
        start_idx = int(start_s * sample_rate)
        end_idx = int(end_s * sample_rate)
        chunk = audio_data[start_idx:end_idx]  # type: ignore[index]
        is_last = end_s >= total_duration
        chunks.append((chunk, start_s, start_s == 0.0, is_last))
        if is_last:
            break
        start_s += stride
    return chunks


def _transcribe_chunk(
    model: object,
    chunk_audio: object,
    sample_rate: int,
    is_first: bool,
    is_last: bool,
) -> str:
    """Write chunk to a temp WAV, transcribe via SALM, return text with boundary trimming."""
    import soundfile as sf

    discard_samples = int(_DISCARD_S * sample_rate)
    audio = chunk_audio
    if not is_first and len(audio) > discard_samples:  # type: ignore[arg-type]
        audio = audio[discard_samples:]  # type: ignore[index]
    if not is_last and len(audio) > discard_samples:  # type: ignore[arg-type]
        audio = audio[:-discard_samples]  # type: ignore[index]

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    sf.write(tmp_path, audio, sample_rate, subtype="PCM_16")

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


def _deduplicate_boundary(left: str, right: str) -> str:
    """Remove overlap between the tail of `left` and the head of `right`."""
    left_words = left.split()
    right_words = right.split()
    max_overlap = min(20, len(left_words), len(right_words))
    for n in range(max_overlap, 0, -1):
        if left_words[-n:] == right_words[:n]:
            return " ".join(left_words + right_words[n:])
    return left + " " + right


def _stitch_chunks(chunk_texts: list[str]) -> str:
    """Join chunk texts, deduplicating repeated phrases at stitch boundaries."""
    if not chunk_texts:
        return ""
    result = chunk_texts[0]
    for text in chunk_texts[1:]:
        result = _deduplicate_boundary(result, text)
    return result


def _build_segments(
    chunks: list[tuple[object, float, bool, bool]],
    chunk_texts: list[str],
    sample_rate: int,
) -> list[TranscriptionSegment]:
    segments = []
    for (chunk_audio, chunk_start_s, _is_first, _is_last), text in zip(chunks, chunk_texts, strict=False):
        if not text:
            continue
        chunk_len = len(chunk_audio) / sample_rate  # type: ignore[arg-type]
        segments.append(
            TranscriptionSegment(
                text=text,
                start=float(chunk_start_s),
                end=float(chunk_start_s) + chunk_len,
                confidence=0.0,  # overwritten after alignment
            )
        )
    return segments


@dataclass
class CanaryQwenTranscriber:
    """Transcriber using NVIDIA Canary Qwen 2.5B via NeMo (English-only, ~8 GB VRAM).

    Best-in-class English ASR: 5.63% WER on Open ASR Leaderboard.
    Does NOT produce word-level timestamps — run NeMo Forced Aligner (Phase 3) afterward.

    Chunking: 40s windows with 5s overlap; boundary deduplication at stitch points.
    VRAM discipline: model is loaded and fully unloaded within each transcribe() call.

    Install NeMo separately (not in pyproject.toml due to complex CUDA deps):
        pip install nemo_toolkit[asr]
    Requires: torch, torchaudio, soundfile
    """

    model_name: str = "nvidia/canary-qwen-2.5b"
    device: str = "cuda"

    def transcribe(self, audio_path: Path, logger: LoggingProtocol) -> TranscriptionResult:
        try:
            import soundfile as sf
            from nemo.collections.speechlm2 import SALM
        except ImportError as e:
            raise ImportError("NeMo and soundfile are required: pip install nemo_toolkit[asr] soundfile") from e

        logger.report_message(f"[blue]Loading Canary Qwen model {self.model_name}...[/blue]")
        model = SALM.from_pretrained(self.model_name)
        model.eval()

        try:
            audio_data, sample_rate = sf.read(str(audio_path), dtype="float32")
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)

            total_duration = len(audio_data) / sample_rate
            logger.report_message(
                f"[blue]Audio duration: {total_duration:.1f}s — chunking into {_CHUNK_DURATION_S}s windows...[/blue]"
            )

            chunks = _build_chunks(audio_data, sample_rate, total_duration)
            logger.report_message(f"[blue]Processing {len(chunks)} chunks...[/blue]")

            chunk_texts: list[str] = []
            with logger.progress("Transcribing chunks", total=len(chunks)) as progress:
                for chunk_audio, _chunk_start, is_first, is_last in chunks:
                    chunk_text = _transcribe_chunk(model, chunk_audio, sample_rate, is_first, is_last)
                    chunk_texts.append(chunk_text)
                    progress.advance(1)

            full_text = _stitch_chunks(chunk_texts)
            logger.report_message("[green]Transcription complete[/green]")

            segments = _build_segments(chunks, chunk_texts, sample_rate)
            return TranscriptionResult(segments=segments, full_text=full_text)

        finally:
            del model
            gc.collect()
            try:
                import torch

                torch.cuda.empty_cache()
            except ImportError:
                pass
