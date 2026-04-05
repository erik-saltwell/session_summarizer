from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from ..processing_results.speech_clip_set import SpeechClip, SpeechClipSet
from ..protocols import (
    EmbeddingFactory,
    GpuLogger,
    LoggingProtocol,
    SessionSettings,
)
from ..speaker_embeddings import get_embeddings_factory


def add_embeddings(
    settings: SessionSettings,
    session_dir: Path,
    clips: SpeechClipSet,
    use_cache_if_present: bool,
    gpu_logger: GpuLogger,
    logger: LoggingProtocol,
) -> SpeechClipSet:
    final_path: Path = session_dir / settings.speech_clips_with_embedding
    if final_path.exists() and use_cache_if_present:
        logger.report_message(f"[yellow]{final_path} already exists, returning cached instance.[/yellow]")
        return SpeechClipSet.load_from_json(final_path)

    audio_path: Path = session_dir / settings.cleaned_audio_file
    audio_data, sample_rate = sf.read(str(audio_path), dtype="float32")
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)

    embedding_factory: EmbeddingFactory
    with logger.status("Loading speaker embedding model."):
        embedding_factory = get_embeddings_factory(settings.device)
    gpu_logger.report_gpu_usage("after loading embedding model")

    clip: SpeechClip
    with logger.progress("Generating embeddings", total=len(clips)) as progress:
        for clip in clips:
            start_sample = int(clip.start_time * sample_rate)
            end_sample = int(clip.end_time * sample_rate)
            chunk = audio_data[start_sample:end_sample]

            # Kaldi fbank requires at least 400 samples (25ms at 16kHz); pad if shorter
            min_samples = 400
            if len(chunk) < min_samples:
                chunk = np.pad(chunk, (0, min_samples - len(chunk)))

            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir) / "chunk.wav"
                sf.write(str(tmp_path), chunk, sample_rate, subtype="PCM_16")
                clip.embedding = embedding_factory.extract(tmp_path, logger)

            progress.advance()

    gpu_logger.report_gpu_usage("after generating embeddings")
    return clips
