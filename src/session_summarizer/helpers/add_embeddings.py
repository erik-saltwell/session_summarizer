from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from ..processing_results.speech_clip import SpeechClip
from ..processing_results.speech_clip_set import SpeechClipSet
from ..protocols import (
    EmbeddingFactory,
    GpuLogger,
    LoggingProtocol,
    SessionSettings,
)
from ..speaker_embeddings import get_embeddings_factory
from ..utils import Tracer, flush_gpu_memory, silence_os_noise


def add_embeddings(
    settings: SessionSettings,
    session_dir: Path,
    clips: SpeechClipSet,
    gpu_logger: GpuLogger,
    logger: LoggingProtocol,
    tracer: Tracer,
) -> SpeechClipSet:
    audio_path: Path = session_dir / settings.paths.cleaned_audio
    audio_data, sample_rate = sf.read(str(audio_path), dtype="float32")
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)

    embedding_factory: EmbeddingFactory
    with logger.status("Loading speaker embedding model."):
        with silence_os_noise():
            embedding_factory = get_embeddings_factory(settings.device)
    gpu_logger.report_gpu_usage("after loading embedding model")

    max_embedding_duration_s = 30.0
    min_samples = 400
    iterations_per_flush_gpu = 100
    max_samples = int(max_embedding_duration_s * sample_rate)
    clip: SpeechClip
    with logger.progress("Generating embeddings", total=len(clips)) as progress:
        for idx, clip in enumerate(clips):
            start_sample = int(clip.start_time * sample_rate)
            end_sample = int(clip.end_time * sample_rate)
            chunk = audio_data[start_sample:end_sample]

            # Cap long clips — speaker embeddings don't benefit from more than ~30s
            if len(chunk) > max_samples:
                chunk = chunk[:max_samples]

            # Kaldi fbank requires at least 400 samples (25ms at 16kHz); pad if shorter
            if len(chunk) < min_samples:
                chunk = np.pad(chunk, (0, min_samples - len(chunk)))

            duration_s = len(chunk) / sample_rate
            gpu_logger.report_gpu_usage(
                f"before clip {clip.start_time:.1f}s-{clip.end_time:.1f}s ({duration_s:.1f}s long)"
            )

            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir) / "chunk.wav"
                sf.write(str(tmp_path), chunk, sample_rate, subtype="PCM_16")
                clip.embedding = embedding_factory.extract(tmp_path, logger)

            if idx % iterations_per_flush_gpu == 0 and idx > 0:
                flush_gpu_memory()
                gpu_logger.report_gpu_usage(f"after clip {clip.start_time:.1f}s-{clip.end_time:.1f}s")

            progress.advance()

    gpu_logger.report_gpu_usage("after generating embeddings")
    return clips
