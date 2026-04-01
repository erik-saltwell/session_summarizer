from __future__ import annotations

import tempfile
from pathlib import Path

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
    use_cache_if_present: bool,
    gpu_logger: GpuLogger,
    logger: LoggingProtocol,
) -> SpeechClipSet:
    import soundfile as sf

    final_path: Path = session_dir / settings.speech_clips_with_embedding
    if final_path.exists() and use_cache_if_present:
        logger.report_message(f"[yellow]{final_path} already exists, returning cached instance.[/yellow]")
        return SpeechClipSet.load_from_json(final_path)

    base_path: Path = session_dir / settings.base_diarized_path
    clips: SpeechClipSet = SpeechClipSet.load_from_json(base_path)

    audio_path: Path = session_dir / settings.cleaned_audio_file
    audio_data, sample_rate = sf.read(str(audio_path), dtype="float32")
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)

    embedding_factory: EmbeddingFactory
    with logger.status("Loading speaker embedding model."):
        embedding_factory = get_embeddings_factory(settings.device)
    gpu_logger.report_gpu_usage("after loading embedding model")

    clip: SpeechClip
    for i, clip in enumerate(clips):
        logger.report_message(f"[blue]Generating embedding for clip {i + 1}/{len(clips)}...[/blue]")
        start_sample = int(clip.start_time * sample_rate)
        end_sample = int(clip.end_time * sample_rate)
        chunk = audio_data[start_sample:end_sample]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            sf.write(str(tmp_path), chunk, sample_rate, subtype="PCM_16")
            clip.embedding = embedding_factory.extract(tmp_path, logger)
        finally:
            tmp_path.unlink(missing_ok=True)

    gpu_logger.report_gpu_usage("after generating embeddings")
    logger.report_message("[blue]Embedding generation complete.[/blue]")
    return clips
