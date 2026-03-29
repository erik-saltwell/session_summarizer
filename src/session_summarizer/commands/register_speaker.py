from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch

import session_summarizer.utils.common_paths as common_paths

from ..audio import (
    convert_to_48k_wav,
    enhance_with_mossformer2,
    measure_loudness,
    normalize_and_export_16k_mono,
)
from ..protocols import EmbeddingFactory, LoggingProtocol, NullLogger
from ..speaker_embeddings import get_embeddings_factory
from ..speaker_embeddings.registered_speakers import RegisteredSpeakers


def _log_gpu_usage(logger: LoggingProtocol, label: str) -> None:
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    logger.report_message(
        f"[dim]GPU RAM ({label}): {allocated:.1f} GB allocated, {reserved:.1f} GB reserved, {total:.1f} GB total[/dim]"
    )


@dataclass
class RegisterSpeakerCommand:
    """Extract an ERes2NetV2 speaker embedding and store it in registered_speakers.yaml."""

    device: str
    speaker_name: str
    session_id: str | None = None
    logger: LoggingProtocol = NullLogger()

    def name(self) -> str:
        return "Register Single Speaker"

    def execute(self, logger: LoggingProtocol) -> None:
        self.logger = logger
        wav_file: Path = common_paths.voice_sample_wav_file(self.speaker_name)
        if not wav_file.exists():
            raise FileNotFoundError(f"WAV file not found: {wav_file}")

        yaml_path: Path = common_paths.build_speakers_file_path(self.session_id)

        speakers: RegisteredSpeakers = RegisteredSpeakers.load(yaml_path)

        action = "Updating" if self.speaker_name in speakers else "Registering"
        self.logger.report_message(f"[blue]{action} speaker '{self.speaker_name}'[/blue]")

        _log_gpu_usage(self.logger, "before processing")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            wav_48k = tmp / "wav_48k.wav"
            cleaned_audio = tmp / "cleaned_audio.wav"
            normalized = tmp / "normalized_audio.wav"

            with self.logger.status("Converting to 48k WAV..."):
                convert_to_48k_wav(wav_file, wav_48k)
            _log_gpu_usage(self.logger, "after 48k conversion")

            with self.logger.status("Enhancing with MossFormer2..."):
                enhance_with_mossformer2(wav_48k, cleaned_audio)
            _log_gpu_usage(self.logger, "after MossFormer2 cleanup")

            with self.logger.status("Measuring loudness..."):
                stats = measure_loudness(cleaned_audio)

            with self.logger.status("Normalizing to 16k mono..."):
                normalize_and_export_16k_mono(cleaned_audio, normalized, stats)

            _log_gpu_usage(self.logger, "before embedding model")
            embedder: EmbeddingFactory = get_embeddings_factory(self.device)
            _log_gpu_usage(self.logger, "after embedding model load")

            with self.logger.status("Extracting speaker embedding..."):
                speakers[self.speaker_name] = embedder.extract(normalized, logger)
            _log_gpu_usage(self.logger, "after embedding extraction")

        speakers.save(yaml_path)
        self.logger.report_message(f"[green]Speaker '{self.speaker_name}' saved to {yaml_path}.[/green]")
