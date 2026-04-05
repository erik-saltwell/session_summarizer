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
from ..audio.speaker_audio import create_combined_speaker_audio_file
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
        f"[dim]vRAM ({label}): {allocated:.1f}GB allocated, {reserved:.1f}GB reserved, {total:.1f}GB total[/dim]"
    )


@dataclass
class RegisterSpeakersCommand:
    logger: LoggingProtocol = NullLogger()
    device: str = "cuda"
    gap_length: float = 0.5
    """Register all speakers found in per-speaker subdirectories of voice_samples/ into registered_speakers.yaml."""

    def execute(self, logger: LoggingProtocol) -> None:
        self.logger = logger
        voice_dir: Path = common_paths.voice_samples_dir()

        # Phase 1 — delete existing root WAVs so stale combined files don't persist
        for wav in voice_dir.glob("*.wav"):
            wav.unlink()

        # Phase 2 — discover speaker directories (skip hidden dirs)
        speaker_dirs: list[Path] = sorted(d for d in voice_dir.iterdir() if d.is_dir() and not d.name.startswith("."))
        if not speaker_dirs:
            logger.report_message("[yellow]No speaker directories found in voice_samples/.[/yellow]")
            return

        # Phase 3 — generate one combined WAV per speaker directory
        for speaker_dir in speaker_dirs:
            logger.report_message(f"[blue]Combining clips for {speaker_dir.name}...[/blue]")
            create_combined_speaker_audio_file(speaker_dir.name, gap_length=self.gap_length)

        # Phase 4 — register embeddings from the newly generated combined WAVs
        wav_files: list[Path] = sorted(voice_dir.glob("*.wav"))
        logger.report_message(f"[blue]Registering {len(wav_files)} speaker(s) from voice_samples/[/blue]")

        speakers: RegisteredSpeakers = RegisteredSpeakers.load(common_paths.build_speakers_file_path())

        for wav_file in wav_files:
            self.register_speaker(wav_file.stem, speakers)

        speakers.save(common_paths.build_speakers_file_path())

        self.logger.report_message(
            f"[green]{len(speakers)} speaker(s) saved to {common_paths.build_speakers_file_path()}.[/green]"
        )

    def name(self) -> str:
        return "Register All Speakers"

    def register_speaker(self, speaker_name: str, speakers: RegisteredSpeakers) -> None:
        wav_file: Path = common_paths.voice_sample_wav_file(speaker_name)
        if not wav_file.exists():
            raise FileNotFoundError(f"WAV file not found: {wav_file}")

        action = "Updating" if speaker_name in speakers else "Registering"
        self.logger.report_message(f"[blue]{action} speaker '{speaker_name}'[/blue]")

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
                speakers[speaker_name] = embedder.extract(normalized, self.logger)
            _log_gpu_usage(self.logger, "after embedding extraction")
