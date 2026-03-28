from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import session_summarizer.utils.common_paths as common_paths

from ..audio.sound_cleaning_helper import (
    convert_to_48k_wav,
    enhance_with_mossformer2,
    measure_loudness,
    normalize_and_export_16k_mono,
)
from ..protocols import CommmandProtocol, EmbeddingFactory, LoggingProtocol
from ..speaker_embeddings import RegisteredSpeakers, get_embeddings_factory
from ..utils.flush_gpu_memory import flush_gpu_memory


@dataclass
class TestCommand(CommmandProtocol):
    session_id: str = "testa"

    def report(self, message: str) -> None:
        self.logger.report_message(f"[blue]{message}[/blue]")

    def execute(self, logger: LoggingProtocol) -> None:
        self.logger = logger
        self.report("converting to 48kwav...")
        convert_to_48k_wav(
            common_paths.audio_file_from_step(common_paths.audio_processing_step.original, self.session_id),
            common_paths.audio_file_from_step(common_paths.audio_processing_step.wav_48k, self.session_id),
        )
        self.report("clean with enhance_with_mossformer2...")
        enhance_with_mossformer2(
            common_paths.audio_file_from_step(common_paths.audio_processing_step.wav_48k, self.session_id),
            common_paths.audio_file_from_step(common_paths.audio_processing_step.cleaned_audio, self.session_id),
        )
        self.report("measure_loudness...")
        stats = measure_loudness(
            common_paths.audio_file_from_step(common_paths.audio_processing_step.cleaned_audio, self.session_id),
        )
        self.report("normalize_and_export_16k_mono...")
        normalize_and_export_16k_mono(
            common_paths.audio_file_from_step(common_paths.audio_processing_step.cleaned_audio, self.session_id),
            common_paths.audio_file_from_step(common_paths.audio_processing_step.normalized_audio, self.session_id),
            stats,
        )

        flush_gpu_memory()

        embedder: EmbeddingFactory = get_embeddings_factory("cpu")
        yaml_path: Path = common_paths.build_speakers_file_path(self.session_id)

        speakers: RegisteredSpeakers = RegisteredSpeakers.load(yaml_path)

        speakers["test"] = embedder.extract(
            common_paths.audio_file_from_step(common_paths.audio_processing_step.normalized_audio, self.session_id),
            logger,
        )
        speakers.save(common_paths.build_speakers_file_path("testa"))
