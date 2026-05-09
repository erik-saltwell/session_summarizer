from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import session_summarizer.utils.common_paths as common_paths

from ..helpers.add_embeddings import add_embeddings
from ..processing_results import SpeechClipSet
from ..settings import SessionSettings
from .clean_audio import CleanAudioCommand
from .diarize_audio_eleven_labs import DiarizeAudioElevenLabsCommand
from .session_processing_command import SessionProcessingCommand


@dataclass
class AddEmbeddingsCommand(SessionProcessingCommand):
    def name(self) -> str:
        return "Add Embeddings"

    def add_dependencies(self, settings: SessionSettings, session_dir: Path) -> None:
        self.inputs.append(session_dir / settings.paths.base_diarization)
        self.inputs.append(session_dir / settings.paths.cleaned_audio)
        self.outputs.append(session_dir / settings.paths.clips_with_embeddings)
        self.dependencies.append(DiarizeAudioElevenLabsCommand(self.session_id, self.tracer))
        self.dependencies.append(CleanAudioCommand(self.session_id, self.tracer))

    def process_session(self, settings: SessionSettings, session_dir: common_paths.Path) -> None:
        clips: SpeechClipSet = SpeechClipSet.load_from_json(session_dir / settings.paths.base_diarization)
        embedded_clips: SpeechClipSet = add_embeddings(settings, session_dir, clips, self, self.logger, self.tracer)
        self.save_speech_clip(embedded_clips, session_dir, settings.paths.clips_with_embeddings)
