from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import session_summarizer.utils.common_paths as common_paths

from ..helpers.speaker_identifier import identify_speakers
from ..processing_results import SpeechClipSet
from ..settings import SessionSettings
from .add_embeddings import AddEmbeddingsCommand
from .session_processing_command import SessionProcessingCommand


@dataclass
class IdentifySpeakersCommand(SessionProcessingCommand):
    def name(self) -> str:
        return "Identify Speakers"

    def add_dependencies(self, settings: SessionSettings, session_dir: Path) -> None:
        self.inputs.append(session_dir / settings.speech_clips_with_embedding)
        self.outputs.append(session_dir / settings.identified_speaker_path)
        self.dependencies.append(AddEmbeddingsCommand(self.session_id))

    def process_session(self, settings: SessionSettings, session_dir: common_paths.Path) -> None:
        clips_with_embeddings: SpeechClipSet = SpeechClipSet.load_from_json(
            session_dir / settings.speech_clips_with_embedding
        )
        identified_speaker_clips: SpeechClipSet = identify_speakers(
            settings, session_dir, clips_with_embeddings, self, self.logger
        )
        self.save_speech_clip(identified_speaker_clips, session_dir, settings.identified_speaker_path)
