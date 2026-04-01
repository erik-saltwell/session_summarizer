from __future__ import annotations

from dataclasses import dataclass

import session_summarizer.utils.common_paths as common_paths
from session_summarizer.processing_results.speech_clip_set import SpeechClipSet
from session_summarizer.protocols.session_settings import SessionSettings

from ..helpers.add_embeddings import add_embeddings
from .session_processing_command import SessionProcessingCommand


@dataclass
class AddEmbeddingsCommand(SessionProcessingCommand):
    def name(self) -> str:
        return "Add Embeddings"

    def process_session(self, settings: SessionSettings, session_dir: common_paths.Path) -> None:
        self.gpu_logging_enabled = True
        clips: SpeechClipSet = add_embeddings(settings, session_dir, False, self, self.logger)
        clips.save_to_json(session_dir / settings.speech_clips_with_embedding)
