from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import session_summarizer.utils.common_paths as common_paths

from ..helpers.add_embeddings import add_embeddings
from ..processing_results import SpeechClipSet
from ..settings import SessionSettings
from .clean_audio import CleanAudioCommand
from .compute_segments import ComputeSegmentsCommand
from .dangling_sentece_fix import DanglingSentenceFixCommand
from .session_processing_command import SessionProcessingCommand


@dataclass
class AddEmbeddingsCommand(SessionProcessingCommand):
    def name(self) -> str:
        return "Add Embeddings"

    def add_dependencies(self, settings: SessionSettings, session_dir: Path) -> None:
        self.inputs.append(session_dir / settings.dangling_sentence_fix_path)
        self.inputs.append(session_dir / settings.cleaned_audio_file)
        self.inputs.append(session_dir / settings.segments_path)
        self.outputs.append(session_dir / settings.speech_clips_with_embedding)
        self.dependencies.append(DanglingSentenceFixCommand(self.session_id))
        self.dependencies.append(ComputeSegmentsCommand(self.session_id))
        self.dependencies.append(CleanAudioCommand(self.session_id))

    def process_session(self, settings: SessionSettings, session_dir: common_paths.Path) -> None:
        clips: SpeechClipSet = SpeechClipSet.load_from_json(session_dir / settings.dangling_sentence_fix_path)
        embedded_clips: SpeechClipSet = add_embeddings(settings, session_dir, clips, self, self.logger)
        self.save_speech_clip(embedded_clips, session_dir, settings.speech_clips_with_embedding)
