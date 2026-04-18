from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import session_summarizer.utils.common_paths as common_paths

from ..helpers.speaker_identifier import identify_speakers
from ..processing_results import SpeechClipSet
from ..settings import SessionSettings
from ..speaker_embeddings import RegisteredSpeakers
from .add_embeddings import AddEmbeddingsCommand
from .session_processing_command import SessionProcessingCommand

_REGISTERED_SPEAKERS_FILE = "registered_speakers.yaml"


def _resolve_speakers_file(session_dir: Path) -> Path:
    session_file = session_dir / _REGISTERED_SPEAKERS_FILE
    if session_file.exists():
        return session_file
    return common_paths.build_speakers_file_path()


@dataclass
class IdentifySpeakersCommand(SessionProcessingCommand):
    def name(self) -> str:
        return "Identify Speakers"

    def add_dependencies(self, settings: SessionSettings, session_dir: Path) -> None:
        self.inputs.append(session_dir / settings.paths.clips_with_embeddings)
        self.outputs.append(session_dir / settings.paths.identified_speakers)
        self.dependencies.append(AddEmbeddingsCommand(self.session_id, self.tracer))

    def process_session(self, settings: SessionSettings, session_dir: common_paths.Path) -> None:
        if not settings.attendees:
            raise ValueError(
                "No attendees are configured in SessionSettings. "
                "Add at least one attendee name to the 'attendees' list in settings.yaml."
            )

        speakers_file = common_paths.speakers_file(self.session_id)
        registered = RegisteredSpeakers.load(speakers_file)
        missing = [name for name in settings.attendees if name not in registered]
        if missing:
            raise ValueError(
                f"The following attendee(s) were not found in {speakers_file}:\n"
                + "".join(f"  - {name}\n" for name in missing)
                + "Run 'register speakers' for each missing attendee before identifying speakers."
            )

        clips_with_embeddings: SpeechClipSet = SpeechClipSet.load_from_json(
            session_dir / settings.paths.clips_with_embeddings
        )
        identified_speaker_clips: SpeechClipSet = identify_speakers(
            settings, session_dir, clips_with_embeddings, self, self.logger, self.tracer
        )
        self.save_speech_clip(identified_speaker_clips, session_dir, settings.paths.identified_speakers)
