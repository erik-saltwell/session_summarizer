from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import session_summarizer.utils.common_paths as common_paths

from ..helpers.summary_generator import generate_summary
from ..processing_results import SpeechClipSet
from ..settings import SessionSettings
from .punctuate_text import PunctuateTextCommand
from .session_processing_command import SessionProcessingCommand


@dataclass
class SummarizeSessionCommand(SessionProcessingCommand):
    def name(self) -> str:
        return "Generate Summary"

    def add_dependencies(self, settings: SessionSettings, session_dir: Path) -> None:
        self.inputs.append(session_dir / settings.paths.punctuated_text)
        self.outputs.append(session_dir / settings.paths.summary_path)
        self.dependencies.append(PunctuateTextCommand(self.session_id))

    def process_session(self, settings: SessionSettings, session_dir: common_paths.Path) -> None:
        input_clips: SpeechClipSet = SpeechClipSet.load_from_json(session_dir / settings.paths.punctuated_text)
        output_path: Path = session_dir / settings.paths.summary_path
        summary_output: str = generate_summary(settings, session_dir, input_clips, self.logger)
        output_path.write_text(summary_output)
