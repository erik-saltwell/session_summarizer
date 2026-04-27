from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dateutil import parser

import session_summarizer.utils.common_paths as common_paths

from ..helpers.summary_generator import generate_summary
from ..processing_results import SpeechClipSet
from ..settings import SessionSettings
from .session_processing_command import SessionProcessingCommand
from .simplify_transcript import SimplifyTranscriptCommand


@dataclass
class SummarizeSessionCommand(SessionProcessingCommand):
    def name(self) -> str:
        return "Generate Summary"

    def append_date_to_filename(self, input_path: Path, date_as_a_string: str) -> Path:
        parsed = parser.parse(date_as_a_string, dayfirst=False)
        normalized_date = parsed.strftime("%Y_%m_%d")

        suffix = "".join(input_path.suffixes)
        base_name = input_path.name[: -len(suffix)] if suffix else input_path.name
        return input_path.with_name(f"{base_name}_{normalized_date}{suffix}")

    def add_dependencies(self, settings: SessionSettings, session_dir: Path) -> None:
        self.inputs.append(session_dir / settings.paths.simplified_transcript)
        self.outputs.append(session_dir / settings.paths.summary_path)
        self.dependencies.append(SimplifyTranscriptCommand(self.session_id, self.tracer))

    def process_session(self, settings: SessionSettings, session_dir: common_paths.Path) -> None:
        input_clips: SpeechClipSet = SpeechClipSet.load_from_json(session_dir / settings.paths.punctuated_text)
        output_path: Path = session_dir / settings.paths.summary_path
        output_path = self.append_date_to_filename(output_path, settings.session_info.session_date)
        summary_output: str = generate_summary(settings, session_dir, input_clips, self.logger, self.tracer)
        output_path.write_text(summary_output)
