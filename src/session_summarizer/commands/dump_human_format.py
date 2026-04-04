from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import session_summarizer.utils.common_paths as common_paths

from ..processing_results import SpeechClipSet
from ..settings.session_settings import SessionSettings
from .session_processing_command import SessionProcessingCommand


@dataclass
class DumpHumanFormatCommand(SessionProcessingCommand):
    json_file: str = ""

    def name(self) -> str:
        return "Dump Human Format"

    def process_session(self, settings: SessionSettings, session_dir: common_paths.Path) -> None:
        input_path: Path = session_dir / self.json_file
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        clips: SpeechClipSet = SpeechClipSet.load_from_json(input_path)
        output_path: Path = input_path.with_stem(input_path.stem + "_human").with_suffix(".txt")

        clips.save_to_human_format(output_path)
        self.logger.report_message(f"[green]Wrote {len(clips)} clips to {output_path}[/green]")
