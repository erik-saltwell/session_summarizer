from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import session_summarizer.utils.common_paths as common_paths

from ..processing_results import SpeechClipSet
from ..settings.session_settings import SessionSettings
from .session_processing_command import SessionProcessingCommand


@dataclass
class DumpHumanFormatCommand(SessionProcessingCommand):
    def name(self) -> str:
        return "Dump Human Format"

    def process_session(self, settings: SessionSettings, session_dir: common_paths.Path) -> None:
        paths = [
            settings.base_diarized_path,
            settings.turn_end_updated_path,
            settings.first_stitched_path,
            settings.identified_speaker_path,
            settings.identity_stitched_path,
        ]
        for relative_path in paths:
            input_path: Path = session_dir / relative_path
            if not input_path.exists():
                self.logger.report_message(f"[yellow]Skipping {input_path} (not found)[/yellow]")
                continue

            clips: SpeechClipSet = SpeechClipSet.load_from_json(input_path)
            output_path: Path = input_path.with_stem(input_path.stem + "_human").with_suffix(".txt")

            clips.save_to_human_format(output_path)
            self.logger.report_message(f"[green]Wrote {len(clips)} clips to {output_path}[/green]")
