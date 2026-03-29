from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from session_summarizer.protocols.session_settings import SessionSettings

from .session_processing_command import SessionProcessingCommand

_SETTINGS_FILE = "settings.yaml"


def _remove_tree(path: Path) -> None:
    """Recursively remove a directory tree."""
    for child in path.iterdir():
        if child.is_dir():
            _remove_tree(child)
        else:
            child.unlink()
    path.rmdir()


@dataclass
class CleanSessionCommand(SessionProcessingCommand):
    def name(self) -> str:
        return "Clean Session"

    def process_session(self, settings: SessionSettings, session_dir: Path) -> None:
        protected: set[Path] = {
            (session_dir / _SETTINGS_FILE).resolve(),
            Path(settings.audio_file).resolve(),
        }

        removed: list[Path] = []
        for item in sorted(session_dir.iterdir()):
            if item.resolve() in protected:
                continue
            if item.is_dir():
                _remove_tree(item)
            else:
                item.unlink()
            removed.append(item)

        if removed:
            self.logger.report_message(f"[green]Removed {len(removed)} item(s) from {session_dir}[/green]")
            for path in removed:
                self.logger.report_message(f"[dim]  deleted: {path.name}[/dim]")
        else:
            self.logger.report_message(f"[yellow]Nothing to remove in {session_dir}[/yellow]")
