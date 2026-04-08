from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import session_summarizer.utils.common_paths as common_paths
from session_summarizer.processing_results.speech_clip_set import SpeechClipSet

from ..settings.session_settings import SessionSettings
from .session_processing_command import SessionProcessingCommand


@dataclass
class TestCommand(SessionProcessingCommand):
    def name(self) -> str:
        return "Test"

    def add_dependencies(self, settings: SessionSettings, session_dir: Path) -> None:
        return

    def process_session(self, settings: SessionSettings, session_dir: common_paths.Path) -> None:
        csv_path = session_dir / "conf_wder.csv"

        with csv_path.open("a") as writer:
            writer.write("confidence,success_rate\n")
            clips: SpeechClipSet = SpeechClipSet.load_from_json(session_dir / settings.identity_stitched_path)
            for clip in clips:
                assert clip.identity is not None
                identity: str = clip.identity.lower()
                if clip.words is None:
                    continue
                for word in clip.words:
                    if word.ground_truth is not None:
                        writer.write(f"{word.confidence},{1.0 if word.ground_truth.lower() == identity else 0.0}\n")
