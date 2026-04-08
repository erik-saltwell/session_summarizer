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
        test_clips = SpeechClipSet.load_from_test_meeting()
        identity_stitched_cips = SpeechClipSet.load_from_json(session_dir / settings.identity_stitched_path)
        test_clips.save_to_rttm(session_dir / "ground_truth.rttm")
        identity_stitched_cips.save_to_rttm(session_dir / "identity_stitched.rttm")
