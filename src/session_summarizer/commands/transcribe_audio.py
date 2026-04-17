from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import session_summarizer.utils.common_paths as common_paths

from ..helpers.audio_transcriber import transcribe_from_cleaned_audio
from ..processing_results.transcriber_protocol import TranscriptionResult
from ..settings.session_settings import SessionSettings
from ..vad.segment_splitter import SegmentSplitResultSet
from .clean_audio import CleanAudioCommand
from .compute_segments import ComputeSegmentsCommand
from .session_processing_command import SessionProcessingCommand


@dataclass
class TranscribeAudioCommand(SessionProcessingCommand):
    def name(self) -> str:
        return "Transcribe"

    def add_dependencies(self, settings: SessionSettings, session_dir: Path) -> None:
        self.inputs.append(session_dir / settings.paths.vad_segments)
        self.inputs.append(session_dir / settings.paths.cleaned_audio)
        self.outputs.append(session_dir / settings.paths.transcript)
        self.dependencies.append(ComputeSegmentsCommand(self.session_id, self.tracer))
        self.dependencies.append(CleanAudioCommand(self.session_id, self.tracer))

    def process_session(self, settings: SessionSettings, session_dir: common_paths.Path) -> None:
        segments: SegmentSplitResultSet = SegmentSplitResultSet.load(session_dir / settings.paths.vad_segments)
        result: TranscriptionResult = transcribe_from_cleaned_audio(settings, session_dir, segments, self, self.logger)
        result.save_to_json(session_dir / settings.paths.transcript)
        self.save_cleaned_text(result, session_dir, settings.paths.transcript)
