from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import session_summarizer.utils.common_paths as common_paths

from ..helpers.audio_diarizer import diarize_audio
from ..helpers.ground_truth_adder import enhance_words_with_ground_truth
from ..processing_results import AlignmentResult, SpeechClipSet
from ..settings.session_settings import SessionSettings
from .align_transcript import AlignTranscriptCommand
from .clean_audio import CleanAudioCommand
from .compute_segments import ComputeSegmentsCommand
from .session_processing_command import SessionProcessingCommand


@dataclass
class DiarizeAudioCommand(SessionProcessingCommand):
    def name(self) -> str:
        return "Diarize Audio"

    def add_dependencies(self, settings: SessionSettings, session_dir: Path) -> None:
        self.inputs.append(session_dir / settings.paths.aligned_transcript)
        self.inputs.append(session_dir / settings.paths.vad_segments)
        self.inputs.append(session_dir / settings.paths.cleaned_audio)
        self.outputs.append(session_dir / settings.paths.base_diarization)
        self.dependencies.append(AlignTranscriptCommand(self.session_id, self.tracer))
        self.dependencies.append(ComputeSegmentsCommand(self.session_id, self.tracer))
        self.dependencies.append(CleanAudioCommand(self.session_id, self.tracer))

    def process_session(self, settings: SessionSettings, session_dir: common_paths.Path) -> None:
        alignment: AlignmentResult = AlignmentResult.load_from_json(session_dir / settings.paths.aligned_transcript)
        clips: SpeechClipSet = diarize_audio(settings, session_dir, alignment, self, self.logger, self.tracer)

        if common_paths.is_test_session(self.session_id):
            self.tracer.add_context("has_ground_truth", True)
            enhance_words_with_ground_truth(clips, self.tracer)
        else:
            self.tracer.add_context("has_ground_truth", False)
        self.save_speech_clip(clips, session_dir, settings.paths.base_diarization)
