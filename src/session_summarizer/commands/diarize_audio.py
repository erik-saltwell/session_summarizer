from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import session_summarizer.utils.common_paths as common_paths

from ..helpers.audio_diarizer import diarize_audio
from ..helpers.ground_truth_adder import enhance_words_with_ground_truth
from ..processing_results import SpeechClipSet
from ..settings.session_settings import SessionSettings
from ..transcription.parakeet_ctc_confidence_scorer import AlignmentResult
from .clean_audio import CleanAudioCommand
from .compute_segments import ComputeSegmentsCommand
from .score_confidence import ScoreConfidenceCommand
from .session_processing_command import SessionProcessingCommand


@dataclass
class DiarizeAudioCommand(SessionProcessingCommand):
    def name(self) -> str:
        return "Diarize Audio"

    def add_dependencies(self, settings: SessionSettings, session_dir: Path) -> None:
        self.inputs.append(session_dir / settings.confidence_transcript_path)
        self.inputs.append(session_dir / settings.segments_path)
        self.inputs.append(session_dir / settings.cleaned_audio_file)
        self.outputs.append(session_dir / settings.base_diarized_path)
        self.dependencies.append(ScoreConfidenceCommand(self.session_id))
        self.dependencies.append(ComputeSegmentsCommand(self.session_id))
        self.dependencies.append(CleanAudioCommand(self.session_id))

    def process_session(self, settings: SessionSettings, session_dir: common_paths.Path) -> None:
        alignment: AlignmentResult = AlignmentResult.load_from_json(session_dir / settings.confidence_transcript_path)
        clips: SpeechClipSet = diarize_audio(settings, session_dir, alignment, self, self.logger)

        if common_paths.is_test_session(self.session_id):
            enhance_words_with_ground_truth(clips)
        self.save_speech_clip(clips, session_dir, settings.base_diarized_path)
