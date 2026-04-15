from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import session_summarizer.utils.common_paths as common_paths

from ..helpers.diarizationlm_refiner import DiarizationLMResult, apply_diarizationlm
from ..processing_results import SpeechClipSet
from ..settings import SessionSettings
from .diarize_audio import DiarizeAudioCommand
from .session_processing_command import SessionProcessingCommand


@dataclass
class DiarizationLMCommand(SessionProcessingCommand):
    def name(self) -> str:
        return "DiarizationLM"

    def add_dependencies(self, settings: SessionSettings, session_dir: Path) -> None:
        self.inputs.append(session_dir / settings.paths.base_diarization)
        self.outputs.append(session_dir / settings.paths.diarizationlm_processed)
        self.dependencies.append(DiarizeAudioCommand(self.session_id, self.tracer))

    def process_session(self, settings: SessionSettings, session_dir: common_paths.Path) -> None:
        clips: SpeechClipSet = SpeechClipSet.load_from_json(session_dir / settings.paths.base_diarization)
        diarizationlm_result: DiarizationLMResult = apply_diarizationlm(settings, session_dir, clips, self, self.logger)
        if diarizationlm_result.prompt_segment_count is None:
            self.logger.report_message("DiarizationLM did not build prompt segments.")
        elif diarizationlm_result.prompt_segment_count == 0:
            self.logger.report_message("DiarizationLM found no words to process.")
        elif diarizationlm_result.prompt_segment_count == 1:
            word_count = diarizationlm_result.prompt_word_count
            self.logger.report_message(f"DiarizationLM processed {word_count} word(s) as one segment; no split needed.")
        else:
            self.logger.report_message(
                f"DiarizationLM split {diarizationlm_result.prompt_word_count} word(s) into "
                f"{diarizationlm_result.prompt_segment_count} segments."
            )

        self.save_speech_clip(diarizationlm_result.clips, session_dir, settings.paths.diarizationlm_processed)
