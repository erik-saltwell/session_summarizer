# from __future__ import annotations

# from dataclasses import dataclass

# import session_summarizer.utils.common_paths as common_paths

# from ..settings import SessionSettings
# from .add_embeddings import AddEmbeddingsCommand
# from .align_transcript import AlignTranscriptCommand
# from .clean_audio import CleanAudioCommand
# from .clean_session import CleanSessionCommand
# from .compute_segments import ComputeSegmentsCommand
# from .diarizationlm_command import DiarizationLMCommand
# from .diarize_audio import DiarizeAudioCommand
# from .identify_speakers import IdentifySpeakersCommand
# from .score_confidence import ScoreConfidenceCommand
# from .session_processing_command import SessionProcessingCommand
# from .transcribe_audio import TranscribeAudioCommand


# @dataclass
# class StitchResults:
#     pre_stitching_segments: int = 0
#     post_stitching_segments: int = 0


# @dataclass
# class ProcessPipelineCommand(SessionProcessingCommand):
#     def name(self) -> str:
#         return "Process Pipeline"

#     def process_session(self, settings: SessionSettings, session_dir: common_paths.Path) -> None:
#         CleanSessionCommand(self.session_id).execute(self.logger)
#         CleanAudioCommand(self.session_id).execute(self.logger)
#         ComputeSegmentsCommand(self.session_id).execute(self.logger)
#         TranscribeAudioCommand(self.session_id).execute(self.logger)
#         AlignTranscriptCommand(self.session_id).execute(self.logger)
#         ScoreConfidenceCommand(self.session_id).execute(self.logger)
#         DiarizeAudioCommand(self.session_id).execute(self.logger)
#         DiarizationLMCommand(self.session_id).execute(self.logger)

#         #        UpdateTurnEndCommand(self.session_id).execute(self.logger)
#         #        FirstStitchClipsCommand(self.session_id).execute(self.logger)
#         AddEmbeddingsCommand(self.session_id).execute(self.logger)
#         IdentifySpeakersCommand(self.session_id).execute(self.logger)
#         #        StitichIdentitiesCommand(self.session_id).execute(self.logger)
#         #        DiarizationLMCommand(self.session_id).execute(self.logger)
#         # DumpAndCompareTextsCommand(self.session_id).execute(self.logger)
#         # DumpHumanFormatCommand(self.session_id).execute(self.logger)
#         # ValidateDiarizationCommand(self.session_id).execute(self.logger)
