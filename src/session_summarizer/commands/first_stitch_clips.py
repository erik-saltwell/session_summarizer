# from __future__ import annotations

# from dataclasses import asdict, dataclass
# from pathlib import Path

# import session_summarizer.utils.common_paths as common_paths

# from ..helpers.first_stitcher import apply_first_stitching
# from ..processing_results import SpeechClipSet
# from ..settings import SessionSettings
# from .session_processing_command import SessionProcessingCommand
# from .stitch_results import StitchResults
# from .update_turn_end import UpdateTurnEndCommand


# # @dataclass
# # class FirstStitchClipsCommand(SessionProcessingCommand):
# #     def name(self) -> str:
# #         return "First Stitch Clips"

# #     def add_dependencies(self, settings: SessionSettings, session_dir: Path) -> None:
# #         self.inputs.append(session_dir / settings.turn_end_updated_path)
# #         self.outputs.append(session_dir / settings.first_stitched_path)
# #         self.dependencies.append(UpdateTurnEndCommand(self.session_id))

# #     def process_session(self, settings: SessionSettings, session_dir: common_paths.Path) -> None:
# #         turn_clips: SpeechClipSet = SpeechClipSet.load_from_json(session_dir / settings.turn_end_updated_path)

# #         stitch_results: StitchResults = StitchResults()
# #         stitch_results.pre_stitching_segments = len(turn_clips)

# #         stitched_clips: SpeechClipSet = apply_first_stitching(settings, session_dir, turn_clips, self, self.logger)
# #         stitch_results.post_stitching_segments = len(stitched_clips)
# #         self.logger.report_table_message(asdict(stitch_results))

# #         self.save_speech_clip(stitched_clips, session_dir, settings.first_stitched_path)
