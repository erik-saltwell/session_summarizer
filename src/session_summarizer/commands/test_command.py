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

    def output_confidence_success_map(self, settings: SessionSettings, session_dir: Path) -> None:
        csv_path = session_dir / "conf_wder.csv"

        with csv_path.open("w") as writer:
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

    def output_success_data(self, settings: SessionSettings, session_dir: common_paths.Path) -> None:
        csv_path = session_dir / "similarity_succes.csv"

        with csv_path.open("w") as writer:
            writer.write(
                "speaker_count,word_count,duration,similarity,similarity_residual,avg_confidence,success_rate,text\n"
            )
            clips: SpeechClipSet = SpeechClipSet.load_from_json(session_dir / settings.identity_stitched_path)
            for clip in clips:
                if clip.cosine_similarity and clip.words:
                    writer.write(
                        f"{len(clip.speakers)},"
                        f"{clip.word_count},"
                        f"{clip.duration},"
                        f"{clip.cosine_similarity},"
                        f"{clip.similarity_residual},"
                        f"{clip.confidence_avg},"
                        f"{1.0 - clip.wder},"
                        f"{clip.text}"
                        "\n"
                    )

    def process_session(self, settings: SessionSettings, session_dir: common_paths.Path) -> None:
        self.output_success_data(settings, session_dir)
