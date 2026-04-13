from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from session_summarizer.utils import common_paths

from ..audio.speaker_tools import save_segment_as_speaker_audio_clip
from ..processing_results import SpeechClipSet
from ..protocols import (
    SessionSettings,
)
from .clean_audio import CleanAudioCommand
from .identify_speakers import IdentifySpeakersCommand
from .session_processing_command import SessionProcessingCommand


@dataclass
class CreateSpeakerClipsCommand(SessionProcessingCommand):
    temp_folder: Path = common_paths.voice_samples_dir()
    use_multi_speaker_clips: bool = False

    def name(self) -> str:
        return "Create Speaker Clips"

    def add_dependencies(self, settings: SessionSettings, session_dir: Path) -> None:
        self.inputs.append(session_dir / settings.identified_speaker_path)
        self.inputs.append(session_dir / settings.cleaned_audio_file)
        self.dependencies.append(IdentifySpeakersCommand(self.session_id))
        self.dependencies.append(CleanAudioCommand(self.session_id))

    def process_session(self, settings: SessionSettings, session_dir: Path) -> None:
        cleaned_audio_path = session_dir / settings.cleaned_audio_file
        identified_clips: SpeechClipSet = SpeechClipSet.load_from_json(session_dir / settings.identified_speaker_path)

        saved_count = 0
        skipped_count = 0
        speaker_clip_counts: dict[str, int] = {}
        speaker_durations: dict[str, float] = {}

        for clip in identified_clips:
            if clip.is_anonymous or clip.identity is None:
                skipped_count += 1
                continue
            if clip.is_multispeaker and not self.use_multi_speaker_clips:
                skipped_count += 1
                continue
            if (
                clip.similarity_residual is None
                or clip.similarity_residual < settings.speaker_clip_minimum_similarity_residual
            ):
                skipped_count += 1
                continue

            save_segment_as_speaker_audio_clip(
                cleaned_audio_path,
                clip,
                clip.identity,
                settings.speaker_clip_lead_in,
                settings.speaker_clip_lead_out,
                temp_folder=self.temp_folder,
            )
            saved_count += 1
            speaker_clip_counts[clip.identity] = speaker_clip_counts.get(clip.identity, 0) + 1
            speaker_durations[clip.identity] = speaker_durations.get(clip.identity, 0.0) + clip.duration

        self.report_message(f"Saved {saved_count} speaker clips, skipped {skipped_count}")

        if speaker_clip_counts:
            headers = ["Speaker", "Clips", "Duration (s)"]
            rows = [
                [speaker, str(speaker_clip_counts[speaker]), f"{speaker_durations[speaker]:.2f}"]
                for speaker in sorted(speaker_clip_counts)
            ]
            self.logger.report_multicolumn_table(headers, rows)
