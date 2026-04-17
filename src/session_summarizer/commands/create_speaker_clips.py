from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from session_summarizer.utils import common_paths

from ..audio.speaker_tools import save_segment_as_speaker_audio_clip
from ..processing_results import SpeechClipSet
from ..processing_results.speech_clip import SpeechClipFlags
from ..protocols import (
    SessionSettings,
)
from .clean_audio import CleanAudioCommand
from .identify_speakers import IdentifySpeakersCommand
from .session_processing_command import SessionProcessingCommand


def _resolve_temp_folder(temp_folder: Path) -> Path:
    voice_samples_root = common_paths.voice_samples_dir().resolve()
    resolved = (voice_samples_root / temp_folder).resolve()
    if resolved == voice_samples_root:
        raise ValueError("Refusing to empty the voice_samples root; pass a temp subdirectory instead.")
    if not resolved.is_relative_to(voice_samples_root):
        raise ValueError(f"Temp folder must resolve inside {voice_samples_root}, got {resolved}")
    return resolved


def _empty_temp_folder(temp_folder: Path) -> bool:
    if not temp_folder.exists():
        return False
    if not temp_folder.is_dir():
        raise ValueError(f"Temp folder exists and is not a directory: {temp_folder}")

    for item in temp_folder.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
    return True


@dataclass
class CreateSpeakerClipsCommand(SessionProcessingCommand):
    temp_folder: Path = common_paths.voice_samples_dir()
    use_multi_speaker_clips: bool = False

    def name(self) -> str:
        return "Create Speaker Clips"

    def add_dependencies(self, settings: SessionSettings, session_dir: Path) -> None:
        self.inputs.append(session_dir / settings.paths.identified_speakers)
        self.inputs.append(session_dir / settings.paths.cleaned_audio)
        self.dependencies.append(IdentifySpeakersCommand(self.session_id, self.tracer))
        self.dependencies.append(CleanAudioCommand(self.session_id, self.tracer))

    def process_session(self, settings: SessionSettings, session_dir: Path) -> None:
        temp_folder = _resolve_temp_folder(self.temp_folder)
        if _empty_temp_folder(temp_folder):
            self.report_message(f"Emptied speaker clip temp folder: {temp_folder}")

        cleaned_audio_path = session_dir / settings.paths.cleaned_audio
        identified_clips: SpeechClipSet = SpeechClipSet.load_from_json(session_dir / settings.paths.identified_speakers)

        saved_count = 0
        skipped_count = 0
        speaker_clip_counts: dict[str, int] = {}
        speaker_durations: dict[str, float] = {}

        for clip in identified_clips:
            is_anonymous: bool = clip.is_anonymous
            is_identity_none: bool = clip.identity is None
            is_multispeaker: bool = clip.is_multispeaker
            is_backchannel: bool = clip.has_flag(SpeechClipFlags.IS_BACKCHANNEL)

            use_multi_speaker_clips: bool = self.use_multi_speaker_clips
            similarity_residual: float = clip.similarity_residual if clip.similarity_residual is not None else 0.0
            target_residual: float = (
                settings.speaker_clips.min_similarity_residual
                if settings.speaker_clips.min_similarity_residual is not None
                else 1.00
            )

            should_save: bool = (
                (not is_anonymous)
                and (not is_backchannel)
                and (not is_identity_none)
                and (use_multi_speaker_clips or (not is_multispeaker))
                and similarity_residual > target_residual
            )

            self.report_message(
                f"Clip analysis: "
                f"anonymous: {is_anonymous}, "
                f"has_identity: {not is_identity_none}, "
                f"multispeaker: {is_multispeaker}, "
                f"backchannel: {is_backchannel}, "
                f"use_multispeaker: {use_multi_speaker_clips}, "
                f"residual: {similarity_residual}, "
                f"target_residual: {target_residual},"
                f"should_save: {should_save}"
            )
            if not should_save:
                skipped_count += 1
                continue

            assert clip.identity is not None

            save_segment_as_speaker_audio_clip(
                cleaned_audio_path,
                clip,
                clip.identity,
                settings.speaker_clips.lead_in_seconds,
                settings.speaker_clips.lead_out_seconds,
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
