from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from session_summarizer.utils import common_paths

from ..helpers.infer_speakers import (
    UNKNOWN_SPEAKER_IDENTITY,
    build_participant_map,
    get_inferred_participants_path,
    load_inferred_participants,
)
from ..helpers.speaker_clip_exporter import SpeakerClipExport, export_speaker_audio_clips
from ..processing_results import SpeechClipSet
from ..processing_results.speech_clip import SpeechClip, SpeechClipFlags
from ..protocols import (
    SessionSettings,
)
from .clean_audio import CleanAudioCommand
from .identify_speakers import IdentifySpeakersCommand
from .infer_speakers import InferSpeakersCommand
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


def should_export_known_speaker_clip(
    clip: SpeechClip, min_similarity_residual: float, use_multi_speaker_clips: bool
) -> bool:
    similarity_residual = clip.similarity_residual if clip.similarity_residual is not None else 0.0
    return (
        (not clip.is_anonymous)
        and (not clip.has_flag(SpeechClipFlags.IS_BACKCHANNEL))
        and clip.identity is not None
        and (use_multi_speaker_clips or (not clip.is_multispeaker))
        and similarity_residual > min_similarity_residual
    )


@dataclass
class CreateKnownSpeakerClipsCommand(SessionProcessingCommand):
    temp_folder: Path = common_paths.voice_samples_dir()
    use_multi_speaker_clips: bool = False

    def name(self) -> str:
        return "Create Known Speaker Clips"

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

        skipped_count = 0
        exports: list[SpeakerClipExport] = []

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

            should_save = should_export_known_speaker_clip(
                clip,
                target_residual,
                use_multi_speaker_clips,
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
            exports.append(SpeakerClipExport(clip=clip, speaker_name=clip.identity))

        export_speaker_audio_clips(
            cleaned_audio_path,
            exports,
            skipped_count,
            settings.speaker_clips.lead_in_seconds,
            settings.speaker_clips.lead_out_seconds,
            self.temp_folder,
            self.logger,
        )


def _inferred_export_speaker_name(clip: SpeechClip, participant_by_label: dict[str, str]) -> str | None:
    if clip.has_flag(SpeechClipFlags.IS_BACKCHANNEL):
        return None
    if clip.identity is None or clip.identity == UNKNOWN_SPEAKER_IDENTITY:
        return None

    participant_names: set[str] = set()
    for speaker_label in clip.speakers:
        participant_name = participant_by_label.get(speaker_label)
        if participant_name is None:
            return None
        participant_names.add(participant_name)

    if len(participant_names) != 1:
        return None
    return next(iter(participant_names))


@dataclass
class CreateSpeakerClipsFromInferredSpeakersCommand(SessionProcessingCommand):
    def name(self) -> str:
        return "Create Speaker Clips From Inferred Speakers"

    def add_dependencies(self, settings: SessionSettings, session_dir: Path) -> None:
        self.inputs.append(session_dir / settings.paths.inferred_speakers)
        self.inputs.append(get_inferred_participants_path(session_dir, settings.paths.inferred_speakers))
        self.inputs.append(session_dir / settings.paths.cleaned_audio)
        self.dependencies.append(InferSpeakersCommand(self.session_id, self.tracer))
        self.dependencies.append(CleanAudioCommand(self.session_id, self.tracer))

    def process_session(self, settings: SessionSettings, session_dir: Path) -> None:
        cleaned_audio_path = session_dir / settings.paths.cleaned_audio
        inferred_clips: SpeechClipSet = SpeechClipSet.load_from_json(session_dir / settings.paths.inferred_speakers)
        participants = load_inferred_participants(
            get_inferred_participants_path(session_dir, settings.paths.inferred_speakers)
        )
        source_labels = {label for clip in inferred_clips for label in clip.speakers}
        participant_by_label = build_participant_map(participants, source_labels)

        skipped_count = 0
        exports: list[SpeakerClipExport] = []
        for clip in inferred_clips:
            speaker_name = _inferred_export_speaker_name(clip, participant_by_label)
            if speaker_name is None:
                skipped_count += 1
                continue
            exports.append(SpeakerClipExport(clip=clip, speaker_name=speaker_name))

        export_speaker_audio_clips(
            cleaned_audio_path,
            exports,
            skipped_count,
            settings.speaker_clips.lead_in_seconds,
            settings.speaker_clips.lead_out_seconds,
            None,
            self.logger,
        )
