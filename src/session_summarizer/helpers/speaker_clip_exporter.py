from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from ..audio.speaker_tools import save_segment_as_speaker_audio_clip
from ..processing_results import SpeechClip
from ..protocols import LoggingProtocol

SaveSpeakerClipFn = Callable[[Path, SpeechClip, str, float, float, Path | None], None]


@dataclass(frozen=True)
class SpeakerClipExport:
    clip: SpeechClip
    speaker_name: str


@dataclass(frozen=True)
class SpeakerClipExportSummary:
    saved_count: int
    skipped_count: int
    speaker_clip_counts: dict[str, int]
    speaker_durations: dict[str, float]


def validate_speaker_sample_name(speaker_name: str) -> str:
    cleaned_name = speaker_name.strip()
    if not cleaned_name:
        raise ValueError("Speaker sample name must be non-empty.")
    if Path(cleaned_name).name != cleaned_name or cleaned_name in {".", ".."}:
        raise ValueError(f"Speaker sample name must not contain path separators or traversal: {speaker_name!r}")
    return cleaned_name


def export_speaker_audio_clips(
    cleaned_audio_path: Path,
    exports: Iterable[SpeakerClipExport],
    skipped_count: int,
    lead_in_seconds: float,
    lead_out_seconds: float,
    temp_folder: Path | None,
    logger: LoggingProtocol,
    save_clip_fn: SaveSpeakerClipFn = save_segment_as_speaker_audio_clip,
) -> SpeakerClipExportSummary:
    saved_count = 0
    speaker_clip_counts: dict[str, int] = {}
    speaker_durations: dict[str, float] = {}

    for export in exports:
        speaker_name = validate_speaker_sample_name(export.speaker_name)
        save_clip_fn(
            cleaned_audio_path,
            export.clip,
            speaker_name,
            lead_in_seconds,
            lead_out_seconds,
            temp_folder,
        )
        saved_count += 1
        speaker_clip_counts[speaker_name] = speaker_clip_counts.get(speaker_name, 0) + 1
        speaker_durations[speaker_name] = speaker_durations.get(speaker_name, 0.0) + export.clip.duration

    logger.report_message(f"Saved {saved_count} speaker clips, skipped {skipped_count}")

    if speaker_clip_counts:
        headers = ["Speaker", "Clips", "Duration (s)"]
        rows = [
            [speaker, str(speaker_clip_counts[speaker]), f"{speaker_durations[speaker]:.2f}"]
            for speaker in sorted(speaker_clip_counts)
        ]
        logger.report_multicolumn_table(headers, rows)

    return SpeakerClipExportSummary(
        saved_count=saved_count,
        skipped_count=skipped_count,
        speaker_clip_counts=speaker_clip_counts,
        speaker_durations=speaker_durations,
    )
