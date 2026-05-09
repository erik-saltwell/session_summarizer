from __future__ import annotations

import shutil
import tempfile
from datetime import datetime
from pathlib import Path

from session_summarizer.utils import common_paths

from ..processing_results import SpeechClip
from ..protocols import LoggingProtocol
from ..utils import run_command


def get_unique_voice_filepath(speaker_label: str, temp_folder: Path | None) -> Path:
    filename = f"{datetime.now().strftime('%Y%m%d_%H_%M_%S_%f')}.wav"
    dir_path = common_paths.voice_samples_for_speaker(speaker_label, root_folder=temp_folder)
    common_paths.ensure_directory(dir_path)
    return dir_path / filename


def _get_audio_duration(audio_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    result = run_command(cmd, capture_output=True)
    return float(result.stdout.strip())


def _concat_audio_files(left: Path, right: Path, output: Path, gap_length: float = 0.0) -> None:
    """Concatenate two WAV files into one using ffmpeg with an optional silence gap."""
    if gap_length > 0:
        filter_complex = (
            "[0]aresample=16000,aformat=sample_fmts=s16:channel_layouts=mono[a0];"
            f"aevalsrc=0:d={gap_length}:s=16000:c=mono[g];"
            "[1]aresample=16000,aformat=sample_fmts=s16:channel_layouts=mono[a1];"
            "[a0][g][a1]concat=n=3:v=0:a=1[out]"
        )
    else:
        filter_complex = (
            "[0]aresample=16000,aformat=sample_fmts=s16:channel_layouts=mono[a0];"
            "[1]aresample=16000,aformat=sample_fmts=s16:channel_layouts=mono[a1];"
            "[a0][a1]concat=n=2:v=0:a=1[out]"
        )
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(left),
        "-i",
        str(right),
        "-filter_complex",
        filter_complex,
        "-map",
        "[out]",
        "-c:a",
        "pcm_s16le",
        str(output),
    ]
    run_command(cmd)


def merge_speaker_clips_to_min_duration(
    input_dir: Path,
    output_dir: Path,
    min_duration: float,
    gap_length: float,
    logger: LoggingProtocol,
) -> None:
    """Greedy-merge shortest clips until all are >= min_duration, writing results to output_dir."""
    input_files = sorted(input_dir.glob("*.wav"))
    if not input_files:
        raise FileNotFoundError(f"No .wav files found in {input_dir}")

    if output_dir.exists() and (not output_dir.is_dir()):
        raise ValueError(f"{output_dir} is not a directory")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"{output_dir} exists and is not empty.")

    state: list[tuple[Path, float]] = [(f, _get_audio_duration(f)) for f in input_files]
    logger.report_message(f"[blue]Found {len(state)} clip(s) in {input_dir}[/blue]")

    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        counter = 0

        while len(state) > 1:
            min_dur = min(d for _, d in state)
            if min_dur >= min_duration:
                break

            shortest_idx = min(range(len(state)), key=lambda i: state[i][1])

            if shortest_idx == 0:
                neighbor_idx = 1
            elif shortest_idx == len(state) - 1:
                neighbor_idx = shortest_idx - 1
            else:
                left_dur = state[shortest_idx - 1][1]
                right_dur = state[shortest_idx + 1][1]
                neighbor_idx = shortest_idx - 1 if left_dur <= right_dur else shortest_idx + 1

            left_idx, right_idx = sorted([shortest_idx, neighbor_idx])
            left_path, left_dur = state[left_idx]
            right_path, right_dur = state[right_idx]

            merged_path = tmp / f"merged_{counter:04d}.wav"
            counter += 1
            _concat_audio_files(left_path, right_path, merged_path, gap_length)
            merged_dur = left_dur + right_dur

            logger.report_message(
                f"Merged {left_path.name} ({left_dur:.2f}s) + {right_path.name} "
                f"({right_dur:.2f}s) → {merged_path.name} ({merged_dur:.2f})s"
            )
            state = state[:left_idx] + [(merged_path, merged_dur)] + state[right_idx + 1 :]

        for i, (path, dur) in enumerate(state):
            dest = output_dir / f"{i:06d}.wav"
            shutil.copy2(str(path), str(dest))
            logger.report_message(f"Wrote {dest.name} ({dur:.2f}s)")

    logger.report_message(
        f"[green]Done — {len(input_files)} original clip(s) merged into {len(state)} clip(s) "
        f"and written to {output_dir}[/green]"
    )


def save_segment_as_speaker_audio_clip(
    cleaned_audio_path: Path,
    clip: SpeechClip,
    speaker_label: str,
    lead_in: float,
    lead_out: float,
    temp_folder: Path | None,
) -> None:
    file_path: Path = get_unique_voice_filepath(speaker_label, temp_folder)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    file_duration = _get_audio_duration(cleaned_audio_path)

    actual_start = max(0.0, clip.start_time - lead_in)
    actual_end = min(file_duration, clip.end_time + lead_out)
    actual_lead_in = clip.start_time - actual_start
    actual_lead_out = actual_end - clip.end_time

    filters: list[str] = []
    if actual_lead_in > 0:
        filters.append(f"afade=t=in:d={actual_lead_in}")
    if actual_lead_out > 0:
        fade_out_start = clip.end_time - actual_start
        filters.append(f"afade=t=out:st={fade_out_start}:d={actual_lead_out}")

    filters.append("aresample=16000")
    filters.append("aformat=sample_fmts=s16:channel_layouts=mono")

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(actual_start),
        "-to",
        str(actual_end),
        "-i",
        str(cleaned_audio_path),
        "-af",
        ",".join(filters),
        "-c:a",
        "pcm_s16le",
        str(file_path),
    ]

    run_command(cmd)
