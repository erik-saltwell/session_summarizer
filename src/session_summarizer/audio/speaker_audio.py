from __future__ import annotations

from datetime import datetime
from pathlib import Path

from session_summarizer.utils import common_paths

from ..processing_results import SpeechClip
from ..utils import run_command


def get_unique_voice_filepath(speaker_label: str) -> Path:
    filename = f"{datetime.now().strftime('%Y%m%d_%H_%M_%S_%f')}.wav"
    speaker_directory = common_paths.voice_samples_dir() / speaker_label
    common_paths.ensure_directory(speaker_directory)
    return speaker_directory / filename


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


def create_combined_speaker_audio_file(speaker_label: str, gap_length: float) -> None:
    audio_directory: Path = common_paths.voice_samples_dir() / speaker_label
    audio_extension: str = ".wav"
    output_filepath: Path = common_paths.voice_samples_dir() / (speaker_label + ".wav")

    input_files = sorted(audio_directory.glob(f"*{audio_extension}"))
    if not input_files:
        raise FileNotFoundError(f"No {audio_extension} files found in {audio_directory}")

    output_filepath.parent.mkdir(parents=True, exist_ok=True)

    cmd: list[str] = ["ffmpeg", "-y"]
    for f in input_files:
        cmd += ["-i", str(f)]

    filter_parts: list[str] = []
    concat_inputs: list[str] = []

    for i in range(len(input_files)):
        label = f"a{i}"
        filter_parts.append(f"[{i}]aresample=16000,aformat=sample_fmts=s16:channel_layouts=mono[{label}]")
        concat_inputs.append(f"[{label}]")

        if i < len(input_files) - 1 and gap_length > 0:
            gap_label = f"g{i}"
            filter_parts.append(f"aevalsrc=0:d={gap_length}:s=16000:c=mono[{gap_label}]")
            concat_inputs.append(f"[{gap_label}]")

    total_segments = len(concat_inputs)
    filter_parts.append(f"{''.join(concat_inputs)}concat=n={total_segments}:v=0:a=1[out]")

    cmd += ["-filter_complex", ";".join(filter_parts)]
    cmd += ["-map", "[out]", "-c:a", "pcm_s16le", str(output_filepath)]

    run_command(cmd)


def save_segment_as_speaker_audio_clip(
    cleaned_audio_path: Path, clip: SpeechClip, speaker_label: str, lead_in: float, lead_out: float
) -> None:
    file_path: Path = get_unique_voice_filepath(speaker_label)
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

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(actual_start),
        "-to",
        str(actual_end),
        "-i",
        str(cleaned_audio_path),
    ]
    if filters:
        cmd += ["-af", ",".join(filters)]
    cmd += ["-c:a", "pcm_s16le", str(file_path)]

    run_command(cmd)
