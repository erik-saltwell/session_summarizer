from __future__ import annotations

import json
import re
from pathlib import Path

from clearvoice import ClearVoice

from ..utils import run_command


def convert_to_48k_wav(input_path: Path, temp_wav_path: Path) -> None:
    """Convert source audio to the mono 48 kHz PCM WAV we will feed into MossFormer2_SE_48K."""
    temp_wav_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "48000",
        "-c:a",
        "pcm_s16le",
        str(temp_wav_path),
    ]
    run_command(cmd)


def enhance_with_mossformer2(model_input_wav: Path, enhanced_wav_path: Path) -> None:
    """Run ClearVoice speech enhancement with MossFormer2_SE_48K."""
    enhanced_wav_path.parent.mkdir(parents=True, exist_ok=True)

    clearvoice = ClearVoice(
        task="speech_enhancement",
        model_names=["MossFormer2_SE_48K"],
    )

    output_wav = clearvoice(input_path=str(model_input_wav), online_write=False)
    clearvoice.write(output_wav, output_path=str(enhanced_wav_path))


def measure_loudness(
    input_wav: Path, target_i: float = -16.0, target_lra: float = 11.0, target_tp: float = -1.5
) -> dict[str, str]:
    """Run loudnorm pass 1 and parse the emitted JSON stats."""
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-i",
        str(input_wav),
        "-af",
        f"loudnorm=I={target_i}:LRA={target_lra}:TP={target_tp}:print_format=json",
        "-f",
        "null",
        "-",
    ]
    result = run_command(cmd, capture_output=True)

    # FFmpeg prints loudnorm JSON to stderr.
    match = re.search(r"\{[\s\S]*?\}", result.stderr)
    if not match:
        raise RuntimeError("Could not parse loudnorm JSON from ffmpeg output.")

    data = json.loads(match.group(0))
    return {k: str(v) for k, v in data.items()}


def normalize_and_export_16k_mono(
    input_wav: Path,
    output_wav: Path,
    stats: dict[str, str],
    *,
    target_i: float = -16.0,
    target_lra: float = 11.0,
    target_tp: float = -1.5,
) -> None:
    """Run loudnorm pass 2 and export the final 16 kHz mono PCM WAV."""
    output_wav.parent.mkdir(parents=True, exist_ok=True)

    loudnorm_filter = (
        "loudnorm="
        f"I={target_i}:"
        f"LRA={target_lra}:"
        f"TP={target_tp}:"
        f"measured_I={stats['input_i']}:"
        f"measured_LRA={stats['input_lra']}:"
        f"measured_TP={stats['input_tp']}:"
        f"measured_thresh={stats['input_thresh']}:"
        f"offset={stats['target_offset']}:"
        "linear=true:"
        "print_format=summary"
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_wav),
        "-af",
        loudnorm_filter,
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_wav),
    ]
    run_command(cmd)
