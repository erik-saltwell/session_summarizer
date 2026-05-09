from __future__ import annotations

import os
import tempfile
from pathlib import Path

from elevenlabs import ElevenLabs

from ..utils import common_paths
from .sound_cleaning import convert_to_16k_mono, measure_loudness, normalize_and_export_16k_mono


def isolate_audio_eleven_labs(input_file: Path, output_file: Path, normalize_audio: bool) -> None:
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY is not set in the environment.")
    common_paths.ensure_directory(output_file.parent)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        isolated_raw = tmp / "isolated.mp3"
        client = ElevenLabs(api_key=api_key)

        with input_file.open("rb") as audio_in:
            audio_stream = client.audio_isolation.convert(
                audio=audio_in,
                file_format="pcm_s16le_16",
            )
            with isolated_raw.open("wb") as out:
                for chunk in audio_stream:
                    if chunk:
                        out.write(chunk)

        if normalize_audio:
            stats = measure_loudness(isolated_raw)
            normalize_and_export_16k_mono(isolated_raw, output_file, stats)
        else:
            convert_to_16k_mono(isolated_raw, output_file)
