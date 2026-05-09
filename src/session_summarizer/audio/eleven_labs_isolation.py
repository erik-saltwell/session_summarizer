from __future__ import annotations

import os
import tempfile
from pathlib import Path

from elevenlabs import ElevenLabs

from ..protocols import LoggingProtocol
from ..utils import Tracer
from .sound_cleaning import convert_to_16k_mono, measure_loudness, normalize_and_export_16k_mono


def isolate_audio(
    input_file: Path, output_file: Path, normalize_audio: bool, logger: LoggingProtocol, tracer: Tracer
) -> None:
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY is not set in the environment.")

    tracer.add_context("input_file", str(input_file))
    tracer.add_context("output_file", str(output_file))
    tracer.add_context("normalize_audio", normalize_audio)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        preprocessed_wav = tmp / "input_16k_mono.wav"
        isolated_raw = tmp / "isolated.mp3"

        logger.report_message("[blue]Preprocessing audio to 16 kHz mono PCM…[/blue]")
        convert_to_16k_mono(input_file, preprocessed_wav)

        logger.report_message("[blue]Calling ElevenLabs voice isolator…[/blue]")
        client = ElevenLabs(api_key=api_key)
        with preprocessed_wav.open("rb") as audio_in:
            audio_stream = client.audio_isolation.convert(
                audio=audio_in,
                file_format="pcm_s16le_16",
            )
            with isolated_raw.open("wb") as out:
                for chunk in audio_stream:
                    if chunk:
                        out.write(chunk)

        logger.report_message(f"[blue]Saving isolated audio as 16 kHz mono WAV to {output_file}…[/blue]")
        if normalize_audio:
            stats = measure_loudness(isolated_raw)
            normalize_and_export_16k_mono(isolated_raw, output_file, stats)
        else:
            convert_to_16k_mono(isolated_raw, output_file)

    tracer.log("eleven_labs_isolation_complete")
    logger.report_message(f"[green]Done — isolated audio written to {output_file}[/green]")
