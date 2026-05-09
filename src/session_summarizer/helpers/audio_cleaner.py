from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from ..audio import (
    convert_to_16k_mono,
    convert_to_48k_wav,
    enhance_with_mossformer2,
    measure_loudness,
    normalize_and_export_16k_mono,
)
from ..protocols import GpuLogger, LoggingProtocol, SessionSettings
from ..utils import Tracer


def clean_audio(
    settings: SessionSettings,
    session_dir: Path,
    normalize_volume: bool,
    gpu_logger: GpuLogger,
    logger: LoggingProtocol,
    tracer: Tracer,
) -> None:
    original_path = session_dir / settings.paths.source_audio
    final_path = session_dir / settings.paths.cleaned_audio

    tracer.add_context("input_audio", original_path)
    if not original_path.exists():
        raise FileNotFoundError(original_path)

    with TemporaryDirectory() as tmpdir:
        tmp_dir = Path(tmpdir)
        wav_48k_path = tmp_dir / "wav_48k.wav"
        post_mosfet_path = tmp_dir / "post_mosfet.wav"

        gpu_logger.report_gpu_usage("before processing")

        with logger.status("Converting to 48k WAV..."):
            convert_to_48k_wav(original_path, wav_48k_path)
        gpu_logger.report_gpu_usage("after 48k esv conversion")

        with logger.status("Enhancing with MossFormer2..."):
            enhance_with_mossformer2(wav_48k_path, post_mosfet_path)
        gpu_logger.report_gpu_usage("after MossFormer2 enhancement")

        if normalize_volume:
            with logger.status("Measuring loudness..."):
                stats = measure_loudness(post_mosfet_path)
            gpu_logger.report_gpu_usage("after loudness measurement")
            with logger.status("Normalizing and converting to 16k mono..."):
                normalize_and_export_16k_mono(post_mosfet_path, final_path, stats)
        else:
            with logger.status("Converting to 16k mono..."):
                convert_to_16k_mono(post_mosfet_path, final_path)
        gpu_logger.report_gpu_usage("after 16k normalization")
        tracer.add_context("cleaned_audio", final_path)
