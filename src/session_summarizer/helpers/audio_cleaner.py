from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from ..audio import convert_to_48k_wav, enhance_with_mossformer2, measure_loudness, normalize_and_export_16k_mono
from ..protocols import GpuLogger, LoggingProtocol, SessionSettings


def clean_audio(
    settings: SessionSettings,
    session_dir: Path,
    use_cache_if_present: bool,
    gpu_logger: GpuLogger,
    logger: LoggingProtocol,
) -> None:
    original_path = session_dir / settings.audio_file
    logger.report_message(f"[blue]Cleaning audio for {original_path}[/blue]")
    final_path = session_dir / settings.cleaned_audio_file
    if final_path.exists() and use_cache_if_present:
        logger.report_message(f"[yellow]{final_path} already exists, skipping processing.[/yellow]")
        return

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

        with logger.status("Measuring loudness..."):
            stats = measure_loudness(post_mosfet_path)
        gpu_logger.report_gpu_usage("after loudness measurement")

        with logger.status("Normalizing to 16k mono..."):
            normalize_and_export_16k_mono(post_mosfet_path, final_path, stats)
        gpu_logger.report_gpu_usage("after 16k normalization")

    logger.report_message(f"[blue]Cleaned audio written to {final_path}.[/blue]")
