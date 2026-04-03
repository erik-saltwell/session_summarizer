from __future__ import annotations

from pathlib import Path

from ..processing_results.speech_clip_set import SpeechClipFlags, SpeechClipSet
from ..protocols import (
    GpuLogger,
    LoggingProtocol,
    SessionSettings,
)
from ..turn_detection.smart_turn_predictor import LocalSmartTurnPredictor
from ..turn_detection.smart_turn_scorer import load_audio_mono_16k, score_clips_with_smart_turn


def update_turn_end(
    settings: SessionSettings,
    session_dir: Path,
    clips: SpeechClipSet,
    use_cache_if_present: bool,
    gpu_logger: GpuLogger,
    logger: LoggingProtocol,
) -> SpeechClipSet:
    final_path: Path = session_dir / settings.turn_end_updated_path
    if final_path.exists() and use_cache_if_present:
        logger.report_message(f"[yellow]{final_path} already exists, returning cached instance.[/yellow]")
        return SpeechClipSet.load_from_json(final_path)

    with logger.status("Loading audio for Smart Turn scoring..."):
        audio = load_audio_mono_16k(session_dir / settings.cleaned_audio_file)

    with logger.status("Loading Smart Turn model..."):
        predictor = LocalSmartTurnPredictor(device=settings.device)
    gpu_logger.report_gpu_usage("after loading Smart Turn model")

    score_clips_with_smart_turn(
        audio_f32_16k=audio,
        clips=clips,
        predictor=predictor,
        silence_reset_seconds=0.8,
        reset_on_speaker_change=True,
        logger=logger,
    )
    gpu_logger.report_gpu_usage("after Smart Turn scoring")

    threshold = settings.diarization_stitching.turn_end_probability_threshold
    for clip in clips:
        if clip.end_of_turn_probability is not None:
            clip.set_flag(
                SpeechClipFlags.END_OF_TURN,
                clip.end_of_turn_probability >= threshold,
            )

    logger.report_message("[blue]Turn-end scoring complete.[/blue]")
    return clips
