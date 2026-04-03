from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

from ..processing_results.speech_clip_set import SpeechClip, SpeechClipSet
from ..protocols import LoggingProtocol
from .smart_turn_predictor import SmartTurnPredictor

SMART_TURN_SR: int = 16_000
SMART_TURN_MAX_SECONDS: float = 8.0
SMART_TURN_MAX_SAMPLES: int = int(SMART_TURN_SR * SMART_TURN_MAX_SECONDS)


def load_audio_mono_16k(path: Path) -> np.ndarray:
    """Load an audio file as mono float32 at 16 kHz, peak-normalized to [-1, 1]."""
    audio, sr = librosa.load(str(path), sr=None, mono=True)
    if sr != SMART_TURN_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SMART_TURN_SR)

    audio = audio.astype(np.float32, copy=False)

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak

    return audio


def seconds_to_sample_index(seconds: float) -> int:
    return max(0, int(round(seconds * SMART_TURN_SR)))


def pad_left_to_8s(audio_f32: np.ndarray) -> np.ndarray:
    """Keep at most the last 8 seconds; left-pad with zeros if shorter."""
    if len(audio_f32) >= SMART_TURN_MAX_SAMPLES:
        return audio_f32[-SMART_TURN_MAX_SAMPLES:]

    padded = np.zeros(SMART_TURN_MAX_SAMPLES, dtype=np.float32)
    padded[-len(audio_f32) :] = audio_f32
    return padded


def should_reset_turn(
    prev_clip: SpeechClip | None,
    current_clip: SpeechClip,
    silence_reset_seconds: float,
    reset_on_speaker_change: bool,
) -> bool:
    """Determine whether the current turn should reset before this clip."""
    if prev_clip is None:
        return True

    gap = current_clip.start_time - prev_clip.end_time
    if gap >= silence_reset_seconds:
        return True

    if reset_on_speaker_change:
        # Only trigger on speaker change when both clips have exactly one
        # non-anonymous speaker and those speakers differ.
        if (
            not prev_clip.is_multispeaker
            and not current_clip.is_multispeaker
            and not prev_clip.is_anonymous
            and not current_clip.is_anonymous
            and prev_clip.speakers != current_clip.speakers
        ):
            return True

    return False


def score_clips_with_smart_turn(
    audio_f32_16k: np.ndarray,
    clips: SpeechClipSet,
    predictor: SmartTurnPredictor,
    silence_reset_seconds: float,
    reset_on_speaker_change: bool,
    logger: LoggingProtocol,
) -> None:
    """Score each clip with end-of-turn probability, mutating clips in place."""
    ordered = sorted(clips, key=lambda c: (c.start_time, c.end_time))

    prev: SpeechClip | None = None
    current_turn_start_time: float | None = None

    for i, clip in enumerate(ordered):
        if should_reset_turn(prev, clip, silence_reset_seconds, reset_on_speaker_change):
            current_turn_start_time = clip.start_time

        assert current_turn_start_time is not None

        turn_start_idx = seconds_to_sample_index(current_turn_start_time)
        clip_end_idx = seconds_to_sample_index(clip.end_time)

        window = audio_f32_16k[turn_start_idx:clip_end_idx]
        model_input = pad_left_to_8s(window)
        result = predictor.predict_endpoint(model_input)

        clip.end_of_turn_probability = float(result["probability"])

        if (i + 1) % 50 == 0 or i == len(ordered) - 1:
            logger.report_message(f"[blue]Scored clip {i + 1}/{len(ordered)}[/blue]")

        prev = clip
