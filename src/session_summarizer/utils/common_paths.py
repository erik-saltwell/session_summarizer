from __future__ import annotations

from enum import StrEnum
from pathlib import Path

_DATA_DIR: Path = Path("data")
_FRAGMENTS_DIR: Path = Path("fragments")
_VOICE_SAMPLES: Path = Path("voice_samples")
_REGISTERED_SPEAKERS_FILE = "registered_speakers.yaml"


class audio_processing_step(StrEnum):
    original = "original.m4a"
    wav_48k = "wav_48k.wav"
    cleaned_audio = "cleaned_audio.wav"
    normalized_audio = "normalized_audio.wav"
    transcript = "transcript.json"
    word_alignments = "word_alignments.json"
    diarization = "diarization.json"


def ensure_directory(dir: Path) -> None:
    dir.mkdir(parents=True, exist_ok=True)


def data_path() -> Path:
    """Return the path to the computed datasets directory under outputs."""
    return _DATA_DIR


def fragments_path() -> Path:
    """Return the shared fragments directory path."""
    return _FRAGMENTS_DIR


def session_path(session_id: str) -> Path:
    return data_path() / session_id


def audio_file_from_step(step: audio_processing_step, session_id: str) -> Path:
    return session_path(session_id) / step


def voice_samples_path() -> Path:
    return data_path() / _VOICE_SAMPLES


def speakers_file(session_id: str) -> Path:
    file_path: Path = session_path(session_id) / _REGISTERED_SPEAKERS_FILE
    if not file_path.exists():
        file_path = voice_samples_path() / _REGISTERED_SPEAKERS_FILE
    return file_path


def build_speakers_file_path(session_id: str | None) -> Path:
    if session_id is None:
        return voice_samples_path() / _REGISTERED_SPEAKERS_FILE
    else:
        return session_path(session_id) / _REGISTERED_SPEAKERS_FILE


def voice_sample_wav_file(speaker_name: str) -> Path:
    return voice_samples_path() / (speaker_name + ".wav")
