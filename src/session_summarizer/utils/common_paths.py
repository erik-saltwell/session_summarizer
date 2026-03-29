from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
_DATA_DIR: Path = _PROJECT_ROOT / "data"
_FRAGMENTS_DIR: Path = _PROJECT_ROOT / "fragments"
_VOICE_SAMPLES: Path = _PROJECT_ROOT / "voice_samples"
_REGISTERED_SPEAKERS_FILE = "registered_speakers.yaml"
_LOGS_DIR = _PROJECT_ROOT / "logs"
_REPORTS_DIR = _PROJECT_ROOT / "test_reports"
_TEST_DATA = _PROJECT_ROOT / "test_meeting"
_TEST_TRANSCRIPT_FILENAME = "SimpleTranscript.json"
_TEST_ORIGINAL_FILEPATH = "original.m4a"


def ensure_directory(dir: Path) -> None:
    dir.mkdir(parents=True, exist_ok=True)


def data_dir() -> Path:
    """Return the path to the computed datasets directory under outputs."""
    return _DATA_DIR


def logs_dir() -> Path:
    return _LOGS_DIR


def timestamp_filename(ext: str = ".txt") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H_%M_%S") + ext
    return timestamp


def generate_logfile_path() -> Path:
    timestamp: str = timestamp_filename(".log")
    return logs_dir() / timestamp


def fragments_dir() -> Path:
    """Return the shared fragments directory path."""
    return _FRAGMENTS_DIR


def reports_dir() -> Path:
    return _REPORTS_DIR


def generate_reportfile_path() -> Path:
    timestamp: str = timestamp_filename(".txt")
    return logs_dir() / timestamp


def session_dir(session_id: str) -> Path:
    return data_dir() / session_id


def voice_samples_dir() -> Path:
    return _VOICE_SAMPLES


def speakers_file(session_id: str) -> Path:
    file_path: Path = session_dir(session_id) / _REGISTERED_SPEAKERS_FILE
    if not file_path.exists():
        file_path = voice_samples_dir() / _REGISTERED_SPEAKERS_FILE
    return file_path


def build_speakers_file_path() -> Path:
    return voice_samples_dir() / _REGISTERED_SPEAKERS_FILE


def voice_sample_wav_file(speaker_name: str) -> Path:
    return voice_samples_dir() / (speaker_name + ".wav")


def ensure_session(session_id: str) -> None:
    path: Path = session_dir(session_id)
    ensure_directory(path)


def delete_session(session_id: str) -> None:
    path = session_dir(session_id)
    shutil.rmtree(path, ignore_errors=True)


def test_data_dir() -> Path:
    return _TEST_DATA


def test_transcript_path() -> Path:
    return test_data_dir() / _TEST_TRANSCRIPT_FILENAME


def test_recording_path() -> Path:
    return test_data_dir() / _TEST_ORIGINAL_FILEPATH
