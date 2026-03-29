from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from session_summarizer.protocols.session_settings import (
    SUPPORTED_AUDIO_SUFFIXES,
    SessionSettings,
)


@pytest.fixture()
def valid_audio_file(tmp_path: Path) -> Path:
    p = tmp_path / "recording.m4a"
    p.touch()
    return p


class TestAudioFileValidation:
    def test_unsupported_suffix_raises(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "recording.xyz"
        bad_file.touch()
        with pytest.raises(ValidationError, match="Unsupported audio format"):
            SessionSettings(attendees=["Alice"], audio_file=bad_file)

    def test_missing_absolute_file_raises(self, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist.m4a"
        with pytest.raises(ValidationError, match="Audio file does not exist"):
            SessionSettings(attendees=["Alice"], audio_file=missing)

    def test_valid_absolute_file_succeeds(self, valid_audio_file: Path) -> None:
        settings = SessionSettings(attendees=["Alice"], audio_file=valid_audio_file)
        assert settings.audio_file == valid_audio_file

    def test_relative_path_skips_existence_check(self) -> None:
        settings = SessionSettings(attendees=["Alice"], audio_file=Path("some_file.wav"))
        assert settings.audio_file == Path("some_file.wav")

    @pytest.mark.parametrize("suffix", sorted(SUPPORTED_AUDIO_SUFFIXES))
    def test_all_supported_suffixes(self, tmp_path: Path, suffix: str) -> None:
        audio = tmp_path / f"recording{suffix}"
        audio.touch()
        settings = SessionSettings(attendees=["Alice"], audio_file=audio)
        assert settings.audio_file == audio

    def test_suffix_check_is_case_insensitive(self, tmp_path: Path) -> None:
        audio = tmp_path / "recording.M4A"
        audio.touch()
        settings = SessionSettings(attendees=["Alice"], audio_file=audio)
        assert settings.audio_file == audio


class TestCleanedAudioAndTranscriptDefaults:
    def test_defaults_are_relative_paths(self, valid_audio_file: Path) -> None:
        settings = SessionSettings(attendees=["Alice"], audio_file=valid_audio_file)
        assert settings.cleaned_audio_file == Path("cleaned_audio.wav")
        assert settings.transcript_file == Path("transcript.json")

    def test_custom_cleaned_audio_file(self, valid_audio_file: Path) -> None:
        settings = SessionSettings(
            attendees=["Alice"],
            audio_file=valid_audio_file,
            cleaned_audio_file=Path("my_clean.wav"),
        )
        assert settings.cleaned_audio_file == Path("my_clean.wav")

    def test_custom_transcript_file(self, valid_audio_file: Path) -> None:
        settings = SessionSettings(
            attendees=["Alice"],
            audio_file=valid_audio_file,
            transcript_file=Path("my_transcript.json"),
        )
        assert settings.transcript_file == Path("my_transcript.json")


class TestDeviceValidation:
    def test_defaults_to_cuda(self, valid_audio_file: Path) -> None:
        settings = SessionSettings(attendees=["Alice"], audio_file=valid_audio_file)
        assert settings.device == "cuda"

    def test_accepts_cpu(self, valid_audio_file: Path) -> None:
        settings = SessionSettings(attendees=["Alice"], audio_file=valid_audio_file, device="cpu")
        assert settings.device == "cpu"

    def test_accepts_cuda(self, valid_audio_file: Path) -> None:
        settings = SessionSettings(attendees=["Alice"], audio_file=valid_audio_file, device="cuda")
        assert settings.device == "cuda"

    def test_rejects_invalid_device(self, valid_audio_file: Path) -> None:
        with pytest.raises(ValidationError, match="Input should be 'cpu' or 'cuda'"):
            SessionSettings(attendees=["Alice"], audio_file=valid_audio_file, device="tpu")  # type: ignore[arg-type]


class TestLoad:
    def test_load_resolves_relative_audio_file(self, tmp_path: Path) -> None:
        audio = tmp_path / "my_audio.flac"
        audio.touch()
        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text(yaml.dump({"attendees": ["Alice", "Bob"], "audio_file": "my_audio.flac"}))

        settings = SessionSettings.load(settings_file)
        assert settings.audio_file == audio.resolve()
        assert settings.audio_file.is_absolute()

    def test_load_preserves_absolute_audio_file(self, tmp_path: Path) -> None:
        audio = tmp_path / "abs_audio.wav"
        audio.touch()
        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text(yaml.dump({"attendees": ["Alice"], "audio_file": str(audio)}))

        settings = SessionSettings.load(settings_file)
        assert settings.audio_file == audio

    def test_load_resolves_cleaned_audio_and_transcript(self, tmp_path: Path) -> None:
        audio = tmp_path / "recording.m4a"
        audio.touch()
        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text(
            yaml.dump(
                {
                    "attendees": ["Alice"],
                    "audio_file": "recording.m4a",
                    "cleaned_audio_file": "clean.wav",
                    "transcript_file": "out.json",
                }
            )
        )

        settings = SessionSettings.load(settings_file)
        assert settings.cleaned_audio_file == (tmp_path / "clean.wav").resolve()
        assert settings.transcript_file == (tmp_path / "out.json").resolve()

    def test_load_uses_defaults_when_not_specified(self, tmp_path: Path) -> None:
        audio = tmp_path / "recording.m4a"
        audio.touch()
        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text(yaml.dump({"attendees": ["Alice"], "audio_file": "recording.m4a"}))

        settings = SessionSettings.load(settings_file)
        assert settings.cleaned_audio_file == Path("cleaned_audio.wav")
        assert settings.transcript_file == Path("transcript.json")

    def test_load_missing_audio_file_key_raises(self, tmp_path: Path) -> None:
        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text(yaml.dump({"attendees": ["Alice"]}))
        with pytest.raises(ValidationError, match="audio_file"):
            SessionSettings.load(settings_file)


class TestLoadCascading:
    def test_session_override_audio_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        session_id = "test_session"
        session_dir = tmp_path / session_id
        session_dir.mkdir()

        audio = session_dir / "override.mp3"
        audio.touch()

        base_yaml = tmp_path / "settings.yaml"
        base_yaml.write_text(yaml.dump({"attendees": ["Alice"], "audio_file": "base.mp3"}))

        session_yaml = session_dir / "settings.yaml"
        session_yaml.write_text(yaml.dump({"audio_file": "override.mp3"}))

        monkeypatch.setattr("session_summarizer.utils.common_paths.data_dir", lambda: tmp_path)
        monkeypatch.setattr("session_summarizer.utils.common_paths.session_dir", lambda sid: tmp_path / sid)

        settings = SessionSettings.load_cascading(session_id)
        assert settings.audio_file == audio.resolve()
