from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from session_summarizer.settings.session_settings import (
    SUPPORTED_AUDIO_SUFFIXES,
    SessionSettings,
)


@pytest.fixture()
def valid_audio_file(tmp_path: Path) -> Path:
    p = tmp_path / "recording.m4a"
    p.touch()
    return p


_VAD_FIELDS: dict = {
    "model_name": "vad_multilingual_frame_marblenet",
    "onset": 0.7,
    "offset": 0.4,
    "min_duration_on": 0.3,
    "min_duration_off": 0.3,
    "pad_onset": 0.1,
    "pad_offset": 0.1,
}

_DIARIZATION_STITCHING_FIELDS: dict = {
    "min_overlap_fraction_word": 0.20,
    "min_overlap_seconds": 0.02,
    "fill_nearest": True,
    "max_nearest_distance": 0.25,
    "anonymous_join_gap": 0.15,
    "merge_gap_seconds": 0.20,
    "unfinished_clip_merge_max_length": 1.0,
    "identity_stitching_max_gap": 10.0,
    "identity_similarity_threshold": 0.65,
    "expand_segments_to_fit_words": False,
    "expansion_limit_seconds": 300,
    "scoring_mode": "overlap_seconds_then_midpoint",
    "prefer_shorter_on_tie": True,
    "max_backchannel_duration": 2.0,
    "max_backchannel_prior_gap": 1.0,
    "max_backchannel_next_gap": 1.0,
    "turn_end_probability_threshold": 0.5,
    "epsilon": 1e-6,
}


def _required_fields(audio_file: Path | str) -> dict:
    """Return a complete set of required fields for constructing a SessionSettings."""
    return {
        "attendees": ["Alice"],
        "audio_file": audio_file,
        "cleaned_audio_file": Path("cleaned_audio.wav"),
        "transcript_file": Path("transcript.json"),
        "aligned_transcript_path": Path("aligned_transcript.json"),
        "confidence_transcript_path": Path("confidence_transcript.json"),
        "base_diarized_path": Path("base_diarization.json"),
        "speech_clips_with_embedding": Path("speech_clips_with_embedding.json"),
        "identified_speaker_path": Path("identified_speakers.json"),
        "turn_end_updated_path": Path("turn_end_updated.json"),
        "first_stitched_path": Path("first_stitched.json"),
        "identity_stitched_path": Path("identity_stitched.json"),
        "high_confidence_similarity_threshold": 0.8,
        "device": "cuda",
        "segments_path": Path("vad_segments.json"),
        "min_segment_length_short": 10.0,
        "max_segment_length_short": 60.0,
        "min_segment_length_long": 30.0,
        "max_segment_length_long": 300.0,
        "vad": _VAD_FIELDS,
        "diarization_stitching": _DIARIZATION_STITCHING_FIELDS,
    }


def _required_yaml_fields(audio_file: str) -> dict:
    """Return a complete set of required fields for writing to a YAML file."""
    return {
        "attendees": ["Alice"],
        "audio_file": audio_file,
        "cleaned_audio_file": "cleaned_audio.wav",
        "transcript_file": "transcript.json",
        "aligned_transcript_path": "aligned_transcript.json",
        "confidence_transcript_path": "confidence_transcript.json",
        "base_diarized_path": "base_diarization.json",
        "speech_clips_with_embedding": "speech_clips_with_embedding.json",
        "identified_speaker_path": "identified_speakers.json",
        "turn_end_updated_path": "turn_end_updated.json",
        "first_stitched_path": "first_stitched.json",
        "identity_stitched_path": "identity_stitched.json",
        "high_confidence_similarity_threshold": 0.8,
        "device": "cuda",
        "segments_path": "vad_segments.json",
        "min_segment_length_short": 10.0,
        "max_segment_length_short": 60.0,
        "min_segment_length_long": 30.0,
        "max_segment_length_long": 300.0,
        "vad": _VAD_FIELDS,
        "diarization_stitching": _DIARIZATION_STITCHING_FIELDS,
    }


class TestAllFieldsRequired:
    def test_missing_cleaned_audio_file_raises(self, valid_audio_file: Path) -> None:
        fields = _required_fields(valid_audio_file)
        del fields["cleaned_audio_file"]
        with pytest.raises(ValidationError, match="cleaned_audio_file"):
            SessionSettings(**fields)

    def test_missing_transcript_file_raises(self, valid_audio_file: Path) -> None:
        fields = _required_fields(valid_audio_file)
        del fields["transcript_file"]
        with pytest.raises(ValidationError, match="transcript_file"):
            SessionSettings(**fields)

    def test_missing_aligned_transcript_path_raises(self, valid_audio_file: Path) -> None:
        fields = _required_fields(valid_audio_file)
        del fields["aligned_transcript_path"]
        with pytest.raises(ValidationError, match="aligned_transcript_path"):
            SessionSettings(**fields)

    def test_missing_confidence_transcript_path_raises(self, valid_audio_file: Path) -> None:
        fields = _required_fields(valid_audio_file)
        del fields["confidence_transcript_path"]
        with pytest.raises(ValidationError, match="confidence_transcript_path"):
            SessionSettings(**fields)

    def test_missing_device_raises(self, valid_audio_file: Path) -> None:
        fields = _required_fields(valid_audio_file)
        del fields["device"]
        with pytest.raises(ValidationError, match="device"):
            SessionSettings(**fields)


class TestAudioFileValidation:
    def test_unsupported_suffix_raises(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "recording.xyz"
        bad_file.touch()
        with pytest.raises(ValidationError, match="Unsupported audio format"):
            SessionSettings(**{**_required_fields(bad_file)})

    def test_missing_absolute_file_raises(self, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist.m4a"
        with pytest.raises(ValidationError, match="Audio file does not exist"):
            SessionSettings(**{**_required_fields(missing)})

    def test_valid_absolute_file_succeeds(self, valid_audio_file: Path) -> None:
        settings = SessionSettings(**_required_fields(valid_audio_file))
        assert settings.audio_file == valid_audio_file

    def test_relative_path_skips_existence_check(self) -> None:
        settings = SessionSettings(**_required_fields(Path("some_file.wav")))
        assert settings.audio_file == Path("some_file.wav")

    @pytest.mark.parametrize("suffix", sorted(SUPPORTED_AUDIO_SUFFIXES))
    def test_all_supported_suffixes(self, tmp_path: Path, suffix: str) -> None:
        audio = tmp_path / f"recording{suffix}"
        audio.touch()
        settings = SessionSettings(**_required_fields(audio))
        assert settings.audio_file == audio

    def test_suffix_check_is_case_insensitive(self, tmp_path: Path) -> None:
        audio = tmp_path / "recording.M4A"
        audio.touch()
        settings = SessionSettings(**_required_fields(audio))
        assert settings.audio_file == audio


class TestDeviceValidation:
    def test_accepts_cpu(self, valid_audio_file: Path) -> None:
        settings = SessionSettings(**{**_required_fields(valid_audio_file), "device": "cpu"})
        assert settings.device == "cpu"

    def test_accepts_cuda(self, valid_audio_file: Path) -> None:
        settings = SessionSettings(**{**_required_fields(valid_audio_file), "device": "cuda"})
        assert settings.device == "cuda"

    def test_rejects_invalid_device(self, valid_audio_file: Path) -> None:
        with pytest.raises(ValidationError, match="Input should be 'cpu' or 'cuda'"):
            SessionSettings(**{**_required_fields(valid_audio_file), "device": "tpu"})


class TestLoad:
    def test_load_resolves_relative_audio_file(self, tmp_path: Path) -> None:
        audio = tmp_path / "my_audio.flac"
        audio.touch()
        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text(yaml.dump({**_required_yaml_fields("my_audio.flac"), "attendees": ["Alice", "Bob"]}))

        settings = SessionSettings.load(settings_file)
        assert settings.audio_file == audio.resolve()
        assert settings.audio_file.is_absolute()

    def test_load_preserves_absolute_audio_file(self, tmp_path: Path) -> None:
        audio = tmp_path / "abs_audio.wav"
        audio.touch()
        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text(yaml.dump(_required_yaml_fields(str(audio))))

        settings = SessionSettings.load(settings_file)
        assert settings.audio_file == audio

    def test_load_resolves_all_paths(self, tmp_path: Path) -> None:
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
                    "aligned_transcript_path": "aligned.json",
                    "confidence_transcript_path": "confidence.json",
                    "base_diarized_path": "base_diarization.json",
                    "speech_clips_with_embedding": "speech_clips_with_embedding.json",
                    "identified_speaker_path": "identified_speakers.json",
                    "turn_end_updated_path": "turn_end_updated.json",
                    "first_stitched_path": "first_stitched.json",
                    "identity_stitched_path": "identity_stitched.json",
                    "high_confidence_similarity_threshold": 0.8,
                    "device": "cuda",
                    "segments_path": "vad_segments.json",
                    "min_segment_length_short": 10.0,
                    "max_segment_length_short": 60.0,
                    "min_segment_length_long": 30.0,
                    "max_segment_length_long": 300.0,
                    "vad": _VAD_FIELDS,
                    "diarization_stitching": _DIARIZATION_STITCHING_FIELDS,
                }
            )
        )

        settings = SessionSettings.load(settings_file)
        assert settings.cleaned_audio_file == (tmp_path / "clean.wav").resolve()
        assert settings.transcript_file == (tmp_path / "out.json").resolve()
        assert settings.aligned_transcript_path == (tmp_path / "aligned.json").resolve()
        assert settings.confidence_transcript_path == (tmp_path / "confidence.json").resolve()
        assert settings.first_stitched_path == (tmp_path / "first_stitched.json").resolve()

    def test_load_missing_audio_file_key_raises(self, tmp_path: Path) -> None:
        settings_file = tmp_path / "settings.yaml"
        fields = _required_yaml_fields("recording.m4a")
        del fields["audio_file"]
        settings_file.write_text(yaml.dump(fields))
        with pytest.raises(ValidationError, match="audio_file"):
            SessionSettings.load(settings_file)

    def test_load_missing_device_raises(self, tmp_path: Path) -> None:
        audio = tmp_path / "recording.m4a"
        audio.touch()
        settings_file = tmp_path / "settings.yaml"
        fields = _required_yaml_fields("recording.m4a")
        del fields["device"]
        settings_file.write_text(yaml.dump(fields))
        with pytest.raises(ValidationError, match="device"):
            SessionSettings.load(settings_file)


class TestLoadCascading:
    def test_session_override_audio_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        session_id = "test_session"
        session_dir = tmp_path / session_id
        session_dir.mkdir()

        audio = session_dir / "override.mp3"
        audio.touch()

        base_yaml = tmp_path / "settings.yaml"
        base_yaml.write_text(yaml.dump(_required_yaml_fields("base.mp3")))

        session_yaml = session_dir / "settings.yaml"
        session_yaml.write_text(yaml.dump({"audio_file": "override.mp3"}))

        monkeypatch.setattr("session_summarizer.utils.common_paths.data_dir", lambda: tmp_path)
        monkeypatch.setattr("session_summarizer.utils.common_paths.session_dir", lambda sid: tmp_path / sid)

        settings = SessionSettings.load_cascading(session_id)
        assert settings.audio_file == audio.resolve()
