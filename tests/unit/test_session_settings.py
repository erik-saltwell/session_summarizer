from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from session_summarizer.settings.session_settings import SessionSettings


def _required_fields() -> dict:
    """Return a complete dict of valid SessionSettings fields using Python types."""
    return {
        "attendees": ["TestPlayer"],
        "paths": {
            "source_audio": Path("session.m4a"),
            "cleaned_audio": Path("cleaned_audio.wav"),
            "transcript": Path("transcript.json"),
            "aligned_transcript": Path("aligned_transcript.json"),
            "confidence_transcript": Path("confidence_transcript.json"),
            "vad_segments": Path("segments.json"),
            "base_diarization": Path("base_diarization.json"),
            "clips_with_embeddings": Path("clips_with_embeddings.json"),
            "identified_speakers": Path("identified_speakers.json"),
            "identity_stitched": Path("identity_stitched.json"),
            "backchannel_marked": Path("backchannel_marked.json"),
            "diarizationlm_processed": Path("diarizationlm_processed.json"),
            "indeterminate_speakers": Path("indeterminate_speakers.json"),
            "dangling_sentence_fix": Path("dangling_sentence_fix.json"),
            "punctuated_text": Path("punctuated_text.json"),
        },
        "segmentation": {
            "short_min_seconds": 10.0,
            "short_max_seconds": 38.0,
            "long_min_seconds": 120.0,
            "long_max_seconds": 300.0,
        },
        "speaker_clips": {
            "lead_in_seconds": 0.25,
            "lead_out_seconds": 0.25,
            "min_similarity_residual": 0.2,
            "min_duration_seconds": 2.0,
            "min_centroid_similarity": 0.75,
        },
        "speaker_identification": {
            "assignment_threshold": 0.70,
        },
        "device": "cpu",
        "vad": {
            "model_name": "vad_multilingual_frame_marblenet",
            "onset": 0.7,
            "offset": 0.4,
            "min_duration_on": 0.3,
            "min_duration_off": 0.3,
            "pad_onset": 0.1,
            "pad_offset": 0.1,
        },
        "stitching": {
            "min_overlap_fraction_word": 0.20,
            "min_overlap_seconds": 0.02,
            "fill_nearest": True,
            "max_nearest_gap_seconds": 0.25,
            "anonymous_join_gap_seconds": 0.15,
            "merge_gap_seconds": 0.20,
            "identity_merge_max_gap_seconds": 10.0,
            "expand_segments_to_fit_words": False,
            "expansion_limit_seconds": 300.0,
            "scoring_mode": "overlap_seconds_then_midpoint",
            "prefer_shorter_on_tie": True,
        },
        "epsilon": 0.000001,
        "seed": 42,
    }


def _required_yaml_fields() -> dict:
    """Return a complete dict of valid SessionSettings fields using plain strings (YAML-style)."""
    return {
        "attendees": ["TestPlayer"],
        "paths": {
            "source_audio": "session.m4a",
            "cleaned_audio": "cleaned_audio.wav",
            "transcript": "transcript.json",
            "aligned_transcript": "aligned_transcript.json",
            "confidence_transcript": "confidence_transcript.json",
            "vad_segments": "segments.json",
            "base_diarization": "base_diarization.json",
            "clips_with_embeddings": "clips_with_embeddings.json",
            "identified_speakers": "identified_speakers.json",
            "identity_stitched": "identity_stitched.json",
            "backchannel_marked": "backchannel_marked.json",
            "diarizationlm_processed": "diarizationlm_processed.json",
            "indeterminate_speakers": "indeterminate_speakers.json",
            "dangling_sentence_fix": "dangling_sentence_fix.json",
            "punctuated_text": "punctuated_text.json",
        },
        "segmentation": {
            "short_min_seconds": 10.0,
            "short_max_seconds": 38.0,
            "long_min_seconds": 120.0,
            "long_max_seconds": 300.0,
        },
        "speaker_clips": {
            "lead_in_seconds": 0.25,
            "lead_out_seconds": 0.25,
            "min_similarity_residual": 0.2,
            "min_duration_seconds": 2.0,
            "min_centroid_similarity": 0.75,
        },
        "speaker_identification": {
            "assignment_threshold": 0.70,
        },
        "device": "cpu",
        "vad": {
            "model_name": "vad_multilingual_frame_marblenet",
            "onset": 0.7,
            "offset": 0.4,
            "min_duration_on": 0.3,
            "min_duration_off": 0.3,
            "pad_onset": 0.1,
            "pad_offset": 0.1,
        },
        "stitching": {
            "min_overlap_fraction_word": 0.20,
            "min_overlap_seconds": 0.02,
            "fill_nearest": True,
            "max_nearest_gap_seconds": 0.25,
            "anonymous_join_gap_seconds": 0.15,
            "merge_gap_seconds": 0.20,
            "identity_merge_max_gap_seconds": 10.0,
            "expand_segments_to_fit_words": False,
            "expansion_limit_seconds": 300.0,
            "scoring_mode": "overlap_seconds_then_midpoint",
            "prefer_shorter_on_tie": True,
        },
        "epsilon": 0.000001,
        "seed": 42,
    }


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_valid_settings_loads() -> None:
    s = SessionSettings(**_required_fields())
    assert s.attendees == ["TestPlayer"]
    assert s.number_of_speakers == 1
    assert s.speaker_clips.min_similarity_residual == pytest.approx(0.2)
    assert s.speaker_clips.min_duration_seconds == pytest.approx(2.0)
    assert s.speaker_clips.min_centroid_similarity == pytest.approx(0.75)


def test_attendees_accepts_valid_list() -> None:
    fields = _required_fields()
    fields["attendees"] = ["Alice", "Bob"]
    s = SessionSettings(**fields)
    assert s.number_of_speakers == 2
    assert s.attendees[0] == "Alice"


def test_load_resolves_all_paths(tmp_path: Path) -> None:
    import yaml

    fields = _required_yaml_fields()
    yaml_text = yaml.dump(fields)
    settings_file = tmp_path / "settings.yaml"
    settings_file.write_text(yaml_text)

    s = SessionSettings.load(settings_file)
    assert s.paths.source_audio.is_absolute()
    assert s.paths.cleaned_audio.is_absolute()
    assert s.paths.transcript.is_absolute()
    assert s.paths.aligned_transcript.is_absolute()
    assert s.paths.confidence_transcript.is_absolute()
    assert s.paths.backchannel_marked.is_absolute()
    assert s.paths.vad_segments.is_absolute()


# ---------------------------------------------------------------------------
# attendees validation
# ---------------------------------------------------------------------------


def test_attendees_rejects_empty_dict() -> None:
    fields = _required_fields()
    fields["attendees"] = {}
    with pytest.raises(ValidationError):
        SessionSettings(**fields)


def test_attendees_rejects_blank_name() -> None:
    fields = _required_fields()
    fields["attendees"] = ["  "]
    with pytest.raises(Exception, match="name is blank"):
        SessionSettings(**fields)


# ---------------------------------------------------------------------------
# Speaker clip settings validation
# ---------------------------------------------------------------------------


def test_min_similarity_residual_rejects_above_one() -> None:
    fields = _required_fields()
    fields["speaker_clips"]["min_similarity_residual"] = 1.1
    with pytest.raises(Exception, match="min_similarity_residual"):
        SessionSettings(**fields)


def test_min_similarity_residual_rejects_negative() -> None:
    fields = _required_fields()
    fields["speaker_clips"]["min_similarity_residual"] = -0.1
    with pytest.raises(Exception, match="min_similarity_residual"):
        SessionSettings(**fields)


def test_min_duration_seconds_rejects_negative() -> None:
    fields = _required_fields()
    fields["speaker_clips"]["min_duration_seconds"] = -1.0
    with pytest.raises(Exception, match="min_duration_seconds"):
        SessionSettings(**fields)


def test_min_centroid_similarity_rejects_negative() -> None:
    fields = _required_fields()
    fields["speaker_clips"]["min_centroid_similarity"] = -0.1
    with pytest.raises(Exception, match="min_centroid_similarity"):
        SessionSettings(**fields)


def test_min_centroid_similarity_rejects_above_one() -> None:
    fields = _required_fields()
    fields["speaker_clips"]["min_centroid_similarity"] = 1.1
    with pytest.raises(Exception, match="min_centroid_similarity"):
        SessionSettings(**fields)


def test_min_centroid_similarity_accepts_zero() -> None:
    fields = _required_fields()
    fields["speaker_clips"]["min_centroid_similarity"] = 0.0
    s = SessionSettings(**fields)
    assert s.speaker_clips.min_centroid_similarity == 0.0


# ---------------------------------------------------------------------------
# Segmentation cross-field validation
# ---------------------------------------------------------------------------


def test_segment_length_short_min_must_be_less_than_max() -> None:
    fields = _required_fields()
    fields["segmentation"]["short_min_seconds"] = 40.0
    fields["segmentation"]["short_max_seconds"] = 38.0
    with pytest.raises(Exception, match="short_min_seconds"):
        SessionSettings(**fields)


def test_segment_length_long_min_must_be_less_than_max() -> None:
    fields = _required_fields()
    fields["segmentation"]["long_min_seconds"] = 400.0
    fields["segmentation"]["long_max_seconds"] = 300.0
    with pytest.raises(Exception, match="long_min_seconds"):
        SessionSettings(**fields)
