from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from session_summarizer.settings.session_settings import SessionSettings


def _required_fields() -> dict:
    """Return a complete dict of valid SessionSettings fields using Python types."""
    return {
        "attendees": ["TestPlayer"],
        "adventure_settings": {
            "pcs": {"TestPlayer": "TestCharacter"},
            "glossary": [],
        },
        "audio_file": Path("session.m4a"),
        "cleaned_audio_file": Path("cleaned_audio.wav"),
        "transcript_file": Path("transcript.json"),
        "aligned_transcript_path": Path("aligned_transcript.json"),
        "confidence_transcript_path": Path("confidence_transcript.json"),
        "base_diarized_path": Path("base_diarization.json"),
        "speech_clips_with_embedding": Path("clips_with_embeddings.json"),
        "identified_speaker_path": Path("identified_speakers.json"),
        "turn_end_updated_path": Path("turn_end_updated.json"),
        "first_stitched_path": Path("first_stitched.json"),
        "identity_stitched_path": Path("identity_stitched.json"),
        "backchannel_marked_path": Path("backchannel_marked.json"),
        "diarizationlm_processed_path": Path("diarizationlm_processed.json"),
        "indeterminate_speakers_path": Path("indeterminate_speakers.json"),
        "dangling_sentence_fix_path": Path("dangling_sentence_fix.json"),
        "punctuated_text_path": Path("punctuated_text.json"),
        "device": "cpu",
        "segments_path": Path("segments.json"),
        "min_segment_length_short": 10.0,
        "max_segment_length_short": 38.0,
        "min_segment_length_long": 120.0,
        "max_segment_length_long": 300.0,
        "high_confidence_similarity_threshold": 0.88,
        "speaker_identity_assignment_threshold": 0.70,
        "vad": {
            "model_name": "vad_multilingual_frame_marblenet",
            "onset": 0.7,
            "offset": 0.4,
            "min_duration_on": 0.3,
            "min_duration_off": 0.3,
            "pad_onset": 0.1,
            "pad_offset": 0.1,
        },
        "speaker_clip_lead_in": 0.25,
        "speaker_clip_lead_out": 0.25,
        "speaker_clip_minimum_similarity_residual": 0.2,
        "minimum_speaker_clip_duration": 2.0,
        "min_speaker_similarity": 0.75,
        "diarization_stitching": {
            "min_overlap_fraction_word": 0.20,
            "min_overlap_seconds": 0.02,
            "fill_nearest": True,
            "max_nearest_distance": 0.25,
            "anonymous_join_gap": 0.15,
            "merge_gap_seconds": 0.20,
            "unfinished_clip_merge_max_length": 2.0,
            "identity_stitching_max_gap": 10.0,
            "identity_similarity_threshold": 0.65,
            "expand_segments_to_fit_words": False,
            "expansion_limit_seconds": 300.0,
            "scoring_mode": "overlap_seconds_then_midpoint",
            "prefer_shorter_on_tie": True,
            "max_backchannel_duration": 0.75,
            "max_backchannel_prior_gap": 0.25,
            "max_backchannel_next_gap": 1.0,
            "max_identity_backchannel_duration": 3.0,
            "max_identity_backchannel_prior_gap": 0.75,
            "max_identity_backchannel_next_gap": 3.0,
            "turn_end_probability_threshold": 0.5,
            "tiny_clip_threshold": 0.1,
        },
        "epsilon": 0.000001,
        "seed": 42,
    }


def _required_yaml_fields() -> dict:
    """Return a complete dict of valid SessionSettings fields using plain strings (YAML-style)."""
    return {
        "attendees": ["TestPlayer"],
        "adventure_settings": {
            "pcs": {"TestPlayer": "TestCharacter"},
            "glossary": [],
        },
        "audio_file": "session.m4a",
        "cleaned_audio_file": "cleaned_audio.wav",
        "transcript_file": "transcript.json",
        "aligned_transcript_path": "aligned_transcript.json",
        "confidence_transcript_path": "confidence_transcript.json",
        "base_diarized_path": "base_diarization.json",
        "speech_clips_with_embedding": "clips_with_embeddings.json",
        "identified_speaker_path": "identified_speakers.json",
        "turn_end_updated_path": "turn_end_updated.json",
        "first_stitched_path": "first_stitched.json",
        "identity_stitched_path": "identity_stitched.json",
        "backchannel_marked_path": "backchannel_marked.json",
        "diarizationlm_processed_path": "diarizationlm_processed.json",
        "indeterminate_speakers_path": "indeterminate_speakers.json",
        "dangling_sentence_fix_path": "dangling_sentence_fix.json",
        "punctuated_text_path": "punctuated_text.json",
        "device": "cpu",
        "segments_path": "segments.json",
        "min_segment_length_short": 10.0,
        "max_segment_length_short": 38.0,
        "min_segment_length_long": 120.0,
        "max_segment_length_long": 300.0,
        "high_confidence_similarity_threshold": 0.88,
        "speaker_identity_assignment_threshold": 0.70,
        "vad": {
            "model_name": "vad_multilingual_frame_marblenet",
            "onset": 0.7,
            "offset": 0.4,
            "min_duration_on": 0.3,
            "min_duration_off": 0.3,
            "pad_onset": 0.1,
            "pad_offset": 0.1,
        },
        "speaker_clip_lead_in": 0.25,
        "speaker_clip_lead_out": 0.25,
        "speaker_clip_minimum_similarity_residual": 0.2,
        "minimum_speaker_clip_duration": 2.0,
        "min_speaker_similarity": 0.75,
        "diarization_stitching": {
            "min_overlap_fraction_word": 0.20,
            "min_overlap_seconds": 0.02,
            "fill_nearest": True,
            "max_nearest_distance": 0.25,
            "anonymous_join_gap": 0.15,
            "merge_gap_seconds": 0.20,
            "unfinished_clip_merge_max_length": 2.0,
            "identity_stitching_max_gap": 10.0,
            "identity_similarity_threshold": 0.65,
            "expand_segments_to_fit_words": False,
            "expansion_limit_seconds": 300.0,
            "scoring_mode": "overlap_seconds_then_midpoint",
            "prefer_shorter_on_tie": True,
            "max_backchannel_duration": 0.75,
            "max_backchannel_prior_gap": 0.25,
            "max_backchannel_next_gap": 1.0,
            "max_identity_backchannel_duration": 3.0,
            "max_identity_backchannel_prior_gap": 0.75,
            "max_identity_backchannel_next_gap": 3.0,
            "turn_end_probability_threshold": 0.5,
            "tiny_clip_threshold": 0.1,
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
    assert s.speaker_clip_minimum_similarity_residual == pytest.approx(0.2)
    assert s.minimum_speaker_clip_duration == pytest.approx(2.0)
    assert s.min_speaker_similarity == pytest.approx(0.75)


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

    # Create a dummy audio file so the existence check passes
    (tmp_path / "session.m4a").touch()

    s = SessionSettings.load(settings_file)
    assert s.audio_file.is_absolute()
    assert s.cleaned_audio_file.is_absolute()
    assert s.transcript_file.is_absolute()
    assert s.aligned_transcript_path.is_absolute()
    assert s.confidence_transcript_path.is_absolute()
    assert s.backchannel_marked_path.is_absolute()
    assert s.segments_path.is_absolute()


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
# adventure_settings validation
# ---------------------------------------------------------------------------


def test_adventure_settings_pcs_rejects_empty_dict() -> None:
    fields = _required_fields()
    fields["adventure_settings"] = {"pcs": {}, "glossary": []}
    with pytest.raises(ValidationError):
        SessionSettings(**fields)


def test_adventure_settings_glossary_accepts_entries_with_and_without_description() -> None:
    fields = _required_fields()
    fields["adventure_settings"] = {
        "pcs": {"TestPlayer": "TestCharacter"},
        "glossary": [
            {"name": "Thornhaven", "description": "Ruined city"},
            {"name": "Dragonbane"},
        ],
    }
    s = SessionSettings(**fields)
    assert len(s.adventure_settings.glossary) == 2
    assert s.adventure_settings.glossary[0].description == "Ruined city"
    assert s.adventure_settings.glossary[1].description is None


# ---------------------------------------------------------------------------
# New speaker clip field validation
# ---------------------------------------------------------------------------


def test_speaker_clip_minimum_similarity_residual_rejects_above_one() -> None:
    fields = _required_fields()
    fields["speaker_clip_minimum_similarity_residual"] = 1.1
    with pytest.raises(Exception, match="speaker_clip_minimum_similarity_residual"):
        SessionSettings(**fields)


def test_speaker_clip_minimum_similarity_residual_rejects_negative() -> None:
    fields = _required_fields()
    fields["speaker_clip_minimum_similarity_residual"] = -0.1
    with pytest.raises(Exception, match="speaker_clip_minimum_similarity_residual"):
        SessionSettings(**fields)


def test_minimum_speaker_clip_duration_rejects_negative() -> None:
    fields = _required_fields()
    fields["minimum_speaker_clip_duration"] = -1.0
    with pytest.raises(Exception, match="minimum_speaker_clip_duration"):
        SessionSettings(**fields)


def test_min_speaker_similarity_rejects_negative() -> None:
    fields = _required_fields()
    fields["min_speaker_similarity"] = -0.1
    with pytest.raises(Exception, match="min_speaker_similarity"):
        SessionSettings(**fields)


def test_min_speaker_similarity_rejects_above_one() -> None:
    fields = _required_fields()
    fields["min_speaker_similarity"] = 1.1
    with pytest.raises(Exception, match="min_speaker_similarity"):
        SessionSettings(**fields)


def test_min_speaker_similarity_accepts_zero() -> None:
    fields = _required_fields()
    fields["min_speaker_similarity"] = 0.0
    s = SessionSettings(**fields)
    assert s.min_speaker_similarity == 0.0


# ---------------------------------------------------------------------------
# Cross-field validation (pre-existing)
# ---------------------------------------------------------------------------


def test_segment_length_short_min_must_be_less_than_max() -> None:
    fields = _required_fields()
    fields["min_segment_length_short"] = 40.0
    fields["max_segment_length_short"] = 38.0
    with pytest.raises(Exception, match="min_segment_length_short"):
        SessionSettings(**fields)


def test_segment_length_long_min_must_be_less_than_max() -> None:
    fields = _required_fields()
    fields["min_segment_length_long"] = 400.0
    fields["max_segment_length_long"] = 300.0
    with pytest.raises(Exception, match="min_segment_length_long"):
        SessionSettings(**fields)
