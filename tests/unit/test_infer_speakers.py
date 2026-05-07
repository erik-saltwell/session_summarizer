from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from session_summarizer.helpers.infer_speakers import (
    UNKNOWN_SPEAKER_IDENTITY,
    InferredParticipant,
    apply_inferred_roles,
    build_participant_role_map,
    build_role_map,
    load_inferred_participants,
    parse_inferred_participants,
    save_inferred_participants,
    update_session_inferred_speaker_settings,
)
from session_summarizer.processing_results import SpeechClip, SpeechClipSet


def _clip(speakers: set[str], identity: str | None = None) -> SpeechClip:
    return SpeechClip(start_time=0.0, end_time=1.0, speakers=speakers, text="hello", identity=identity)


def test_parse_inferred_participants_extracts_final_results_json() -> None:
    output = """
    Reasoning omitted.
    <final_results>
    [
      {
        "input_speaker_labels": ["1", "3"],
        "real_name": "Morgan",
        "role": "Game Master",
        "confidence": "high"
      }
    ]
    </final_results>
    """

    assert parse_inferred_participants(output) == [
        InferredParticipant(
            input_speaker_labels=["1", "3"],
            real_name="Morgan",
            role="Game Master",
            confidence="high",
        )
    ]


def test_parse_inferred_participants_errors_on_malformed_json() -> None:
    with pytest.raises(ValueError, match="malformed JSON"):
        parse_inferred_participants("<final_results>[</final_results>")


def test_build_role_map_errors_on_unknown_output_label() -> None:
    participants = [
        InferredParticipant(input_speaker_labels=["9"], real_name="Speaker 9", role="Character 1", confidence="low")
    ]

    with pytest.raises(ValueError, match="unknown input speaker label"):
        build_role_map(participants, {"1"})


def test_build_participant_role_map_preserves_participant_order() -> None:
    participants = [
        InferredParticipant(input_speaker_labels=["1"], real_name="Morgan", role="Game Master", confidence="high"),
        InferredParticipant(input_speaker_labels=["2"], real_name="Avery", role="Selene", confidence="low"),
    ]

    assert build_participant_role_map(participants) == {"Morgan": "Game Master", "Avery": "Selene"}


def test_build_participant_role_map_errors_on_conflicting_duplicate_participant() -> None:
    participants = [
        InferredParticipant(input_speaker_labels=["1"], real_name="Morgan", role="Game Master", confidence="high"),
        InferredParticipant(input_speaker_labels=["3"], real_name="Morgan", role="Selene", confidence="medium"),
    ]

    with pytest.raises(ValueError, match="mapped participant"):
        build_participant_role_map(participants)


def test_apply_inferred_roles_sets_role_for_single_speaker_clip() -> None:
    clips = SpeechClipSet([_clip({"1"})])

    result = apply_inferred_roles(clips, {"1": "Game Master"})

    assert result[0].identity == "Game Master"


def test_apply_inferred_roles_uses_unknown_for_unmapped_speaker() -> None:
    clips = SpeechClipSet([_clip({"2"})])

    result = apply_inferred_roles(clips, {"1": "Game Master"})

    assert result[0].identity == UNKNOWN_SPEAKER_IDENTITY


def test_apply_inferred_roles_preserves_identity_for_multi_role_clip() -> None:
    clips = SpeechClipSet([_clip({"1", "2"}, identity="prior identity")])

    result = apply_inferred_roles(clips, {"1": "Game Master", "2": "Selene"})

    assert result[0].identity == "prior identity"


def test_clipset_markdown_renderer_is_reused_for_prompt_format() -> None:
    clips = SpeechClipSet([_clip({"1"}, identity="Game Master")])

    assert (
        clips.to_markdown(include_speaker=True, include_timestamps=False, included_utterance_ids=False)
        == "**Game Master** — hello\n\n"
    )


def test_update_session_inferred_speaker_settings_preserves_existing_settings(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(
        """
attendees:
  - Old
session_info:
  session_date: "2026-02-03"
campaign_info:
  players:
    Old: Old Character
  glossary:
    - term: Delta Green
      description: The organization
""".lstrip(),
        encoding="utf-8",
    )
    participants = [
        InferredParticipant(input_speaker_labels=["1"], real_name="Morgan", role="Game Master", confidence="high"),
        InferredParticipant(input_speaker_labels=["2"], real_name="Avery", role="Selene", confidence="low"),
    ]

    update_session_inferred_speaker_settings(settings_path, participants)

    data = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    assert data["attendees"] == ["Morgan", "Avery"]
    assert data["session_info"] == {"session_date": "2026-02-03"}
    assert data["campaign_info"]["players"] == {"Morgan": "Game Master", "Avery": "Selene"}
    assert data["campaign_info"]["glossary"] == [{"term": "Delta Green", "description": "The organization"}]


def test_update_session_inferred_speaker_settings_creates_session_file(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.yaml"
    participants = [
        InferredParticipant(input_speaker_labels=["1"], real_name="Speaker 1", role="Character 1", confidence="low")
    ]

    update_session_inferred_speaker_settings(settings_path, participants)

    assert yaml.safe_load(settings_path.read_text(encoding="utf-8")) == {
        "campaign_info": {"players": {"Speaker 1": "Character 1"}},
        "attendees": ["Speaker 1"],
    }


def test_save_and_load_inferred_participants_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "inferred_speakers_participants.json"
    participants = [
        InferredParticipant(input_speaker_labels=["1", "3"], real_name="Morgan", role="Game Master", confidence="high")
    ]

    save_inferred_participants(path, participants)

    assert load_inferred_participants(path) == participants
