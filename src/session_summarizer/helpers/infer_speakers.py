from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from ..completions import PromptData, get_completion
from ..completions.model_settings import ModelEffort, ModelString
from ..processing_results import SpeechClipSet
from ..protocols import LoggingProtocol
from ..settings.session_settings import SessionSettings
from ..utils import FragmentID, Tracer, get_fragment

UNKNOWN_SPEAKER_IDENTITY = "unknown speaker"

CompletionFn = Callable[[PromptData, ModelString, ModelEffort, Tracer | None], str]

_FINAL_RESULTS_RE = re.compile(r"<final_results>\s*(?P<json>.*?)\s*</final_results>", re.DOTALL)


@dataclass(frozen=True)
class InferredParticipant:
    input_speaker_labels: list[str]
    real_name: str
    role: str
    confidence: str


@dataclass(frozen=True)
class InferSpeakersResult:
    clips: SpeechClipSet
    participants: list[InferredParticipant]


def _construct_input(clips: SpeechClipSet) -> str:
    return clips.to_markdown()


def parse_inferred_participants(output: str) -> list[InferredParticipant]:
    match = _FINAL_RESULTS_RE.search(output)
    if match is None:
        raise ValueError("Completion output did not include a <final_results> JSON block.")

    try:
        raw_results: Any = json.loads(match.group("json"))
    except json.JSONDecodeError as exc:
        raise ValueError("Completion <final_results> block contained malformed JSON.") from exc

    if not isinstance(raw_results, list):
        raise ValueError("Completion <final_results> JSON must be an array.")

    participants: list[InferredParticipant] = []
    for index, item in enumerate(raw_results):
        if not isinstance(item, dict):
            raise ValueError(f"Completion result at index {index} must be an object.")

        raw_labels = item.get("input_speaker_labels")
        real_name = item.get("real_name")
        role = item.get("role")
        confidence = item.get("confidence")

        if not isinstance(raw_labels, list) or not all(isinstance(label, str) for label in raw_labels):
            raise ValueError(f"Completion result at index {index} has invalid input_speaker_labels.")
        if not isinstance(real_name, str) or not real_name.strip():
            raise ValueError(f"Completion result at index {index} has invalid real_name.")
        if not isinstance(role, str) or not role.strip():
            raise ValueError(f"Completion result at index {index} has invalid role.")
        if confidence not in {"high", "medium", "low"}:
            raise ValueError(f"Completion result at index {index} has invalid confidence.")

        participants.append(
            InferredParticipant(
                input_speaker_labels=raw_labels,
                real_name=real_name,
                role=role,
                confidence=confidence,
            )
        )

    return participants


def build_role_map(participants: list[InferredParticipant], source_labels: set[str]) -> dict[str, str]:
    role_by_label: dict[str, str] = {}
    for participant in participants:
        for label in participant.input_speaker_labels:
            if label not in source_labels:
                raise ValueError(f"Completion result included unknown input speaker label: {label!r}.")
            existing_role = role_by_label.get(label)
            if existing_role is not None and existing_role != participant.role:
                raise ValueError(
                    f"Completion result mapped input speaker label {label!r} to both "
                    f"{existing_role!r} and {participant.role!r}."
                )
            role_by_label[label] = participant.role
    return role_by_label


def build_participant_map(participants: list[InferredParticipant], source_labels: set[str]) -> dict[str, str]:
    participant_by_label: dict[str, str] = {}
    for participant in participants:
        for label in participant.input_speaker_labels:
            if label not in source_labels:
                raise ValueError(f"Completion result included unknown input speaker label: {label!r}.")
            existing_participant = participant_by_label.get(label)
            if existing_participant is not None and existing_participant != participant.real_name:
                raise ValueError(
                    f"Completion result mapped input speaker label {label!r} to both "
                    f"{existing_participant!r} and {participant.real_name!r}."
                )
            participant_by_label[label] = participant.real_name
    return participant_by_label


def build_participant_role_map(participants: list[InferredParticipant]) -> dict[str, str]:
    role_by_participant: dict[str, str] = {}
    for participant in participants:
        real_name = participant.real_name.strip()
        role = participant.role.strip()
        existing_role = role_by_participant.get(real_name)
        if existing_role is not None and existing_role != role:
            raise ValueError(
                f"Completion result mapped participant {real_name!r} to both {existing_role!r} and {role!r}."
            )
        role_by_participant[real_name] = role
    return role_by_participant


def get_inferred_participants_path(session_dir: Path, inferred_speakers_path: Path) -> Path:
    return session_dir / f"{inferred_speakers_path.stem}_participants.json"


def save_inferred_participants(path: Path, participants: list[InferredParticipant]) -> None:
    path.write_text(json.dumps([asdict(participant) for participant in participants], indent=2), encoding="utf-8")


def load_inferred_participants(path: Path) -> list[InferredParticipant]:
    try:
        raw_participants: Any = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Inferred participants file contained malformed JSON: {path}") from exc

    if not isinstance(raw_participants, list):
        raise ValueError(f"Inferred participants file must contain a JSON array: {path}")

    participants: list[InferredParticipant] = []
    for index, item in enumerate(raw_participants):
        if not isinstance(item, dict):
            raise ValueError(f"Inferred participant at index {index} must be an object.")
        raw_labels = item.get("input_speaker_labels")
        real_name = item.get("real_name")
        role = item.get("role")
        confidence = item.get("confidence")
        if not isinstance(raw_labels, list) or not all(isinstance(label, str) for label in raw_labels):
            raise ValueError(f"Inferred participant at index {index} has invalid input_speaker_labels.")
        if not isinstance(real_name, str) or not real_name.strip():
            raise ValueError(f"Inferred participant at index {index} has invalid real_name.")
        if not isinstance(role, str) or not role.strip():
            raise ValueError(f"Inferred participant at index {index} has invalid role.")
        if confidence not in {"high", "medium", "low"}:
            raise ValueError(f"Inferred participant at index {index} has invalid confidence.")
        participants.append(
            InferredParticipant(
                input_speaker_labels=raw_labels,
                real_name=real_name,
                role=role,
                confidence=confidence,
            )
        )
    return participants


def apply_inferred_roles(clips: SpeechClipSet, role_by_label: dict[str, str]) -> SpeechClipSet:
    for clip in clips:
        if any(label not in role_by_label for label in clip.speakers):
            clip.identity = UNKNOWN_SPEAKER_IDENTITY
            continue

        mapped_roles: set[str] = {role_by_label[label] for label in clip.speakers}
        if len(mapped_roles) == 1:
            clip.identity = next(iter(mapped_roles))
    return clips


def update_session_inferred_speaker_settings(
    session_settings_path: Path, participants: list[InferredParticipant]
) -> None:
    role_by_participant = build_participant_role_map(participants)
    data: dict[str, Any] = {}

    if session_settings_path.exists():
        with session_settings_path.open("r", encoding="utf-8") as f:
            raw_data: Any = yaml.safe_load(f)
        if raw_data is not None:
            if not isinstance(raw_data, dict):
                raise ValueError(f"Session settings file must contain a YAML mapping: {session_settings_path}")
            data = raw_data

    campaign_info = data.get("campaign_info")
    if campaign_info is None:
        campaign_info = {}
        data["campaign_info"] = campaign_info
    if not isinstance(campaign_info, dict):
        raise ValueError(f"campaign_info must be a YAML mapping in session settings: {session_settings_path}")

    data["attendees"] = list(role_by_participant.keys())
    campaign_info["players"] = role_by_participant

    session_settings_path.parent.mkdir(parents=True, exist_ok=True)
    with session_settings_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def infer_speakers(
    settings: SessionSettings,
    session_dir: Path,
    clips: SpeechClipSet,
    logger: LoggingProtocol,
    tracer: Tracer,
    completion_fn: CompletionFn = get_completion,
) -> InferSpeakersResult:
    system_prompt: str = get_fragment(FragmentID.INFER_PLAYERS_PROMPT)
    prompt = PromptData(system_prompt, _construct_input(clips))
    llm_settings = settings.llm.infer_players

    output = completion_fn(prompt, llm_settings.model, llm_settings.effort, tracer)
    participants = parse_inferred_participants(output)
    source_labels = {label for clip in clips for label in clip.speakers}
    role_by_label = build_role_map(participants, source_labels)

    logger.report_message(f"[blue]Inferred speaker roles for {len(role_by_label)} input label(s).[/blue]")
    return InferSpeakersResult(clips=apply_inferred_roles(clips, role_by_label), participants=participants)
