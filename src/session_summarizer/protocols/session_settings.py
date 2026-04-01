from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

_SETTINGS_FILE = "settings.yaml"

SUPPORTED_AUDIO_SUFFIXES: frozenset[str] = frozenset(
    {".m4a", ".mp3", ".wav", ".flac", ".ogg", ".opus", ".wma", ".aac", ".webm"}
)


class VadSettings(BaseModel, frozen=True):
    """Hyperparameters for NeMo VAD post-processing."""

    model_name: str = Field(description="Pretrained NeMo VAD model name")
    onset: float = Field(description="Speech onset probability threshold (0.0–1.0)")
    offset: float = Field(description="Speech offset probability threshold (0.0–1.0)")
    min_duration_on: float = Field(description="Minimum speech segment duration in seconds")
    min_duration_off: float = Field(description="Minimum silence segment duration in seconds")
    pad_onset: float = Field(description="Padding added before speech onset in seconds")
    pad_offset: float = Field(description="Padding added after speech offset in seconds")


class SessionSettings(BaseModel, frozen=True):
    attendees: Annotated[
        list[str],
        Field(min_length=1, description="Names of all speakers present in the session"),
    ]
    audio_file: Annotated[
        Path,
        Field(description="Path to the audio file for the session"),
    ]
    cleaned_audio_file: Annotated[
        Path,
        Field(description="Path to the cleaned audio file (created during processing)"),
    ]
    transcript_file: Annotated[
        Path,
        Field(description="Path to the transcript JSON file (created during processing)"),
    ]
    aligned_transcript_path: Annotated[
        Path,
        Field(description="Path to the word-aligned transcript JSON (created during processing)"),
    ]
    confidence_transcript_path: Annotated[
        Path,
        Field(
            description="Path to the transcript JSON annotated with per-word confidence scores"
            " (created during processing)"
        ),
    ]
    base_diarized_path: Annotated[
        Path,
        Field(description="Path to the list of diarized segments generated from audio. (created during processing)"),
    ]
    speech_clips_with_embedding: Annotated[
        Path,
        Field(
            description="Path to SpeechClipSet JSON file with speaker embeddings added (read/written during processing)"
        ),
    ]
    device: Annotated[
        Literal["cpu", "cuda"],
        Field(description="Device for model inference — 'cpu' or 'cuda'"),
    ]

    segments_path: Annotated[
        Path,
        Field(description="Path to the VAD segments JSON output (created during processing)"),
    ]
    min_segment_length_short: Annotated[
        float,
        Field(
            description=(
                "Minimum audio segment length in seconds for short VAD-based chunking (used for Canary transcription)"
            ),
        ),
    ]
    max_segment_length_short: Annotated[
        float,
        Field(
            description=(
                "Maximum audio segment length in seconds for short VAD-based chunking (used for Canary transcription)"
            ),
        ),
    ]
    min_segment_length_long: Annotated[
        float,
        Field(
            description=(
                "Minimum audio segment length in seconds for long VAD-based chunking "
                "(used for operations that are sensitive to OOM, e.g. diarization)"
            ),
        ),
    ]
    max_segment_length_long: Annotated[
        float,
        Field(
            description=(
                "Maximum audio segment length in seconds for long VAD-based chunking "
                "(used for operations that are sensitive to OOM, e.g. diarization)"
            ),
        ),
    ]
    high_confidence_similarity_threshold: Annotated[
        float,
        Field(
            description=(
                "Minimum cosine similarity score (0.0–1.0) for a speaker embedding match to be "
                "considered high-confidence during initial speaker identification"
            ),
        ),
    ]
    vad: Annotated[
        VadSettings,
        Field(description="VAD model and post-processing hyperparameters"),
    ]

    @field_validator("high_confidence_similarity_threshold")
    @classmethod
    def _similarity_threshold_must_be_in_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"high_confidence_similarity_threshold must be between 0.0 and 1.0, got {v!r}")
        return v

    @field_validator("attendees")
    @classmethod
    def _attendee_names_must_be_non_empty(cls, names: list[str]) -> list[str]:
        for i, name in enumerate(names):
            stripped = name.strip()
            if not stripped:
                raise ValueError(
                    f"attendees[{i}] is blank — every attendee name must be a non-empty string, got {name!r}"
                )
        return names

    @model_validator(mode="after")
    def _validate_audio_file(self) -> Self:
        path = self.audio_file
        if path.suffix.lower() not in SUPPORTED_AUDIO_SUFFIXES:
            raise ValueError(f"Unsupported audio format {path.suffix!r}. Supported: {sorted(SUPPORTED_AUDIO_SUFFIXES)}")
        if path.is_absolute() and not path.exists():
            raise ValueError(f"Audio file does not exist: {path}")
        if self.min_segment_length_short >= self.max_segment_length_short:
            raise ValueError(
                f"min_segment_length_short ({self.min_segment_length_short}) must be less than "
                f"max_segment_length_short ({self.max_segment_length_short})"
            )
        if self.min_segment_length_long >= self.max_segment_length_long:
            raise ValueError(
                f"min_segment_length_long ({self.min_segment_length_long}) must be less than "
                f"max_segment_length_long ({self.max_segment_length_long})"
            )
        return self

    @property
    def number_of_speakers(self) -> int:
        """Derived from the length of attendees; used by the diarizer."""
        return len(self.attendees)

    @staticmethod
    def _resolve_paths(data: dict, base_dir: Path) -> None:
        for key in (
            "audio_file",
            "cleaned_audio_file",
            "transcript_file",
            "aligned_transcript_path",
            "confidence_transcript_path",
            "segments_path",
            "base_diarized_path",
            "speech_clips_with_embedding",
        ):
            raw = data.get(key)
            if raw is None:
                continue
            p = Path(raw)
            if not p.is_absolute():
                p = (base_dir / p).resolve()
            data[key] = p

    @classmethod
    def load(cls, path: Path) -> SessionSettings:
        with path.open("r", encoding="utf-8") as f:
            data: dict = yaml.safe_load(f) or {}
        cls._resolve_paths(data, path.parent)
        return cls(**data)

    @classmethod
    def load_cascading(cls, session_id: str) -> SessionSettings:
        from session_summarizer.utils.common_paths import data_dir, session_dir

        base_file = data_dir() / _SETTINGS_FILE
        session_file = session_dir(session_id) / _SETTINGS_FILE

        if not base_file.exists() and not session_file.exists():
            raise FileNotFoundError(
                f"No settings file found — looked in:\n"
                f"  {base_file}\n"
                f"  {session_file}\n"
                f"Place a {_SETTINGS_FILE} in either location."
            )

        base: dict = {}
        if base_file.exists():
            with base_file.open("r", encoding="utf-8") as f:
                base = yaml.safe_load(f) or {}

        override: dict = {}
        if session_file.exists():
            with session_file.open("r", encoding="utf-8") as f:
                override = yaml.safe_load(f) or {}

        merged = {**base, **override}
        cls._resolve_paths(merged, session_dir(session_id))
        return cls(**merged)


# Backwards-compatible alias
session_settings = SessionSettings
