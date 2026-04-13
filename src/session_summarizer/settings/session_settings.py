from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

from .diarization_stitching_settings import DiarizationStitchingSettings
from .vad_settings import VadSettings

_SETTINGS_FILE = "settings.yaml"


class GlossaryEntry(BaseModel, frozen=True):
    name: Annotated[str, Field(description="A proper noun that may appear in the session transcript")]
    description: Annotated[
        str | None,
        Field(description="Optional description or context for this term"),
    ] = None


class AdventureSettings(BaseModel, frozen=True):
    pcs: Annotated[
        dict[str, str],
        Field(min_length=1, description="Mapping of player name to character name for all PCs in the adventure"),
    ]
    glossary: Annotated[
        list[GlossaryEntry],
        Field(
            description=(
                "List of proper nouns (places, NPCs, items, factions, etc.) that may appear in the session transcript"
            )
        ),
    ]

    @field_validator("pcs")
    @classmethod
    def _pc_names_must_be_non_empty(cls, v: dict[str, str]) -> dict[str, str]:
        for player, character in v.items():
            if not player.strip():
                raise ValueError(
                    f"PC player name is blank — every player name must be a non-empty string, got {player!r}"
                )
            if not character.strip():
                raise ValueError(
                    f"PC character name for '{player}' is blank — character names must be non-empty strings, "
                    f"got {character!r}"
                )
        return v


SUPPORTED_AUDIO_SUFFIXES: frozenset[str] = frozenset(
    {".m4a", ".mp3", ".wav", ".flac", ".ogg", ".opus", ".wma", ".aac", ".webm"}
)


class SessionSettings(BaseModel, frozen=True):
    attendees: Annotated[
        list[str],
        Field(
            min_length=1, description="List of player names present in the session; drives diarization speaker count"
        ),
    ]
    adventure_settings: Annotated[
        AdventureSettings,
        Field(description="Adventure-specific metadata: PC roster and glossary of proper nouns"),
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
    identified_speaker_path: Annotated[
        Path,
        Field(description="Path to SpeechClipSet JSON file with identified speakers (read/written during processing)"),
    ]
    turn_end_updated_path: Annotated[
        Path,
        Field(
            description=(
                "Path to SpeechClipSet JSON file with END_OF_TURN flags applied (read/written during processing)"
            )
        ),
    ]
    first_stitched_path: Annotated[
        Path,
        Field(
            description=(
                "Path to SpeechClipSet JSON file that has had initial stitching (read/written during processing)"
            )
        ),
    ]
    identity_stitched_path: Annotated[
        Path,
        Field(description="Path to SpeechClipSet JSON file with speakers identified (read/written during processing)"),
    ]
    diarizationlm_processed_path: Annotated[
        Path,
        Field(
            description=(
                "Path to SpeechClipSet JSON file after DiarizationLM post-processing (read/written during processing)"
            )
        ),
    ]
    indeterminate_speakers_path: Annotated[
        Path,
        Field(
            description=(
                "Path to SpeechClipSet JSON file with speakers that could not be identified assigned "
                "(read/written during processing)"
            )
        ),
    ]
    dangling_sentence_fix_path: Annotated[
        Path,
        Field(
            description=(
                "Path to SpeechClipSet JSON file where dangling sentences have been fixed (written during processing)"
            )
        ),
    ]
    punctuated_text_path: Annotated[
        Path,
        Field(
            description=(
                "Path to SpeechClipSet JSON file written after punctuation and capitalisation "
                "have been restored to clip transcripts (written during processing)"
            )
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
    speaker_identity_assignment_threshold: Annotated[
        float,
        Field(
            description=(
                "Minimum cosine similarity score (0.0–1.0) required to assign a speaker identity "
                "to a clip; clips below this threshold are left as indeterminate"
            ),
        ),
    ]
    vad: Annotated[
        VadSettings,
        Field(description="VAD model and post-processing hyperparameters"),
    ]

    speaker_clip_lead_in: Annotated[
        float,
        Field(description="Seconds of audio padding before each speaker clip when creating individual audio files"),
    ]
    speaker_clip_lead_out: Annotated[
        float,
        Field(description="Seconds of audio padding after each speaker clip when creating individual audio files"),
    ]

    speaker_clip_minimum_similarity_residual: Annotated[
        float,
        Field(
            description="Minimum cosine similarity residual a clip must have to be included as a speaker clip sample"
        ),
    ]
    minimum_speaker_clip_duration: Annotated[
        float,
        Field(
            description=(
                "Target minimum speaker clip duration in seconds; short clips are merged until none fall below "
                "this threshold"
            )
        ),
    ]
    min_speaker_similarity: Annotated[
        float,
        Field(
            description=(
                "Minimum cosine similarity (0.0–1.0) a speaker clip must have to the group centroid "
                "to be kept; clips below this threshold are removed as outliers"
            )
        ),
    ]
    speaker_clip_gap_length: Annotated[
        float,
        Field(
            description="Seconds of silence inserted between clips when combining or merging speaker audio",
        ),
    ] = 0.5

    diarization_stitching: Annotated[
        DiarizationStitchingSettings,
        Field(
            description="Policy knobs for assigning ASR words to diarized speaker segments",
        ),
    ]

    epsilon: Annotated[
        float,
        Field(description="Small floating-point tolerance."),
    ]

    seed: Annotated[
        int,
        Field(
            description="Random seed for reproducible model inference across all frameworks (Python, NumPy, PyTorch)"
        ),
    ]

    @field_validator("epsilon")
    @classmethod
    def _epsilon_must_be_non_negative(cls, v: float) -> float:
        if v < 0.0:
            raise ValueError(f"epsilon must be >= 0.0, got {v!r}")
        return v

    @field_validator("speaker_clip_lead_in", "speaker_clip_lead_out")
    @classmethod
    def _lead_must_be_non_negative(cls, v: float, info: ValidationInfo) -> float:
        if v < 0.0:
            raise ValueError(f"{info.field_name} must be >= 0.0, got {v!r}")
        return v

    @field_validator("high_confidence_similarity_threshold", "speaker_identity_assignment_threshold")
    @classmethod
    def _similarity_threshold_must_be_in_range(cls, v: float, info: ValidationInfo) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"{info.field_name} must be between 0.0 and 1.0, got {v!r}")
        return v

    @field_validator("attendees")
    @classmethod
    def _attendee_names_must_be_non_empty(cls, v: list[str]) -> list[str]:
        for name in v:
            if not name.strip():
                raise ValueError(f"attendee name is blank — every name must be a non-empty string, got {name!r}")
        return v

    @field_validator("speaker_clip_minimum_similarity_residual")
    @classmethod
    def _similarity_residual_must_be_in_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"speaker_clip_minimum_similarity_residual must be between 0.0 and 1.0, got {v!r}")
        return v

    @field_validator("minimum_speaker_clip_duration", "speaker_clip_gap_length")
    @classmethod
    def _speaker_clip_params_must_be_non_negative(cls, v: float, info: ValidationInfo) -> float:
        if v < 0.0:
            raise ValueError(f"{info.field_name} must be >= 0.0, got {v!r}")
        return v

    @field_validator("min_speaker_similarity")
    @classmethod
    def _min_speaker_similarity_must_be_in_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"min_speaker_similarity must be between 0.0 and 1.0, got {v!r}")
        return v

    @model_validator(mode="after")
    def _validate_audio_file(self) -> Self:
        path = self.audio_file
        if path.suffix.lower() not in SUPPORTED_AUDIO_SUFFIXES:
            raise ValueError(f"Unsupported audio format {path.suffix!r}. Supported: {sorted(SUPPORTED_AUDIO_SUFFIXES)}")
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
            "identified_speaker_path",
            "turn_end_updated_path",
            "first_stitched_path",
            "identity_stitched_path",
            "diarizationlm_processed_path",
            "indeterminate_speakers_path",
            "dangling_sentence_fix_path",
            "punctuated_text_path",
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
