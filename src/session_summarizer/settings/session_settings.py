from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

from ..completions.model_settings import ModelEffort, ModelString
from .diarization_stitching_settings import DiarizationStitchingSettings
from .vad_settings import VadSettings

_SETTINGS_FILE = "settings.yaml"


SUPPORTED_AUDIO_SUFFIXES: frozenset[str] = frozenset(
    {".m4a", ".mp3", ".wav", ".flac", ".ogg", ".opus", ".wma", ".aac", ".webm"}
)


class PipelinePaths(BaseModel, frozen=True):
    """File paths for every processing artifact in the pipeline.

    All paths can be specified as relative (resolved against the settings
    file's parent directory) or absolute.
    """

    source_audio: Annotated[
        Path,
        Field(
            description=(
                "Path to the original session recording. Read by: clean_audio command (input to noise reduction)."
            )
        ),
    ]
    cleaned_audio: Annotated[
        Path,
        Field(
            description=(
                "Path to the noise-reduced audio WAV. "
                "Written by: clean_audio. Read by: most downstream commands "
                "(transcription, alignment, diarization, embedding, etc.)."
            )
        ),
    ]
    transcript: Annotated[
        Path,
        Field(
            description=("Path to the initial ASR transcript JSON. Written by: transcribe. Read by: align_transcript.")
        ),
    ]
    aligned_transcript: Annotated[
        Path,
        Field(
            description=(
                "Path to the word-aligned transcript JSON with per-word timestamps. "
                "Written by: align_transcript. Read by: score_confidence."
            )
        ),
    ]
    confidence_transcript: Annotated[
        Path,
        Field(
            description=(
                "Path to the transcript JSON annotated with per-word confidence scores (0.0–1.0). "
                "Written by: score_confidence. Read by: diarize_audio."
            )
        ),
    ]
    vad_segments: Annotated[
        Path,
        Field(
            description=(
                "Path to the VAD segments JSON containing silence-aware cut points. "
                "Written by: compute_vad_segments. Read by: transcription, diarization, "
                "alignment, embedding, and other segment-aware commands."
            )
        ),
    ]
    base_diarization: Annotated[
        Path,
        Field(
            description=(
                "Path to the initial SpeechClipSet JSON from diarization + word assignment. "
                "Written by: diarize_audio. Read by: add_embeddings, validate_diarization."
            )
        ),
    ]
    clips_with_embeddings: Annotated[
        Path,
        Field(
            description=(
                "Path to the SpeechClipSet JSON with speaker embedding vectors added. "
                "Written by: add_embeddings. Read by: identify_speakers, create_speaker_clips."
            )
        ),
    ]
    identified_speakers: Annotated[
        Path,
        Field(
            description=(
                "Path to the SpeechClipSet JSON with speaker identities assigned via "
                "cosine similarity matching. "
                "Written by: identify_speakers. Read by: stitch_identities, create_speaker_clips."
            )
        ),
    ]
    identity_stitched: Annotated[
        Path,
        Field(
            description=(
                "Path to the SpeechClipSet JSON after same-identity clip merging. "
                "Written by: stitch_identities. Read by: mark_backchannels, punctuate_text."
            )
        ),
    ]
    backchannel_marked: Annotated[
        Path,
        Field(
            description=(
                "Path to the SpeechClipSet JSON with backchannel flags applied. "
                "Written by: mark_backchannels. Read by: punctuate_text."
            )
        ),
    ]
    diarizationlm_processed: Annotated[
        Path,
        Field(
            description=(
                "Path to the SpeechClipSet JSON after DiarizationLM post-processing. "
                "Written by: diarizationlm. Read by: dangling_sentence_fix."
            )
        ),
    ]
    indeterminate_speakers: Annotated[
        Path,
        Field(
            description=(
                "Path to the SpeechClipSet JSON with unidentifiable speakers assigned "
                "an indeterminate label. "
                "Written by: indeterminate_speaker_assignment."
            )
        ),
    ]
    dangling_sentence_fix: Annotated[
        Path,
        Field(
            description=(
                "Path to the SpeechClipSet JSON after mid-sentence clip boundaries "
                "have been repaired. "
                "Written by: dangling_sentence_fix command."
            )
        ),
    ]
    punctuated_text: Annotated[
        Path,
        Field(
            description=(
                "Path to the final SpeechClipSet JSON after punctuation and capitalisation "
                "have been restored to clip transcripts. "
                "Written by: punctuate_text. Read by: simplify_transcript and summarize_session."
            )
        ),
    ]
    simplified_transcript: Annotated[
        Path,
        Field(
            description=(
                "Path where the LLM-cleaned transcript generated from punctuated text is written. "
                "Written by: simplify_transcript command."
            )
        ),
    ]
    summary_path: Annotated[
        Path,
        Field(
            description=(
                "Path where the final session summary generated by Claude is written. "
                "Written by: summarize_session command."
            )
        ),
    ]

    @field_validator("source_audio")
    @classmethod
    def _validate_audio_suffix(cls, v: Path) -> Path:
        if v.suffix.lower() not in SUPPORTED_AUDIO_SUFFIXES:
            raise ValueError(f"Unsupported audio format {v.suffix!r}. Supported: {sorted(SUPPORTED_AUDIO_SUFFIXES)}")
        return v


class SegmentationSettings(BaseModel, frozen=True):
    """Duration bounds for VAD-based audio chunking.

    Two chunking modes exist — short (for ASR transcription) and long
    (for GPU-heavy operations like diarization).  Each mode has a min/max
    bound in seconds.
    """

    short_min_seconds: Annotated[
        float,
        Field(
            description=(
                "Minimum audio segment length in seconds for short VAD-based chunking. "
                "Chunks shorter than this are merged with neighbours. "
                "Used by: audio_segmenter.py for Canary transcription segments."
            )
        ),
    ]
    short_max_seconds: Annotated[
        float,
        Field(
            description=(
                "Maximum audio segment length in seconds for short VAD-based chunking. "
                "Continuous speech longer than this is hard-cut. "
                "Used by: audio_segmenter.py for Canary transcription segments."
            )
        ),
    ]
    long_min_seconds: Annotated[
        float,
        Field(
            description=(
                "Minimum audio segment length in seconds for long VAD-based chunking. "
                "Chunks shorter than this are merged with neighbours. "
                "Used by: audio_segmenter.py for OOM-sensitive operations (diarization)."
            )
        ),
    ]
    long_max_seconds: Annotated[
        float,
        Field(
            description=(
                "Maximum audio segment length in seconds for long VAD-based chunking. "
                "Continuous speech longer than this is hard-cut. "
                "Used by: audio_segmenter.py for OOM-sensitive operations (diarization)."
            )
        ),
    ]

    @model_validator(mode="after")
    def _validate_bounds(self) -> Self:
        if self.short_min_seconds >= self.short_max_seconds:
            raise ValueError(
                f"short_min_seconds ({self.short_min_seconds}) must be less than "
                f"short_max_seconds ({self.short_max_seconds})"
            )
        if self.long_min_seconds >= self.long_max_seconds:
            raise ValueError(
                f"long_min_seconds ({self.long_min_seconds}) must be less than "
                f"long_max_seconds ({self.long_max_seconds})"
            )
        return self


class SpeakerClipSettings(BaseModel, frozen=True):
    """Settings for speaker clip creation, filtering, and merging.

    Controls how individual speaker audio clips are extracted from the
    cleaned audio, how outliers are removed, and how short clips are merged.
    """

    lead_in_seconds: Annotated[
        float,
        Field(
            description=(
                "Seconds of audio padding before each speaker clip. "
                "Used by: create_speaker_clips command when extracting individual audio files."
            )
        ),
    ]
    lead_out_seconds: Annotated[
        float,
        Field(
            description=(
                "Seconds of audio padding after each speaker clip. "
                "Used by: create_speaker_clips command when extracting individual audio files."
            )
        ),
    ]
    min_similarity_residual: Annotated[
        float,
        Field(
            description=(
                "Minimum cosine similarity residual (0.0–1.0) a clip must have to be "
                "included as a speaker sample. The residual is the difference between "
                "a clip's best-match similarity and the mean similarity across all speakers. "
                "Used by: create_speaker_clips to filter ambiguous clips."
            )
        ),
    ]
    min_duration_seconds: Annotated[
        float,
        Field(
            description=(
                "Target minimum speaker clip duration in seconds. Short clips are "
                "iteratively merged until none fall below this threshold. "
                "Used by: register_speakers and merge_speaker_clips commands."
            )
        ),
    ]
    min_centroid_similarity: Annotated[
        float,
        Field(
            description=(
                "Minimum cosine similarity (0.0–1.0) a speaker clip must have to the "
                "group centroid to be kept. Clips below this are removed as outliers. "
                "Used by: remove_outlier_speakers and remove_outlier_speaker_clips."
            )
        ),
    ]
    silence_gap_seconds: Annotated[
        float,
        Field(
            description=(
                "Seconds of silence inserted between clips when combining or merging "
                "speaker audio. Prevents utterance bleed-through for better embedding quality. "
                "Used by: register_speakers and merge_speaker_clips commands."
            )
        ),
    ] = 0.5

    @field_validator("lead_in_seconds", "lead_out_seconds")
    @classmethod
    def _lead_must_be_non_negative(cls, v: float, info: ValidationInfo) -> float:
        if v < 0.0:
            raise ValueError(f"{info.field_name} must be >= 0.0, got {v!r}")
        return v

    @field_validator("min_similarity_residual")
    @classmethod
    def _similarity_residual_must_be_in_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"min_similarity_residual must be between 0.0 and 1.0, got {v!r}")
        return v

    @field_validator("min_duration_seconds", "silence_gap_seconds")
    @classmethod
    def _duration_must_be_non_negative(cls, v: float, info: ValidationInfo) -> float:
        if v < 0.0:
            raise ValueError(f"{info.field_name} must be >= 0.0, got {v!r}")
        return v

    @field_validator("min_centroid_similarity")
    @classmethod
    def _min_centroid_similarity_must_be_in_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"min_centroid_similarity must be between 0.0 and 1.0, got {v!r}")
        return v


class SpeakerIdentificationSettings(BaseModel, frozen=True):
    """Thresholds for assigning speaker identities to clips."""

    assignment_threshold: Annotated[
        float,
        Field(
            description=(
                "Minimum similarity residual (0.0–1.0) required to assign a speaker "
                "identity to a clip. Clips below this threshold are left as indeterminate. "
                "Used by: indeterminate_speakers.py to decide identity assignment."
            )
        ),
    ]

    @field_validator("assignment_threshold")
    @classmethod
    def _must_be_in_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"assignment_threshold must be between 0.0 and 1.0, got {v!r}")
        return v


class GlossaryEntry(BaseModel, frozen=True):
    """A single proper-noun entry in the campaign glossary."""

    term: Annotated[
        str,
        Field(description="The proper noun or name to recognise in transcripts."),
    ]
    description: str = ""

    @field_validator("term")
    @classmethod
    def _term_must_be_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError(f"glossary term must be a non-empty string, got {v!r}")
        return v


class CampaignInfo(BaseModel, frozen=True):
    """Campaign-level context used when generating session summaries."""

    players: Annotated[
        dict[str, str],
        Field(description="Mapping of player names to their character names."),
    ]
    glossary: Annotated[
        list[GlossaryEntry],
        Field(description="List of proper nouns (places, NPCs, items) that may appear in transcripts."),
    ]

    @field_validator("players")
    @classmethod
    def _players_must_have_non_empty_entries(cls, v: dict[str, str]) -> dict[str, str]:
        for player, character in v.items():
            if not player.strip():
                raise ValueError(f"player name must be a non-empty string, got {player!r}")
            if not character.strip():
                raise ValueError(f"character name for player {player!r} must be a non-empty string, got {character!r}")
        return v


class SessionInfo(BaseModel, frozen=True):
    """Metadata about the current TTRPG session being processed."""

    session_date: Annotated[
        str,
        Field(description="Date of the recorded session in YYYY-MM-DD format."),
    ]
    adventure_name: Annotated[
        str,
        Field(description="Name of the adventure or module being played in this session."),
    ]
    campaign_name: Annotated[
        str,
        Field(description="Name of the overarching campaign this session belongs to."),
    ]

    @field_validator("session_date", "adventure_name", "campaign_name")
    @classmethod
    def _must_be_non_empty(cls, v: str, info: ValidationInfo) -> str:
        if not v.strip():
            raise ValueError(f"{info.field_name} must be a non-empty string, got {v!r}")
        return v


class LlmCallSettings(BaseModel, frozen=True):
    """Model configuration for a single LLM call site."""

    model: Annotated[
        ModelString,
        Field(description="Model identifier passed to LiteLLM for this LLM call."),
    ]
    effort: Annotated[
        ModelEffort,
        Field(description="Reasoning effort passed to LiteLLM for this LLM call."),
    ]


class LlmSettings(BaseModel, frozen=True):
    """Per-call LLM model configuration."""

    session_logs: Annotated[
        LlmCallSettings,
        Field(description="Model configuration for generating session logs."),
    ]
    session_summary: Annotated[
        LlmCallSettings,
        Field(description="Model configuration for generating the final session summary."),
    ]


class SessionSettings(BaseModel, frozen=True):
    attendees: Annotated[
        list[str],
        Field(
            min_length=1,
            description=(
                "List of speaker names present in the session. Drives diarization speaker count "
                "and speaker identification matching. "
                "Used by: identify_speakers (identity matching), diarize_audio (speaker count)."
            ),
        ),
    ]
    session_info: Annotated[
        SessionInfo,
        Field(description="Metadata about the current session: date, adventure, and campaign"),
    ]
    campaign_info: Annotated[
        CampaignInfo,
        Field(description="Campaign context: player-to-character mapping and glossary of proper nouns"),
    ]
    llm: Annotated[
        LlmSettings,
        Field(description="Model and effort configuration for each LLM call in the pipeline"),
    ]
    paths: Annotated[
        PipelinePaths,
        Field(description="File paths for all processing artifacts in the pipeline"),
    ]
    segmentation: Annotated[
        SegmentationSettings,
        Field(description="Duration bounds for VAD-based audio chunking (short and long modes)"),
    ]
    speaker_clips: Annotated[
        SpeakerClipSettings,
        Field(description="Settings for speaker clip creation, filtering, and merging"),
    ]
    speaker_identification: Annotated[
        SpeakerIdentificationSettings,
        Field(description="Thresholds for assigning speaker identities to clips"),
    ]
    vad: Annotated[
        VadSettings,
        Field(description="VAD model and post-processing hyperparameters"),
    ]
    stitching: Annotated[
        DiarizationStitchingSettings,
        Field(description="Policy knobs for assigning ASR words to diarized speaker segments"),
    ]
    device: Annotated[
        Literal["cpu", "cuda"],
        Field(
            description=(
                "Compute device for model inference — 'cpu' or 'cuda'. "
                "Used by: all commands that load ML models (transcription, alignment, "
                "diarization, embedding, confidence scoring, etc.)."
            )
        ),
    ]
    epsilon: Annotated[
        float,
        Field(
            description=(
                "Small floating-point tolerance for time-boundary comparisons. "
                "Used by: speech_clip_factory, candidate_pool, anonymous_clips, "
                "identity_stitch, diarizationlm_refiner, validate_diarization."
            )
        ),
    ]
    seed: Annotated[
        int,
        Field(
            description=(
                "Random seed for reproducible model inference across all frameworks "
                "(Python random, NumPy, PyTorch). "
                "Used by: main.py _set_seed() at CLI startup."
            )
        ),
    ]

    @field_validator("epsilon")
    @classmethod
    def _epsilon_must_be_non_negative(cls, v: float) -> float:
        if v < 0.0:
            raise ValueError(f"epsilon must be >= 0.0, got {v!r}")
        return v

    @field_validator("attendees")
    @classmethod
    def _attendee_names_must_be_non_empty(cls, v: list[str]) -> list[str]:
        for name in v:
            if not name.strip():
                raise ValueError(f"attendee name is blank — every name must be a non-empty string, got {name!r}")
        return v

    @property
    def number_of_speakers(self) -> int:
        """Derived from the length of attendees; used by the diarizer."""
        return len(self.attendees)

    @staticmethod
    def _resolve_paths(data: dict, base_dir: Path) -> None:
        paths_dict = data.get("paths")
        if paths_dict is None:
            return
        for key in list(paths_dict.keys()):
            raw = paths_dict[key]
            if raw is None:
                continue
            p = Path(raw)
            if not p.is_absolute():
                p = (base_dir / p).resolve()
            paths_dict[key] = p

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
