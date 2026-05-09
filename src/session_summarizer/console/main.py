from __future__ import annotations

import os

# Suppress TensorFlow C++ and oneDNN log spam before any ML library is imported
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import random
import sys
from importlib.metadata import PackageNotFoundError, metadata
from importlib.metadata import version as dist_version
from pathlib import Path

import numpy as np
import torch
import typer
from dotenv import load_dotenv
from rich.console import Console

from session_summarizer.commands.punctuate_text import PunctuateTextCommand
from session_summarizer.utils import common_paths

from ..commands.add_embeddings import AddEmbeddingsCommand
from ..commands.align_transcript import AlignTranscriptCommand
from ..commands.assign_utterance_ids import AssignUtteranceIdsCommand
from ..commands.clean_audio import CleanAudioCommand
from ..commands.clean_session import CleanSessionCommand
from ..commands.clean_session_step import CleanSessionStepCommand
from ..commands.clear_logs import ClearLogsCommand
from ..commands.compare_fulltext import CompareFullTextCommand
from ..commands.compute_segments import ComputeSegmentsCommand
from ..commands.create_speaker_clips import (
    CreateKnownSpeakerClipsCommand,
    CreateSpeakerClipsFromInferredSpeakersCommand,
)
from ..commands.diarizationlm_command import DiarizationLMCommand
from ..commands.diarize_audio import DiarizeAudioCommand
from ..commands.document_dependencies import DocumentDependenciesCommand
from ..commands.identify_speakers import IdentifySpeakersCommand
from ..commands.infer_speakers import InferSpeakersCommand
from ..commands.mark_backchannels import MarkBackchannelsCommand
from ..commands.merge_speaker_clips import MergeSpeakerClipsCommand
from ..commands.register_speakers import RegisterSpeakersCommand
from ..commands.remove_outlier_speaker_clips import RemoveOutlierSpeakerClipsCommand
from ..commands.save_session_clipset import SaveSessionClipsetCommand
from ..commands.simplify_transcript import SimplifyTranscriptCommand
from ..commands.stitch_identities import StitichIdentitiesCommand
from ..commands.summarize_session import SummarizeSessionCommand
from ..commands.test_command import TestCommand
from ..commands.transcribe_audio import TranscribeAudioCommand
from ..commands.validate_diarization import ValidateDiarizationCommand
from ..commands.validate_transcribers import ValidateTranscribersCommand
from ..logging import CompositeLogger, FileLogger, RichConsoleLogger
from ..protocols import LoggingProtocol
from ..settings.session_settings import SessionSettings
from ..utils import Tracer, configure_logging, flush_gpu_memory, initialize_request, initialize_tracing
from .console_validation import _validate_directory_exists

load_dotenv()
configure_logging()
initialize_tracing()

# Set random seeds for reproducible model inference


def _set_seed(session_id: str) -> None:
    settings: SessionSettings = SessionSettings.load_cascading(session_id)
    seed: int = settings.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


flush_gpu_memory()

app = typer.Typer(
    name="session-summarizer",
    add_completion=True,
    help="CLI for session-summarizer",
)


def initialize_logging() -> tuple[LoggingProtocol, Tracer]:
    logger: LoggingProtocol
    tracer: Tracer

    request_id = initialize_request()
    tracer = Tracer()

    console = Console(file=sys.__stdout__)
    # error_console = Console(file=sys.__stderr__)
    console_logger: RichConsoleLogger = RichConsoleLogger(console)

    logfile_path = common_paths.generate_logfile_path()
    file_logger: FileLogger = FileLogger(logfile_path, verbose_training=True)
    logger = CompositeLogger([console_logger, file_logger])

    # logger = console_logger
    logger.report_message(f"[blue]Session id: {request_id}[/blue]")
    return logger, tracer


def confirm_session(session_id: str) -> None:
    session_dir = common_paths.session_dir(session_id)
    errors: list[str] = _validate_directory_exists(session_dir)
    if errors and len(errors) > 0:
        console: Console = Console()
        for error in errors:
            console.print(f"[red]Error: {error}[/red]")
        raise typer.Exit(1)


@app.command("add-embeddings")
def add_embeddings(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to process"),
) -> None:
    """Generate speaker embeddings for each speech clip and save to disk."""
    confirm_session(session)
    _set_seed(session)
    logger, tracer = initialize_logging()
    command: AddEmbeddingsCommand = AddEmbeddingsCommand(session, tracer, force=True)
    command.execute(logger)


@app.command("align-transcription")
def align_transcription(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to transcribe"),
) -> None:
    confirm_session(session)
    _set_seed(session)
    logger, tracer = initialize_logging()
    command: AlignTranscriptCommand = AlignTranscriptCommand(session, tracer, force=True)
    command.execute(logger)


@app.command("apply-identity-stitching")
def apply_identity_stitiching(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to process"),
) -> None:
    """Score each speech clip with end-of-turn probability and set the END_OF_TURN flag."""
    confirm_session(session)
    _set_seed(session)
    logger, tracer = initialize_logging()
    command: StitichIdentitiesCommand = StitichIdentitiesCommand(session, tracer, force=True)
    command.execute(logger)


@app.command("assign-utterance-ids")
def assign_utterance_ids(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to process"),
) -> None:
    """Stamp each speech clip with a stable <campaign_id>_<session_id>_<n> utterance id."""
    confirm_session(session)
    _set_seed(session)
    logger, tracer = initialize_logging()
    command: AssignUtteranceIdsCommand = AssignUtteranceIdsCommand(session, tracer, force=True)
    command.execute(logger)


@app.command("save-session-clipset")
def save_session_clipset(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to process"),
) -> None:
    """Save the utterance-id-stamped clipset under <session_id>.json."""
    confirm_session(session)
    _set_seed(session)
    logger, tracer = initialize_logging()
    command: SaveSessionClipsetCommand = SaveSessionClipsetCommand(session, tracer, force=True)
    command.execute(logger)


@app.command("compare-texts")
def compare_texts(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to transcribe"),
) -> None:
    confirm_session(session)
    _set_seed(session)
    logger, tracer = initialize_logging()
    command: CompareFullTextCommand = CompareFullTextCommand(session, tracer, force=True)
    command.execute(logger)


@app.command("compute-vad-segments")
def compute_vad_segments(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to segment"),
) -> None:
    """Run VAD on cleaned audio and compute optimal cut points for chunked processing."""
    confirm_session(session)
    _set_seed(session)
    logger, tracer = initialize_logging()
    command: ComputeSegmentsCommand = ComputeSegmentsCommand(session, tracer, force=True)
    command.execute(logger)


@app.command("create-known-speaker-clips")
def create_known_speaker_clips(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to process"),
    temp_folder: str = typer.Option(
        ..., "--temp-folder", "-t", help="Name of temp folder inside voice samples to hold output"
    ),
) -> None:
    """Save known identified-speaker clips as individual audio files."""
    confirm_session(session)
    _set_seed(session)
    logger, tracer = initialize_logging()
    command: CreateKnownSpeakerClipsCommand = CreateKnownSpeakerClipsCommand(
        session, tracer, use_multi_speaker_clips=False, temp_folder=Path(temp_folder)
    )
    command.execute(logger)


@app.command("create-speaker-clips-from-inferred-speakers")
def create_speaker_clips_from_inferred_speakers(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to process"),
) -> None:
    """Save inferred-speaker clips into top-level voice sample folders."""
    confirm_session(session)
    _set_seed(session)
    logger, tracer = initialize_logging()
    command: CreateSpeakerClipsFromInferredSpeakersCommand = CreateSpeakerClipsFromInferredSpeakersCommand(
        session, tracer, force=True
    )
    command.execute(logger)


@app.command("clean-audio")
def clean_audio(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to clean"),
) -> None:
    confirm_session(session)
    _set_seed(session)
    logger, tracer = initialize_logging()

    command: CleanAudioCommand = CleanAudioCommand(session, tracer, force=True)
    command.execute(logger)


@app.command("clean-diarization")
def clean_diarization(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to clean"),
) -> None:
    """Delete all generated files in a session folder, keeping settings.yaml and the original audio."""
    confirm_session(session)
    _set_seed(session)
    logger, tracer = initialize_logging()
    command: CleanSessionStepCommand = CleanSessionStepCommand(session, tracer, force=True)
    command.commands_to_clean.append(DiarizeAudioCommand(session, tracer, force=True))
    command.execute(logger)


@app.command("clean-session")
def clean_session(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to clean"),
) -> None:
    """Delete all generated files in a session folder, keeping settings.yaml and the original audio."""
    confirm_session(session)
    _set_seed(session)
    logger, tracer = initialize_logging()
    command: CleanSessionCommand = CleanSessionCommand(session, tracer, force=True)
    command.execute(logger)


@app.command("clear-logs")
def clear_logs() -> None:
    """Delete all files in the logs directory."""
    logger, tracer = initialize_logging()
    ClearLogsCommand().execute(logger)


@app.command("diarizationlm")
def diarizationlm(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to process"),
) -> None:
    """Post-process diarization with DiarizationLM to correct speaker attribution errors."""
    confirm_session(session)
    _set_seed(session)
    logger, tracer = initialize_logging()
    command: DiarizationLMCommand = DiarizationLMCommand(session, tracer, force=True)
    command.execute(logger)


@app.command("merge-speaker-clips")
def merge_speaker_clips(
    speaker: str = typer.Option(
        ..., "--speaker", "-s", help="Speaker label — must match a subdirectory in voice_samples/"
    ),
    output_folder: str = typer.Option(..., "--output-folder", "-o", help="Folder to write the merged clips into"),
) -> None:
    """Merge short clips for a speaker until all are >= speaker_clips.min_duration_seconds."""
    logger, tracer = initialize_logging()
    command: MergeSpeakerClipsCommand = MergeSpeakerClipsCommand(
        speaker_label=speaker,
        output_folder=Path(output_folder),
    )
    command.execute(logger)


@app.command("remove-outlier-speaker-clips")
def remove_outlier_speaker_clips(
    speaker: str = typer.Option(
        ..., "--speaker", "-s", help="Speaker label — must match a subdirectory in voice_samples/"
    ),
    output_folder: str = typer.Option(..., "--output-folder", "-o", help="Folder to write the merged clips into"),
) -> None:
    """Merge short clips for a speaker until all are >= speaker_clips.min_duration_seconds."""
    logger, tracer = initialize_logging()
    command: RemoveOutlierSpeakerClipsCommand = RemoveOutlierSpeakerClipsCommand(
        speaker_label=speaker,
        output_folder=Path(output_folder),
    )
    command.execute(logger)


@app.command("diarize-audio")
def diarize_audio(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to transcribe"),
) -> None:
    confirm_session(session)
    _set_seed(session)
    logger, tracer = initialize_logging()
    command: DiarizeAudioCommand = DiarizeAudioCommand(session, tracer, force=True)
    command.execute(logger)


_SAMPLE_SETTINGS = """\
# ============================================================================
# Session Summarizer — settings.yaml
# ============================================================================
#
# This file configures a session-summarizer run. It can live in two places:
#
#   1. data/settings.yaml           — shared defaults for every session
#   2. data/<session-id>/settings.yaml — per-session overrides
#
# When both exist, per-session values override the shared defaults.
# ============================================================================


# ---------------------------------------------------------------------------
# session_info  (REQUIRED)
# ---------------------------------------------------------------------------
# Metadata about the current TTRPG session. Used by the summarize_session
# command to provide context to Claude when generating the session summary.
#
# Fields:
#   session_date    — Date of the recorded session (YYYY-MM-DD).
#   session_id      — Stable short identifier for this session, used as the
#                     second segment of utterance ids
#                     (<campaign_id>_<session_id>_<n>).
#                     Used by: assign_utterance_ids.
#   adventure_name  — Name of the adventure or module being played.
#   campaign_name   — Name of the overarching campaign.
#
# Example:
#   session_info:
#     session_date: "2026-04-13"
#     session_id: "s07"
#     adventure_name: "Curse of Strahd"
#     campaign_name: "Ravenloft Chronicles"
session_info:
  session_date: "2026-01-01"
  session_id: "session-001"
  adventure_name: "My Adventure"
  campaign_name: "My Campaign"


# ---------------------------------------------------------------------------
# campaign_info  (REQUIRED)
# ---------------------------------------------------------------------------
# Campaign-level context provided to Claude when generating session summaries.
#
# Fields:
#   campaign_id — Stable short identifier for the campaign, used as the first
#                 segment of utterance ids (<campaign_id>_<session_id>_<n>).
#                 Used by: assign_utterance_ids.
#   players   — A mapping of player names (as they appear in the transcript)
#               to their character names. Used to personalise the summary.
#   glossary  — A list of proper nouns (NPCs, locations, factions, items)
#               that may appear in the transcript. Each entry has a required
#               'term' field and an optional 'description' field.
#
# Example:
#   campaign_info:
#     campaign_id: "ravenloft"
#     players:
#       Alice: Elara the Wise
#       Bob: Grog the Strong
#     glossary:
#       - term: Strahd
#         description: The vampire lord of Barovia
#       - term: Barovia
campaign_info:
  campaign_id: "my-campaign"
  players:
    Player1: Character1
    Player2: Character2
  glossary:
    - term: ExamplePlace
      description: A location in the adventure


# ---------------------------------------------------------------------------
# attendees  (REQUIRED)
# ---------------------------------------------------------------------------
# A list of player names present in the session. This drives diarization
# (number of speakers). Names must match entries in registered_speakers.yaml
# and must be non-empty strings.
# Used by: identify_speakers (identity matching), diarize_audio (speaker count).
#
# Example:
#   attendees:
#     - Alice
#     - Bob
#     - Charlie
attendees:
  - Speaker1
  - Speaker2


# ---------------------------------------------------------------------------
# llm  (REQUIRED)
# ---------------------------------------------------------------------------
# Model and effort configuration for each LLM call in the pipeline.
#
# Fields:
#   infer_players    — Used by infer-speakers when inferring role labels.
#   session_logs     — Used when generating session logs.
#   session_summary  — Used by summarize_session when generating summary.md.
#
# Each call has:
#   model   — One of: gpt-5.4, claude-opus-4-6-20260205,
#             claude-sonnet-4-5-20250929, claude-opus-4-7,
#             gemini/gemini-3.1-pro-preview
#   effort  — One of: minimal, low, medium, high, xhigh
#
# Example:
#   llm:
#     session_summary:
#       model: claude-opus-4-7
#       effort: medium
llm:
  infer_players:
    model: gpt-5.5
    effort: medium
  session_logs:
    model: claude-sonnet-4-6
    effort: medium
  session_summary:
    model: claude-opus-4-7
    effort: medium


# ---------------------------------------------------------------------------
# paths  (REQUIRED)
# ---------------------------------------------------------------------------
# File paths for all processing artifacts in the pipeline.
# Relative paths are resolved from the directory that contains this file.
# Absolute paths are used as-is.
paths:

  # Path to the original session recording.
  # Supported formats: .m4a .mp3 .wav .flac .ogg .opus .wma .aac .webm
  # Read by: clean_audio command (input to noise reduction).
  source_audio: original.m4a

  # Path to the noise-reduced audio WAV.
  # Written by: clean_audio. Read by: most downstream commands.
  cleaned_audio: cleaned_audio.wav

  # Path to the initial ASR transcript JSON.
  # Written by: transcribe. Read by: align_transcript.
  transcript: transcript.json

  # Path to the word-aligned transcript JSON with per-word timestamps.
  # Written by: align_transcript. Read by: score_confidence.
  aligned_transcript: aligned_transcript.json

  # Path to the transcript JSON annotated with per-word confidence scores.
  # Written by: score_confidence. Read by: diarize_audio.
  confidence_transcript: confidence_transcript.json

  # Path to the VAD segments JSON containing silence-aware cut points.
  # Written by: compute_vad_segments. Read by: transcription, diarization, etc.
  vad_segments: segments.json

  # Path to the initial SpeechClipSet JSON from diarization + word assignment.
  # Written by: diarize_audio. Read by: add_embeddings, validate_diarization.
  base_diarization: base_diarization.json

  # Path to the SpeechClipSet JSON with speaker embedding vectors added.
  # Written by: add_embeddings. Read by: identify_speakers, create_speaker_clips.
  clips_with_embeddings: clips_with_embeddings.json

  # Path to the SpeechClipSet JSON with speaker identities assigned.
  # Written by: identify_speakers. Read by: stitch_identities, create_speaker_clips.
  identified_speakers: identified_speakers.json

  # Path to the SpeechClipSet JSON after same-identity clip merging.
  # Written by: stitch_identities. Read by: mark_backchannels, punctuate_text.
  identity_stitched: identity_stitched.json

  # Path to the SpeechClipSet JSON with backchannel flags applied.
  # Written by: mark_backchannels. Read by: punctuate_text.
  backchannel_marked: backchannel_marked.json

  # Path to the SpeechClipSet JSON after DiarizationLM post-processing.
  # Written by: diarizationlm. Read by: dangling_sentence_fix.
  diarizationlm_processed: diarizationlm_processed.json

  # Path to the SpeechClipSet JSON with unidentifiable speakers assigned
  # an indeterminate label.
  # Written by: indeterminate_speaker_assignment.
  indeterminate_speakers: indeterminate_speakers.json

  # Path to the SpeechClipSet JSON after mid-sentence clip boundaries
  # have been repaired.
  # Written by: dangling_sentence_fix command.
  dangling_sentence_fix: dangling_sentence_fix.json

  # Path to the SpeechClipSet JSON with role-based speaker identities inferred from transcript text.
  # Written by: infer-speakers command.
  inferred_speakers: inferred_speakers.json

  # Path to the final SpeechClipSet JSON after punctuation and capitalisation
  # have been restored.
  # Written by: punctuate_text. Read by: assign_utterance_ids.
  punctuated_text: punctuated_text.json

  # Path to the SpeechClipSet JSON after assign_utterance_ids stamps each clip
  # with a <campaign_id>_<session_id>_<n> utterance id.
  # Written by: assign_utterance_ids. Read by: simplify_transcript and summarize_session.
  utterance_ids_annotated: utterance_ids_annotated.json

  # Path where the LLM-cleaned transcript generated from punctuated text is written.
  # Written by: simplify_transcript command.
  simplified_transcript: simplified_transcript.json

  # Path where the final session summary generated by Claude is written.
  # Written by: summarize_session command.
  summary_path: summary.md


# ---------------------------------------------------------------------------
# segmentation  (REQUIRED)
# ---------------------------------------------------------------------------
# Duration bounds (in seconds) for VAD-based audio chunking.
# Two modes exist: short (for Canary ASR) and long (for GPU-heavy ops).
# Used by: audio_segmenter.py.
segmentation:

  # Minimum segment length for short chunks (Canary transcription).
  # Chunks shorter than this are merged with neighbours.
  short_min_seconds: 10

  # Maximum segment length for short chunks (Canary transcription).
  # Continuous speech longer than this is hard-cut.
  short_max_seconds: 38

  # Minimum segment length for long chunks (diarization, embedding).
  # Chunks shorter than this are merged with neighbours.
  long_min_seconds: 120

  # Maximum segment length for long chunks (diarization, embedding).
  # Tune down if you see CUDA OOM errors.
  long_max_seconds: 300


# ---------------------------------------------------------------------------
# speaker_clips  (REQUIRED)
# ---------------------------------------------------------------------------
# Settings for speaker clip creation, filtering, and merging.
speaker_clips:

  # Seconds of audio padding before each speaker clip.
  # Used by: create_speaker_clips command.
  # Allowed values: >= 0.0 (seconds). Reasonable default: 0.25
  lead_in_seconds: 0.25

  # Seconds of audio padding after each speaker clip.
  # Used by: create_speaker_clips command.
  # Allowed values: >= 0.0 (seconds). Reasonable default: 0.25
  lead_out_seconds: 0.25

  # Minimum cosine similarity residual (0.0–1.0) a clip must have to be
  # included as a speaker sample. The residual is the difference between
  # a clip's best-match similarity and the mean similarity across speakers.
  # Used by: create_speaker_clips to filter ambiguous clips.
  # Allowed values: 0.0–1.0. Reasonable default: 0.2
  min_similarity_residual: 0.2

  # Target minimum speaker clip duration in seconds. Short clips are
  # iteratively merged until none fall below this threshold.
  # Used by: register_speakers and merge_speaker_clips.
  # Allowed values: >= 0.0 (seconds). Reasonable default: 2.0
  min_duration_seconds: 2.25

  # Minimum cosine similarity (0.0–1.0) a speaker clip must have to the
  # group centroid to be kept. Clips below this are removed as outliers.
  # Used by: remove_outlier_speakers and remove_outlier_speaker_clips.
  # Allowed values: 0.0–1.0. Reasonable default: 0.75
  min_centroid_similarity: 0.6

  # Seconds of silence inserted between clips when combining or merging
  # speaker audio. Prevents utterance bleed-through for better embedding quality.
  # Used by: register_speakers and merge_speaker_clips.
  # Allowed values: >= 0.0 (seconds). Reasonable default: 0.5
  silence_gap_seconds: 0.5


# ---------------------------------------------------------------------------
# speaker_identification  (REQUIRED)
# ---------------------------------------------------------------------------
# Thresholds for assigning speaker identities to clips.
speaker_identification:

  # Minimum similarity residual (0.0–1.0) required to assign a speaker
  # identity to a clip. Clips below this are left as indeterminate.
  # Used by: indeterminate_speakers.py.
  # Allowed values: 0.0–1.0. Reasonable default: 0.08
  assignment_threshold: 0.08


# ---------------------------------------------------------------------------
# device  (REQUIRED)
# ---------------------------------------------------------------------------
# Compute device for model inference. Allowed values:
#   cuda  — use the GPU (requires a CUDA-capable NVIDIA GPU)
#   cpu   — use the CPU (much slower, but works everywhere)
# Used by: all commands that load ML models.
device: cuda


# ---------------------------------------------------------------------------
# vad  (VAD model hyperparameters)
# ---------------------------------------------------------------------------
# Controls the NeMo Voice Activity Detection model used to find speech and
# silence boundaries. Used by: audio_segmenter.py.
vad:

  # Pretrained NeMo VAD model to load.
  model_name: vad_multilingual_frame_marblenet

  # Probability threshold to START a speech region.
  # Allowed values: 0.0–1.0. Reasonable default: 0.7
  onset: 0.7

  # Probability threshold to END a speech region.
  # Allowed values: 0.0–1.0. Reasonable default: 0.4
  offset: 0.4

  # Speech regions shorter than this (seconds) are discarded.
  # Reasonable default: 0.3
  min_duration_on: 0.3

  # Silence regions shorter than this (seconds) are bridged.
  # Reasonable default: 0.3
  min_duration_off: 0.3

  # Seconds of audio to include BEFORE each speech onset.
  # Reasonable default: 0.1
  pad_onset: 0.1

  # Seconds of audio to include AFTER each speech offset.
  # Reasonable default: 0.1
  pad_offset: 0.1


# ---------------------------------------------------------------------------
# stitching  (word-to-speaker-segment assignment)
# ---------------------------------------------------------------------------
# Controls how ASR words are assigned to diarized speaker segments.
# See .research/speaker_segment_assignment.md for the full design rationale.
# Used by: speech_clip_factory.py, candidate_pool.py, anonymous_clips.py,
#          identity_stitch.py, candidate_score.py.
stitching:

  # ── Overlap acceptance thresholds ──────────────────────────────────

  # Minimum fraction of the word's duration that must be overlapped.
  # Allowed values: 0.0–1.0. Reasonable default: 0.20
  min_overlap_fraction_word: 0.20

  # Absolute floor: overlaps shorter than this (seconds) are ignored.
  # Allowed values: >= 0.0 (seconds). Reasonable default: 0.02
  min_overlap_seconds: 0.02

  # ── Fallback: nearest-segment assignment ─────────────────────────────

  # Whether to enable nearest-segment fallback.
  fill_nearest: true

  # Maximum gap (seconds) between a word and a non-overlapping segment
  # for nearest-assignment to apply.
  # Allowed values: >= 0.0 (seconds). Reasonable default: 0.25
  max_nearest_gap_seconds: 0.25

  # ── Fallback: anonymous segments ─────────────────────────────────────

  # Maximum gap (seconds) between consecutive anonymous words that will
  # be merged into the same anonymous segment.
  # Allowed values: >= 0.0 (seconds). Reasonable default: 0.15
  anonymous_join_gap_seconds: 0.15

  # ── Post-processing ──────────────────────────────────────────────────

  # Maximum gap (seconds) between same-speaker adjacent segments that
  # will be merged.
  # Allowed values: >= 0.0 (seconds). Reasonable default: 0.20
  merge_gap_seconds: 0.20

  # Maximum gap (seconds) between two clips with the same identified
  # speaker that can still be merged during identity stitching.
  # Allowed values: >= 0.0 (seconds). Reasonable default: 10.0
  identity_merge_max_gap_seconds: 10.0

  # Widen each segment's time boundaries to fully contain its assigned
  # words.
  expand_segments_to_fit_words: false

  # Cap on how far (seconds) a segment boundary may be expanded.
  # Allowed values: >= 0.0 (seconds). Reasonable default: 300
  expansion_limit_seconds: 300

  # ── Candidate scoring ────────────────────────────────────────────────

  # How to rank candidate segments that overlap a word.
  # Allowed values:
  #   overlap_seconds_then_midpoint
  #   overlap_fraction_word_then_midpoint
  #   iou_then_midpoint
  scoring_mode: overlap_seconds_then_midpoint

  # When two candidates score identically, prefer the shorter segment.
  prefer_shorter_on_tie: true


# ---------------------------------------------------------------------------
# epsilon  (REQUIRED)
# ---------------------------------------------------------------------------
# Small floating-point tolerance used when comparing time boundaries.
# Used by: speech_clip_factory, candidate_pool, anonymous_clips,
#          identity_stitch, diarizationlm_refiner, validate_diarization.
# Allowed values: >= 0.0. Reasonable default: 0.000001
epsilon: 0.000001


# ---------------------------------------------------------------------------
# seed  (REQUIRED)
# ---------------------------------------------------------------------------
# Random seed for reproducible model inference across all frameworks.
# Used by: main.py _set_seed() at CLI startup.
# Allowed values: any integer. Reasonable default: 42
seed: 43
"""


@app.command("document-dependencies")
def document_dependencies_cmd() -> None:
    """Inspect all pipeline commands and write a Mermaid file/command dependency graph."""
    logger, _tracer = initialize_logging()
    command: DocumentDependenciesCommand = DocumentDependenciesCommand()
    command.execute(logger)


@app.command("generate-sample-settings")
def generate_sample_settings() -> None:
    """Generate a well-documented sample settings.yaml in the data directory."""
    console = Console()
    target = common_paths.data_dir() / "settings.yaml"

    common_paths.ensure_directory(common_paths.data_dir())
    target.write_text(_SAMPLE_SETTINGS, encoding="utf-8")
    console.print(f"[green]Sample settings written to {target}[/green]")
    console.print("[dim]Edit the file to match your session before running other commands.[/dim]")


@app.command("identify-speakers")
def identify_speakers(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to process"),
) -> None:
    """Identify speakers in each speech clip by comparing embeddings to registered attendees."""
    confirm_session(session)
    _set_seed(session)
    logger, tracer = initialize_logging()

    command: IdentifySpeakersCommand = IdentifySpeakersCommand(session, tracer, force=True)
    command.execute(logger)


@app.command("infer-speakers")
def infer_speakers(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to process"),
) -> None:
    """Infer role-based speaker identities from transcript text."""
    confirm_session(session)
    _set_seed(session)
    logger, tracer = initialize_logging()

    command: InferSpeakersCommand = InferSpeakersCommand(session, tracer, force=True)
    command.execute(logger)


@app.command("mark-backchannels")
def mark_backchannels(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to process"),
) -> None:
    """Mark short acknowledgement clips as backchannels."""
    confirm_session(session)
    _set_seed(session)
    logger, tracer = initialize_logging()

    command: MarkBackchannelsCommand = MarkBackchannelsCommand(session, tracer, force=True)
    command.execute(logger)


@app.command("punctuate-text")
def punctuate_text(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to process"),
) -> None:
    """Identify speakers in each speech clip by comparing embeddings to registered attendees."""
    confirm_session(session)
    _set_seed(session)
    logger, tracer = initialize_logging()

    command: PunctuateTextCommand = PunctuateTextCommand(session, tracer, force=True)
    command.execute(logger)


@app.command("register-speakers")
def register_speakers() -> None:
    """Merge clips, remove outliers, and register centroid embeddings into registered_speakers.yaml."""
    logger, tracer = initialize_logging()
    RegisterSpeakersCommand().execute(logger)


@app.command("simplify-transcript")
def simplify_transcript(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to process"),
) -> None:
    """Generate a cleaned narrative transcript from punctuated speech clips."""
    confirm_session(session)
    _set_seed(session)
    logger, tracer = initialize_logging()

    command: SimplifyTranscriptCommand = SimplifyTranscriptCommand(session, tracer, force=True)
    command.execute(logger)


@app.command("summarize-session")
def summarize_session(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to transcribe"),
) -> None:
    confirm_session(session)
    _set_seed(session)
    logger, tracer = initialize_logging()

    command: SummarizeSessionCommand = SummarizeSessionCommand(session, tracer, force=True)
    command.execute(logger)


@app.command("test")
def test(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to transcribe"),
) -> None:
    confirm_session(session)
    _set_seed(session)
    logger, tracer = initialize_logging()

    command: TestCommand = TestCommand(session, tracer, force=True)
    command.execute(logger)


@app.command("transcribe")
def transcribe(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to transcribe"),
) -> None:
    confirm_session(session)
    _set_seed(session)
    logger, tracer = initialize_logging()

    command: TranscribeAudioCommand = TranscribeAudioCommand(session, tracer, force=True)
    command.execute(logger)


@app.command("validate-diarization")
def validate_diarization(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to use for validation"),
) -> None:
    """Evaluate diarization quality across pipeline stages and display a metrics comparison table."""
    confirm_session(session)
    _set_seed(session)
    logger, tracer = initialize_logging()

    command: ValidateDiarizationCommand = ValidateDiarizationCommand(session, tracer, force=True)
    command.execute(logger)


@app.command("validate-transcribers")
def validate_transcribers(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to use for validation"),
) -> None:
    """Transcribe test audio with every registered transcriber and compare accuracy metrics."""
    confirm_session(session)
    _set_seed(session)
    logger, tracer = initialize_logging()

    command: ValidateTranscribersCommand = ValidateTranscribersCommand(session, tracer, force=True)
    command.execute(logger)


def _version_callback(value: bool) -> None:
    """Print version and exit."""
    if not value:
        return

    # IMPORTANT: distribution name (pyproject.toml [project].name), often hyphenated.
    # Example: "my-tool" even if your import package is "my_tool".
    DIST_NAME = "session-summarizer"

    console = Console()

    try:
        pkg_version = dist_version(DIST_NAME)
        md = metadata(DIST_NAME)
        try:
            pkg_name = md["Name"]
        except KeyError:
            pkg_name = DIST_NAME

        console.print(f"{pkg_name} {pkg_version}")
    except PackageNotFoundError:
        # Running from source without an installed distribution
        console.print(f"{DIST_NAME} 0.0.0+unknown")

    raise typer.Exit()


@app.callback()
def _callback(
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Show version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """Root command group for reddit_rpg_miner."""
    # Intentionally empty: this forces Typer to keep subcommands like `test`.
    pass


if __name__ == "__main__":
    app()
