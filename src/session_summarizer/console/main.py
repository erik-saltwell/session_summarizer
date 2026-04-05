from __future__ import annotations

from importlib.metadata import PackageNotFoundError, metadata
from importlib.metadata import version as dist_version

import typer
from dotenv import load_dotenv
from rich.console import Console

from session_summarizer.commands.add_embeddings import AddEmbeddingsCommand
from session_summarizer.commands.align_transcript import AlignTranscriptCommand
from session_summarizer.commands.clean_audio import CleanAudioCommand
from session_summarizer.commands.clean_session import CleanSessionCommand
from session_summarizer.commands.compute_segments import ComputeSegmentsCommand
from session_summarizer.commands.diarize_audio import DiarizeAudioCommand
from session_summarizer.commands.dump_and_compare_texts import DumpAndCompareTextsCommand
from session_summarizer.commands.dump_human_format import DumpHumanFormatCommand
from session_summarizer.commands.identify_speakers import IdentifySpeakersCommand
from session_summarizer.commands.score_confidence import ScoreConfidenceCommand
from session_summarizer.commands.transcribe_audio import TranscribeAudioCommand
from session_summarizer.commands.update_turn_end import UpdateTurnEndCommand
from session_summarizer.commands.validate_transcribers import ValidateTranscribersCommand
from session_summarizer.utils import common_paths

from ..commands.first_stitch_clips import FirstStitchClipsCommand
from ..commands.process_pipeline import ProcessPipelineCommand
from ..commands.register_speakers import RegisterSpeakersCommand
from ..commands.stitch_identities import StitichIdentitiesCommand
from ..logging import CompositeLogger, FileLogger, RichConsoleLogger
from ..protocols import LoggingProtocol
from ..utils import flush_gpu_memory
from ..utils.logging_config import configure_logging
from .console_validation import _validate_directory_exists

load_dotenv()
configure_logging()

flush_gpu_memory()

app = typer.Typer(
    name="session-summarizer",
    add_completion=True,
    help="CLI for session-summarizer",
)


def create_logger() -> LoggingProtocol:
    console = Console()
    console_logger: RichConsoleLogger = RichConsoleLogger(console)
    logfile_path = common_paths.generate_logfile_path()
    file_logger: FileLogger = FileLogger(logfile_path, verbose_training=True)
    return CompositeLogger([console_logger, file_logger])


def confirm_session(session_id: str) -> None:
    session_dir = common_paths.session_dir(session_id)
    errors: list[str] = _validate_directory_exists(session_dir)
    if errors and len(errors) > 0:
        console: Console = Console()
        for error in errors:
            console.print(f"[red]Error: {error}[/red]")
        raise typer.Exit(1)


@app.command("transcribe")
def transcribe(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to transcribe"),
) -> None:
    confirm_session(session)
    logger: LoggingProtocol = create_logger()
    command: TranscribeAudioCommand = TranscribeAudioCommand(session)
    command.execute(logger)


@app.command("align-transcription")
def align_transcription(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to transcribe"),
) -> None:
    confirm_session(session)
    logger: LoggingProtocol = create_logger()
    command: AlignTranscriptCommand = AlignTranscriptCommand(session)
    command.execute(logger)


@app.command("score-confidence")
def score_confidence(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to transcribe"),
) -> None:
    confirm_session(session)
    logger: LoggingProtocol = create_logger()
    command: ScoreConfidenceCommand = ScoreConfidenceCommand(session)
    command.execute(logger)


@app.command("add-embeddings")
def add_embeddings(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to process"),
) -> None:
    """Generate speaker embeddings for each speech clip and save to disk."""
    confirm_session(session)
    logger: LoggingProtocol = create_logger()
    command: AddEmbeddingsCommand = AddEmbeddingsCommand(session)
    command.execute(logger)


@app.command("identify-speakers")
def identify_speakers(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to process"),
) -> None:
    """Identify speakers in each speech clip by comparing embeddings to registered attendees."""
    confirm_session(session)
    logger: LoggingProtocol = create_logger()
    command: IdentifySpeakersCommand = IdentifySpeakersCommand(session)
    command.execute(logger)


@app.command("update-turn-end")
def update_turn_end(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to process"),
) -> None:
    """Score each speech clip with end-of-turn probability and set the END_OF_TURN flag."""
    confirm_session(session)
    logger: LoggingProtocol = create_logger()
    command: UpdateTurnEndCommand = UpdateTurnEndCommand(session)
    command.execute(logger)


@app.command("apply-first-stitching")
def apply_first_stitching(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to process"),
) -> None:
    """Score each speech clip with end-of-turn probability and set the END_OF_TURN flag."""
    confirm_session(session)
    logger: LoggingProtocol = create_logger()
    command: FirstStitchClipsCommand = FirstStitchClipsCommand(session)
    command.execute(logger)


@app.command("apply-identity-stitching")
def apply_identity_stitiching(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to process"),
) -> None:
    """Score each speech clip with end-of-turn probability and set the END_OF_TURN flag."""
    confirm_session(session)
    logger: LoggingProtocol = create_logger()
    command: StitichIdentitiesCommand = StitichIdentitiesCommand(session)
    command.execute(logger)


@app.command("dump-human-format")
def dump_human_format(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session"),
) -> None:
    """Export base_diarization, update_turn, and first_stitch to human-readable text format."""
    confirm_session(session)
    logger: LoggingProtocol = create_logger()
    command: DumpHumanFormatCommand = DumpHumanFormatCommand(session)
    command.execute(logger)


@app.command("diarize-audio")
def diarize_audio(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to transcribe"),
) -> None:
    confirm_session(session)
    logger: LoggingProtocol = create_logger()
    command: DiarizeAudioCommand = DiarizeAudioCommand(session)
    command.execute(logger)


@app.command("clean-audio")
def clean_audio(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to clean"),
) -> None:
    confirm_session(session)
    logger: LoggingProtocol = create_logger()
    command: CleanAudioCommand = CleanAudioCommand(session)
    command.execute(logger)


@app.command("compute-vad-segments")
def compute_vad_segments(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to segment"),
) -> None:
    """Run VAD on cleaned audio and compute optimal cut points for chunked processing."""
    confirm_session(session)
    logger: LoggingProtocol = create_logger()
    command: ComputeSegmentsCommand = ComputeSegmentsCommand(session)
    command.execute(logger)


@app.command("process-pipeline")
def process_pipeline(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to use for validation"),
) -> None:
    """Clean the session and run the full pipeline."""
    confirm_session(session)
    logger: LoggingProtocol = create_logger()
    command: ProcessPipelineCommand = ProcessPipelineCommand(session)
    command.execute(logger)


@app.command("validate-transcribers")
def validate_transcribers(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to use for validation"),
) -> None:
    """Transcribe test audio with every registered transcriber and compare accuracy metrics."""
    confirm_session(session)
    logger: LoggingProtocol = create_logger()
    command: ValidateTranscribersCommand = ValidateTranscribersCommand(session)
    command.execute(logger)


@app.command("dump-and-compare-texts")
def dump_and_compare_texts(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to transcribe"),
) -> None:
    confirm_session(session)
    logger: LoggingProtocol = create_logger()
    command: DumpAndCompareTextsCommand = DumpAndCompareTextsCommand(session)
    command.execute(logger)


@app.command("clean-session")
def clean_session(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to clean"),
) -> None:
    """Delete all generated files in a session folder, keeping settings.yaml and the original audio."""
    confirm_session(session)
    logger: LoggingProtocol = create_logger()
    command: CleanSessionCommand = CleanSessionCommand(session)
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
# attendees  (REQUIRED)
# ---------------------------------------------------------------------------
# A list of speaker names expected in the session. This drives diarization
# (the number of speakers the model should look for) and labels in the final
# transcript. Every entry must be a non-empty string.
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
# audio_file  (REQUIRED)
# ---------------------------------------------------------------------------
# Path to the original recording. Supported formats:
#   .m4a  .mp3  .wav  .flac  .ogg  .opus  .wma  .aac  .webm
#
# Relative paths are resolved from the directory that contains this file.
# Absolute paths are used as-is (and must point to an existing file).
#
# Example:
#   audio_file: meeting_2025-03-29.m4a
audio_file: original.m4a


# ---------------------------------------------------------------------------
# cleaned_audio_file  (REQUIRED)
# ---------------------------------------------------------------------------
# Where the noise-reduced audio is written (or read from, if it already
# exists). Relative paths are resolved from this file's directory.
cleaned_audio_file: cleaned_audio.wav


# ---------------------------------------------------------------------------
# transcript_file  (REQUIRED)
# ---------------------------------------------------------------------------
# Where the transcript JSON is written (or read from, if it already exists).
# Relative paths are resolved from this file's directory.
transcript_file: transcript.json


# ---------------------------------------------------------------------------
# aligned_transcript_path  (REQUIRED)
# ---------------------------------------------------------------------------
# Where the word-aligned transcript is written (or read from, if it already
# exists). Word alignment maps each word to a precise start/end timestamp
# using CTC forced alignment — more accurate than segment-level timing.
# Relative paths are resolved from this file's directory.
aligned_transcript_path: aligned_transcript.json


# ---------------------------------------------------------------------------
# confidence_transcript_path  (REQUIRED)
# ---------------------------------------------------------------------------
# Where the transcript annotated with per-word confidence scores is written
# (or read from, if it already exists). Confidence scores (0.0–1.0) indicate
# how certain the model was about each word; useful for post-processing,
# review prioritisation, and filtering low-confidence segments.
# Relative paths are resolved from this file's directory.
confidence_transcript_path: confidence_transcript.json

# ---------------------------------------------------------------------------
# base_diarized_path  (REQUIRED)
# ---------------------------------------------------------------------------
# Where the list of diarized segments generated from audio is written
# (or read from, if it already exists). Contains auto-generated speaker labels
# and timestamps for each speech segment, used as a basis for final diarization output.
# Relative paths are resolved from this file's directory.
base_diarized_path: base_diarization.json

# ---------------------------------------------------------------------------
# speech_clips_with_embedding  (REQUIRED)
# ---------------------------------------------------------------------------
# Path to the SpeechClipSet JSON file that stores speech clips with speaker
# embeddings attached. The app writes this file after computing embeddings and
# reads it back in subsequent steps (e.g. speaker identification, merging).
# Relative paths are resolved from this file's directory.
#
# Default: clips_with_embeddings.json
#
# Example:
#   speech_clips_with_embedding: clips_with_embeddings.json
speech_clips_with_embedding: clips_with_embeddings.json

# ---------------------------------------------------------------------------
# identified_speaker_path  (REQUIRED)
# ---------------------------------------------------------------------------
# Path to the SpeechClipSet JSON file with speaker identities assigned.
# Written by the identify-speakers command after matching clip embeddings
# against registered attendee embeddings using cosine similarity.
# Relative paths are resolved from this file's directory.
#
# Default: identified_speakers.json
#
# Example:
#   identified_speaker_path: identified_speakers.json
identified_speaker_path: identified_speakers.json

# ---------------------------------------------------------------------------
# turn_end_updated_path  (REQUIRED)
# ---------------------------------------------------------------------------
# Path to the SpeechClipSet JSON file with END_OF_TURN flags applied.
# Written by the update-turn-end command.
#
# Example:
#   turn_end_updated_path: turn_end_updated.json
turn_end_updated_path: turn_end_updated.json

# ---------------------------------------------------------------------------
# first_stitched_path  (REQUIRED)
# ---------------------------------------------------------------------------
# Path to the SpeechClipSet JSON file with first stitching applied.
# Written by the first-stitching command.
#
# Example:
#   first_stitched_path: first_stitched.json
first_stitched_path: first_stitched.json

# ---------------------------------------------------------------------------
# identity_stitched_path  (REQUIRED)
# ---------------------------------------------------------------------------
# Path to the SpeechClipSet JSON file with speakers identified. Written
# after speaker identity has been resolved and stitched into the clip set.
# Relative paths are resolved from this file's directory.
#
# Default: identity_stitched.json
#
# Example:
#   identity_stitched_path: identity_stitched.json
identity_stitched_path: identity_stitched.json

# ---------------------------------------------------------------------------
# device  (REQUIRED)
# ---------------------------------------------------------------------------
# Compute device for model inference. Allowed values:
#   cuda  — use the GPU (requires a CUDA-capable NVIDIA GPU)
#   cpu   — use the CPU (much slower, but works everywhere)
device: cuda


# ---------------------------------------------------------------------------
# segments_path
# ---------------------------------------------------------------------------
# Where the segment plan is written. This JSON file contains silence-aware
# cut points that downstream commands use to process long audio in chunks
# without splitting mid-speech.
# Relative paths are resolved from this file's directory.
segments_path: segments.json


# ---------------------------------------------------------------------------
# min_segment_length_short / max_segment_length_short
# ---------------------------------------------------------------------------
# Bounds (in seconds) for SHORT audio chunks used by Canary transcription.
# Canary processes audio in ~40 s internal windows, so keeping segments short
# reduces latency and avoids feeding the model more context than it needs.
#
#   min_segment_length_short — chunks shorter than this are merged with neighbours.
#   max_segment_length_short — no chunk will exceed this duration. If continuous
#       speech runs longer than this with no silence gap, a hard cut is made.
min_segment_length_short: 10
max_segment_length_short: 38


# ---------------------------------------------------------------------------
# min_segment_length_long / max_segment_length_long
# ---------------------------------------------------------------------------
# Bounds (in seconds) for LONG audio chunks used by operations that load large
# models (e.g. diarization, speaker embedding). Longer segments mean fewer
# model load/unload cycles, reducing overhead — but segments that are too long
# can exhaust GPU memory (OOM). Tune max_segment_length_long down if you see
# CUDA out-of-memory errors, or up if your GPU has headroom to spare.
#
#   min_segment_length_long — chunks shorter than this are merged with neighbours.
#   max_segment_length_long — no chunk will exceed this duration. A hard cut is
#       made when no silence gap falls within the window.
min_segment_length_long: 120
max_segment_length_long: 300


# ---------------------------------------------------------------------------
# high_confidence_similarity_threshold  (REQUIRED)
# ---------------------------------------------------------------------------
# Minimum cosine similarity score for a speaker embedding comparison to be
# treated as a confident match during initial speaker identification. Matches
# at or above this threshold are used to merge speech clips and assign speaker
# labels before the full diarization pass.
#
# Allowed values: 0.0–1.0  (cosine similarity; higher = stricter matching)
#
# Default: 0.88
# Reasonable range: 0.80–0.95
#   Lower values accept more matches (may merge different speakers).
#   Higher values accept fewer matches (may leave clips unlabelled).
#
# Example:
#   high_confidence_similarity_threshold: 0.88
high_confidence_similarity_threshold: 0.88


# ---------------------------------------------------------------------------
# vad  (VAD model hyperparameters)
# ---------------------------------------------------------------------------
# Controls the NeMo Voice Activity Detection model used to find speech and
# silence boundaries. Tune these if you see too many false speech detections
# (lower onset) or missed speech (raise onset / lower offset).
vad:

  # Pretrained NeMo VAD model to load.
  #
  # Allowed values: any NeMo-registered VAD model name (string)
  # Reasonable default: vad_multilingual_frame_marblenet
  model_name: vad_multilingual_frame_marblenet

  # Probability threshold to START a speech region. Higher = fewer false
  # positives but may miss quiet speech. Must be >= offset (hysteresis).
  #
  # Allowed values: 0.0–1.0
  # Reasonable default: 0.7
  onset: 0.7

  # Probability threshold to END a speech region. Lower = speech regions
  # extend further into trailing silence. Must be <= onset.
  #
  # Allowed values: 0.0–1.0
  # Reasonable default: 0.4
  offset: 0.4

  # Speech regions shorter than this (seconds) are discarded. Filters out
  # clicks, coughs, and transient noise.
  #
  # Allowed values: >= 0.0 (seconds)
  # Reasonable default: 0.3
  min_duration_on: 0.3

  # Silence regions shorter than this (seconds) are bridged (treated as
  # speech). Prevents choppy segmentation from brief pauses within sentences.
  #
  # Allowed values: >= 0.0 (seconds)
  # Reasonable default: 0.3
  min_duration_off: 0.3

  # Seconds of audio to include BEFORE each speech onset. Captures plosive
  # consonants and breath that precede speech.
  #
  # Allowed values: >= 0.0 (seconds)
  # Reasonable default: 0.1
  pad_onset: 0.1

  # Seconds of audio to include AFTER each speech offset. Captures
  # word-final sounds and natural trailing silence.
  #
  # Allowed values: >= 0.0 (seconds)
  # Reasonable default: 0.1
  pad_offset: 0.1


# ---------------------------------------------------------------------------
# diarization_stitching
# ---------------------------------------------------------------------------
# Controls how ASR words (with timestamps) are assigned to diarized speaker
# segments (with timestamps and speaker labels). The algorithm iterates words
# in time order and scores candidate segments by overlap. When no segment
# overlaps acceptably, a fallback chain applies: nearest-segment assignment,
# then anonymous-segment creation. After all words are assigned, optional
# post-processing merges and expands segments.
#
# See .research/speaker_segment_assignment.md for the full design rationale.
diarization_stitching:

  # ── Overlap acceptance thresholds ──────────────────────────────────
  # A candidate segment may pass *either* thresholds to count as an
  # "in-range" overlap.  Relaxed defaults accommodate the boundary jitter
  # inherent in both ASR word timestamps and diarization segment edges.

  # Minimum fraction of the word's duration that must be overlapped by
  # the candidate segment. 0.20 = at least 20 %% of the word must fall
  # inside the segment. Prevents "barely touching" overlaps caused by
  # boundary jitter.
  #
  # Allowed values: 0.0–1.0
  # Reasonable default: 0.20
  min_overlap_fraction_word: 0.20

  # Absolute floor: overlaps shorter than this (seconds) are ignored.
  # 20 ms matches typical speech-processing frame sizes (~25 ms); overlaps
  # below one frame are not acoustically meaningful.
  #
  # Allowed values: >= 0.0 (seconds)
  # Reasonable default: 0.02
  min_overlap_seconds: 0.02


  # ── Fallback: nearest-segment assignment ─────────────────────────────
  # When no candidate passes the overlap thresholds, the algorithm can
  # assign the word to the closest segment by midpoint distance, as long
  # as the gap between intervals is within max_nearest_distance.

  # Whether to enable nearest-segment fallback.
  #
  # Allowed values: true / false
  # Reasonable default: true
  fill_nearest: true

  # Maximum gap (seconds) between a word and a non-overlapping segment
  # for nearest-assignment to apply. 250 ms is a common tolerance scale
  # in speech scoring; keeps the fallback conservative so it won't jump
  # speakers across long silences.
  #
  # Allowed values: >= 0.0 (seconds)
  # Reasonable default: 0.25
  max_nearest_distance: 0.25


  # ── Fallback: anonymous segments ─────────────────────────────────────
  # If nearest-assignment also fails (or is disabled), words are placed
  # into auto-created "anonymous" segments so that every word is covered.
  # Consecutive anonymous words close in time are merged into one span.

  # Maximum gap (seconds) between consecutive anonymous words that will
  # be merged into the same anonymous segment. Keeps output clean.
  #
  # Allowed values: >= 0.0 (seconds)
  # Reasonable default: 0.15
  anonymous_join_gap: 0.15


  # ── Post-processing ──────────────────────────────────────────────────

  # Maximum gap (seconds) between same-speaker adjacent segments that will be
  # merged.
  #
  # Allowed values: >= 0.0 (seconds)
  # Reasonable default: 0.20
  merge_gap_seconds: 0.20

  # Maximum gap (seconds) between an unfinished speech clip (not marked
  # as an end-of-turn) and a following clip with the same speaker, for
  # them to be merged. Helps preserve conversational flow by avoiding
  # artificial breaks in ongoing speech, while respecting turn boundaries.
  #
  # Allowed values: >= 0.0 (seconds)
  # Reasonable default: 2.0
  unfinished_clip_merge_max_length: 2.0

  # Maximum gap (seconds) between two clips with the same identified
  # speaker that can still be merged into a single clip during identity
  # stitching. Larger values allow merging across longer pauses; smaller
  # values keep clips separate when the same speaker resumes after silence.
  #
  # Allowed values: >= 0.0 (seconds)
  # Reasonable default: 10.0
  identity_stitching_max_gap: 10.0

  # Minimum cosine similarity (0.0–1.0) between two clip embeddings for
  # them to be considered the same speaker during identity stitching.
  # Lower values accept weaker matches; higher values require stronger
  # acoustic similarity before merging.
  #
  # Allowed values: 0.0–1.0
  # Reasonable default: 0.65
  # Reasonable range: 0.50–0.85
  identity_similarity_threshold: 0.65

  # Widen each segment's time boundaries to fully contain its assigned
  # words. Useful for UI rendering where words must not extend beyond
  # their parent segment, but reduces diarization boundary fidelity.
  #
  # Allowed values: true / false
  # Reasonable default: false
  expand_segments_to_fit_words: false

  # Cap on how far (seconds) a segment boundary may be expanded.
  #
  # Allowed values: >= 0.0 (seconds)
  # Reasonable default: 300
  expansion_limit_seconds: 300


  # ── Candidate scoring ────────────────────────────────────────────────

  # How to rank candidate segments that overlap a word. Each mode scores
  # by its primary metric first; ties are broken by midpoint distance
  # (word midpoint vs. segment midpoint).
  #
  # Allowed values:
  #   overlap_seconds_then_midpoint          — rank by raw overlap seconds
  #   overlap_fraction_word_then_midpoint    — rank by overlap / word duration
  #   iou_then_midpoint                      — rank by intersection-over-union
  #
  # Reasonable default: overlap_seconds_then_midpoint
  scoring_mode: overlap_seconds_then_midpoint

  # When two candidates score identically, prefer the shorter segment.
  # Avoids bias toward long segments that span many words.
  #
  # Allowed values: true / false
  # Reasonable default: true
  prefer_shorter_on_tie: true

  # ── Backchannel detection ────────────────────────────────────────────
  # Backchannel utterances are short, reactive sounds one speaker makes
  # while another is talking — "mm-hmm", "right", "yeah", "uh-huh".
  # They are not independent turns; they acknowledge or encourage the
  # primary speaker without interrupting their turn. All three thresholds
  # below must be satisfied simultaneously for a clip to be treated as a
  # backchannel.

  # Maximum duration (seconds) a clip may be to still qualify as a
  # backchannel. Short utterances are candidates; longer ones are almost
  # certainly independent contributions rather than backchannels.
  #
  # Allowed values: >= 0.0 (seconds)
  # Reasonable default: 0.75
  max_backchannel_duration: 0.75

  # Maximum gap (seconds) between a clip and the clip that precedes it
  # for the clip to be considered a backchannel. Backchannels typically
  # occur during or immediately after another speaker's speech. If the
  # prior clip ended a long time ago the short utterance is more likely
  # a new turn opener than a reactive backchannel.
  #
  # Allowed values: >= 0.0 (seconds)
  # Reasonable default: 0.25
  max_backchannel_prior_gap: 0.25

  # Maximum gap (seconds) between a clip and the clip that follows it
  # for the clip to be considered a backchannel. If the next speech
  # arrives after a long silence the utterance probably stands alone;
  # true backchannels are surrounded by active conversation.
  #
  # Allowed values: >= 0.0 (seconds)
  # Reasonable default: 1.0
  max_backchannel_next_gap: 1.0


  # ── Turn detection ──────────────────────────────────────────────────

  # Probability threshold for classifying a speech clip as the end of a
  # conversational turn.  A clip whose AI-model turn-end probability meets
  # or exceeds this value is flagged as a turn boundary.
  #
  # Allowed values: 0.0 to 1.0
  # Reasonable default: 0.5
  turn_end_probability_threshold: 0.8

  # ── Numeric tolerance ────────────────────────────────────────────────

  # Small value used when comparing floating-point time boundaries to
  # avoid edge cases from imprecision and quantization.
  #
  # Allowed values: >= 0.0
  # Reasonable default: 0.000001
  epsilon: 0.000001
"""


@app.command("generate-sample-settings")
def generate_sample_settings() -> None:
    """Generate a well-documented sample settings.yaml in the data directory."""
    console = Console()
    target = common_paths.data_dir() / "settings.yaml"

    common_paths.ensure_directory(common_paths.data_dir())
    target.write_text(_SAMPLE_SETTINGS, encoding="utf-8")
    console.print(f"[green]Sample settings written to {target}[/green]")
    console.print("[dim]Edit the file to match your session before running other commands.[/dim]")


@app.command("register-speakers")
def register_speakers() -> None:
    """Register all speakers from voice_samples directory into registered_speakers.yaml."""
    logger: LoggingProtocol = create_logger()
    RegisterSpeakersCommand().execute(logger)


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
