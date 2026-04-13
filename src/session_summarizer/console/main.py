from __future__ import annotations

import random
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
from ..commands.clean_audio import CleanAudioCommand
from ..commands.clean_session import CleanSessionCommand
from ..commands.clear_logs import ClearLogsCommand
from ..commands.compare_fulltext import CompareFullTextCommand
from ..commands.compute_segments import ComputeSegmentsCommand
from ..commands.create_speaker_clips import CreateSpeakerClipsCommand
from ..commands.diarizationlm_command import DiarizationLMCommand
from ..commands.diarize_audio import DiarizeAudioCommand
from ..commands.identify_speakers import IdentifySpeakersCommand
from ..commands.mark_backchannels import MarkBackchannelsCommand
from ..commands.merge_speaker_clips import MergeSpeakerClipsCommand
from ..commands.register_speakers import RegisterSpeakersCommand
from ..commands.remove_outlier_speaker_clips import RemoveOutlierSpeakerClipsCommand
from ..commands.score_confidence import ScoreConfidenceCommand
from ..commands.stitch_identities import StitichIdentitiesCommand
from ..commands.test_command import TestCommand
from ..commands.transcribe_audio import TranscribeAudioCommand
from ..commands.validate_diarization import ValidateDiarizationCommand
from ..commands.validate_transcribers import ValidateTranscribersCommand
from ..logging import CompositeLogger, FileLogger, RichConsoleLogger
from ..protocols import LoggingProtocol
from ..settings.session_settings import SessionSettings
from ..utils import flush_gpu_memory
from ..utils.logging_config import configure_logging
from .console_validation import _validate_directory_exists

load_dotenv()
configure_logging()

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
# A list of player names present in the session. This drives diarization
# (number of speakers). Names must match entries in registered_speakers.yaml
# and must be non-empty strings.
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
# adventure_settings  (REQUIRED)
# ---------------------------------------------------------------------------
# Adventure-specific metadata used for transcript labelling and context.
#
# pcs:
#   A mapping of player name to character name for all PCs in the adventure.
#   Player names must match entries in registered_speakers.yaml. Both player
#   names and character names must be non-empty strings.
#
# glossary:
#   A list of proper nouns (places, NPCs, items, factions, spells, etc.)
#   that may appear in the session transcript. Each entry has a required
#   name and an optional description for additional context.
#
# Example:
#   adventure_settings:
#     pcs:
#       Alice: Aethon the Bold
#       Bob: Rogdar
#       Charlie: Sylvara
#     glossary:
#       - name: Thornhaven
#         description: "The ruined city the party is currently exploring"
#       - name: Dragonbane
adventure_settings:
  pcs:
    Speaker1: Character1
    Speaker2: Character2
  glossary:
    - name: ExampleProperNoun
      description: "Optional description of this term"


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
# backchannel_marked_path  (REQUIRED)
# ---------------------------------------------------------------------------
# Path to the SpeechClipSet JSON file with IS_BACKCHANNEL flags applied.
# Written by the mark-backchannels command and read by the punctuate-text
# command. Relative paths are resolved from this file's directory.
#
# Default: backchannel_marked.json
#
# Example:
#   backchannel_marked_path: backchannel_marked.json
backchannel_marked_path: backchannel_marked.json

# ---------------------------------------------------------------------------
# diarizationlm_processed_path  (REQUIRED)
# ---------------------------------------------------------------------------
# Path to the SpeechClipSet JSON file after DiarizationLM post-processing.
# DiarizationLM uses a fine-tuned LLM to correct speaker attribution errors
# in the diarized transcript. Written by the diarizationlm command.
# Relative paths are resolved from this file's directory.
#
# Default: diarizationlm_processed.json
#
# Example:
#   diarizationlm_processed_path: diarizationlm_processed.json
diarizationlm_processed_path: diarizationlm_processed.json

# ---------------------------------------------------------------------------
# indeterminate_speakers_path  (REQUIRED)
# ---------------------------------------------------------------------------
# Path to the SpeechClipSet JSON file where speakers that could not be
# identified are assigned an indeterminate label. Written after the
# indeterminate-speaker assignment step. Relative paths are resolved from
# this file's directory.
#
# Default: indeterminate_speakers.json
#
# Example:
#   indeterminate_speakers_path: indeterminate_speakers.json
indeterminate_speakers_path: indeterminate_speakers.json

# ---------------------------------------------------------------------------
# dangling_sentence_fix_path  (REQUIRED)
# ---------------------------------------------------------------------------
# Path to the SpeechClipSet JSON file written after dangling sentences have
# been fixed. A "dangling sentence" is a speech clip whose transcript ends
# mid-sentence; this step merges or extends such clips so every clip ends on
# a sentence boundary.
# Relative paths are resolved from this file's directory.
#
# Default: dangling_sentence_fix.json
#
# Example:
#   dangling_sentence_fix_path: dangling_sentence_fix.json
dangling_sentence_fix_path: dangling_sentence_fix.json

# ---------------------------------------------------------------------------
# punctuated_text_path  (REQUIRED)
# ---------------------------------------------------------------------------
# Path to the SpeechClipSet JSON file written after punctuation and
# capitalisation have been restored to clip transcripts by the punctuate-text
# command. Relative paths are resolved from this file's directory.
#
# Default: punctuated_text.json
#
# Example:
#   punctuated_text_path: punctuated_text.json
punctuated_text_path: punctuated_text.json

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
# speaker_identity_assignment_threshold  (REQUIRED)
# ---------------------------------------------------------------------------
# Minimum cosine similarity score required for a speaker embedding match to
# result in an actual speaker identity being assigned to a clip. Clips whose
# best match falls below this threshold are left as indeterminate and written
# to indeterminate_speakers_path for manual review.
#
# Allowed values: 0.0–1.0  (cosine similarity; higher = stricter assignment)
#
# Default: 0.70
# Reasonable range: 0.55–0.85
#   Lower values assign an identity to more clips (may mislabel ambiguous clips).
#   Higher values leave more clips indeterminate (safer but reduces coverage).
#
# Example:
#   speaker_identity_assignment_threshold: 0.08
speaker_identity_assignment_threshold: 0.08


# ---------------------------------------------------------------------------
# speaker_clip_lead_in / speaker_clip_lead_out  (REQUIRED)
# ---------------------------------------------------------------------------
# Seconds of audio to include before/after each speech clip when creating
# individual speaker audio files via the create-speaker-clips command.
# Padding is faded in/out to avoid hard audio edges.
#
# Allowed values: >= 0.0 (seconds)
# Reasonable default: 0.25
speaker_clip_lead_in: 0.25
speaker_clip_lead_out: 0.25


# ---------------------------------------------------------------------------
# speaker_clip_minimum_similarity_residual  (REQUIRED)
# ---------------------------------------------------------------------------
# Minimum cosine similarity residual a clip must have to be included as a
# speaker sample during clip selection. The residual is the difference between
# a clip's best-match similarity and the mean similarity across all speakers.
# Clips with a low residual are ambiguous between speakers and are excluded.
#
# Allowed values: 0.0–1.0
# Default: 0.2
# Reasonable range: 0.05–0.5
#   Lower values include more clips (may admit ambiguous clips).
#   Higher values admit only clear, unambiguous clips.
#
# Example:
#   speaker_clip_minimum_similarity_residual: 0.2
speaker_clip_minimum_similarity_residual: 0.2


# ---------------------------------------------------------------------------
# minimum_speaker_clip_duration  (REQUIRED)
# ---------------------------------------------------------------------------
# Target minimum duration (seconds) for speaker clip samples. When building
# a set of clips for a speaker, the system iteratively merges the shortest
# clips with their neighbours until no clip falls below this threshold.
#
# Allowed values: >= 0.0 (seconds)
# Default: 2.0
# Reasonable range: 0.5–5.0
#
# Example:
#   minimum_speaker_clip_duration: 2.0
minimum_speaker_clip_duration: 2.25


# ---------------------------------------------------------------------------
# min_speaker_similarity  (REQUIRED)
# ---------------------------------------------------------------------------
# Minimum cosine similarity (0.0–1.0) a speaker clip must have to the group
# centroid to be kept. Clips below this threshold are iteratively removed as
# outliers, starting with the worst. The centroid is recomputed after each
# removal.
#
# Allowed values: 0.0–1.0
# Default: 0.75
# Reasonable range: 0.70–0.85
#
# Example:
#   min_speaker_similarity: 0.6
min_speaker_similarity: 0.6


# ---------------------------------------------------------------------------
# speaker_clip_gap_length  (OPTIONAL — default: 0.5)
# ---------------------------------------------------------------------------
# Seconds of silence inserted between adjacent clips when combining or
# merging speaker audio files. This gap appears in two places:
#
#   1. When individual speaker clips are concatenated into a single combined
#      WAV during speaker registration (register-speakers command).
#   2. When short clips are merged together to meet the
#      minimum_speaker_clip_duration threshold (merge-speaker-clips and
#      remove-outlier-speaker-clips commands).
#
# A small gap prevents the tail of one utterance from bleeding into the head
# of the next, which improves speaker-embedding quality.
#
# Allowed values: >= 0.0 (seconds). Set to 0.0 for no gap.
# Reasonable range: 0.25–1.0
#
# Example:
#   speaker_clip_gap_length: 0.5
speaker_clip_gap_length: 0.5


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

  # ── Identity-based backchannel detection ────────────────────────────
  # Same concept as backchannel detection above, but applied during
  # identity stitching where speaker labels come from embedding
  # similarity rather than diarization labels.  All three thresholds
  # must be satisfied simultaneously.

  # Maximum duration (seconds) a clip may be to still qualify as a
  # backchannel during identity stitching.
  #
  # Allowed values: >= 0.0 (seconds)
  # Reasonable default: 3.0
  max_identity_backchannel_duration: 3.0

  # Maximum gap (seconds) between a clip and its predecessor for the
  # clip to be considered a backchannel during identity stitching.
  #
  # Allowed values: >= 0.0 (seconds)
  # Reasonable default: 0.5
  max_identity_backchannel_prior_gap: 0.75

  # Maximum gap (seconds) between a clip and its successor for the
  # clip to be considered a backchannel during identity stitching.
  #
  # Allowed values: >= 0.0 (seconds)
  # Reasonable default: 2.0
  max_identity_backchannel_next_gap: 3.0


  # ── Turn detection ──────────────────────────────────────────────────

  # Probability threshold for classifying a speech clip as the end of a
  # conversational turn.  A clip whose AI-model turn-end probability meets
  # or exceeds this value is flagged as a turn boundary.
  #
  # Allowed values: 0.0 to 1.0
  # Reasonable default: 0.5
  turn_end_probability_threshold: 0.8

  # Clips shorter than this duration are merged into the closest adjacent
  # clip rather than kept as standalone segments. Eliminates very short
  # fragments that typically result from diarization jitter or brief
  # silence mis-attribution.
  #
  # Allowed values: >= 0.0 (seconds)
  # Reasonable default: 0.5
  # Reasonable range: 0.0–2.0
  #
  # Example:
  #   tiny_clip_threshold: 0.5
  tiny_clip_threshold: 0.1



# ---------------------------------------------------------------------------
# epsilon  (REQUIRED)
# ---------------------------------------------------------------------------
# Small floating-point tolerance used when comparing time boundaries to
# avoid edge cases from imprecision and quantization.
#
# Allowed values: >= 0.0
# Reasonable default: 0.000001
#
# Example:
#   epsilon: 0.000001
epsilon: 0.000001


# ---------------------------------------------------------------------------
# seed  (REQUIRED)
# ---------------------------------------------------------------------------
# Random seed for reproducible model inference across all frameworks
# (Python random, NumPy, PyTorch). Set to any integer for deterministic
# results; change to get a different random sequence.
#
# Allowed values: any integer
# Reasonable default: 42
#
# Example:
#   seed: 43
seed: 43
"""


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


@app.command("add-embeddings")
def add_embeddings(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to process"),
) -> None:
    """Generate speaker embeddings for each speech clip and save to disk."""
    confirm_session(session)
    _set_seed(session)
    logger: LoggingProtocol = create_logger()
    command: AddEmbeddingsCommand = AddEmbeddingsCommand(session, force=True)
    command.execute(logger)


@app.command("align-transcription")
def align_transcription(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to transcribe"),
) -> None:
    confirm_session(session)
    _set_seed(session)
    logger: LoggingProtocol = create_logger()
    command: AlignTranscriptCommand = AlignTranscriptCommand(session, force=True)
    command.execute(logger)


# @app.command("apply-first-stitching")
# def apply_first_stitching(
#     session: str = typer.Option(..., "--session", "-s", help="ID of the session to process"),
# ) -> None:
#     """Score each speech clip with end-of-turn probability and set the END_OF_TURN flag."""
#     confirm_session(session)
#     _set_seed(session)
#     logger: LoggingProtocol = create_logger()
#     command: FirstStitchClipsCommand = FirstStitchClipsCommand(session, force=True)
#     command.execute(logger)


@app.command("apply-identity-stitching")
def apply_identity_stitiching(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to process"),
) -> None:
    """Score each speech clip with end-of-turn probability and set the END_OF_TURN flag."""
    confirm_session(session)
    _set_seed(session)
    logger: LoggingProtocol = create_logger()
    command: StitichIdentitiesCommand = StitichIdentitiesCommand(session, force=True)
    command.execute(logger)


@app.command("diarizationlm")
def diarizationlm(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to process"),
) -> None:
    """Post-process diarization with DiarizationLM to correct speaker attribution errors."""
    confirm_session(session)
    _set_seed(session)
    logger: LoggingProtocol = create_logger()
    command: DiarizationLMCommand = DiarizationLMCommand(session, force=True)
    command.execute(logger)


@app.command("clean-audio")
def clean_audio(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to clean"),
) -> None:
    confirm_session(session)
    _set_seed(session)
    logger: LoggingProtocol = create_logger()
    command: CleanAudioCommand = CleanAudioCommand(session, force=True)
    command.execute(logger)


@app.command("clean-session")
def clean_session(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to clean"),
) -> None:
    """Delete all generated files in a session folder, keeping settings.yaml and the original audio."""
    confirm_session(session)
    _set_seed(session)
    logger: LoggingProtocol = create_logger()
    command: CleanSessionCommand = CleanSessionCommand(session, force=True)
    command.execute(logger)


@app.command("clear-logs")
def clear_logs() -> None:
    """Delete all files in the logs directory."""
    logger: LoggingProtocol = create_logger()
    ClearLogsCommand().execute(logger)


@app.command("compare-texts")
def compare_texts(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to transcribe"),
) -> None:
    confirm_session(session)
    _set_seed(session)
    logger: LoggingProtocol = create_logger()
    command: CompareFullTextCommand = CompareFullTextCommand(session, force=True)
    command.execute(logger)


@app.command("compute-vad-segments")
def compute_vad_segments(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to segment"),
) -> None:
    """Run VAD on cleaned audio and compute optimal cut points for chunked processing."""
    confirm_session(session)
    _set_seed(session)
    logger: LoggingProtocol = create_logger()
    command: ComputeSegmentsCommand = ComputeSegmentsCommand(session, force=True)
    command.execute(logger)


@app.command("create-speaker-clips")
def create_speaker_clips(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to process"),
    temp_folder: str = typer.Option(
        ..., "--temp-folder", "-t", help="Name of temp folder inside voice samples to hold output"
    ),
) -> None:
    """Save each identified speaker clip as an individual audio file."""
    confirm_session(session)
    _set_seed(session)
    logger: LoggingProtocol = create_logger()
    command: CreateSpeakerClipsCommand = CreateSpeakerClipsCommand(
        session, use_multi_speaker_clips=False, temp_folder=Path(temp_folder)
    )
    command.execute(logger)


@app.command("merge-speaker-clips")
def merge_speaker_clips(
    speaker: str = typer.Option(
        ..., "--speaker", "-s", help="Speaker label — must match a subdirectory in voice_samples/"
    ),
    output_folder: str = typer.Option(..., "--output-folder", "-o", help="Folder to write the merged clips into"),
) -> None:
    """Merge short clips for a speaker until all are >= minimum_speaker_clip_duration."""
    logger: LoggingProtocol = create_logger()
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
    """Merge short clips for a speaker until all are >= minimum_speaker_clip_duration."""
    logger: LoggingProtocol = create_logger()
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
    logger: LoggingProtocol = create_logger()
    command: DiarizeAudioCommand = DiarizeAudioCommand(session, force=True)
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
    logger: LoggingProtocol = create_logger()
    command: IdentifySpeakersCommand = IdentifySpeakersCommand(session, force=True)
    command.execute(logger)


@app.command("mark-backchannels")
def mark_backchannels(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to process"),
) -> None:
    """Mark short acknowledgement clips as backchannels."""
    confirm_session(session)
    _set_seed(session)
    logger: LoggingProtocol = create_logger()
    command: MarkBackchannelsCommand = MarkBackchannelsCommand(session, force=True)
    command.execute(logger)


@app.command("punctuate-text")
def punctuate_text(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to process"),
) -> None:
    """Identify speakers in each speech clip by comparing embeddings to registered attendees."""
    confirm_session(session)
    _set_seed(session)
    logger: LoggingProtocol = create_logger()
    command: PunctuateTextCommand = PunctuateTextCommand(session, force=True)
    command.execute(logger)


@app.command("register-speakers")
def register_speakers() -> None:
    """Merge clips, remove outliers, and register centroid embeddings into registered_speakers.yaml."""
    logger: LoggingProtocol = create_logger()
    RegisterSpeakersCommand().execute(logger)


@app.command("score-confidence")
def score_confidence(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to transcribe"),
) -> None:
    confirm_session(session)
    _set_seed(session)
    logger: LoggingProtocol = create_logger()
    command: ScoreConfidenceCommand = ScoreConfidenceCommand(session, force=True)
    command.execute(logger)


@app.command("test")
def test(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to transcribe"),
) -> None:
    confirm_session(session)
    _set_seed(session)
    logger: LoggingProtocol = create_logger()
    command: TestCommand = TestCommand(session, force=True)
    command.execute(logger)


@app.command("transcribe")
def transcribe(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to transcribe"),
) -> None:
    confirm_session(session)
    _set_seed(session)
    logger: LoggingProtocol = create_logger()
    command: TranscribeAudioCommand = TranscribeAudioCommand(session, force=True)
    command.execute(logger)


# @app.command("update-turn-end")
# def update_turn_end(
#     session: str = typer.Option(..., "--session", "-s", help="ID of the session to process"),
# ) -> None:
#     """Score each speech clip with end-of-turn probability and set the END_OF_TURN flag."""
#     confirm_session(session)
#     _set_seed(session)
#     logger: LoggingProtocol = create_logger()
#     command: UpdateTurnEndCommand = UpdateTurnEndCommand(session, force=True)
#     command.execute(logger)


@app.command("validate-diarization")
def validate_diarization(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to use for validation"),
) -> None:
    """Evaluate diarization quality across pipeline stages and display a metrics comparison table."""
    confirm_session(session)
    _set_seed(session)
    logger: LoggingProtocol = create_logger()
    command: ValidateDiarizationCommand = ValidateDiarizationCommand(session, force=True)
    command.execute(logger)


@app.command("validate-transcribers")
def validate_transcribers(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to use for validation"),
) -> None:
    """Transcribe test audio with every registered transcriber and compare accuracy metrics."""
    confirm_session(session)
    _set_seed(session)
    logger: LoggingProtocol = create_logger()
    command: ValidateTranscribersCommand = ValidateTranscribersCommand(session, force=True)
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
