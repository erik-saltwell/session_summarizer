from __future__ import annotations

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

  # Path to the SpeechClipSet JSON with unidentifiable speakers assigned
  # an indeterminate label.
  # Written by: indeterminate_speaker_assignment.
  indeterminate_speakers: indeterminate_speakers.json

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
# device  (REQUIRED)
# ---------------------------------------------------------------------------
# Compute device for model inference. Allowed values:
#   cuda  — use the GPU (requires a CUDA-capable NVIDIA GPU)
#   cpu   — use the CPU (much slower, but works everywhere)
# Used by: all commands that load ML models.
device: cuda


# ---------------------------------------------------------------------------
# eleven_labs  (REQUIRED)
# ---------------------------------------------------------------------------
# Settings for the ElevenLabs Scribe v2 diarization path. Used by the
# diarize-audio-eleven-labs command, which calls ElevenLabs' speech-to-text
# API with diarization enabled and builds a SpeechClipSet directly from the
# returned word-level speaker assignments. This path bypasses the DiariZen
# diarizer and the word-to-segment stitching pipeline entirely — speaker
# assignments come straight from ElevenLabs at word granularity.
# Used by: helpers/audio_diarizaer_eleven_labs.py.
eleven_labs:

  # Maximum gap (seconds) between two consecutive same-speaker words that
  # will still be kept inside a single SpeechClip. Pauses longer than this
  # split the clip in two even when the speaker_id is unchanged. Without
  # this, one speaker's extended monologue (e.g. a Game Master narration
  # block between player questions) would collapse into a single
  # multi-minute clip and frustrate downstream per-utterance steps such as
  # backchannel detection and identity stitching. A speaker change always
  # splits, regardless of this value.
  # Used by: diarize_audio_eleven_labs to decide intra-speaker clip
  #          boundaries when grouping ElevenLabs words into clips.
  # Allowed values: >= 0.0 (seconds). Reasonable default: 2.0
  clip_gap_seconds: 2.0


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


def get_sample_settings() -> str:
    return _SAMPLE_SETTINGS
