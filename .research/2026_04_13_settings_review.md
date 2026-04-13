# Settings Review — 2026-04-13

Comprehensive audit of every field in `SessionSettings` and its nested objects
(`VadSettings`, `DiarizationStitchingSettings`), including which commands use
each setting and what the code actually does with the value.

---

## Table of Contents

- [SessionSettings — Top-Level Fields](#sessionsettings--top-level-fields)
  - [attendees](#attendees)
  - [adventure_settings](#adventure_settings)
  - [audio_file](#audio_file)
  - [cleaned_audio_file](#cleaned_audio_file)
  - [transcript_file](#transcript_file)
  - [aligned_transcript_path](#aligned_transcript_path)
  - [confidence_transcript_path](#confidence_transcript_path)
  - [base_diarized_path](#base_diarized_path)
  - [speech_clips_with_embedding](#speech_clips_with_embedding)
  - [identified_speaker_path](#identified_speaker_path)
  - [turn_end_updated_path](#turn_end_updated_path)
  - [first_stitched_path](#first_stitched_path)
  - [identity_stitched_path](#identity_stitched_path)
  - [diarizationlm_processed_path](#diarizationlm_processed_path)
  - [indeterminate_speakers_path](#indeterminate_speakers_path)
  - [dangling_sentence_fix_path](#dangling_sentence_fix_path)
  - [punctuated_text_path](#punctuated_text_path)
  - [device](#device)
  - [segments_path](#segments_path)
  - [min_segment_length_short](#min_segment_length_short)
  - [max_segment_length_short](#max_segment_length_short)
  - [min_segment_length_long](#min_segment_length_long)
  - [max_segment_length_long](#max_segment_length_long)
  - [high_confidence_similarity_threshold](#high_confidence_similarity_threshold)
  - [speaker_identity_assignment_threshold](#speaker_identity_assignment_threshold)
  - [vad](#vad)
  - [speaker_clip_lead_in](#speaker_clip_lead_in)
  - [speaker_clip_lead_out](#speaker_clip_lead_out)
  - [speaker_clip_minimum_similarity_residual](#speaker_clip_minimum_similarity_residual)
  - [minimum_speaker_clip_duration](#minimum_speaker_clip_duration)
  - [min_speaker_similarity](#min_speaker_similarity)
  - [speaker_clip_gap_length](#speaker_clip_gap_length)
  - [diarization_stitching](#diarization_stitching)
  - [epsilon](#epsilon)
  - [seed](#seed)
  - [number_of_speakers (property)](#number_of_speakers-property)
- [DiarizationStitchingSettings Fields](#diarizationstitchingsettings-fields)
- [VadSettings Fields](#vadsettings-fields)
- [Summary](#summary)

---

## SessionSettings — Top-Level Fields

### attendees

| Attribute | Value |
|---|---|
| **Type** | `list[str]` (min_length=1) |
| **Description** | List of player names present in the session |
| **Default** | None (required) |

**Used by commands:**
- `IdentifySpeakersCommand` (`commands/identify_speakers.py:36,44`) — validates list is non-empty, checks all names exist in registered speakers file

**Used by helpers:**
- `helpers/speaker_identifier.py:32` — creates a set of attendee names to filter registered speaker embeddings; only embeddings for attendees are used during cosine-similarity matching

**What code does:** Determines which registered speakers participate in the session. The identify-speakers step filters the global registered-speakers file to only match against attendees. Also drives a validation check that all attendee names exist in the registered speakers file.

---

### adventure_settings

| Attribute | Value |
|---|---|
| **Type** | `AdventureSettings` (contains `pcs: dict[str,str]`, `glossary: list[GlossaryEntry]`) |
| **Description** | Adventure-specific metadata: PC roster and glossary of proper nouns |
| **Default** | None (required) |

**Used by commands:** NONE

**Used by helpers:** NONE

**What code does:** The `AdventureSettings` class has a `to_prompt_fragment()` method (session_settings.py:52) that formats PCs and glossary into an XML-tagged prompt fragment. However, `to_prompt_fragment()` is **never called** anywhere in the codebase. The entire `adventure_settings` field, including `pcs` and `glossary`, is **defined but unused**.

**Note:** There is a bug in `to_prompt_fragment()` — line 67 constructs a string with the glossary description but does not concatenate it to `result` (the f-string is a bare expression, not `result +=`).

---

### audio_file

| Attribute | Value |
|---|---|
| **Type** | `Path` |
| **Description** | Path to the audio file for the session |
| **Default** | None (required) |

**Used by commands:**
- `CleanAudioCommand` (`commands/clean_audio.py:19`) — input file, reads raw audio for noise cleaning
- `CleanSessionCommand` (`commands/clean_session.py:33`) — used to resolve absolute path during session cleanup

**Used by helpers:**
- `helpers/audio_cleaner.py:16` — reads the original audio file for cleaning

**What code does:** The starting-point audio file for all processing. Only directly consumed by the clean-audio step, which reads it and produces `cleaned_audio_file`.

---

### cleaned_audio_file

| Attribute | Value |
|---|---|
| **Type** | `Path` |
| **Description** | Path to the cleaned audio file |
| **Default** | None (required) |

**Used by commands:**
- `CleanAudioCommand` (`commands/clean_audio.py:20`) — output path
- `ComputeSegmentsCommand` (`commands/compute_segments.py:21`) — input
- `TranscribeAudioCommand` (`commands/transcribe_audio.py:24`) — input
- `DiarizeAudioCommand` (`commands/diarize_audio.py:27`) — input
- `AddEmbeddingsCommand` (`commands/add_embeddings.py:24`) — input
- `ValidateTranscribersCommand` (`commands/validate_transcribers.py:59`) — input
- `CreateSpeakerClipsCommand` (`commands/create_speaker_clips.py:28,33`) — input

**Used by helpers:**
- `helpers/audio_cleaner.py:18` — output path
- `helpers/confidence_scorer.py:69` — loads audio for confidence scoring
- `helpers/update_turn_end.py:24` — loads audio for turn-end prediction
- `helpers/audio_transcriber.py:22` — loads audio for transcription
- `helpers/audio_diarizer.py:24` — loads audio for diarization
- `helpers/audio_segmenter.py:42` — loads audio for VAD
- `helpers/transcript_aligner.py:78` — loads audio for alignment
- `helpers/add_embeddings.py:27` — loads audio for embedding extraction

**What code does:** The central audio file consumed by virtually all processing steps. Produced by `clean_audio` and used by transcription, alignment, confidence scoring, diarization, embedding extraction, turn-end detection, and speaker clip creation.

---

### transcript_file

| Attribute | Value |
|---|---|
| **Type** | `Path` |
| **Description** | Path to the transcript JSON file |
| **Default** | None (required) |

**Used by commands:**
- `TranscribeAudioCommand` (`commands/transcribe_audio.py:25,32,33`) — output path, writes transcription JSON
- `AlignTranscriptCommand` (`commands/align_transcript.py:25`) — input
- `CompareFulltextCommand` (`commands/compare_fulltext.py:49`) — lists available transcript files

**What code does:** Stores initial ASR transcription output from CanaryQwen. Read by align-transcript for word-level alignment. Intermediate file in the transcription -> alignment -> confidence pipeline.

---

### aligned_transcript_path

| Attribute | Value |
|---|---|
| **Type** | `Path` |
| **Description** | Path to the word-aligned transcript JSON |
| **Default** | None (required) |

**Used by commands:**
- `AlignTranscriptCommand` (`commands/align_transcript.py:28`) — output path
- `ScoreConfidenceCommand` (`commands/score_confidence.py:24`) — input
- `CompareFulltextCommand` (`commands/compare_fulltext.py:50`) — lists available files

**What code does:** Stores word-aligned transcription with per-word start/end timestamps from CTC forced alignment. Created by align-transcript, consumed by score-confidence.

---

### confidence_transcript_path

| Attribute | Value |
|---|---|
| **Type** | `Path` |
| **Description** | Path to transcript JSON with per-word confidence scores |
| **Default** | None (required) |

**Used by commands:**
- `ScoreConfidenceCommand` (`commands/score_confidence.py:26`) — output path
- `DiarizeAudioCommand` (`commands/diarize_audio.py:25`) — input
- `CompareFulltextCommand` (`commands/compare_fulltext.py:51`) — lists available files

**What code does:** Stores transcription with per-word confidence scores (0.0-1.0). Created by score-confidence, consumed by diarize-audio which uses the aligned words with confidence annotations for word-to-segment assignment.

---

### base_diarized_path

| Attribute | Value |
|---|---|
| **Type** | `Path` |
| **Description** | Path to diarized segments JSON |
| **Default** | None (required) |

**Used by commands:**
- `DiarizeAudioCommand` (`commands/diarize_audio.py:28`) — output path
- `DiarizationLMCommand` (`commands/diarizationlm_command.py:21`) — input
- `CompareFulltextCommand` (`commands/compare_fulltext.py:52`) — lists files
- `ValidateDiarizationCommand` (`commands/validate_diarization.py:28`) — used in validation

**What code does:** Stores initial SpeechClipSet output from the diarization pipeline (DiariZen diarizer + word assignment via speech_clip_factory). Consumed by DiarizationLM for refinement.

---

### speech_clips_with_embedding

| Attribute | Value |
|---|---|
| **Type** | `Path` |
| **Description** | Path to SpeechClipSet JSON with speaker embeddings |
| **Default** | None (required) |

**Used by commands:**
- `AddEmbeddingsCommand` (`commands/add_embeddings.py:26`) — output path
- `IdentifySpeakersCommand` (`commands/identify_speakers.py:31`) — input
- `CompareFulltextCommand` (`commands/compare_fulltext.py:54`) — lists files
- `ValidateDiarizationCommand` (`commands/validate_diarization.py:30`) — validation

**What code does:** Stores SpeechClipSet with speaker embedding vectors attached to each clip. Created by add-embeddings, consumed by identify-speakers for cosine-similarity matching against registered speakers.

---

### identified_speaker_path

| Attribute | Value |
|---|---|
| **Type** | `Path` |
| **Description** | Path to SpeechClipSet JSON with identified speakers |
| **Default** | None (required) |

**Used by commands:**
- `IdentifySpeakersCommand` (`commands/identify_speakers.py:32`) — output path
- `StitichIdentitiesCommand` (`commands/stitch_identities.py:22`) — input
- `CreateSpeakerClipsCommand` (`commands/create_speaker_clips.py:27`) — input
- `CompareFulltextCommand` (`commands/compare_fulltext.py:55`) — lists files
- `ValidateDiarizationCommand` (`commands/validate_diarization.py:31`) — validation

**What code does:** Stores SpeechClipSet with speaker identity (name), cosine_similarity, and similarity_residual assigned to each clip. Created by identify-speakers. Consumed by stitch-identities and create-speaker-clips.

---

### turn_end_updated_path

| Attribute | Value |
|---|---|
| **Type** | `Path` |
| **Description** | Path to SpeechClipSet JSON with END_OF_TURN flags |
| **Default** | None (required) |

**Used by commands:**
- `UpdateTurnEndCommand` (`commands/update_turn_end.py:22`) — output path
- `FirstStitchClipsCommand` (`commands/first_stitch_clips.py:22`) — input

**What code does:** Stores SpeechClipSet with END_OF_TURN flags set based on the Smart Turn model's turn-end probability vs `turn_end_probability_threshold`. Created by update-turn-end, consumed by first-stitch-clips.

---

### first_stitched_path

| Attribute | Value |
|---|---|
| **Type** | `Path` |
| **Description** | Path to SpeechClipSet JSON after initial stitching |
| **Default** | None (required) |

**Used by commands:**
- `FirstStitchClipsCommand` (`commands/first_stitch_clips.py:23`) — output path

**What code does:** Stores SpeechClipSet after backchannel merging and unfinished-segment merging. Created by first-stitch-clips. **Note:** This path is written but never read as input by any other command in the current pipeline — it appears to be an intermediate artifact that was part of an older pipeline ordering.

---

### identity_stitched_path

| Attribute | Value |
|---|---|
| **Type** | `Path` |
| **Description** | Path to SpeechClipSet JSON with speakers identified |
| **Default** | None (required) |

**Used by commands:**
- `StitichIdentitiesCommand` (`commands/stitch_identities.py:23`) — output path
- `PunctuateTextCommand` (`commands/punctuate_text.py:21`) — input
- `IndeterminantSpeakerAssignmentCommand` (`commands/indeterminate_speaker_assignment.py:21`) — input
- `TestCommand` (`commands/test_command.py:27,44,60`) — validation
- `ValidateDiarizationCommand` (`commands/validate_diarization.py:32`) — validation

**What code does:** Stores SpeechClipSet after identity-based stitching (merging adjacent clips with same identified speaker within `identity_stitching_max_gap`). Created by stitch-identities, consumed by punctuate-text and indeterminate-speaker-assignment.

---

### diarizationlm_processed_path

| Attribute | Value |
|---|---|
| **Type** | `Path` |
| **Description** | Path to SpeechClipSet JSON after DiarizationLM processing |
| **Default** | None (required) |

**Used by commands:**
- `DiarizationLMCommand` (`commands/diarizationlm_command.py:22`) — output path
- `UpdateTurnEndCommand` (`commands/update_turn_end.py:21`) — input
- `DanglingSentenceFixCommand` (`commands/dangling_sentece_fix.py:21`) — input
- `CompareFulltextCommand` (`commands/compare_fulltext.py:53`) — lists files
- `ValidateDiarizationCommand` (`commands/validate_diarization.py:29`) — validation

**What code does:** Stores SpeechClipSet after DiarizationLM (LLM-based speaker attribution correction). Created by diarizationlm, consumed by update-turn-end and dangling-sentence-fix.

---

### indeterminate_speakers_path

| Attribute | Value |
|---|---|
| **Type** | `Path` |
| **Description** | Path to SpeechClipSet with unidentified speakers assigned |
| **Default** | None (required) |

**Used by commands:**
- `IndeterminantSpeakerAssignmentCommand` (`commands/indeterminate_speaker_assignment.py:22`) — output path
- `ValidateDiarizationCommand` (`commands/validate_diarization.py:33`) — validation

**What code does:** Stores SpeechClipSet where clips with `similarity_residual < speaker_identity_assignment_threshold` have been reassigned to `UNASSIGNED_SPEAKER_NAME`. Created by indeterminate-speaker-assignment. **Note:** This path is written but never read as input by any downstream command — it is a terminal output file.

---

### dangling_sentence_fix_path

| Attribute | Value |
|---|---|
| **Type** | `Path` |
| **Description** | Path to SpeechClipSet with dangling sentences fixed |
| **Default** | None (required) |

**Used by commands:**
- `DanglingSentenceFixCommand` (`commands/dangling_sentece_fix.py:22`) — output path
- `AddEmbeddingsCommand` (`commands/add_embeddings.py:23`) — input

**What code does:** Stores SpeechClipSet with corrected dangling sentences (clips ending mid-sentence are merged/extended). Created by dangling-sentence-fix, consumed by add-embeddings.

---

### punctuated_text_path

| Attribute | Value |
|---|---|
| **Type** | `Path` |
| **Description** | Path to SpeechClipSet with punctuation/capitalisation restored |
| **Default** | None (required) |

**Used by commands:**
- `PunctuateTextCommand` (`commands/punctuate_text.py:22`) — output path

**What code does:** Final SpeechClipSet with punctuation and capitalisation applied to clip transcripts. Created by punctuate-text. Terminal output file — not read as input by any downstream command.

---

### device

| Attribute | Value |
|---|---|
| **Type** | `Literal["cpu", "cuda"]` |
| **Description** | Device for model inference |
| **Default** | None (required) |

**Used by commands:**
- `ValidateTranscribersCommand` (`commands/validate_transcribers.py:76`) — creates transcriber on device

**Used by helpers:**
- `helpers/confidence_scorer.py:75` — creates ParakeetCTC scorer
- `helpers/remove_outlier_speakers.py:20` — creates embeddings factory
- `helpers/audio_transcriber.py:33` — creates CanaryQwen transcriber
- `helpers/add_embeddings.py:34` — creates embeddings factory
- `helpers/audio_segmenter.py:33` — creates NemoVadDetector
- `helpers/diarizationlm_refiner.py:21` — creates DiarizationLM model
- `helpers/update_turn_end.py:27` — creates LocalSmartTurnPredictor
- `helpers/transcript_aligner.py:74` — creates ParakeetCTC aligner

**What code does:** Passed to every neural model constructor to control where tensors are allocated (GPU vs CPU). Affects all model inference steps: VAD, transcription, alignment, confidence scoring, diarization LM, turn detection, speaker embeddings, and outlier removal.

---

### segments_path

| Attribute | Value |
|---|---|
| **Type** | `Path` |
| **Description** | Path to the VAD segments JSON output |
| **Default** | None (required) |

**Used by commands:**
- `ComputeSegmentsCommand` (`commands/compute_segments.py:22`) — output path
- `TranscribeAudioCommand` (`commands/transcribe_audio.py:23`) — input
- `DiarizeAudioCommand` (`commands/diarize_audio.py:26`) — input
- `AddEmbeddingsCommand` (`commands/add_embeddings.py:25`) — input
- `AlignTranscriptCommand` (`commands/align_transcript.py:26`) — input
- `ScoreConfidenceCommand` (`commands/score_confidence.py:25`) — input
- `ValidateTranscribersCommand` (`commands/validate_transcribers.py:60`) — input

**What code does:** Stores VAD-based segment boundaries (silence-aware cut points). Created by compute-segments. The `short` segment set is consumed by transcription and confidence scoring; the `long` segment set is consumed by forced alignment. Diarization and add-embeddings declare `segments_path` as an input dependency, but their current helper code does not read segment chunks directly.

---

### min_segment_length_short

| Attribute | Value |
|---|---|
| **Type** | `float` (seconds) |
| **Description** | Minimum segment length for short VAD-based chunking |
| **Default** | None (required); typical value: 10 |

**Used by commands:** None directly

**Used by helpers:**
- `helpers/audio_segmenter.py:50` — passed to `compute_segments()` as `min_length` when `mode="short"`

**What code does:** In short-chunking mode (used for Canary transcription and confidence scoring), this is the earliest eligible distance from the current segment start where the splitter will consider a silence cut. The splitter does not run a separate "merge short segments" post-pass; a trailing final segment can still be shorter than this. Works as a pair with `max_segment_length_short`. Validated that `min < max` in model validator.

---

### max_segment_length_short

| Attribute | Value |
|---|---|
| **Type** | `float` (seconds) |
| **Description** | Maximum segment length for short VAD-based chunking |
| **Default** | None (required); typical value: 38 |

**Used by commands:** None directly

**Used by helpers:**
- `helpers/audio_segmenter.py:51` — passed to `compute_segments()` as `max_length` when `mode="short"`

**What code does:** In short-chunking mode, no segment exceeds this duration. A hard cut is made if continuous speech runs longer with no silence gap. Set to 38s because Canary processes audio in ~40s internal windows.

---

### min_segment_length_long

| Attribute | Value |
|---|---|
| **Type** | `float` (seconds) |
| **Description** | Minimum segment length for long VAD-based chunking |
| **Default** | None (required); typical value: 120 |

**Used by commands:** None directly

**Used by helpers:**
- `helpers/audio_segmenter.py:54` — passed to `compute_segments()` as `min_length` when `mode="long"`

**What code does:** In long-chunking mode (currently used by forced alignment), this is the earliest eligible distance from the current segment start where the splitter will consider a silence cut. The splitter does not run a separate "merge short segments" post-pass; a trailing final segment can still be shorter than this.

---

### max_segment_length_long

| Attribute | Value |
|---|---|
| **Type** | `float` (seconds) |
| **Description** | Maximum segment length for long VAD-based chunking |
| **Default** | None (required); typical value: 300 |

**Used by commands:** None directly

**Used by helpers:**
- `helpers/audio_segmenter.py:54` — passed to `compute_segments()` as `max_length` when `mode="long"`

**What code does:** In long-chunking mode, no non-final segment exceeds this duration; if no silence gap exists within the allowed window, the splitter force-cuts at `max_segment_length_long`. Tune down if you see CUDA OOM errors in forced alignment, or up if GPU has headroom.

---

### high_confidence_similarity_threshold

| Attribute | Value |
|---|---|
| **Type** | `float` (0.0-1.0) |
| **Description** | Minimum cosine similarity for high-confidence speaker match |
| **Default** | None (required); typical value: 0.88 |

**Used by commands:** NONE

**Used by helpers:** NONE

**What code does:** **DEFINED BUT NEVER USED.** The field is validated in the settings model (0.0-1.0 range check at session_settings.py:321) but is never referenced by any processing code. Appears to be a legacy or planned field.

---

### speaker_identity_assignment_threshold

| Attribute | Value |
|---|---|
| **Type** | `float` (0.0-1.0) |
| **Description** | Minimum cosine similarity to assign a speaker identity |
| **Default** | None (required); typical value: 0.08 |

**Used by commands:** None directly

**Used by helpers:**
- `helpers/indeterminate_speakers.py:16` — in `_should_unassign_speaker()`, clips with `similarity_residual < threshold` are reassigned to `UNASSIGNED_SPEAKER_NAME`

**What code does:** Despite the description saying "cosine similarity score", the code actually compares it against `clip.similarity_residual` (the difference between best-match similarity and mean similarity across all speakers). Clips with residual below this threshold are considered ambiguous and unassigned. **Note:** The description says "similarity score" but the code uses it as a residual threshold — this is misleading.

---

### vad

| Attribute | Value |
|---|---|
| **Type** | `VadSettings` (nested object) |
| **Description** | VAD model and post-processing hyperparameters |
| **Default** | None (required) |

**Used by commands:** None directly (settings are consumed via the nested object)

**Used by helpers:**
- `helpers/audio_segmenter.py:32-39` — all 7 VadSettings fields are unpacked and passed to `NemoVadDetector` constructor

**What code does:** Container for Voice Activity Detection configuration. All fields are passed through to the NeMo VAD model during segment computation. See [VadSettings Fields](#vadsettings-fields) below for per-field details.

---

### speaker_clip_lead_in

| Attribute | Value |
|---|---|
| **Type** | `float` (seconds, >= 0.0) |
| **Description** | Audio padding before each speaker clip |
| **Default** | None (required); typical value: 0.25 |

**Used by commands:**
- `CreateSpeakerClipsCommand` (`commands/create_speaker_clips.py:59`) — passed to `save_segment_as_speaker_audio_clip()`

**What code does:** Seconds of audio included before each clip's start time when extracting individual speaker audio WAV files. Padding is faded in to avoid hard audio edges.

---

### speaker_clip_lead_out

| Attribute | Value |
|---|---|
| **Type** | `float` (seconds, >= 0.0) |
| **Description** | Audio padding after each speaker clip |
| **Default** | None (required); typical value: 0.25 |

**Used by commands:**
- `CreateSpeakerClipsCommand` (`commands/create_speaker_clips.py:60`) — passed to `save_segment_as_speaker_audio_clip()`

**What code does:** Seconds of audio included after each clip's end time when extracting individual speaker audio WAV files. Padding is faded out to avoid hard audio edges.

---

### speaker_clip_minimum_similarity_residual

| Attribute | Value |
|---|---|
| **Type** | `float` (0.0-1.0) |
| **Description** | Minimum similarity residual to include a clip as a speaker sample |
| **Default** | None (required); typical value: 0.2 |

**Used by commands:**
- `CreateSpeakerClipsCommand` (`commands/create_speaker_clips.py:50`) — clips with `similarity_residual < threshold` are skipped (not saved as speaker audio samples)

**What code does:** Filters which identified clips are saved as speaker audio samples during the create-speaker-clips step. Clips with low residuals (ambiguous between speakers) are excluded.

---

### minimum_speaker_clip_duration

| Attribute | Value |
|---|---|
| **Type** | `float` (seconds, >= 0.0) |
| **Description** | Target minimum speaker clip duration |
| **Default** | None (required); typical value: 2.25 |

**Used by commands:**
- `MergeSpeakerClipsCommand` (`commands/merge_speaker_clips.py:27`) — passed to `merge_speaker_clips_to_min_duration()`
- `RegisterSpeakersCommand` (`commands/register_speakers.py:52`) — passed to `merge_speaker_clips_to_min_duration()`

**What code does:** During speaker registration, short audio clips are iteratively merged with neighbours until no clip falls below this duration threshold.

---

### min_speaker_similarity

| Attribute | Value |
|---|---|
| **Type** | `float` (0.0-1.0) |
| **Description** | Minimum cosine similarity to group centroid to keep a clip |
| **Default** | None (required); typical value: 0.6 |

**Used by commands:**
- `RemoveOutlierSpeakerClipsCommand` (`commands/remove_outlier_speaker_clips.py:36`) — logged in info message

**Used by helpers:**
- `helpers/remove_outlier_speakers.py:26,48,50` — iteratively removes the worst clip (lowest similarity to centroid) until all remaining clips meet or exceed this threshold

**What code does:** During speaker registration, clips whose cosine similarity to the group centroid falls below this threshold are removed as outliers. The centroid is recomputed after each removal.

---

### speaker_clip_gap_length

| Attribute | Value |
|---|---|
| **Type** | `float` (seconds, >= 0.0) |
| **Description** | Silence between clips when merging speaker audio |
| **Default** | 0.5 |

**Used by commands:**
- `MergeSpeakerClipsCommand` (`commands/merge_speaker_clips.py:37`) — passed to `merge_speaker_clips_to_min_duration()`
- `RegisterSpeakersCommand` (`commands/register_speakers.py:53`) — passed to `merge_speaker_clips_to_min_duration()`

**What code does:** Seconds of silence inserted between adjacent clips when merging short clips together during speaker registration. Prevents audio bleeding between utterances, improving embedding quality.

---

### diarization_stitching

| Attribute | Value |
|---|---|
| **Type** | `DiarizationStitchingSettings` (nested object) |
| **Description** | Policy knobs for assigning ASR words to diarized speaker segments |
| **Default** | None (required) |

**Used by commands/helpers:** Container — individual fields accessed via `settings.diarization_stitching.<field>`. See [DiarizationStitchingSettings Fields](#diarizationstitchingsettings-fields) below.

---

### epsilon

| Attribute | Value |
|---|---|
| **Type** | `float` (>= 0.0) |
| **Description** | Small floating-point tolerance |
| **Default** | None (required); typical value: 0.000001 |

**Used by commands:**
- `ValidateDiarizationCommand` (`commands/validate_diarization.py:94`) — passed to `evaluate_diarization_result()`

**Used by helpers/core:**
- `diarization/speech_clip_factory.py` (throughout) — used in overlap calculations, boundary comparisons, candidate pool radius, acceptable overlap checks
- `diarization/clip_merger.py:36` — gap distance comparison tolerance
- `diarization/candidate_pool.py:30` — added to search radius
- `diarization/candidate_score.py:35` — minimum meaningful length
- `diarization/anonymous_clips.py:17` — anonymous join gap tolerance
- `helpers/tiny_clip_merger.py:23-24` — gap distance calculations
- `helpers/identity_stitch.py:50,59,86` — gap distance comparisons
- `helpers/first_stitcher.py:44,53,78` — gap distance comparisons
- `helpers/diarizationlm_refiner.py:25` — passed to DiarizationLM processor
- `diarizationlm/clip_set_converter.py:46` — speaker computation
- `diarizationlm/speaker_mapping.py:26` — speaker computation
- `diarizationlm/diarizationlm_processor.py:29` — passed through to all conversion steps
- `processing_results/speech_clip.py:98-123` — gap distance and speaker computation
- `processing_results/speech_clip.py:242-253` — expand bounds comparison
- `processing_results/segment_protocol.py:30-31` — meaningful duration calculation
- `evaluation/evaluate_word_diarization.py:66,83` — WDER evaluation
- `evaluation/evaluate_diatization.py:26,63` — diarization evaluation

**What code does:** Pervasive floating-point tolerance used throughout the system for time boundary comparisons, overlap calculations, gap distance computations, and boundary expansion. Prevents false negatives from floating-point imprecision and quantization.

---

### seed

| Attribute | Value |
|---|---|
| **Type** | `int` |
| **Description** | Random seed for reproducible inference |
| **Default** | None (required); typical value: 43 |

**Used by commands:** None directly

**Used by:**
- `console/main.py:52-59` — `_set_seed()` function sets seeds for Python `random`, NumPy, PyTorch (CPU + CUDA) before most session-scoped CLI command invocations

**What code does:** Sets deterministic random seeds across all frameworks for most session-scoped commands. Commands that do not take a session ID or operate on global speaker clips/settings (for example `clear-logs`, `merge-speaker-clips`, `remove-outlier-speaker-clips`, `register-speakers`, and `generate-sample-settings`) do not call `_set_seed()`.

---

### number_of_speakers (property)

| Attribute | Value |
|---|---|
| **Type** | `int` (derived, `= len(self.attendees)`) |
| **Description** | Number of speakers, derived from attendees list |

**Used by commands:** NONE

**Used by helpers:** NONE

**What code does:** **DEFINED BUT NEVER USED.** This is a `@property` on `SessionSettings` (session_settings.py:375) that returns `len(self.attendees)`. It is never called anywhere in the codebase. The diarizer (DiariZen) infers speaker count automatically rather than using this property.

---

## DiarizationStitchingSettings Fields

All fields below live in `settings/diarization_stitching_settings.py` and are accessed via `settings.diarization_stitching.<field>`.

### min_overlap_fraction_word

| Attribute | Value |
|---|---|
| **Type** | `float` (0.0-1.0) |
| **Typical value** | 0.20 |

**Used in:**
- `diarization/speech_clip_factory.py:32` — in `_is_acceptable_overlap()`: word passes if `(overlap / word_duration) >= min_overlap_fraction_word - epsilon`

**What code does:** One of two acceptance thresholds for word-segment overlap. A word is acceptably overlapped if the overlap fraction of the word's duration meets this threshold. Prevents "barely touching" overlaps from boundary jitter.

---

### min_overlap_seconds

| Attribute | Value |
|---|---|
| **Type** | `float` (seconds, >= 0.0) |
| **Typical value** | 0.02 |

**Used in:**
- `diarization/speech_clip_factory.py:30` — in `_is_acceptable_overlap()`: word passes if `overlap >= min_overlap_seconds - epsilon`

**What code does:** Absolute floor on overlap duration. A word is acceptably overlapped if the raw overlap in seconds meets this threshold. Either this OR `min_overlap_fraction_word` passing is sufficient.

---

### fill_nearest

| Attribute | Value |
|---|---|
| **Type** | `bool` |
| **Typical value** | true |

**Used in:**
- `diarization/speech_clip_factory.py:130` — cached as `should_fill_nearest`; controls whether nearest-segment fallback is enabled in `_find_best_candidate()`
- `diarization/candidate_pool.py:29` — if True, affects candidate pool search radius

**What code does:** When True and no segment passes overlap thresholds, the algorithm falls back to assigning the word to the nearest segment (by midpoint distance) within `max_nearest_distance`.

---

### max_nearest_distance

| Attribute | Value |
|---|---|
| **Type** | `float` (seconds, >= 0.0) |
| **Typical value** | 0.25 |

**Used in:**
- `diarization/speech_clip_factory.py:131` — cached as `max_nearest_distance + epsilon`; used in `_find_best_candidate()` to cap the gap for nearest-segment fallback
- `diarization/candidate_pool.py:29` — adjusts candidate pool search radius

**What code does:** Maximum gap between a word and a non-overlapping segment for nearest-assignment to apply. Keeps fallback conservative to avoid jumping speakers across long silences.

---

### anonymous_join_gap

| Attribute | Value |
|---|---|
| **Type** | `float` (seconds, >= 0.0) |
| **Typical value** | 0.15 |

**Used in:**
- `diarization/anonymous_clips.py:17` — radius for merging consecutive anonymous words: `radius = anonymous_join_gap + epsilon`

**What code does:** When a word has no segment assignment (neither overlap nor nearest), it becomes anonymous. Consecutive anonymous words within this gap are merged into a single anonymous segment.

---

### merge_gap_seconds

| Attribute | Value |
|---|---|
| **Type** | `float` (seconds, >= 0.0) |
| **Typical value** | 0.20 |

**Used in:**
- `diarization/speech_clip_factory.py:68` — in `SimpleMergeSelector.ShouldMerge()`: merges adjacent same-speaker segments separated by ≤ this gap

**What code does:** Post-processing merge during initial speech clip creation: adjacent clips with the same speaker label separated by ≤ this gap are merged. Reduces over-segmentation from diarization.

---

### unfinished_clip_merge_max_length

| Attribute | Value |
|---|---|
| **Type** | `float` (seconds, >= 0.0) |
| **Typical value** | 2.0 |

**Used in:**
- `helpers/first_stitcher.py:77-78` — in `MergeUnfinishedSegmentsWithSameSpeakerOrAnonymous.ShouldMerge()`: merges an unfinished clip (no END_OF_TURN flag) with the following same-speaker clip if gap ≤ this value

**What code does:** During first-stitching, clips that are not flagged as turn-ends are merged with the next clip from the same speaker if the gap is small enough. Preserves conversational flow while respecting turn boundaries.

---

### identity_stitching_max_gap

| Attribute | Value |
|---|---|
| **Type** | `float` (seconds, >= 0.0) |
| **Typical value** | 10.0 |

**Used in:**
- `helpers/identity_stitch.py:85` — in `IdentityMergeSelector.ShouldMerge()`: merges clips with the same identified speaker if gap ≤ this value

**What code does:** During identity stitching, adjacent clips assigned to the same speaker (by name/identity) are merged if separated by at most this gap. Larger values allow merging across longer pauses.

---

### identity_similarity_threshold

| Attribute | Value |
|---|---|
| **Type** | `float` (0.0-1.0) |
| **Typical value** | 0.65 |

**Used by commands:** NONE

**Used by helpers:** NONE

**What code does:** **DEFINED BUT NEVER USED.** Validated in settings (0.0-1.0 range at diarization_stitching_settings.py:199) but never referenced by any processing code. Intended as a cosine similarity threshold for identity stitching merging but not implemented.

---

### expand_segments_to_fit_words

| Attribute | Value |
|---|---|
| **Type** | `bool` |
| **Typical value** | false |

**Used in:**
- `diarization/speech_clip_factory.py:161` — conditional: if True, iterates all clips and calls `clip.expand_bounds_to_include_words()`

**What code does:** When True, after all word assignment, each segment's time boundaries are widened to fully contain all assigned words. Useful for UI rendering but reduces diarization boundary fidelity. Gated — code path is skipped when False (the default).

---

### expansion_limit_seconds

| Attribute | Value |
|---|---|
| **Type** | `float` (seconds, >= 0.0) |
| **Typical value** | 300 |

**Used in:**
- `diarization/speech_clip_factory.py:163` — passed to `clip.expand_bounds_to_include_words()`
- `processing_results/speech_clip.py:252,254` — caps how far start/end time can shift during expansion

**What code does:** Maximum distance (seconds) a segment boundary can be expanded. Only relevant when `expand_segments_to_fit_words` is True. Prevents runaway expansion.

---

### scoring_mode

| Attribute | Value |
|---|---|
| **Type** | `ScoringMode` enum |
| **Typical value** | `overlap_seconds_then_midpoint` |

**Used in:**
- `diarization/speech_clip_factory.py:132` — cached; passed to `score_candidate()` in `_find_best_candidate()`
- `diarization/candidate_score.py:46-51` — determines primary scoring metric: overlap_seconds, overlap_fraction_word, or IOU

**What code does:** Controls how candidate diarization segments are ranked for each word:
- `overlap_seconds_then_midpoint` — raw overlap in seconds (ties broken by midpoint distance)
- `overlap_fraction_word_then_midpoint` — overlap as fraction of word duration
- `iou_then_midpoint` — intersection-over-union of intervals

---

### prefer_shorter_on_tie

| Attribute | Value |
|---|---|
| **Type** | `bool` |
| **Typical value** | true |

**Used in:**
- `diarization/speech_clip_factory.py:133` — cached; passed to `score_candidate()`
- `diarization/candidate_score.py:53` — when True, adds `-segment_duration` as an additional tiebreaker component

**What code does:** When candidates score identically on primary metric + midpoint distance, prefers the shorter segment. Avoids bias toward long segments that happen to contain the word.

---

### max_backchannel_duration

| Attribute | Value |
|---|---|
| **Type** | `float` (seconds, >= 0.0) |
| **Typical value** | 0.75 |

**Used in:**
- `helpers/first_stitcher.py:37` — in `BackchannelMerger.ShouldMerge()`: clips longer than this are never treated as backchannels

**What code does:** Maximum duration for a clip to qualify as a backchannel utterance (e.g., "mm-hmm", "right"). All three backchannel thresholds (duration, prior gap, next gap) must be met simultaneously.

---

### max_backchannel_prior_gap

| Attribute | Value |
|---|---|
| **Type** | `float` (seconds, >= 0.0) |
| **Typical value** | 0.25 |

**Used in:**
- `helpers/first_stitcher.py:43` — in `BackchannelMerger.ShouldMerge()`: maximum gap between clip and predecessor

**What code does:** Backchannels must occur close to the preceding clip. If the prior gap is too large, the utterance is an independent contribution rather than a reactive backchannel.

---

### max_backchannel_next_gap

| Attribute | Value |
|---|---|
| **Type** | `float` (seconds, >= 0.0) |
| **Typical value** | 1.0 |

**Used in:**
- `helpers/first_stitcher.py:52` — in `BackchannelMerger.ShouldMerge()`: maximum gap between clip and successor

**What code does:** Backchannels must be followed by nearby speech. If the next clip is far away, the short utterance is standalone rather than mid-stream backchannel.

---

### max_identity_backchannel_duration

| Attribute | Value |
|---|---|
| **Type** | `float` (seconds, >= 0.0) |
| **Typical value** | 3.0 |

**Used in:**
- `helpers/identity_stitch.py:43` — in `IdentityBackchannelMerger.ShouldMerge()`: clips longer than this are not merged as backchannels

**What code does:** **DEFINED BUT EFFECTIVELY UNUSED.** Same concept as `max_backchannel_duration`, but intended for identity-based stitching. It is referenced only by `IdentityBackchannelMerger`; `apply_identity_stitching()` currently instantiates only `IdentityMergeSelector`, so this setting is not used by any reachable command path.

---

### max_identity_backchannel_prior_gap

| Attribute | Value |
|---|---|
| **Type** | `float` (seconds, >= 0.0) |
| **Typical value** | 0.75 |

**Used in:**
- `helpers/identity_stitch.py:49` — in `IdentityBackchannelMerger.ShouldMerge()`: maximum gap to predecessor

**What code does:** **DEFINED BUT EFFECTIVELY UNUSED.** Identity-stitching version of `max_backchannel_prior_gap`. It is referenced only by `IdentityBackchannelMerger`; `apply_identity_stitching()` currently instantiates only `IdentityMergeSelector`, so this setting is not used by any reachable command path.

---

### max_identity_backchannel_next_gap

| Attribute | Value |
|---|---|
| **Type** | `float` (seconds, >= 0.0) |
| **Typical value** | 3.0 |

**Used in:**
- `helpers/identity_stitch.py:58` — in `IdentityBackchannelMerger.ShouldMerge()`: maximum gap to successor

**What code does:** **DEFINED BUT EFFECTIVELY UNUSED.** Identity-stitching version of `max_backchannel_next_gap`. It is referenced only by `IdentityBackchannelMerger`; `apply_identity_stitching()` currently instantiates only `IdentityMergeSelector`, so this setting is not used by any reachable command path.

---

### turn_end_probability_threshold

| Attribute | Value |
|---|---|
| **Type** | `float` (0.0-1.0) |
| **Typical value** | 0.8 |

**Used in:**
- `helpers/update_turn_end.py:40` — clips with `end_of_turn_probability >= threshold` get the `END_OF_TURN` flag set

**What code does:** Threshold for the Smart Turn model's turn-end probability. Clips meeting or exceeding this probability are flagged as turn boundaries. These flags drive `unfinished_clip_merge_max_length` behavior in first-stitching.

---

### tiny_clip_threshold

| Attribute | Value |
|---|---|
| **Type** | `float` (seconds, >= 0.0) |
| **Typical value** | 0.1 |

**Used in:**
- `helpers/tiny_clip_merger.py:20` — in `TinyClipMergeSelector.ShouldMerge()`: clips shorter than this are merged with the closest adjacent clip

**What code does:** **DEFINED BUT EFFECTIVELY UNUSED.** The `TinyClipMergeSelector` class and `apply_tiny_stitching()` function exist in `helpers/tiny_clip_merger.py`, but `apply_tiny_stitching` is **never imported or called** from any command. The code is dead/unreachable.

---

## VadSettings Fields

All fields below live in `settings/vad_settings.py` and are accessed via `settings.vad.<field>`. They are all consumed in a single place: `helpers/audio_segmenter.py:32-39`, where they are unpacked and passed to the `NemoVadDetector` constructor, which passes them to the NeMo VAD model for inference and post-processing.

### model_name

| Attribute | Value |
|---|---|
| **Type** | `str` |
| **Typical value** | `vad_multilingual_frame_marblenet` |

**Used in:**
- `helpers/audio_segmenter.py:32` → `vad/nemo_vad_detector.py:53` — loads pretrained NeMo VAD model by name

### onset

| Attribute | Value |
|---|---|
| **Type** | `float` (0.0-1.0) |
| **Typical value** | 0.7 |

**Used in:**
- `helpers/audio_segmenter.py:34` → `vad/nemo_vad_detector.py:78` — probability threshold to transition from silence to speech (hysteresis: must be >= offset)

### offset

| Attribute | Value |
|---|---|
| **Type** | `float` (0.0-1.0) |
| **Typical value** | 0.4 |

**Used in:**
- `helpers/audio_segmenter.py:35` → `vad/nemo_vad_detector.py:80` — probability threshold to transition from speech to silence

### min_duration_on

| Attribute | Value |
|---|---|
| **Type** | `float` (seconds) |
| **Typical value** | 0.3 |

**Used in:**
- `helpers/audio_segmenter.py:36` → NeMo VAD post-processing — speech regions shorter than this are discarded (filters clicks, coughs, transient noise)

### min_duration_off

| Attribute | Value |
|---|---|
| **Type** | `float` (seconds) |
| **Typical value** | 0.3 |

**Used in:**
- `helpers/audio_segmenter.py:37` → NeMo VAD post-processing — silence regions shorter than this are bridged (treated as speech, prevents choppy segmentation)

### pad_onset

| Attribute | Value |
|---|---|
| **Type** | `float` (seconds) |
| **Typical value** | 0.1 |

**Used in:**
- `helpers/audio_segmenter.py:38` → `vad/nemo_vad_detector.py:81` — padding added before speech onset to capture plosive consonants and breath

### pad_offset

| Attribute | Value |
|---|---|
| **Type** | `float` (seconds) |
| **Typical value** | 0.1 |

**Used in:**
- `helpers/audio_segmenter.py:39` → `vad/nemo_vad_detector.py:82` — padding added after speech offset to capture word-final sounds

---

## Summary

### Statistics

| Category | Total | Actively Used | Unused |
|---|---|---|---|
| SessionSettings top-level | 36 (incl. property) | 33 | 3 |
| DiarizationStitchingSettings | 20 | 15 | 5 |
| VadSettings | 7 | 7 | 0 |
| **Total** | **63** | **55** | **8** |

### Unused Settings

| Setting | Location | Notes |
|---|---|---|
| `high_confidence_similarity_threshold` | SessionSettings | Validated but never referenced. Legacy/planned. |
| `adventure_settings` (pcs, glossary) | SessionSettings | `to_prompt_fragment()` exists but is never called. Required in YAML but serves no purpose. Also has a bug (line 67 does not concatenate). |
| `number_of_speakers` | SessionSettings (property) | Property exists but never called. DiariZen infers speaker count automatically. |
| `identity_similarity_threshold` | DiarizationStitchingSettings | Validated but never referenced. Intended for identity stitching but not implemented. |
| `max_identity_backchannel_duration` | DiarizationStitchingSettings | Referenced only by `IdentityBackchannelMerger`, which is not wired into `apply_identity_stitching()`. |
| `max_identity_backchannel_prior_gap` | DiarizationStitchingSettings | Referenced only by `IdentityBackchannelMerger`, which is not wired into `apply_identity_stitching()`. |
| `max_identity_backchannel_next_gap` | DiarizationStitchingSettings | Referenced only by `IdentityBackchannelMerger`, which is not wired into `apply_identity_stitching()`. |
| `tiny_clip_threshold` | DiarizationStitchingSettings | Code exists in `tiny_clip_merger.py` but `apply_tiny_stitching()` is never imported or called — dead code. |

### Path Settings (16 total)

These 16 settings are all `Path` fields that define input/output file locations for pipeline steps. They form the wiring of the pipeline DAG:

```
audio_file
  → cleaned_audio_file
      → segments_path
      → transcript_file → aligned_transcript_path → confidence_transcript_path
      → base_diarized_path → diarizationlm_processed_path
          → turn_end_updated_path → first_stitched_path
          → dangling_sentence_fix_path → speech_clips_with_embedding
              → identified_speaker_path → identity_stitched_path
                  → punctuated_text_path
                  → indeterminate_speakers_path
```

### Terminal Output Files (not consumed by any downstream command)

- `first_stitched_path` — written by first-stitch-clips but not read by any downstream command
- `punctuated_text_path` — final output
- `indeterminate_speakers_path` — final output
