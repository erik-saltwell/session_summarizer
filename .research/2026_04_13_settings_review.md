# Settings Review - 2026-04-13

Audit of the current `SessionSettings` model and nested settings objects:
`AdventureSettings`, `VadSettings`, and `DiarizationStitchingSettings`.

This document describes runtime use in the current codebase. References from
tests, sample YAML, and fully commented-out code are not counted as active use;
commented references are called out when they explain stale settings.

## Loading And Validation

- `SessionSettings.load(path)` reads one YAML file and resolves path settings relative to `path.parent`.
- `SessionSettings.load_cascading(session_id)` merges `data/settings.yaml` with `data/<session-id>/settings.yaml`, then resolves path settings relative to `data/<session-id>`.
- Resolved path settings are:
  `audio_file`, `cleaned_audio_file`, `transcript_file`, `aligned_transcript_path`,
  `confidence_transcript_path`, `segments_path`, `base_diarized_path`,
  `speech_clips_with_embedding`, `identified_speaker_path`,
  `turn_end_updated_path`, `first_stitched_path`, `identity_stitched_path`,
  `backchannel_marked_path`, `diarizationlm_processed_path`,
  `indeterminate_speakers_path`, `dangling_sentence_fix_path`, and
  `punctuated_text_path`.
- `audio_file` suffix is validated against:
  `.m4a`, `.mp3`, `.wav`, `.flac`, `.ogg`, `.opus`, `.wma`, `.aac`, `.webm`.
- `attendees` must contain at least one non-blank name.
- `adventure_settings.pcs` must contain at least one player -> character pair, and both sides must be non-blank.
- `device` is limited to `"cpu"` or `"cuda"`.
- `epsilon`, `speaker_clip_lead_in`, `speaker_clip_lead_out`,
  `minimum_speaker_clip_duration`, and `speaker_clip_gap_length` must be non-negative.
- `high_confidence_similarity_threshold`, `speaker_identity_assignment_threshold`,
  `speaker_clip_minimum_similarity_residual`, and `min_speaker_similarity` must be in `0.0..1.0`.
- `min_segment_length_short < max_segment_length_short`.
- `min_segment_length_long < max_segment_length_long`.
- `VadSettings` currently has no field validators; its values are passed through to the VAD detector.
- `DiarizationStitchingSettings` validates selected duration/gap fields as non-negative and selected threshold fields as `0.0..1.0`.

## Current Pipeline Paths

The active JSON/audio artifact flow is:

```text
audio_file
  -> cleaned_audio_file
      -> segments_path
      -> transcript_file
          -> aligned_transcript_path
              -> confidence_transcript_path
                  -> base_diarized_path
                      -> diarizationlm_processed_path
                          -> dangling_sentence_fix_path
                              -> speech_clips_with_embedding
                                  -> identified_speaker_path
                                      -> identity_stitched_path
                                          -> backchannel_marked_path
                                              -> punctuated_text_path
                                          -> indeterminate_speakers_path
```

Side consumers:

- `CreateSpeakerClipsCommand` reads `identified_speaker_path` and `cleaned_audio_file` to create per-speaker audio clips.
- `CompareFullTextCommand` compares fulltext sidecars for the transcript, alignment, confidence, diarization, DiarizationLM, embeddings, and identified-speaker artifacts.
- `ValidateDiarizationCommand` evaluates several `SpeechClipSet` artifacts. Note: its registry currently joins `session_dir` for some paths but not for `identity_stitched_path` and `indeterminate_speakers_path`.
- `turn_end_updated_path` and `first_stitched_path` are still settings, but no active command writes or reads them.

## SessionSettings Top-Level Fields

### `attendees`

- Type: `list[str]`, required, min length 1.
- Active use:
  - `IdentifySpeakersCommand` validates that each attendee exists in the registered speaker file.
  - `helpers/speaker_identifier.py` filters registered speaker embeddings down to the attendee set before cosine-similarity matching.
- Meaning: the set of speaker identities eligible for assignment in this session.

### `adventure_settings`

- Type: `AdventureSettings`, required.
- Active use: none in runtime processing.
- Validation use: `pcs` must be non-empty and contain non-blank player/character names.
- Notes:
  - `AdventureSettings.to_prompt_fragment()` exists but is not called anywhere.
  - `to_prompt_fragment()` still appears to have a bug: the glossary description f-string is a bare expression and is not appended to `result`.

### `audio_file`

- Type: `Path`, required.
- Active use:
  - `CleanAudioCommand` declares it as input.
  - `helpers/audio_cleaner.py` reads it and writes `cleaned_audio_file`.
  - `CleanSessionCommand` preserves the resolved original audio path during cleanup.
- Validation: suffix must be one of the supported audio suffixes.

### `cleaned_audio_file`

- Type: `Path`, required.
- Active use:
  - `CleanAudioCommand` writes it.
  - `ComputeSegmentsCommand`, `TranscribeAudioCommand`, `AlignTranscriptCommand`,
    `DiarizeAudioCommand`, `AddEmbeddingsCommand`, `ValidateTranscribersCommand`,
    and `CreateSpeakerClipsCommand` declare or read it.
  - Helpers that read it include `audio_segmenter`, `audio_transcriber`,
    `transcript_aligner`, `confidence_scorer`, `audio_diarizer`, and `add_embeddings`.
- Meaning: the central cleaned audio artifact used by most downstream model steps.

### `transcript_file`

- Type: `Path`, required.
- Active use:
  - `TranscribeAudioCommand` writes the initial ASR transcript JSON and fulltext sidecar.
  - `AlignTranscriptCommand` reads it.
  - `CompareFullTextCommand` includes it in the comparison set.

### `aligned_transcript_path`

- Type: `Path`, required.
- Active use:
  - `AlignTranscriptCommand` writes it.
  - `ScoreConfidenceCommand` reads it.
  - `CompareFullTextCommand` includes it in the comparison set.
- Meaning: word-aligned transcript with per-word start/end timestamps.

### `confidence_transcript_path`

- Type: `Path`, required.
- Active use:
  - `ScoreConfidenceCommand` writes it.
  - `DiarizeAudioCommand` reads it.
  - `CompareFullTextCommand` includes it in the comparison set.
- Meaning: aligned transcript with per-word confidence values.

### `base_diarized_path`

- Type: `Path`, required.
- Active use:
  - `DiarizeAudioCommand` writes it.
  - `DiarizationLMCommand` reads it.
  - `CompareFullTextCommand` includes it in the comparison set.
  - `ValidateDiarizationCommand` includes it in the hypothesis registry.
- Meaning: initial `SpeechClipSet` created from diarization plus word assignment.

### `speech_clips_with_embedding`

- Type: `Path`, required.
- Active use:
  - `AddEmbeddingsCommand` writes it.
  - `IdentifySpeakersCommand` reads it.
  - `CompareFullTextCommand` includes it in the comparison set.
  - `ValidateDiarizationCommand` includes it in the hypothesis registry.
- Meaning: `SpeechClipSet` with speaker embedding vectors attached.

### `identified_speaker_path`

- Type: `Path`, required.
- Active use:
  - `IdentifySpeakersCommand` writes it.
  - `StitichIdentitiesCommand` reads it.
  - `CreateSpeakerClipsCommand` reads it.
  - `CompareFullTextCommand` includes it in the comparison set.
  - `ValidateDiarizationCommand` includes it in the hypothesis registry.
- Meaning: `SpeechClipSet` with `identity`, `cosine_similarity`, and `similarity_residual` assigned where embeddings are available.

### `turn_end_updated_path`

- Type: `Path`, required.
- Active use: none.
- Commented references:
  - `commands/update_turn_end.py`
  - `commands/first_stitch_clips.py`
- Meaning in old pipeline: intended output of turn-end scoring before first stitching. The command is currently commented out, so this is a stale path setting.

### `first_stitched_path`

- Type: `Path`, required.
- Active use: none.
- Commented references:
  - `commands/first_stitch_clips.py`
- Meaning in old pipeline: intended output of first-stitching after turn-end scoring. The command is currently commented out, so this is a stale path setting.

### `identity_stitched_path`

- Type: `Path`, required.
- Active use:
  - `StitichIdentitiesCommand` writes it.
  - `MarkBackchannelsCommand` reads it.
  - `IndeterminantSpeakerAssignmentCommand` reads it.
  - `TestCommand` reads it for diagnostic outputs.
  - `ValidateDiarizationCommand` includes it in the hypothesis registry, but currently without prefixing `session_dir`.
- Meaning: `SpeechClipSet` after adjacent same-identity clips have been merged by identity stitching.

### `backchannel_marked_path`

- Type: `Path`, required.
- Active use:
  - `MarkBackchannelsCommand` writes it after applying `SpeechClipFlags.IS_BACKCHANNEL`.
  - `PunctuateTextCommand` reads it.
- Meaning: `SpeechClipSet` after text-based one-word backchannel marking. The current marker does not use the diarization-stitching backchannel gap/duration settings; it checks whether a clip has exactly one word and that cleaned word appears in `helpers/backchannel_marker.py`.

### `diarizationlm_processed_path`

- Type: `Path`, required.
- Active use:
  - `DiarizationLMCommand` writes it.
  - `DanglingSentenceFixCommand` reads it.
  - `CompareFullTextCommand` includes it in the comparison set.
  - `ValidateDiarizationCommand` includes it in the hypothesis registry.
- Commented references:
  - `UpdateTurnEndCommand` used to read it, but that command is currently commented out.

### `indeterminate_speakers_path`

- Type: `Path`, required.
- Active use:
  - `IndeterminantSpeakerAssignmentCommand` writes it.
  - `ValidateDiarizationCommand` includes it in the hypothesis registry, but currently without prefixing `session_dir`.
- Meaning: terminal `SpeechClipSet` where ambiguous identities are reassigned to the unassigned speaker label.

### `dangling_sentence_fix_path`

- Type: `Path`, required.
- Active use:
  - `DanglingSentenceFixCommand` writes it.
  - `AddEmbeddingsCommand` reads it.
- Meaning: `SpeechClipSet` after dangling sentence repair.

### `punctuated_text_path`

- Type: `Path`, required.
- Active use:
  - `PunctuateTextCommand` writes it.
- Meaning: terminal `SpeechClipSet` after punctuation and capitalization restoration.

### `device`

- Type: `Literal["cpu", "cuda"]`, required.
- Active use:
  - `ValidateTranscribersCommand` passes it to transcriber factories.
  - `helpers/audio_segmenter.py` passes it to `NemoVadDetector`.
  - `helpers/audio_transcriber.py` passes it to `CanaryQwenTranscriber`.
  - `helpers/transcript_aligner.py` passes it to `ParakeetCTCWordAligner`.
  - `helpers/confidence_scorer.py` passes it to `ParakeetCTCConfidenceScorer`.
  - `helpers/diarizationlm_refiner.py` passes it to `DiarizationLMModel`.
  - `helpers/add_embeddings.py` and `helpers/remove_outlier_speakers.py` pass it to the embedding factory.
- Commented references:
  - `helpers/update_turn_end.py` used it for Smart Turn inference, but that helper is currently commented out.

### `segments_path`

- Type: `Path`, required.
- Active use:
  - `ComputeSegmentsCommand` writes it.
  - `TranscribeAudioCommand`, `AlignTranscriptCommand`, `ScoreConfidenceCommand`,
    `DiarizeAudioCommand`, `AddEmbeddingsCommand`, and `ValidateTranscribersCommand`
    declare it as an input/dependency.
  - `TranscribeAudioCommand`, `AlignTranscriptCommand`, and `ScoreConfidenceCommand` load and use the segment data directly.
- Notes:
  - `DiarizeAudioCommand` and `AddEmbeddingsCommand` declare `segments_path` as an input so segment computation runs first, but their current helper code does not load the segment data directly.

### `min_segment_length_short`

- Type: `float`, required.
- Active use:
  - `helpers/audio_segmenter.py` passes it as `min_length` when computing short VAD cut points.
- Validation: must be less than `max_segment_length_short`.
- Meaning: earliest eligible duration for a short-mode segment cut. Short segments are used by transcription and confidence scoring.

### `max_segment_length_short`

- Type: `float`, required.
- Active use:
  - `helpers/audio_segmenter.py` passes it as `max_length` when computing short VAD cut points.
- Validation: must be greater than `min_segment_length_short`.
- Meaning: maximum short-mode segment duration before a cut is forced.

### `min_segment_length_long`

- Type: `float`, required.
- Active use:
  - `helpers/audio_segmenter.py` passes it as `min_length` when computing long VAD cut points.
- Validation: must be less than `max_segment_length_long`.

### `max_segment_length_long`

- Type: `float`, required.
- Active use:
  - `helpers/audio_segmenter.py` passes it as `max_length` when computing long VAD cut points.
- Validation: must be greater than `min_segment_length_long`.

### `high_confidence_similarity_threshold`

- Type: `float`, required, `0.0..1.0`.
- Active use: none.
- Meaning: legacy or planned setting. It is validated and present in sample settings, but no runtime code reads it.

### `speaker_identity_assignment_threshold`

- Type: `float`, required, `0.0..1.0`.
- Active use:
  - `helpers/indeterminate_speakers.py` compares it to `clip.similarity_residual`.
- Important accuracy note:
  - The settings description says "cosine similarity score", but the active code uses it as a similarity residual threshold. Clips with `similarity_residual` that is `NaN` or below this value are assigned to `UNASSIGNED_SPEAKER_NAME`.

### `vad`

- Type: `VadSettings`, required.
- Active use:
  - `helpers/audio_segmenter.py` passes every field to `NemoVadDetector`.
- See [VadSettings Fields](#vadsettings-fields).

### `speaker_clip_lead_in`

- Type: `float`, required, non-negative.
- Active use:
  - `CreateSpeakerClipsCommand` passes it to `save_segment_as_speaker_audio_clip()`.
- Meaning: audio padding before extracted speaker clips.

### `speaker_clip_lead_out`

- Type: `float`, required, non-negative.
- Active use:
  - `CreateSpeakerClipsCommand` passes it to `save_segment_as_speaker_audio_clip()`.
- Meaning: audio padding after extracted speaker clips.

### `speaker_clip_minimum_similarity_residual`

- Type: `float`, required, `0.0..1.0`.
- Active use:
  - `CreateSpeakerClipsCommand` uses it to decide whether a clip is clean enough to save as a speaker sample.
- Meaning: clips are saved only when `similarity_residual > speaker_clip_minimum_similarity_residual`, and only if they also have an identity, are not anonymous, are not flagged as backchannels, and either multispeaker clips are allowed or the clip is single-speaker.

### `minimum_speaker_clip_duration`

- Type: `float`, required, non-negative.
- Active use:
  - `MergeSpeakerClipsCommand` passes it to `merge_speaker_clips_to_min_duration()`.
  - `RegisterSpeakersCommand` passes it to `merge_speaker_clips_to_min_duration()`.
- Meaning: target minimum duration when merging short speaker sample clips.

### `min_speaker_similarity`

- Type: `float`, required, `0.0..1.0`.
- Active use:
  - `RemoveOutlierSpeakerClipsCommand` logs it.
  - `helpers/remove_outlier_speakers.py` removes the lowest-centroid-similarity clip until all remaining clips meet this threshold.
- Meaning: speaker sample outlier-removal threshold.

### `speaker_clip_gap_length`

- Type: `float`, optional default `0.5`, non-negative.
- Active use:
  - `MergeSpeakerClipsCommand` and `RegisterSpeakersCommand` pass it to `merge_speaker_clips_to_min_duration()`.
- Meaning: silence inserted between adjacent clips when merging speaker sample audio.

### `diarization_stitching`

- Type: `DiarizationStitchingSettings`, required.
- Active use: container for nested stitching/word-assignment settings.
- See [DiarizationStitchingSettings Fields](#diarizationstitchingsettings-fields).

### `epsilon`

- Type: `float`, required, non-negative.
- Active use:
  - `diarization/speech_clip_factory.py` uses it in word/segment overlap and expansion decisions.
  - `diarization/candidate_pool.py`, `diarization/anonymous_clips.py`, and `diarization/clip_merger.py` use it for candidate radius or gap comparisons.
  - `helpers/identity_stitch.py` uses it in same-identity gap checks.
  - `helpers/diarizationlm_refiner.py` passes it into the DiarizationLM processor.
  - `diarizationlm` conversion/mapping code uses it during speaker computation and segment conversion.
  - `ValidateDiarizationCommand` passes it into diarization evaluation.
  - Evaluation and `processing_results` utilities use it for time-boundary comparisons.
- Meaning: tolerance for floating-point and timestamp boundary comparisons.

### `seed`

- Type: `int`, required.
- Active use:
  - `console/main.py` uses it in `_set_seed()` to seed Python `random`, NumPy, PyTorch CPU, and PyTorch CUDA.
- Notes:
  - `_set_seed()` is called by session-scoped CLI commands.
  - Global/non-session commands such as `clear-logs`, `generate-sample-settings`, `merge-speaker-clips`, `remove-outlier-speaker-clips`, and `register-speakers` do not call `_set_seed()`.

### `number_of_speakers` property

- Type: derived `int` property returning `len(attendees)`.
- Active runtime use: none.
- Test use: settings unit tests assert it.
- Meaning: convenience property only; the current diarizer does not use it.

## DiarizationStitchingSettings Fields

All fields below live under `settings.diarization_stitching`.

### `min_overlap_fraction_word`

- Type: `float`, `0.0..1.0`.
- Active use:
  - `diarization/speech_clip_factory.py` uses it in `_is_acceptable_overlap()`.
- Meaning: a word/segment candidate is acceptable if overlap divided by word duration meets this threshold. This is an OR condition with `min_overlap_seconds`.

### `min_overlap_seconds`

- Type: `float`, non-negative.
- Active use:
  - `diarization/speech_clip_factory.py` uses it in `_is_acceptable_overlap()`.
- Meaning: a word/segment candidate is acceptable if raw overlap seconds meet this threshold. This is an OR condition with `min_overlap_fraction_word`.

### `fill_nearest`

- Type: `bool`.
- Active use:
  - `diarization/speech_clip_factory.py` enables/disables nearest-segment fallback.
  - `diarization/candidate_pool.py` includes nearest-distance search radius only when enabled.
- Meaning: when no overlap candidate passes, allow fallback to the nearest segment within `max_nearest_distance`.

### `max_nearest_distance`

- Type: `float`, non-negative.
- Active use:
  - `diarization/speech_clip_factory.py` caps nearest-segment fallback distance.
  - `diarization/candidate_pool.py` uses it in candidate search radius when `fill_nearest` is true.
- Meaning: maximum allowed gap for assigning an otherwise unassigned word to a nearby segment.

### `anonymous_join_gap`

- Type: `float`, non-negative.
- Active use:
  - `diarization/anonymous_clips.py` uses it, plus `epsilon`, to merge consecutive anonymous word spans.
- Meaning: maximum gap for joining anonymous words into one anonymous clip.

### `merge_gap_seconds`

- Type: `float`, non-negative.
- Active use:
  - `diarization/speech_clip_factory.py` uses it in `SimpleMergeSelector` to merge adjacent same-speaker clips.
- Meaning: initial post-processing merge gap for adjacent clips with the same speaker label.

### `unfinished_clip_merge_max_length`

- Type: `float`, non-negative.
- Active use: none.
- Commented references:
  - `helpers/first_stitcher.py` references it inside commented-out first-stitching logic.
- Meaning: stale first-stitching setting in the current runtime.

### `identity_stitching_max_gap`

- Type: `float`, non-negative.
- Active use:
  - `helpers/identity_stitch.py` uses it in `IdentityMergeSelector` to merge adjacent clips with the same `identity`.
- Meaning: maximum gap allowed when identity stitching same-speaker clips.

### `identity_similarity_threshold`

- Type: `float`, `0.0..1.0`.
- Active use: none.
- Meaning: legacy or planned threshold. The current identity stitching implementation only checks identity equality and `identity_stitching_max_gap`.

### `expand_segments_to_fit_words`

- Type: `bool`.
- Active use:
  - `diarization/speech_clip_factory.py` gates whether each clip expands bounds to include assigned words.
- Meaning: optional post-processing that expands segment boundaries to contain all assigned word timings.

### `expansion_limit_seconds`

- Type: `float`, non-negative.
- Active use:
  - `diarization/speech_clip_factory.py` passes it into `SpeechClip.expand_bounds_to_include_words()`.
  - `processing_results/speech_clip.py` uses it to cap boundary expansion.
- Meaning: maximum amount each boundary may expand when `expand_segments_to_fit_words` is true.

### `scoring_mode`

- Type: `ScoringMode` enum.
- Active use:
  - `diarization/speech_clip_factory.py` passes it into candidate scoring.
  - `diarization/candidate_score.py` selects between overlap seconds, overlap fraction, and intersection-over-union scoring.
- Allowed values:
  - `overlap_seconds_then_midpoint`
  - `overlap_fraction_word_then_midpoint`
  - `iou_then_midpoint`

### `prefer_shorter_on_tie`

- Type: `bool`.
- Active use:
  - `diarization/speech_clip_factory.py` passes it into candidate scoring.
  - `diarization/candidate_score.py` uses it as an additional tie breaker.
- Meaning: when candidate scores are otherwise tied, prefer the shorter segment.

### `max_backchannel_duration`

- Type: `float`, non-negative.
- Active use: none.
- Commented references:
  - `helpers/first_stitcher.py` references it inside commented-out backchannel merging logic.
- Important note:
  - Current `MarkBackchannelsCommand` does not use this setting. It uses text-only detection in `helpers/backchannel_marker.py`.

### `max_backchannel_prior_gap`

- Type: `float`, non-negative.
- Active use: none.
- Commented references:
  - `helpers/first_stitcher.py` references it inside commented-out backchannel merging logic.
- Important note:
  - Current `MarkBackchannelsCommand` does not use this setting.

### `max_backchannel_next_gap`

- Type: `float`, non-negative.
- Active use: none.
- Commented references:
  - `helpers/first_stitcher.py` references it inside commented-out backchannel merging logic.
- Important note:
  - Current `MarkBackchannelsCommand` does not use this setting.

### `max_identity_backchannel_duration`

- Type: `float`, non-negative.
- Active use: none.
- Commented references:
  - `helpers/identity_stitch.py` references it inside commented-out `IdentityBackchannelMerger` logic.

### `max_identity_backchannel_prior_gap`

- Type: `float`, non-negative.
- Active use: none.
- Commented references:
  - `helpers/identity_stitch.py` references it inside commented-out `IdentityBackchannelMerger` logic.

### `max_identity_backchannel_next_gap`

- Type: `float`, non-negative.
- Active use: none.
- Commented references:
  - `helpers/identity_stitch.py` references it inside commented-out `IdentityBackchannelMerger` logic.

### `turn_end_probability_threshold`

- Type: `float`, `0.0..1.0`.
- Active use: none.
- Commented references:
  - `helpers/update_turn_end.py` references it inside commented-out Smart Turn logic.
- Meaning in old pipeline: probability threshold for setting an end-of-turn flag.

### `tiny_clip_threshold`

- Type: `float`, non-negative.
- Active use: none.
- Meaning: stale setting. No active helper or command currently reads it.

## VadSettings Fields

All `VadSettings` fields are passed from `helpers/audio_segmenter.py` to
`NemoVadDetector`. The settings model itself does not validate their ranges.

### `model_name`

- Type: `str`.
- Active use:
  - `helpers/audio_segmenter.py` passes it to `NemoVadDetector`.
- Meaning: pretrained NeMo VAD model name.

### `onset`

- Type: `float`.
- Active use:
  - `helpers/audio_segmenter.py` passes it to `NemoVadDetector`.
- Meaning: probability threshold to enter speech.

### `offset`

- Type: `float`.
- Active use:
  - `helpers/audio_segmenter.py` passes it to `NemoVadDetector`.
- Meaning: probability threshold to leave speech.

### `min_duration_on`

- Type: `float`.
- Active use:
  - `helpers/audio_segmenter.py` passes it to `NemoVadDetector`.
- Meaning: minimum speech region duration used by VAD post-processing.

### `min_duration_off`

- Type: `float`.
- Active use:
  - `helpers/audio_segmenter.py` passes it to `NemoVadDetector`.
- Meaning: minimum silence duration used by VAD post-processing.

### `pad_onset`

- Type: `float`.
- Active use:
  - `helpers/audio_segmenter.py` passes it to `NemoVadDetector`.
- Meaning: padding added before speech onset.

### `pad_offset`

- Type: `float`.
- Active use:
  - `helpers/audio_segmenter.py` passes it to `NemoVadDetector`.
- Meaning: padding added after speech offset.

## Runtime-Unused Settings

Top-level `SessionSettings` fields with no active runtime use:

- `adventure_settings`
- `turn_end_updated_path`
- `first_stitched_path`
- `high_confidence_similarity_threshold`

Runtime-unused derived property:

- `number_of_speakers`

`DiarizationStitchingSettings` fields with no active runtime use:

- `unfinished_clip_merge_max_length`
- `identity_similarity_threshold`
- `max_backchannel_duration`
- `max_backchannel_prior_gap`
- `max_backchannel_next_gap`
- `max_identity_backchannel_duration`
- `max_identity_backchannel_prior_gap`
- `max_identity_backchannel_next_gap`
- `turn_end_probability_threshold`
- `tiny_clip_threshold`

## Counts

| Category | Total settings | Active runtime use | Runtime-unused |
|---|---:|---:|---:|
| `SessionSettings` fields | 36 | 32 | 4 |
| `DiarizationStitchingSettings` fields | 21 | 11 | 10 |
| `VadSettings` fields | 7 | 7 | 0 |
| **Total settings fields** | **64** | **50** | **14** |

`number_of_speakers` is a derived property, not a settings field; it is also runtime-unused.
