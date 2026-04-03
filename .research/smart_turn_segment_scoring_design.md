# Smart Turn Segment Scoring Design

## Overview

This document describes a design for using the local **Smart Turn** model in an offline pipeline to assign an **end-of-turn probability** to each speech segment.

The goal is not to decide turn boundaries in real time, but to walk through an existing list of segments and annotate each segment with a score representing how likely that segment marks the end of a conversational turn.

This design is intended for Python-based processing of already-segmented audio, such as diarization segments, ASR segments, or merged utterance fragments.

---

## Goal

For each segment in chronological order, compute and store:

- `end_of_turn_probability`: probability that the segment ends a turn
- `end_of_turn_prediction`: binary prediction from the model

The resulting data can be used for:

- turn-boundary analysis
- downstream segmentation refinement
- conversational behavior research
- threshold tuning for later online or batch systems

---

## Key Insight

Do **not** score each segment in isolation unless each segment already represents the full current turn.

Instead, for each segment, run Smart Turn on the **current turn so far**, ending at that segment’s `end` time.

In practice, this means:

1. Walk segments in chronological order.
2. Determine where the current turn begins.
3. Build an audio window from the current turn start through the current segment end.
4. Keep only the most recent 8 seconds of that window.
5. Left-pad with zeros if the audio is shorter than 8 seconds.
6. Run the local Smart Turn model.
7. Store the returned probability on the segment.

This is the closest offline approximation of how Smart Turn is intended to operate in a live voice system.

---

## Why Segment-Only Scoring Is Not Recommended

A segment in the middle of an utterance often does not contain enough context to know whether the speaker is finished.

For example:

- Segment 1: "I was wondering..."
- Segment 2: "...if maybe..."
- Segment 3: "...you could help me with that?"

If Segment 2 is scored by itself, the model sees only a short fragment. It does not see the buildup of the utterance. The probability may therefore be misleading.

Scoring the **accumulated current-turn audio through the endpoint of Segment 2** preserves the pacing and completion cues that Smart Turn is designed to use.

---

## Model Assumptions and Constraints

The local Smart Turn inference path should be treated with these assumptions:

- input audio must be **mono**
- input sample rate must be **16 kHz**
- the model should receive the **full current turn**, not arbitrary short fragments
- the model uses at most the **most recent 8 seconds** of audio
- if the window is shorter than 8 seconds, it should be **left-padded with zeros**

Because of those constraints, the batch pipeline should resample the source waveform once at load time and then slice from that normalized in-memory waveform.

---

## Offline Pipeline Design

### Inputs

The pipeline expects:

- a single audio waveform file for the conversation or channel being scored
- a list of segments in chronological order, where each segment has at least:
  - `start: float`
  - `end: float`
- optionally:
  - `speaker: str | None`
  - `text: str | None`

### Outputs

The pipeline returns a new list of segments with added fields:

- `end_of_turn_probability: float | None`
- `end_of_turn_prediction: int | None`

Optional debugging fields are also recommended:

- `smart_turn_window_start: float`
- `smart_turn_window_end: float`
- `smart_turn_reset_reason: str`

---

## Defining the Current Turn

A core design problem is determining where the **current turn** begins for any given segment.

### Recommended reset rules

Start a new current turn when either of the following is true:

1. **Long silence gap** before the segment
2. **Speaker changes**, if diarization labels are available

### Why these rules are reasonable

For offline multi-speaker audio, a turn should usually not include:

- speech from a previous speaker
- speech separated by a large silence

These reset rules let the scoring window align more closely to what a human would intuitively describe as “the current utterance so far.”

### Suggested default values

- `silence_reset_seconds = 0.8`
- `reset_on_speaker_change = True`

These are heuristics and should be tuned against real data.

---

## Core Algorithm

For each segment in chronological order:

1. Determine whether the current turn should reset.
2. If reset is needed, set `current_turn_start_time = segment.start`.
3. Slice audio from `current_turn_start_time` through `segment.end`.
4. Keep only the last 8 seconds.
5. Left-pad with zeros if needed.
6. Call `predict_endpoint(model_input)`.
7. Store:
   - `probability`
   - `prediction`
8. Continue to the next segment.

---

## Python Data Model

A simple segment structure:

```python
from dataclasses import dataclass

@dataclass(slots=True)
class Segment:
    start: float
    end: float
    text: str | None = None
    speaker: str | None = None
    end_of_turn_probability: float | None = None
    end_of_turn_prediction: int | None = None
```

---

## Reference Implementation Shape

```python
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable
import numpy as np
import librosa

from inference import predict_endpoint

SMART_TURN_SR = 16_000
SMART_TURN_MAX_SECONDS = 8.0
SMART_TURN_MAX_SAMPLES = int(SMART_TURN_SR * SMART_TURN_MAX_SECONDS)


@dataclass(slots=True)
class Segment:
    start: float
    end: float
    text: str | None = None
    speaker: str | None = None
    end_of_turn_probability: float | None = None
    end_of_turn_prediction: int | None = None


def load_audio_mono_16k(path: str) -> np.ndarray:
    audio, sr = librosa.load(path, sr=None, mono=True)
    if sr != SMART_TURN_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SMART_TURN_SR)

    audio = audio.astype(np.float32, copy=False)

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak

    return audio


def seconds_to_sample_index(seconds: float, sample_rate: int = SMART_TURN_SR) -> int:
    return max(0, int(round(seconds * sample_rate)))


def pad_left_to_8s(audio_f32: np.ndarray) -> np.ndarray:
    if len(audio_f32) >= SMART_TURN_MAX_SAMPLES:
        return audio_f32[-SMART_TURN_MAX_SAMPLES:]

    padded = np.zeros(SMART_TURN_MAX_SAMPLES, dtype=np.float32)
    padded[-len(audio_f32):] = audio_f32
    return padded


def should_reset_turn(
    prev_segment: Segment | None,
    current_segment: Segment,
    *,
    silence_reset_seconds: float = 0.8,
    reset_on_speaker_change: bool = True,
) -> bool:
    if prev_segment is None:
        return True

    gap = current_segment.start - prev_segment.end
    if gap >= silence_reset_seconds:
        return True

    if (
        reset_on_speaker_change
        and prev_segment.speaker is not None
        and current_segment.speaker is not None
        and prev_segment.speaker != current_segment.speaker
    ):
        return True

    return False


def score_segments_with_smart_turn(
    *,
    audio_f32_16k: np.ndarray,
    segments: Iterable[Segment],
    silence_reset_seconds: float = 0.8,
    reset_on_speaker_change: bool = True,
) -> list[Segment]:
    ordered = sorted(segments, key=lambda s: (s.start, s.end))

    scored: list[Segment] = []
    prev: Segment | None = None
    current_turn_start_time: float | None = None

    for seg in ordered:
        if should_reset_turn(
            prev,
            seg,
            silence_reset_seconds=silence_reset_seconds,
            reset_on_speaker_change=reset_on_speaker_change,
        ):
            current_turn_start_time = seg.start

        assert current_turn_start_time is not None

        turn_start_idx = seconds_to_sample_index(current_turn_start_time)
        seg_end_idx = seconds_to_sample_index(seg.end)

        window = audio_f32_16k[turn_start_idx:seg_end_idx]
        model_input = pad_left_to_8s(window)
        result = predict_endpoint(model_input)

        scored.append(
            replace(
                seg,
                end_of_turn_probability=float(result["probability"]),
                end_of_turn_prediction=int(result["prediction"]),
            )
        )

        prev = seg

    return scored
```

---

## Configuration Parameters

The pipeline should expose the following configuration values:

### Audio / model configuration

- `sample_rate = 16000`
- `max_window_seconds = 8.0`

These should remain fixed unless the underlying model requirements change.

### Turn reset configuration

- `silence_reset_seconds = 0.8`
- `reset_on_speaker_change = True`

These are tuning parameters and should be exposed to the caller.

### Optional post-processing threshold

Even though the model returns a binary prediction, the offline pipeline should primarily preserve the raw probability. A downstream threshold can later convert the score into a yes/no label.

Possible threshold candidates to evaluate:

- `0.5`
- `0.6`
- `0.7`

The best threshold depends on the dataset and the cost of false positives versus false negatives.

---

## Recommended Stored Fields

At minimum, store:

```python
segment.end_of_turn_probability
segment.end_of_turn_prediction
```

Recommended debugging fields:

```python
segment.smart_turn_window_start
segment.smart_turn_window_end
segment.smart_turn_reset_reason
```

Why store these?

- makes debugging easier
- lets you inspect why a probability was unexpectedly high or low
- helps compare heuristic reset logic across experiments

---

## Example Interpretation

Suppose the segments are:

| Segment | Start | End | Speaker | Text |
|---|---:|---:|---|---|
| 1 | 0.0 | 0.7 | A | "I was wondering" |
| 2 | 0.8 | 1.5 | A | "if maybe" |
| 3 | 1.6 | 2.8 | A | "you could help me with that" |
| 4 | 4.0 | 4.8 | B | "Sure" |

Processing would look like this:

- Segment 1: current turn starts at 0.0, score audio from 0.0 to 0.7
- Segment 2: same turn, score audio from 0.0 to 1.5
- Segment 3: same turn, score audio from 0.0 to 2.8
- Segment 4: new turn because of speaker change and silence gap, score audio from 4.0 to 4.8

This produces a score for each endpoint while still preserving turn context.

---

## Performance Considerations

### Resample once

Do not repeatedly reload or resample the file for each segment. Load the waveform once, convert once, and slice from memory.

### Avoid repeated copies

Use NumPy slices where possible. Only allocate a new array when building the final padded model input.

### Batch processing

If many files are being scored, process one file at a time and serialize the scored segment metadata to disk before moving to the next file.

### Caching

If experimentation involves repeated rescoring with different thresholds but the same Smart Turn outputs, store the raw probability once and avoid rerunning inference.

---

## Failure Modes and Pitfalls

### Pitfall 1: scoring only the isolated segment

This discards prior context in the same utterance and can make probabilities less meaningful.

### Pitfall 2: failing to reset turns

If the current-turn window spans multiple speakers or a long silence, the score may reflect the wrong conversational unit.

### Pitfall 3: very short segments

Very short fragments are especially likely to produce unstable scores if treated as standalone inputs.

### Pitfall 4: mismatched sample rate or channel layout

Always convert to mono 16 kHz before slicing windows for inference.

### Pitfall 5: trusting only the binary prediction

For offline analysis, the probability is often more valuable than the model’s immediate binary result.

---

## Evaluation Strategy

To validate this design, build a small labeled dataset where human annotators mark whether each segment truly ends a turn.

Then compare:

- isolated segment scoring
- accumulated current-turn scoring
- different silence reset thresholds
- reset vs no reset on speaker change
- different probability cutoffs

Useful evaluation measures:

- precision
- recall
- F1
- ROC / PR analysis if enough data exists
- manual review of borderline cases

---

## Recommended Next Steps

1. Implement the scoring function against the real segment class.
2. Add optional debug fields to the output.
3. Run the scorer on a small validation file.
4. Inspect probabilities around obvious turn boundaries.
5. Tune:
   - silence reset threshold
   - speaker-change reset behavior
   - downstream decision threshold
6. Add an evaluation notebook or script for quantitative analysis.

---

## Final Recommendation

The recommended design is:

**For each segment endpoint, run Smart Turn on the entire current-turn audio up to that endpoint, capped to the most recent 8 seconds, and store the returned probability on the segment.**

This preserves the context Smart Turn needs, fits the model’s expected input pattern, and provides a strong offline approximation of live turn-completion scoring.
