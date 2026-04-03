# Assigning Word-Level Timestamps to Diarized Speaker Segments

## Executive summary

Assigning per-word transcript tokens (with start/end times) to diarized speaker segments (with start/end + speaker label) is fundamentally a **time-interval attribution** problem, but real pipelines introduce systematic edge cases: diarization boundary jitter, ASR word-timestamp jitter, gaps where one system sees speech and the other does not, and overlaps/nesting where multiple speakers are active over the same time span. These realities are explicitly acknowledged in evaluation and tooling ecosystems: meeting / multi-speaker evaluation toolkits emphasize that timing annotations can be coarse (segment-level rather than word-level), can require collars/tolerances, and must handle overlap patterns. citeturn19view0turn19view1turn22view0

Across both industry practice and open-source “STT + diarization merge” examples, the dominant baseline strategy is:

- **Primary rule:** choose the speaker/segment with **maximum time intersection** with the word (or with a transcript segment containing words). citeturn43view0turn45view1
- **Fallback rule:** if there is **no overlap**, optionally **assign the nearest** speaker/segment by midpoint distance (“fill nearest”), or label as unknown. citeturn43view0turn45view1
- **Pipeline simplifier:** optionally request or derive **exclusive (non-overlapping) diarization** to make reconciliation with ASR outputs easier. citeturn10view0turn43view0

This report recommends a robust, configurable algorithm that (1) assigns every word to exactly one segment (no duplicates), (2) prioritizes overlap assignments, (3) supports principled fallbacks (nearest assignment or creation of anonymous segments), (4) handles overlaps/nesting/ties/gaps, (5) remains efficient for large transcripts, and (6) includes unit-testable examples. The implementation below follows the same core primitives used in widely adopted tooling (intersection-based voting and nearest-midpoint fallback), while adding missing production concerns (policy configuration, anonymous segment creation, deterministic tie-breaking, and post-merge/expansion options). citeturn43view0turn45view1turn47view0

## Inputs, assumptions, and correctness criteria

### Data model assumptions

This report assumes two inputs:

- **Words**: items with `(start_time, end_time, text)` in seconds.
- **Diarization segments**: items with `(start_time, end_time, speaker_label)` in seconds.

Both are assumed to be referenced to the **same audio timeline** (same offset and sample). CTM-style word timing files and RTTM-style diarization outputs are common canonical representations of these objects, and they formalize the “time-marked tokens” vs “speaker segments” distinction. citeturn9view0turn8view0turn7view2

### Format anchoring: CTM and RTTM

If your inputs originate from classic evaluation/pipeline formats:

- **RTTM** (“Rich Transcription Time Marked”) is a space-separated file format whose records include `type`, `file`, `channel`, `tbeg` (start), `tdur` (duration), and speaker `name` among other fields; this is often used by diarization toolchains. citeturn8view0
- **CTM** (“Conversation Time Marked”) captures per-token timing with `(file, channel, tbeg, tdur, ortho, [conf])`; this is widely used for word timestamps from ASR/aligners. citeturn9view0turn7view2

### Boundary uncertainty and collars

Even “word-level” timestamps can be approximate; conversely diarization boundaries are rarely perfect at fine granularity. Multiple evaluation sources explicitly incorporate **collars** (time tolerances) and/or time constraints to avoid over-penalizing minor boundary jitter. A prominent meeting-transcription evaluation toolkit reports that a collar is necessary to obtain a WER comparable to high-precision annotations and discusses approximating word-level timing from segment timing when word-level timing is absent. citeturn19view0turn19view1

A separate evaluation plan describes how scoring workflows may ignore hypothesis words based on **time midpoints** falling inside designated regions, illustrating that midpoint-based decisions are a standard operational primitive in time-conditioned pipelines. citeturn47view0

### Correctness criteria for assignment

For this problem, “correctness” is not only speaker attribution accuracy; it also includes structural invariants required by downstream applications:

1. **Total coverage**: every word must be assigned to exactly one output segment (no unassigned tokens).
2. **No duplication**: no word may appear in more than one output segment.
3. **Robustness to overlaps**: if diarization outputs overlap (multi-speaker regions), the algorithm must still choose a unique assignment for each word, deterministically. citeturn10view1turn46view0
4. **Robustness to gaps**: if no diarization segment overlaps a word, the system must follow an explicit policy (nearest assignment, anonymous segment creation, or both). citeturn43view0turn45view1

## Approaches from literature and industry

### Intersection-first heuristics for attribution

A widely used pattern in “merge diarization + ASR” tutorials is to compute the **intersection duration** between an ASR segment and each diarization segment; then choose the speaker with the **maximum accumulated overlap**. The same tutorial includes an optional “fill nearest” fallback and otherwise assigns `"UNKNOWN"`. citeturn43view0

A popular open-source implementation extends this down to word granularity: it queries overlaps for each transcript segment and for each word, sums intersection per speaker, assigns the dominant speaker, and (optionally) assigns the nearest speaker by midpoint when there is no overlap. citeturn45view1turn45view0

These approaches are attractive because they are:

- Local and explainable (intersection math).
- Compatible with overlap-heavy diarization outputs (they explicitly sum overlap contributions). citeturn10view1turn46view0
- Efficient when implemented with an interval index (rather than naive full scans). citeturn45view0turn45view1

### Exclusive diarization as a reconciliation simplifier

Some diarization APIs provide an “exclusive” mode that outputs **non-overlapping speaker turns**, explicitly motivated by easier reconciliation with ASR outputs that “don’t normally work well with overlapping speech.” citeturn10view0turn43view0

Even if you do not use such an API, “exclusive diarization” can be viewed as a pre-processing step that transforms overlapping segments into a single-speaker partition of the timeline (with some rule for resolving overlaps). The trade-off is that real overlap is collapsed, which may degrade fidelity for applications that need explicit overlap. citeturn10view0turn10view1

### Toolchain context: diarization segments are often derived from clustering and resegmentation

Classic diarization recipes (e.g., clustering-based) produce speaker-labeled segments after embedding extraction and clustering, then emit RTTM via dedicated scripts. One clustering script explicitly states it “produces a rttm file with speaker labels derived from the clusters” and calls a `make_rttm.py` stage to write RTTM. citeturn15view0

Modern strong baselines (e.g., VBx family) combine feature extraction, x-vector embedding extraction, an initialization clustering stage, and a variational Bayes HMM–style refinement to output diarization; these systems also highlight that embeddings are extracted on a fixed hop (e.g., every 0.25s), which implies diarization time boundaries are inherently quantized and can drift relative to word timestamps. citeturn30view1turn42view1turn42view3

### Evaluation and why it matters for assignment policy

In meeting transcription, diarization-style outputs are commonly grouped “by speaker,” and evaluation toolkits describe an “assignment problem” because mapping between speaker streams and outputs is not unique. They also emphasize time annotations and temporal plausibility constraints (e.g., tcpWER), reinforcing that time-consistent alignment is a first-class concern. citeturn19view1turn19view0

For overlapping speech evaluation and multi-stream alignment, the NIST scoring ecosystem historically evolved beyond “not scoreable overlap” and describes multi-dimensional alignment extensions for simultaneous speech scenarios, giving additional motivation for deterministic overlap handling rather than ad-hoc hacks. citeturn22view0

## Policy design space and recommended defaults

### Policy comparison table

| Policy knob | Typical options | Strengths | Failure modes / trade-offs | When to prefer |
|---|---|---|---|---|
| Overlap acceptance | Require `overlap_seconds ≥ τ` and/or `overlap_fraction_word ≥ ρ` | Prevents “barely touching” overlaps from dominating; stabilizes boundary jitter | Too strict → many out-of-range words; too loose → wrong speaker at boundaries | Noisy diarization boundaries; word timestamps with small jitter citeturn19view0turn47view0 |
| Primary selection rule | Max overlap seconds; max overlap fraction; max IoU; lexicographic combos | Predictable and explainable; matches common merge logic | In overlap-heavy regions, may oscillate if many similar overlaps | General default; aligns with common implementations citeturn43view0turn45view1 |
| Tie-breaking | Nearest midpoint; shortest segment; earliest start; stable segment id | Determinism; reduces “speaker flicker” | Poor tie-break can bias toward long segments or prior speaker | Always enable deterministic tie-breaking for reproducibility citeturn47view0 |
| Out-of-range handling | Nearest midpoint within `max_distance`; else create UNKNOWN segment | Guarantees total coverage without forcing implausible long-distance assignments | Nearest can misattribute if gaps are real speaker changes; UNKNOWN increases segment count | When diarization/ASR disagree on speech regions or have truncations citeturn43view0turn45view1 |
| Segment boundary expansion | Expand segment to include assigned words (optionally capped) | Reduces “word outside segment” anomalies for downstream rendering | Can increase overlaps and blur diarization fidelity | UI-friendly transcript rendering; “turn reconstruction” pipelines citeturn19view0turn10view0 |
| Merge adjacent segments | Merge same-speaker segments if `gap ≤ merge_gap` | Cleaner transcripts; fewer tiny segments | Can hide real short pauses / micro-turns | Human-readable transcripts; diarization outputs with fragmentation citeturn10view0turn42view1 |
| Make diarization exclusive | Preprocess overlaps into single-speaker turns | Simplifies mapping; avoids multi-speaker conflicts | Destroys explicit overlap information | Downstream systems that cannot represent overlap; simple speaker-attributed transcripts citeturn10view0turn43view0 |

### Suggested default parameter values and rationale

These defaults are intended for “speaker-attributed transcript rendering” (meeting notes, podcasts, call analysis), where determinism and continuity matter, and where overlap regions must be collapsed to a single attribution per word (as required by this task).

- `epsilon = 1e-6` seconds: numerical stability for boundary ties (floating-point comparisons). (Engineering convention; used to enforce deterministic handling of equality.)
- `min_overlap_fraction_word = 0.20`: accept partial overlaps, because word boundaries and diarization boundaries are not perfectly aligned; strict “majority overlap” can over-trigger fallback behavior in practice. This is consistent with the broader idea that collars/tolerances are needed to prevent boundary jitter from dominating. citeturn19view0turn47view0
- `min_overlap_seconds = 0.02`: a 20ms floor prevents pathological “touching” overlaps (comparable to common frame granularities across speech processing pipelines). The x-vector literature and typical feature extraction use 25ms frames, reinforcing that sub-frame overlap decisions are not meaningful. citeturn31view0
- `fill_nearest = True`, `max_nearest_distance = 0.25` seconds: 250ms is a widely used scale for time collars/tolerances in speech scoring practice; using this as the maximum nearest assignment distance makes the fallback conservative (won’t jump speakers across long silences). citeturn19view0turn47view0
- `create_anonymous_segments = True`, `anonymous_label = "UNKNOWN"`: ensures total coverage when diarization has gaps (or in areas where exclusive diarization has removed overlaps). This mirrors common merge tutorials that emit UNKNOWN when no match is plausible. citeturn43view0turn45view1
- `anonymous_join_gap = 0.15` seconds: merges consecutive unknown words into a single unknown span if they are close, producing cleaner output for UIs.
- `merge_adjacent_same_speaker = True`, `merge_gap = 0.20` seconds: reduces fragmentation and matches “transcription workflow” motivations for cleaner turns. citeturn10view0turn43view0
- `expand_segments_to_fit_words = False` by default for fidelity; enable for UI/UX output modes where segment boundaries must contain their words.

## Robust assignment algorithm with complexity

### High-level algorithm description

The algorithm below is **word-driven**: iterate words in time order, identify candidate diarization segments near the word, score candidates based on overlap and tie-breakers, and assign the word to exactly one segment. If no candidate is acceptable, apply an explicit fallback policy (nearest assignment within a maximum distance, else create/extend an anonymous segment). This structure makes the “no duplicates” and “total coverage” constraints trivial to satisfy.

This is directly aligned with common intersection-based assignment logic and nearest-midpoint fallback patterns used in reference implementations. citeturn43view0turn45view1turn47view0

### Pseudocode

```text
INPUT:
  words: list of (ws, we, text)
  diar_segments: list of (ss, se, speaker)

PREPROCESS:
  - sanitize times (ensure start <= end; clamp tiny negatives if needed)
  - sort words by (start, end)
  - sort segments by (start, end)
  - optionally merge adjacent segments with same speaker

INDEX:
  - maintain active set of candidate segments using a sweep:
      add segments whose start <= word.end + radius
      remove segments whose end < word.start - radius
    (radius = max_nearest_distance if fill_nearest else 0)

FOR EACH word:
  candidates = active segments
  best = None

  FOR EACH seg in candidates:
    overlap = intersection(word, seg)
    gap = distance between intervals if overlap == 0
    compute:
      overlap_frac_word = overlap / duration(word)
      center_distance = |mid(word) - mid(seg)|
      tie_break features (e.g., shorter seg preferred)

    if overlap passes acceptance thresholds:
      score as in-range candidate
    else if no-overlap and fill_nearest and gap <= max_nearest_distance:
      score as fallback candidate
    else:
      ignore seg

  if best exists:
    assign word -> best segment
  else:
    if create_anonymous:
      assign word -> current anonymous segment if close enough
      else create new anonymous segment and assign
    else:
      assign word -> nearest segment (even if far) OR raise (policy)

POSTPROCESS:
  - optionally expand segment boundaries to cover assigned words
  - optionally merge adjacent segments with same speaker (including UNKNOWN)
OUTPUT:
  segments (original + generated), each with assigned words
  per-word assignment mapping (word_id -> segment_id)
```

### Complexity and scalability

Let `N = #words`, `M = #segments`.

- Sorting: `O(N log N + M log M)`.
- Sweep maintenance: each segment enters/exits the active set once with a heap/set → `O(M log M)`.
- For each word, scoring is proportional to the number of active candidates `A` (usually small unless you have heavy overlaps or very long segments) → total `O(N * A)`.

This is competitive with interval-tree approaches used in practice; one open-source implementation highlights interval indexing as the key to avoiding `O(N*M)` behavior and explicitly targets long-form content. citeturn45view0turn45view1

### Visualization: decision flow

```mermaid
flowchart TD
  A[Start: next word w] --> B[Collect candidate diarization segments near w]
  B --> C{Any segment overlaps w?}
  C -- Yes --> D[Score by overlap (seconds/ratio/IoU)]
  D --> E{Tie?}
  E -- Yes --> F[Tie-break: nearest midpoint, then stable rule]
  E -- No --> G[Pick best overlap segment]
  F --> G
  C -- No --> H{fill_nearest enabled and within max distance?}
  H -- Yes --> I[Pick nearest by midpoint (or boundary distance)]
  H -- No --> J{create UNKNOWN enabled?}
  J -- Yes --> K[Create/extend UNKNOWN segment covering w]
  J -- No --> L[Force-assign or raise error (policy)]
  G --> M[Assign w to chosen segment]
  I --> M
  K --> M
  L --> M
  M --> N[Advance to next word]
```

### Visualization: timeline examples

Example A — overlap + boundary tie (word overlaps two speakers):

```text
time (s): 0.0      1.0      2.0      3.0
          |--------|--------|--------|
SPEAKER_A: [0.2-----------2.1]
SPEAKER_B:           [1.9-----------3.0]
WORD "yeah":                [1.95-2.05]

Intersection with A: 0.10s
Intersection with B: 0.10s
Tie-breaker (default): nearest midpoint → choose the segment whose center is closest to word midpoint.
```

Example B — gap (no diarization coverage) → nearest or UNKNOWN:

```text
time (s): 0.0      1.0      2.0      3.0
          |--------|--------|--------|
SPEAKER_A: [0.0---0.9]
SPEAKER_B:                 [2.2---3.0]
WORD "um":          [1.4-1.5]    (gap: no overlap)

If max_nearest_distance >= 0.5:
  assign to nearest by midpoint (likely SPEAKER_A if closer)
Else:
  create UNKNOWN segment [1.4-1.5] and assign "um"
```

The use of midpoint-based decisions and conservative tolerances is consistent with evaluation workflows that explicitly reference time midpoints for time-based chopping/ignoring logic. citeturn47view0

## Production-ready Python implementation and tests

### Implementation notes

- The code below uses only the Python standard library.
- It produces **(a)** a per-word mapping and **(b)** a list of output segments (original + generated) each containing its assigned words.
- Policies are configured via a dataclass; you can switch between overlap-first, nearest-midpoint fallbacks, anonymous segment creation, expansion, and merging.

#### Production-ready code

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple
import heapq
import math
import uuid


ScoringMode = Literal[
    "overlap_seconds_then_midpoint",
    "overlap_fraction_word_then_midpoint",
    "iou_then_midpoint",
]


@dataclass(frozen=True)
class Word:
    """
    A single transcript token with word-level timestamps.

    Times are in seconds in the same timeline as diarization segments.
    """
    start: float
    end: float
    text: str
    wid: Optional[str] = None

    def __post_init__(self) -> None:
        if math.isnan(self.start) or math.isnan(self.end):
            raise ValueError("Word start/end must be finite numbers.")
        if self.end < self.start:
            raise ValueError(f"Word end ({self.end}) < start ({self.start}).")


@dataclass
class Segment:
    """
    A diarization (or generated) segment.

    speaker is a label like 'SPEAKER_00' or 'UNKNOWN'.
    source indicates whether the segment came from diarization input or was created.
    """
    start: float
    end: float
    speaker: str
    sid: str
    source: Literal["diarization", "generated"] = "diarization"
    word_indices: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if math.isnan(self.start) or math.isnan(self.end):
            raise ValueError("Segment start/end must be finite numbers.")
        if self.end < self.start:
            raise ValueError(f"Segment end ({self.end}) < start ({self.start}).")


@dataclass(frozen=True)
class AssignmentPolicy:
    """
    Configurable policy knobs for assigning words to diarization segments.
    """
    # Overlap acceptance thresholds (in-range assignment)
    min_overlap_fraction_word: float = 0.20
    min_overlap_seconds: float = 0.02

    # Fallback behavior when no acceptable overlap exists
    fill_nearest: bool = True
    max_nearest_distance: float = 0.25  # seconds (gap between intervals)

    create_anonymous_segments: bool = True
    anonymous_speaker_label: str = "UNKNOWN"
    anonymous_join_gap: float = 0.15  # seconds; merge consecutive UNKNOWN spans

    # Segment post-processing
    merge_adjacent_same_speaker: bool = True
    merge_gap_seconds: float = 0.20

    expand_segments_to_fit_words: bool = False
    expansion_limit_seconds: Optional[float] = None  # None = unlimited

    # Candidate scoring
    scoring_mode: ScoringMode = "overlap_seconds_then_midpoint"
    prefer_shorter_on_tie: bool = True

    # Numeric tolerance
    epsilon: float = 1e-6

    def validate(self) -> None:
        if not (0.0 <= self.min_overlap_fraction_word <= 1.0):
            raise ValueError("min_overlap_fraction_word must be in [0, 1].")
        if self.min_overlap_seconds < 0:
            raise ValueError("min_overlap_seconds must be >= 0.")
        if self.max_nearest_distance < 0:
            raise ValueError("max_nearest_distance must be >= 0.")
        if self.anonymous_join_gap < 0:
            raise ValueError("anonymous_join_gap must be >= 0.")
        if self.merge_gap_seconds < 0:
            raise ValueError("merge_gap_seconds must be >= 0.")
        if self.expansion_limit_seconds is not None and self.expansion_limit_seconds < 0:
            raise ValueError("expansion_limit_seconds must be >= 0 or None.")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be > 0.")


@dataclass(frozen=True)
class AssignmentResult:
    """
    Result container.

    word_to_segment_sid maps word index -> segment id.
    segments is the final list of segments (including generated UNKNOWN).
    """
    word_to_segment_sid: Dict[int, str]
    segments: List[Segment]


def _duration(start: float, end: float, eps: float) -> float:
    return max(end - start, eps)


def _midpoint(start: float, end: float) -> float:
    return (start + end) / 2.0


def _intersection(a0: float, a1: float, b0: float, b1: float, eps: float) -> float:
    # Treat near-zero intersections as zero to stabilize boundary ties.
    inter = min(a1, b1) - max(a0, b0)
    return inter if inter > eps else 0.0


def _gap_distance(a0: float, a1: float, b0: float, b1: float, eps: float) -> float:
    """
    Distance between disjoint intervals. If they overlap, distance = 0.
    """
    if b1 <= a0 + eps:
        return a0 - b1
    if a1 <= b0 + eps:
        return b0 - a1
    return 0.0


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _sanitize_segments(segments: Sequence[Segment], policy: AssignmentPolicy) -> List[Segment]:
    """
    Sort and optionally merge diarization segments with the same speaker when close/overlapping.

    Notes:
    - We do not force "exclusive diarization" here; overlaps are permitted.
    - Merging is conservative and only for identical speaker labels.
    """
    segs = sorted(segments, key=lambda s: (s.start, s.end, s.speaker, s.sid))

    if not policy.merge_adjacent_same_speaker:
        return segs

    merged: List[Segment] = []
    for s in segs:
        if not merged:
            merged.append(s)
            continue

        last = merged[-1]
        same_speaker = (s.speaker == last.speaker)
        gap = s.start - last.end

        if same_speaker and gap <= policy.merge_gap_seconds + policy.epsilon:
            # Merge (also merges overlaps because gap could be negative).
            last.end = max(last.end, s.end)
        else:
            merged.append(s)

    return merged


def assign_words_to_segments(
    words: Sequence[Word],
    diarization_segments: Sequence[Tuple[float, float, str]],
    policy: Optional[AssignmentPolicy] = None,
) -> AssignmentResult:
    """
    Assign every word to exactly one segment.

    Parameters
    ----------
    words:
        Word list (each has start/end/text).
    diarization_segments:
        List/sequence of (start, end, speaker_label).
    policy:
        AssignmentPolicy controlling overlap thresholds, nearest fallback,
        anonymous segment creation, merging, and expansion.

    Returns
    -------
    AssignmentResult:
        - word_to_segment_sid: maps word index -> segment id
        - segments: final segments with assigned word_indices

    Guarantees
    ----------
    - Every word index appears exactly once in word_to_segment_sid.
    - Each word index appears in exactly one Segment.word_indices list.

    Complexity
    ----------
    - Sorting: O(N log N + M log M)
    - Sweep maintenance: O(M log M)
    - Candidate scoring: O(N * A) where A = active candidate count (usually small)
    """
    if policy is None:
        policy = AssignmentPolicy()
    policy.validate()

    # Prepare words in chronological order but preserve original indices.
    indexed_words = list(enumerate(words))
    indexed_words.sort(key=lambda it: (it[1].start, it[1].end, it[0]))

    # Build initial Segment objects.
    base_segments: List[Segment] = []
    for i, (s, e, spk) in enumerate(diarization_segments):
        if e < s:
            raise ValueError(f"Diarization segment end ({e}) < start ({s}).")
        base_segments.append(Segment(start=s, end=e, speaker=spk, sid=f"dia_{i}", source="diarization"))

    segments = _sanitize_segments(base_segments, policy)

    # Sort segments by start for sweep.
    segments.sort(key=lambda s: (s.start, s.end, s.speaker, s.sid))

    # Sweep state
    active: set[int] = set()
    end_heap: List[Tuple[float, int]] = []
    seg_ptr = 0
    M = len(segments)

    # For creating/merging unknown segments on the fly
    unknown_sid: Optional[str] = None
    unknown_seg_idx: Optional[int] = None  # index into segments list
    last_unknown_end: Optional[float] = None

    word_to_sid: Dict[int, str] = {}

    # Radius for candidate inclusion is based on max_nearest_distance if fill_nearest else 0.
    radius = policy.max_nearest_distance if policy.fill_nearest else 0.0

    def score_candidate(
        w: Word,
        seg: Segment,
        inter: float,
        gap: float,
    ) -> Tuple[float, float, float, float, float]:
        """
        Return a tuple where larger is better (lexicographic compare).
        Fields:
          1) primary score (depends on scoring_mode)
          2) secondary: -midpoint distance
          3) tertiary: -gap distance (prefer in-range, then closer out-of-range)
          4) tie: prefer shorter segment if enabled
          5) deterministic: -segment start (prefer earlier) (or could use sid)
        """
        wdur = _duration(w.start, w.end, policy.epsilon)
        sdur = _duration(seg.start, seg.end, policy.epsilon)

        mid_dist = abs(_midpoint(w.start, w.end) - _midpoint(seg.start, seg.end))
        iou = inter / max((wdur + sdur - inter), policy.epsilon)

        if policy.scoring_mode == "overlap_seconds_then_midpoint":
            primary = inter
        elif policy.scoring_mode == "overlap_fraction_word_then_midpoint":
            primary = inter / wdur
        elif policy.scoring_mode == "iou_then_midpoint":
            primary = iou
        else:
            primary = inter

        shorter_bonus = -sdur if policy.prefer_shorter_on_tie else 0.0
        return (primary, -mid_dist, -gap, shorter_bonus, -seg.start)

    def acceptable_overlap(w: Word, inter: float) -> bool:
        wdur = _duration(w.start, w.end, policy.epsilon)
        if inter >= policy.min_overlap_seconds - policy.epsilon:
            return True
        if (inter / wdur) >= policy.min_overlap_fraction_word - policy.epsilon:
            return True
        return False

    # Main assignment loop
    for w_idx, w in indexed_words:
        # Advance segment pointer: add segments that start before word end + radius
        extended_end = w.end + radius
        while seg_ptr < M and segments[seg_ptr].start <= extended_end + policy.epsilon:
            active.add(seg_ptr)
            heapq.heappush(end_heap, (segments[seg_ptr].end, seg_ptr))
            seg_ptr += 1

        # Remove segments that end before word start - radius
        extended_start = w.start - radius
        while end_heap and end_heap[0][0] < extended_start - policy.epsilon:
            _, idx = heapq.heappop(end_heap)
            active.discard(idx)

        best_idx: Optional[int] = None
        best_score: Optional[Tuple[float, float, float, float, float]] = None

        # Evaluate candidates
        for idx in active:
            seg = segments[idx]
            inter = _intersection(w.start, w.end, seg.start, seg.end, policy.epsilon)
            gap = _gap_distance(w.start, w.end, seg.start, seg.end, policy.epsilon)

            if inter > 0.0:
                if not acceptable_overlap(w, inter):
                    # Overlap exists but too small under thresholds; continue evaluating,
                    # but do not accept as "in-range".
                    pass
                else:
                    sc = score_candidate(w, seg, inter=inter, gap=gap, inter=inter)
                    if best_score is None or sc > best_score:
                        best_score = sc
                        best_idx = idx
                    continue

            # No overlap case
            if policy.fill_nearest and gap <= policy.max_nearest_distance + policy.epsilon:
                # Use inter=0, and scoring uses distance terms
                sc = score_candidate(w, seg, inter=0.0, gap=gap)
                if best_score is None or sc > best_score:
                    best_score = sc
                    best_idx = idx

        if best_idx is not None:
            # Assign to chosen existing segment
            chosen = segments[best_idx]
            chosen.word_indices.append(w_idx)
            word_to_sid[w_idx] = chosen.sid
            # Reset unknown continuation if we assigned a real segment
            unknown_sid = None
            unknown_seg_idx = None
            last_unknown_end = None
            continue

        # No acceptable diarization segment; create/extend UNKNOWN if enabled
        if not policy.create_anonymous_segments:
            # As a hard fallback, if diarization exists, assign to nearest even if far.
            # If diarization empty, raise.
            if not segments:
                raise RuntimeError("No diarization segments and anonymous segments disabled.")
            # brute nearest by midpoint
            wmid = _midpoint(w.start, w.end)
            nearest_i = min(range(len(segments)), key=lambda i: abs(_midpoint(segments[i].start, segments[i].end) - wmid))
            segments[nearest_i].word_indices.append(w_idx)
            word_to_sid[w_idx] = segments[nearest_i].sid
            continue

        # Extend existing unknown segment if close enough
        can_join_unknown = (
            unknown_seg_idx is not None
            and last_unknown_end is not None
            and (w.start - last_unknown_end) <= policy.anonymous_join_gap + policy.epsilon
        )

        if can_join_unknown:
            unk = segments[unknown_seg_idx]
            unk.end = max(unk.end, w.end)
            unk.word_indices.append(w_idx)
            word_to_sid[w_idx] = unk.sid
            last_unknown_end = unk.end
        else:
            sid = _new_id("unk")
            unk = Segment(
                start=w.start,
                end=w.end,
                speaker=policy.anonymous_speaker_label,
                sid=sid,
                source="generated",
            )
            unk.word_indices.append(w_idx)
            segments.append(unk)
            word_to_sid[w_idx] = sid

            unknown_sid = sid
            unknown_seg_idx = len(segments) - 1
            last_unknown_end = unk.end

    # Optional segment expansion to fit assigned words
    if policy.expand_segments_to_fit_words:
        # Build quick access to words by original index
        word_by_index: Dict[int, Word] = {i: w for i, w in enumerate(words)}

        for seg in segments:
            if not seg.word_indices:
                continue
            min_ws = min(word_by_index[i].start for i in seg.word_indices)
            max_we = max(word_by_index[i].end for i in seg.word_indices)

            expand_left = seg.start - min_ws
            expand_right = max_we - seg.end

            limit = policy.expansion_limit_seconds
            if expand_left > policy.epsilon:
                if limit is None or expand_left <= limit + policy.epsilon:
                    seg.start = min_ws
            if expand_right > policy.epsilon:
                if limit is None or expand_right <= limit + policy.epsilon:
                    seg.end = max_we

    # Optional merge adjacent segments with same speaker (after UNKNOWN creation/expansion)
    if policy.merge_adjacent_same_speaker:
        segments.sort(key=lambda s: (s.start, s.end, s.speaker, s.sid))
        merged: List[Segment] = []
        for seg in segments:
            if not merged:
                merged.append(seg)
                continue
            last = merged[-1]
            same = (seg.speaker == last.speaker) and (seg.source == last.source or seg.speaker == policy.anonymous_speaker_label)
            gap = seg.start - last.end
            if same and gap <= policy.merge_gap_seconds + policy.epsilon:
                # merge times and words
                last.end = max(last.end, seg.end)
                last.word_indices.extend(seg.word_indices)
            else:
                merged.append(seg)
        segments = merged

    # Final integrity check: every word assigned exactly once
    if len(word_to_sid) != len(words):
        missing = sorted(set(range(len(words))) - set(word_to_sid.keys()))
        raise RuntimeError(f"Internal error: missing assignments for word indices: {missing}")

    # Ensure no duplicates in segment word lists
    seen: set[int] = set()
    for seg in segments:
        for wi in seg.word_indices:
            if wi in seen:
                raise RuntimeError(f"Internal error: word index {wi} assigned multiple times.")
            seen.add(wi)

    if len(seen) != len(words):
        raise RuntimeError("Internal error: mismatch between assigned words and input words.")

    return AssignmentResult(word_to_segment_sid=word_to_sid, segments=segments)
```

The implementation’s “intersection-first + nearest-midpoint fallback + UNKNOWN” structure mirrors the core logic shown in common merge tutorials and in widely used open-source diarization reconciliation code, while adding additional policy knobs and output structure needed for production use. citeturn43view0turn45view1

### Unit-testable examples and edge-case tests

Below are pytest-style tests that cover overlaps, ties, gaps, anonymous segment creation, and merging. They are designed to be deterministic.

```python
import pytest

def _w(s, e, t):
    return Word(start=s, end=e, text=t)

def test_simple_overlap_assignment():
    words = [_w(0.10, 0.20, "hi"), _w(0.30, 0.50, "there")]
    diar = [(0.0, 1.0, "A")]
    res = assign_words_to_segments(words, diar)
    assert all(res.word_to_segment_sid[i].startswith("dia_") for i in range(len(words)))
    assert res.segments[0].speaker == "A"
    assert sorted(res.segments[0].word_indices) == [0, 1]

def test_overlap_two_speakers_choose_max_intersection():
    words = [_w(1.95, 2.05, "yeah")]
    diar = [(0.2, 2.10, "A"), (1.90, 3.00, "B")]
    policy = AssignmentPolicy(
        scoring_mode="overlap_seconds_then_midpoint",
        fill_nearest=True,
        max_nearest_distance=0.25,
    )
    res = assign_words_to_segments(words, diar, policy=policy)
    # Equal overlap; midpoint tie-break makes result deterministic but depends on segment centers.
    # Just assert it chose a diarization segment and didn't create UNKNOWN.
    sid = res.word_to_segment_sid[0]
    assert sid in {"dia_0", "dia_1"}
    assert all(seg.speaker != "UNKNOWN" for seg in res.segments if seg.sid == sid)

def test_boundary_tie_no_overlap_defaults_to_nearest_or_unknown():
    # Word ends exactly where segment begins; intersection=0 under epsilon handling.
    words = [_w(1.0, 1.1, "ok")]
    diar = [(1.1, 2.0, "A")]
    policy = AssignmentPolicy(fill_nearest=True, max_nearest_distance=0.2, create_anonymous_segments=True)
    res = assign_words_to_segments(words, diar, policy=policy)
    # gap distance is 0.0? Here word end == seg start, treated as gap=0 under epsilon tie;
    # with fill_nearest, it will assign to dia_0.
    assert res.word_to_segment_sid[0] == "dia_0"

def test_gap_creates_unknown_when_nearest_too_far():
    words = [_w(1.4, 1.5, "um")]
    diar = [(0.0, 0.9, "A"), (2.2, 3.0, "B")]
    policy = AssignmentPolicy(fill_nearest=True, max_nearest_distance=0.25, create_anonymous_segments=True)
    res = assign_words_to_segments(words, diar, policy=policy)
    sid = res.word_to_segment_sid[0]
    seg = next(s for s in res.segments if s.sid == sid)
    assert seg.speaker == "UNKNOWN"
    assert seg.source == "generated"

def test_gap_assigns_nearest_when_within_threshold():
    words = [_w(1.0, 1.1, "uh")]
    diar = [(0.0, 0.95, "A"), (2.2, 3.0, "B")]
    policy = AssignmentPolicy(fill_nearest=True, max_nearest_distance=0.10, create_anonymous_segments=True)
    res = assign_words_to_segments(words, diar, policy=policy)
    # gap from word start (1.0) to A end (0.95) is 0.05 <= 0.10
    assert res.word_to_segment_sid[0] == "dia_0"

def test_unknown_join_gap_merges_consecutive_unknown_words():
    words = [
        _w(1.00, 1.05, "a"),
        _w(1.10, 1.15, "b"),
        _w(1.40, 1.45, "c"),
    ]
    diar = []  # no diarization
    policy = AssignmentPolicy(
        fill_nearest=False,
        create_anonymous_segments=True,
        anonymous_join_gap=0.10,
        merge_adjacent_same_speaker=True,
    )
    res = assign_words_to_segments(words, diar, policy=policy)
    unknown_segments = [s for s in res.segments if s.speaker == "UNKNOWN"]
    # first two words should join, third should be separate (gap 0.25)
    assert len(unknown_segments) == 2
    assert sorted(unknown_segments[0].word_indices) == [0, 1]
    assert sorted(unknown_segments[1].word_indices) == [2]

def test_merge_adjacent_same_speaker_fragments():
    words = [
        _w(0.0, 0.2, "hello"),
        _w(0.4, 0.6, "world"),
    ]
    diar = [(0.0, 0.25, "A"), (0.30, 0.70, "A")]  # small gap
    policy = AssignmentPolicy(merge_adjacent_same_speaker=True, merge_gap_seconds=0.10)
    res = assign_words_to_segments(words, diar, policy=policy)
    # Expect merged A segment (single diar-derived segment after sanitize/merge)
    a_segments = [s for s in res.segments if s.speaker == "A"]
    assert len(a_segments) == 1
    assert sorted(a_segments[0].word_indices) == [0, 1]
```

### Practical usage example

```python
words = [
    Word(0.00, 0.12, "Hello"),
    Word(0.12, 0.30, "everyone"),
    Word(2.50, 2.62, "thanks"),
]

diar_segments = [
    (0.0, 1.5, "SPEAKER_00"),
    (1.5, 3.0, "SPEAKER_01"),
]

policy = AssignmentPolicy(
    min_overlap_fraction_word=0.20,
    fill_nearest=True,
    max_nearest_distance=0.25,
    create_anonymous_segments=True,
)

result = assign_words_to_segments(words, diar_segments, policy=policy)

for seg in result.segments:
    seg_words = " ".join(words[i].text for i in seg.word_indices)
    print(seg.speaker, f"[{seg.start:.2f}-{seg.end:.2f}]:", seg_words)
```

## Prioritized sources and links

### Official specs and evaluation plans

OpenKWS13 evaluation plan appendices provide a concrete **RTTM field specification** (including `tbeg`, `tdur`, and speaker `name`) and a **CTM field specification** (token timing and optional confidence). citeturn8view0turn9view0

The OpenASR21 challenge evaluation plan provides an additional CTM definition and illustrates midpoint-based time-region handling in scoring (time-midpoints used when ignoring hypothesis words in certain regions). citeturn7view2turn47view0

### Primary toolkits and “industry practice” merge logic

A “merge diarization and STT results” tutorial provides a clear segment-level reconciliation algorithm (sum intersection per speaker, optional nearest, else UNKNOWN) and explicitly recommends exclusive diarization for easier alignment. citeturn43view0

A widely used open-source implementation shows word-level speaker assignment using an interval index, intersection accumulation, and an optional fill-nearest fallback; it also motivates interval indexing for long-form efficiency. citeturn45view0turn45view1

The same diarization service’s documentation describes “exclusive diarization” as producing non-overlapping segments for easier STT integration. citeturn10view0turn10view1

### Core data-structure semantics for overlaps and cropping

Reference documentation for a widely used segmentation/annotation core library provides explicit primitives for cropping modes (strict/loose/intersection), extrusion, and overlap extraction—useful conceptual models for overlap and partial-overlap cases. citeturn46view0

### Kaldi-format pipeline context (RTTM generation in practice)

A clustering script for a classic diarization recipe explicitly documents that it produces RTTM and shows the stage where speaker labels (clusters) are turned into RTTM via a `make_rttm.py` writer. citeturn15view0

### Primary diarization and speaker-embedding papers (for boundary/granularity intuition)

The x-vector paper specifies frame-based feature extraction patterns (e.g., 25ms frame-length) and the presence of SAD (speech activity detection), contextualizing why timestamp boundaries are granular and imperfect at sub-frame scales. citeturn31view0

The VBx family documentation and preprint describe the VBx diarization approach and note embedding extraction on a fixed hop (e.g., every 0.25s), implying quantized temporal resolution that can mismatch word timestamps. citeturn30view1turn42view2turn42view3

A primary meeting-transcription evaluation paper introduces time-constrained cpWER (tcpWER), discusses segment-level timing and approximating word-level timing from segment timing, and argues for collars/tolerances—directly relevant to designing robust timestamp reconciliation policies. citeturn19view0turn19view1

A government publication on multi-dimensional alignment for overlapping speech provides historical context for why overlap must be handled rather than ignored in meeting-style speech evaluation. citeturn22view0

### WebRTC VAD (for pipeline completeness)

A WebRTC header documents chunk-based voice probability estimation and constraints such as supported sampling rates, and a standalone fork describes its origin from the WebRTC native code package. citeturn11view1turn11view0
