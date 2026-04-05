# Single-document diarization and speaker-attributed transcript evaluation in Python

[Download the Markdown guide](sandbox:/mnt/data/speaker_diarization_eval_guide.md)

## Executive summary

You asked for a rigorous but pragmatic development guide for computing **three** “high impact” metrics on **one document at a time** (no overlap speech), using your existing per-word/per-segment timing + speaker labels and an easy conversion path to RTTM/CTM.

The most robust, reproducible stack for your scenario is:

- **DER (Diarization Error Rate)** + component breakdown (miss / false alarm / confusion) computed with the entity["organization","pyannote","speaker diarization project"] metrics toolkit. citeturn0search7turn0search11turn0search0  
- **Confusion rate** defined as `confusion_seconds / total_reference_speech_seconds` (extracted from DER components). Humans notice speaker swaps more than tiny boundary jitter. citeturn0search7turn0search11  
- **tcpWER** (time-constrained minimum permutation WER) computed with entity["organization","MeetEval","meeting transcription eval"]. tcpWER is designed to evaluate “who said what when” while being **permutation-invariant** (your predicted speaker labels don’t need to match reference IDs) and **time-aware** via a temporal constraint. citeturn12view1turn0search6turn3search17  

Two best-practice defaults for **DER collar** are:
- **`collar=0.0`** (strict, DIHARD-like / modern reproducibility) and  
- **`collar=0.25` seconds** (legacy RT scoring practice; reduces sensitivity to minor boundary jitter). citeturn8view0turn7view0turn7view1  

For **tcpWER**, the collar is conceptually different: it tolerates timing inaccuracies in the time-constrained word matching and usually needs to be much larger (often seconds, not milliseconds). The tcpWER docs recommend choosing it to avoid penalizing harmless merges while still penalizing “tens-of-seconds-apart” mistakes, and note they found **~2–5s** useful on LibriCSS; their CLI examples use `--collar 5`. citeturn12view1turn11search3  

## Recommended evaluation pipeline

The pipeline below is optimized for *single-document iteration* and also scales naturally to corpus-level regression.

```mermaid
flowchart TD
  A[Inputs: words/segments + times + speaker labels] --> B[Validate: sorted, finite times, no overlap]
  B --> C[Create atomic single-speaker turns]
  C --> D[Build pyannote.core Annotation (ref + hyp)]
  D --> E[DER + components via pyannote.metrics]
  E --> F[Confusion rate = confusion / total]
  C --> G[Write SegLST or STM for MeetEval]
  G --> H[tcpWER via meeteval-wer tcpwer]
  C --> I[Hungarian mapping + overlap matrix]
  I --> J[Confusion heatmap + debugging plots]
```

Key best practices (especially for one-file scoring):

- **Always evaluate “atomic” single-speaker turns**, not container segments that may include sequential speakers. If your segment contains “multiple speakers, no overlap,” you must split it into sequential single-speaker spans for meaningful diarization scoring.  
- **Always define a scoring region explicitly (UEM)** when using NIST-style tools or wrappers. A known failure mode is ignoring false alarms outside automatically generated regions (e.g., false alarms before the first reference speech). citeturn4view1turn11search2turn8view0  
- **Report two DER conditions** for human-impact decisions: strict `collar=0.0` and tolerant `collar=0.25`. Those two numbers usually tell you whether you improved “real” speaker assignment versus boundary polishing. citeturn8view0turn7view0  

Optional corpus aggregation (if/when you evaluate multiple documents):
- Aggregate DER by summing components across files and dividing once (don’t average per-file rates). citeturn0search7turn3search4  
- For uncertainty: file-level bootstrap CI (95%, 5000 resamples) and paired permutation tests on per-file metric deltas. citeturn3search3  

## Exact format specs and examples

This section gives the **minimum** you need for reproducible scoring and interoperability with scorers.

### RTTM for diarization (turns)

RTTM is a space-delimited format with **ten fields** per speaker turn line, commonly used by diarization scorers; `dscore` documents the required fields and provides validation tooling. citeturn4view1turn0search1  

**RTTM SPEAKER line template** (dscore-compatible): citeturn4view1  
```
SPEAKER <file_id> 1 <turn_onset> <turn_duration> <NA> <NA> <speaker_name> <NA> <NA>
```

**Example**: citeturn4view1  
```
SPEAKER rec1 1 130.430000 2.350 <NA> <NA> spk1 <NA> <NA>
```

### UEM for scoring regions

UEM lines define scored intervals per recording; `dscore` provides the minimal four-field format. citeturn4view1  
```
<file_id> 1 <onset_seconds> <offset_seconds>
```

**Example**: citeturn4view1  
```
rec1 1 0.000000 615.330000
```

RT evaluation documentation also describes UEM as `<F> <C> <BT> <ET>` with examples and intent (“specify time regions”). citeturn7view1turn6view0  

**Best practice for single-document regression:**  
Set UEM to at least `[0.0, max(ref_end, hyp_end)]` to ensure scoring includes early false alarms and trailing outputs. This mitigates known “auto-UEM from ref only” pitfalls. citeturn11search2turn4view1  

### CTM/STM/TRN for word scoring interoperability

You said you can convert to CTM/STM/TRN. Here are “official-tool” definitions:

**CTM (SCTK)** defines a word stream record as: citeturn4view0  
```
CTM :== <F> <C> <BT> <DUR> <WORD> [ <CONF> ]
```
where `<BT>` is begin time in seconds from file start, `<DUR>` is duration. citeturn4view0  

**TRN (SCTK)** is “word sequence + utterance ID in parentheses”. citeturn5view0  
```
she had your dark suit ... (cmh_sa01)
```

**STM (SCTK)** is segment-level with speaker, start/end, transcript; SCTK defines: citeturn5view0  
```
STM :== <F> <C> <S> <BT> <ET> [ <LABEL> ] transcript...
```

For MeetEval, the repo documents a simplified STM line form: citeturn11search0  
```
<filename> <channel> <speaker_id> <begin_time> <end_time> <transcript>
```

### SegLST JSON for MeetEval tcpWER

MeetEval strongly supports SegLST (segment list) JSON for time-constrained metrics; their example file shows the concrete keys: `session_id`, `start_time`, `end_time`, `speaker`, `words`. citeturn10view0turn11search0  

**Example (minimal)**: citeturn10view0  
```json
[
  {"session_id": "recordingA", "start_time": "0", "end_time": "1", "speaker": "Alice", "words": "hello world"},
  {"session_id": "recordingA", "start_time": "1", "end_time": "2.5", "speaker": "Bob", "words": "hi there"}
]
```

## Metric definitions and formulas

### DER and components

The classic DER definition (also used in the entity["organization","dscore","diarization scoring toolkit"] family via md-eval) is: citeturn0search11turn8view0  
\[
\mathrm{DER} = \frac{T_{FA} + T_{MISS} + T_{CONF}}{T_{TOTAL}}
\]
Where: citeturn0search11turn0search7  
- \(T_{FA}\): false alarm speech time  
- \(T_{MISS}\): missed speech time  
- \(T_{CONF}\): speaker confusion time  
- \(T_{TOTAL}\): total reference speech time (in “who spoke when,” overlap conventions can matter in general; you have no overlap) citeturn0search11  

`pyannote.metrics` exposes exactly these components via `detailed=True`, and documents the keys (`'false alarm'`, `'missed detection'`, `'confusion'`, `'total'`, etc.). citeturn0search7  

Collar and mapping:
- Collar is a forgiveness region around boundaries; `dscore` documents collar=0ms default for DER via md-eval and shows how to change it. citeturn8view0  
- Older RT scoring used a 0.25s collar in md-eval command lines, illustrating the legacy tolerance convention. citeturn7view0turn7view1  
- Optimal speaker mapping is computed via Hungarian assignment in `pyannote.metrics` “optimal DER,” minimizing confusion after mapping. citeturn0search0turn0search8turn0search11  

### Confusion rate

Define confusion rate for your regression dashboard as:

\[
\mathrm{ConfusionRate} = \frac{T_{CONF}}{T_{TOTAL}}
\]

You compute this from the same DER component dictionary. `pyannote.metrics` makes it easy to extract `metric['confusion']` and `metric['total']`. citeturn0search7turn0search8  

### cpWER and tcpWER overview

**cpWER (concatenated minimum-permutation WER)**:
- Concatenate all words by speaker (ref and system), then compute the minimum WER over all permutations between system speakers and reference speakers. This is permutation-invariant when speaker identities are unknown. citeturn3search17turn0search6  

**tcpWER** adds temporal constraints:
- MeetEval’s tcpWER documentation defines it as “similar to cpWER, but uses temporal information to prevent matching words that are far apart temporally.” citeturn12view1turn0search2  
- tcpWER requires pseudo-word-level timing strategies when word-level timestamps are missing and describes multiple strategies (character-based defaults, equidistant intervals, etc.). citeturn12view1turn0search6  
- tcpWER introduces a collar concept (seconds-scale) to avoid penalizing small diarization timing errors while still penalizing large implausible merges; MeetEval notes typical values depend on data and reports ranges like ~2–5s in an example domain. citeturn12view1turn11search3  

## Reproducible Python implementation

This section provides “ready-to-run” code blocks you can paste into a single script (or split into modules). It assumes: **no overlap speech** and one document (one `uri`) at a time.

### Recommended packages and install commands

```bash
pip install pyannote.core pyannote.metrics
pip install meeteval
pip install numpy scipy pandas matplotlib plotly
```

For optimal assignment + CIs: `scipy.optimize.linear_sum_assignment` and `scipy.stats.bootstrap`. citeturn3search6turn3search3  

Optional external fallback for SCTK:
- Install Asclite from the entity["organization","SCTK","nist scoring toolkit"] distribution if you need legacy scoring outputs; Asclite supports multiple input formats and SASTT evaluation. citeturn4view3  

### Adapting your existing data structures

From your uploaded code, you have:

- `WordAlignment(word, start_time, end_time, confidence)` with helper methods and JSON serialization in an `AlignmentResult(words=[...])`. fileciteturn0file0  
- `SpeechClip(start_time, end_time, speakers: set[str], identity, words: list[WordAlignment], ...)` and `SpeechClipSet` that can be saved/loaded as JSON. fileciteturn0file1  

**Important evaluation constraint:** DER and tcpWER require **single-speaker turns** (one speaker label per time span). If a `SpeechClip` can contain multiple sequential speakers (but no overlap), split it into atomic turns first.

Below I use a neutral `WordToken` representation because it cleanly supports:
- turn building for DER
- segment building for SegLST/STM for tcpWER

### Core dataclasses and validation

```python
from __future__ import annotations

import json
import math
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import bootstrap

from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate
```

```python
@dataclass(frozen=True)
class WordToken:
    word: str
    start: float
    end: float
    speaker: str
    confidence: float = 0.0
```

```python
@dataclass(frozen=True)
class SpeakerTurn:
    start: float
    end: float
    speaker: str
```

```python
def validate_words(words: List[WordToken], eps: float = 1e-6) -> None:
    if not words:
        raise ValueError("No words.")
    for w in words:
        if not (math.isfinite(w.start) and math.isfinite(w.end)):
            raise ValueError(f"Non-finite time: {w}")
        if w.end <= w.start + eps:
            raise ValueError(f"Degenerate duration: {w}")
        if not w.speaker:
            raise ValueError(f"Missing speaker: {w}")
    for a, b in zip(words, words[1:]):
        if b.start + eps < a.start:
            raise ValueError("Words must be sorted by time.")
```

### Convert words to atomic turns

This merges consecutive same-speaker words into turns (useful even if your pipeline emits only word-level labels). You can tune `merge_gap` to bridge tiny pauses.

```python
def build_turns_from_words(words: List[WordToken], merge_gap: float = 0.2) -> List[SpeakerTurn]:
    validate_words(words)
    turns: List[SpeakerTurn] = []
    cur = SpeakerTurn(words[0].start, words[0].end, words[0].speaker)

    for w in words[1:]:
        gap = w.start - cur.end
        if w.speaker == cur.speaker and gap <= merge_gap:
            cur = SpeakerTurn(cur.start, max(cur.end, w.end), cur.speaker)
        else:
            turns.append(cur)
            cur = SpeakerTurn(w.start, w.end, w.speaker)

    turns.append(cur)
    return turns
```

Because you have **no overlap**, add a hard check:

```python
def assert_no_overlap(turns: List[SpeakerTurn], eps: float = 1e-6) -> None:
    t = sorted(turns, key=lambda x: (x.start, x.end))
    for a, b in zip(t, t[1:]):
        if b.start + eps < a.end:
            raise ValueError(f"Overlap detected: {a} vs {b}")
```

### Convert turns to pyannote Annotation

`pyannote.core.Annotation` supports writing RTTM directly (`Annotation.write_rttm`) and is the native input form for pyannote.metrics. citeturn1search10turn0search7  

```python
def turns_to_annotation(uri: str, turns: List[SpeakerTurn]) -> Annotation:
    ann = Annotation(uri=uri)
    for t in turns:
        ann[Segment(t.start, t.end)] = t.speaker
    return ann
```

### Write RTTM and UEM

RTTM speaker line follows the dscore spec (10 columns). citeturn4view1  

```python
def write_rttm(uri: str, turns: List[SpeakerTurn], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        for t in turns:
            dur = t.end - t.start
            f.write(f"SPEAKER {uri} 1 {t.start:.6f} {dur:.6f} <NA> <NA> {t.speaker} <NA> <NA\n")
```

UEM is 4 fields as documented in dscore. citeturn4view1  

```python
def write_uem(uri: str, start: float, end: float, out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"{uri} 1 {start:.6f} {end:.6f}\n")
```

**Why UEM matters:** a documented issue shows that if a tool generates UEM from reference-only boundaries, it can ignore false alarms outside those boundaries. citeturn11search2  

### Compute DER, components, and confusion rate with pyannote.metrics

pyannote.metrics documents that `detailed=True` returns component dicts and accumulated components are accessible by key lookup. citeturn0search7  

```python
def compute_der_components(ref: Annotation, hyp: Annotation, collar: float = 0.0) -> Dict[str, float]:
    metric = DiarizationErrorRate(collar=collar, skip_overlap=False)  # no overlap => skip_overlap irrelevant
    d = metric(ref, hyp, detailed=True)

    return {
        "der": float(d["diarization error rate"]),
        "total": float(d["total"]),
        "miss": float(d["missed detection"]),
        "fa": float(d["false alarm"]),
        "conf": float(d["confusion"]),
        "correct": float(d["correct"]),
    }
```

```python
def confusion_rate(components: Dict[str, float]) -> float:
    total = components["total"]
    return float("nan") if total <= 0 else components["conf"] / total
```

**Recommended default settings:**
- Strict: `collar=0.0` (and always report it). `dscore` uses collar 0ms by default for DER scoring with md-eval. citeturn8view0  
- Tolerant: `collar=0.25` for comparability with classic RT evaluation practice. citeturn7view0turn7view1  

### Compute tcpWER with MeetEval

MeetEval documents tcpWER goals, pseudo-word timing strategies, and the `--collar` parameter (required). citeturn12view1turn11search3turn0search6  

For tcpWER input, the easiest format to generate is **SegLST JSON** (the example shows the keys). citeturn10view0turn11search0  

**Write SegLST JSON from turns + words**

```python
def turns_to_seglst_json(uri: str, turns: List[SpeakerTurn], turn_text: Dict[Tuple[float, float, str], str], out_path: Path) -> None:
    segs = []
    for t in turns:
        segs.append({
            "session_id": uri,
            "start_time": float(t.start),
            "end_time": float(t.end),
            "speaker": t.speaker,
            "words": turn_text[(t.start, t.end, t.speaker)],
        })
    out_path.write_text(json.dumps(segs, indent=2, ensure_ascii=False), encoding="utf-8")
```

**Call meeteval-wer tcpwer**

```python
def run_meeteval_tcpwer(ref_json: Path, hyp_json: Path, collar_seconds: float = 5.0) -> Dict:
    with tempfile.TemporaryDirectory() as td:
        avg_out = Path(td) / "tcpwer_average.json"
        per_out = Path(td) / "tcpwer_per_reco.json"
        cmd = [
            "meeteval-wer", "tcpwer",
            "-r", str(ref_json),
            "-h", str(hyp_json),
            "--collar", str(collar_seconds),
            "--average-out", str(avg_out),
            "--per-reco-out", str(per_out),
        ]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(f"meeteval failed:\n{p.stderr}\nSTDOUT:\n{p.stdout}")
        return json.loads(avg_out.read_text(encoding="utf-8"))
```

**Default tcpWER collar recommendation:** start with `collar_seconds=5.0` and tune to your transcript segmentation granularity. MeetEval explains why the collar must be relatively large when pseudo-word timing is used and suggests typical seconds-scale ranges in at least one domain. citeturn12view1turn11search3  

### Asclite fallback wrapper

Asclite is a multi-dimensional Levenshtein alignment tool and supports RTTM/CTM/STM/TRN input formats. citeturn4view3turn5view0  

```python
def run_asclite_on_stm(ref_stm: Path, hyp_stm: Path) -> Tuple[int, str, str]:
    cmd = [
        "asclite",
        "-r", str(ref_stm), "stm",
        "-h", str(hyp_stm), "stm", "sys",
        "-o", "sum", "stdout",
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr
```

If you want time-pruned behavior, tcpWER notes the idea is similar (not identical) to asclite’s `-time-prune` option. citeturn12view1  

### Hungarian mapping and confusion heatmap

For debugging speaker swaps, build the duration overlap matrix \(O_{ij}\) between reference speaker \(i\) and hypothesis speaker \(j\), then solve optimal assignment with Hungarian (SciPy’s `linear_sum_assignment`). citeturn3search6turn3search2  

```python
def overlap_matrix_seconds(ref_turns: List[SpeakerTurn], hyp_turns: List[SpeakerTurn]):
    ref_spk = sorted({t.speaker for t in ref_turns})
    hyp_spk = sorted({t.speaker for t in hyp_turns})
    r_index = {s: i for i, s in enumerate(ref_spk)}
    h_index = {s: j for j, s in enumerate(hyp_spk)}
    O = np.zeros((len(ref_spk), len(hyp_spk)), dtype=float)

    for r in ref_turns:
        for h in hyp_turns:
            inter = max(0.0, min(r.end, h.end) - max(r.start, h.start))
            if inter > 0:
                O[r_index[r.speaker], h_index[h.speaker]] += inter
    return ref_spk, hyp_spk, O
```

```python
def hungarian_speaker_map(ref_spk: List[str], hyp_spk: List[str], O: np.ndarray) -> Dict[str, str]:
    row_ind, col_ind = linear_sum_assignment(-O)  # maximize overlap
    return {hyp_spk[j]: ref_spk[i] for i, j in zip(row_ind, col_ind) if O[i, j] > 0}
```

Plot (matplotlib):

```python
import matplotlib.pyplot as plt

def plot_overlap_heatmap(ref_spk, hyp_spk, O):
    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(O, aspect="auto")
    ax.set_xticks(range(len(hyp_spk)))
    ax.set_xticklabels(hyp_spk, rotation=45, ha="right")
    ax.set_yticks(range(len(ref_spk)))
    ax.set_yticklabels(ref_spk)
    ax.set_title("Overlap seconds: reference (rows) vs hypothesis (cols)")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    return fig
```

## Uncertainty, paired testing, and aggregation

### Aggregation for multiple files

**DER aggregation:** sum the components across files and divide once (this is what component-based metric accumulators enable). pyannote.metrics explicitly supports accumulated values via indexing. citeturn0search7  

**MeetEval aggregation:** MeetEval’s README notes that the combined WER “is not the average” but the error rate from combining errors and lengths. citeturn3search4  

### Bootstrap confidence intervals

For corpus-level CI (95%, 5000 resamples), SciPy’s `bootstrap` provides a standard approach and documents the resampling process and parameters (including `confidence_level` and `n_resamples`). citeturn3search3  

```python
from scipy.stats import bootstrap
import numpy as np

def bootstrap_mean_ci(values: np.ndarray, n_resamples: int = 5000, confidence_level: float = 0.95, seed: int = 0):
    rng = np.random.default_rng(seed)
    res = bootstrap((values,), np.mean, n_resamples=n_resamples, confidence_level=confidence_level,
                    random_state=rng, method="percentile")
    ci = res.confidence_interval
    return float(values.mean()), float(ci.low), float(ci.high)
```

For **single-document only**, file-wise bootstrap isn’t meaningful. If you *must* quantify uncertainty within one long recording, use a **block bootstrap** over fixed time windows (e.g., 30s windows) using your sliding-window DER series as the resampling unit; treat it as a heuristic, not a formal statistical guarantee.

### Paired significance test for pipeline A vs B

For multiple files (or multiple blocks), compute per-unit deltas and run a paired permutation test (sign-flip test). This avoids distribution assumptions and is easy to implement.

```python
import numpy as np

def paired_sign_flip_pvalue(deltas: np.ndarray, n: int = 10000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    observed = deltas.mean()
    count = 0
    for _ in range(n):
        signs = rng.choice([-1, 1], size=len(deltas))
        stat = (deltas * signs).mean()
        if abs(stat) >= abs(observed):
            count += 1
    return (count + 1) / (n + 1)
```

## Visualization and debugging

image_group{"layout":"carousel","aspect_ratio":"16:9","query":["diarization error rate components miss false alarm confusion diagram","speaker diarization timeline plot reference vs hypothesis","confusion matrix heatmap speaker diarization overlap seconds","DER sliding window plot example"],"num_per_query":1}

### Timeline plots for human inspection

For notebook visualization, pyannote.core supports direct rendering of `Annotation` objects (with `pyannote.core[notebook]`). citeturn1search10  

If you want pure matplotlib (no notebook special repr), use `broken_barh`:

```python
import matplotlib.pyplot as plt

def plot_turns(ax, turns: List[SpeakerTurn], y: float, h: float, label: str):
    ax.broken_barh([(t.start, t.end - t.start) for t in turns], (y, h))
    ax.text(0.0, y + h + 0.1, label, va="bottom")

def plot_ref_hyp_timelines(ref_turns, hyp_turns):
    fig, ax = plt.subplots(figsize=(12, 3))
    plot_turns(ax, ref_turns, y=1.0, h=0.8, label="Reference turns")
    plot_turns(ax, hyp_turns, y=0.0, h=0.8, label="Hypothesis turns")
    ax.set_xlabel("Time (s)")
    ax.set_yticks([])
    ax.set_title("Reference vs hypothesis turn timelines")
    plt.tight_layout()
    return fig
```

### Sliding-window DER component chart

Use this to localize regressions (“it got worse in the last 2 minutes”) and to compare pipeline A vs B windows.

```python
from pyannote.core import Segment
from pyannote.metrics.diarization import DiarizationErrorRate

def sliding_der_components(ref_ann, hyp_ann, win=10.0, hop=1.0, collar=0.0):
    m = DiarizationErrorRate(collar=collar, skip_overlap=False)
    extent = ref_ann.get_timeline().extent()
    t0, t1 = extent.start, extent.end

    rows = []
    t = t0
    while t + win <= t1:
        seg = Segment(t, t + win)
        d = m(ref_ann.crop(seg), hyp_ann.crop(seg), detailed=True)
        rows.append({
            "t_start": t,
            "miss": d["missed detection"],
            "fa": d["false alarm"],
            "conf": d["confusion"],
            "total": d["total"],
            "der": d["diarization error rate"],
        })
        t += hop
    return pd.DataFrame(rows)
```

Then plot:

```python
def plot_sliding_der(df):
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(df["t_start"].values, df["der"].values)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("DER (window)")
    ax.set_title("Sliding-window DER")
    plt.tight_layout()
    return fig
```

## Human-impact interpretation and regression thresholds

Because you want *noticeable* improvements (not 0.0001 score changes), treat the metrics as a “scorecard”:

- **Confusion rate** (most speaker-visible):  
  - If it drops meaningfully, readers see fewer misattributed turns.  
  - If it rises, you’ll see “speaker swaps” and distrust in the transcript.

- **tcpWER** (end-to-end “who said what”):  
  - Improvements here usually mean your transcript is more correct *and* better attributed.  
  - If DER improves but tcpWER worsens, you may have improved boundaries but harmed speaker grouping or label stability.

- **DER** as overall “time wrong”:  
  - Use it as the headline diarization number, but interpret *via components* (conf vs miss/FA). pyannote.metrics explicitly supports component access. citeturn0search7  

Practical “go/no-go” heuristics for single-document iteration:
- Require improvements under **both** `collar=0.0` and `collar=0.25` before calling it a real win, unless your product explicitly tolerates boundary jitter. citeturn8view0turn7view0  
- When in doubt, weight confusion more heavily than miss/false alarm for transcript UX (wrong-speaker attribution is more harmful than small missing fragments).

Recommended default hyperparameters (per your request):
- DER: `collar=0.0` and optional `collar=0.25s` citeturn8view0turn7view0  
- Overlap: `skip_overlap=False` (irrelevant in your data, but keep explicit) citeturn0search7  
- frame_step: `0.01s` (for sliding-window plotting / block bootstrap; dscore documents 10ms as a default frame step for frame-based metrics) citeturn8view0  
- bootstrap: `n_resamples=5000`, `CI=95%` citeturn3search3  

