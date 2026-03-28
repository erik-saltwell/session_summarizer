# Word-Level Confidence Scores with NeMo SALM / canary-qwen-2.5b

**Date:** 2026-03-27
**Context:** `src/session_summarizer/transcription/canary_qwen_transcriber.py`

---

## Architectural Decision

**Use Option 3a: Parakeet-CTC free transcription pass mapped to SALM words by time window.**

See [Implementation Plan](#implementation-plan-option-3a) below.

---

## TL;DR

`SALM.generate()` has **no native confidence API**. Three viable paths exist:

1. **HF `output_scores=True`** — token log probs from SALM via HuggingFace generate kwargs (rough, LLM-mixed uncertainty)
2. **Parallel CTC model** — run `nvidia/parakeet-ctc-0.6b-v2` with NeMo's `ConfidenceConfig` (most reliable, acoustic-calibrated; requires word string alignment between Parakeet and SALM outputs)
3. **Forced alignment confidence** — use the existing Parakeet-CTC forced alignment pass to produce confidence; map to SALM words by time window rather than string matching

---

## 1. SALM.generate() and the HuggingFace `output_scores` Path

`SALM.generate()` internally calls `self.llm.generate(...)` and passes `**generation_kwargs` straight through to HuggingFace Transformers. You can therefore use HF's `return_dict_in_generate` / `output_scores` flags:

```python
output = model.generate(
    prompts=[[{
        "role": "user",
        "content": f"Transcribe: {model.audio_locator_tag}",
        "audio": [tmp_path],
    }]],
    max_new_tokens=512,
    return_dict_in_generate=True,   # returns GenerateOutput instead of tensor
    output_scores=True,             # includes per-step logit tensors
)

# output.sequences: (batch, total_seq_len) token IDs
# output.scores:    tuple of (batch, vocab_size) tensors, one per generated step

import torch
log_probs = model.llm.compute_transition_scores(
    output.sequences, output.scores, normalize_logits=True
)
# log_probs[0] shape: (num_new_tokens,) — log prob of each generated token
```

**Aggregating token log probs to word level** — Qwen's BPE tokenizer uses `▁` (U+2581) prefix to mark word starts. Group subword tokens by that boundary and aggregate (sum or min):

```python
prompt_len = output.sequences.shape[1] - log_probs.shape[1]
new_token_ids = output.sequences[0][prompt_len:]

words, word_log_probs = [], []
current_tokens, current_lp = [], []

for tid, lp in zip(new_token_ids.tolist(), log_probs[0].tolist()):
    token_str = model.tokenizer.ids_to_tokens([tid])[0]
    if token_str.startswith("▁") and current_tokens:
        words.append("".join(current_tokens).lstrip("▁"))
        word_log_probs.append(min(current_lp))  # or sum / mean
        current_tokens, current_lp = [], []
    current_tokens.append(token_str)
    current_lp.append(lp)
if current_tokens:
    words.append("".join(current_tokens).lstrip("▁"))
    word_log_probs.append(min(current_lp))

word_confidence = [float(torch.exp(torch.tensor(lp))) for lp in word_log_probs]
```

**Caveat:** SALM's LLM logits mix acoustic evidence with language model priors. A low-probability token may just be an unusual word for the LLM, not a transcription error. These confidence values are **less calibrated** than CTC-based confidence.

**Note on return type:** When `return_dict_in_generate=True`, `model.generate()` returns a `GenerateOutput` object rather than a bare `torch.Tensor`. The current `canary_qwen_transcriber.py` code does `answer_ids[0].tolist()` — this must change to `output.sequences[0][prompt_len:].tolist()`.

---

## 2. CTC / RNNT Models with Native `ConfidenceConfig` (Recommended)

NeMo's traditional ASR models (`nemo.collections.asr`) have a mature, first-class confidence system. `nvidia/parakeet-ctc-0.6b-v2` or `nvidia/parakeet-tdt-1.1b` provide **acoustic-calibrated word-level confidence natively**:

```python
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.utils.asr_confidence_utils import (
    ConfidenceConfig, ConfidenceMethodConfig
)
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCBPEDecodingConfig

model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained("nvidia/parakeet-ctc-0.6b-v2")

confidence_cfg = ConfidenceConfig(
    preserve_frame_confidence=True,
    preserve_token_confidence=True,
    preserve_word_confidence=True,
    aggregation="min",              # "min", "mean", "max", or "prod"
    method_cfg=ConfidenceMethodConfig(name="max_prob"),  # or "entropy"
)
decoding_cfg = CTCBPEDecodingConfig(
    preserve_alignments=True,
    compute_timestamps=True,
    confidence_cfg=confidence_cfg,
)
model.change_decoding_strategy(decoding_cfg)

hypotheses = model.transcribe(["audio.wav"], return_hypotheses=True)
hyp = hypotheses[0]

print(hyp.text)             # full transcript
print(hyp.word_confidence)  # List[float], one per word, e.g. [0.94, 0.87, 0.23]
print(hyp.timestamp['word'])  # List[{word, start_offset, end_offset}]
```

The `Hypothesis` dataclass fields relevant here:
- `word_confidence: Optional[List[float]]`
- `token_confidence: Optional[List[float]]`
- `frame_confidence: Optional[List[float]]`
- `timestamp: dict` with keys `'char'`, `'word'`, `'segment'`

**RNNT/TDT models** (e.g. `nvidia/parakeet-tdt-1.1b`) work identically but use `RNNTDecodingConfig`. Research suggests that **entropy-based methods** (`method_cfg=ConfidenceMethodConfig(name="entropy", entropy_type="gibbs")`) outperform `max_prob` for detecting errors.

---

## 3. Forced Alignment Confidence (Option 3a and 3b)

The project already uses `get_forced_aligned_timestamps_with_external_model()` (NeMo's internal forced alignment utility, not the standalone NFA tool) in `src/session_summarizer/alignment/parakeet_ctc_aligner.py`. This function runs Parakeet-CTC and Viterbi-aligns SALM's exact transcript to the audio, producing word timestamps.

**Key finding from code inspection:** The `Hypothesis` dataclass (`nemo/collections/asr/parts/utils/rnnt_utils.py`) has `word_confidence`, `token_confidence`, and `frame_confidence` fields built in — but `get_forced_aligned_timestamps_with_external_model` **never populates them**. It only writes timestamps. Confirmed by grep: the word "confidence" does not appear anywhere in `timestamp_utils.py`.

This means two sub-options exist:

### Option 3a — Time-window mapping (chosen approach)

Run a second Parakeet-CTC pass inside the existing `align()` method using `model.transcribe()` with `ConfidenceConfig`. This produces Parakeet's own word list with confidence scores and per-word timestamps. Map those confidence values onto SALM's words using **time-window overlap** (not string matching): for each SALM word's `[start, end]` from the forced alignment, find all overlapping Parakeet words and take the minimum of their confidence scores.

This avoids string matching entirely. The time-window map is reliable because both passes use the same audio and the forced alignment already anchors SALM's words to accurate timestamps.

### Option 3b — True Viterbi path confidence (not chosen)

Patch `get_forced_aligned_timestamps_with_external_model` or reimplement the Viterbi step to expose the per-frame CTC log-probs along the alignment path. This gives confidence scores derived from Parakeet judging SALM's exact words (not its own), which is marginally more principled. Rejected because: requires working with NeMo internals, is fragile against NeMo version updates, and the accuracy gain over 3a is negligible for cross-talk detection (which produces such strong low-confidence signals that both approaches will flag it clearly).

---

## Confidence Quality Comparison

| Model type | Confidence source | Word-level native | Quality |
|---|---|---|---|
| **CTC** (Parakeet-CTC) | Frame-level CTC softmax, aggregated | Yes, via `ConfidenceConfig` | Most reliable; directly tied to acoustic probability |
| **RNNT/TDT** (Parakeet-TDT) | Joiner output probs, aggregated | Yes, via `ConfidenceConfig` | Also reliable; entropy methods outperform max_prob |
| **AED / Canary-1B** | Cross-attention decoder probs | Partial (timestamps via NFA) | Less calibrated than CTC |
| **Speech-LLM / SALM (canary-qwen-2.5b)** | LLM token log probs via HF `output_scores` | No, manual aggregation | Mixes acoustic + LM uncertainty; lowest calibration |

---

## Implementation Plan: Option 3a

All changes are confined to `src/session_summarizer/alignment/parakeet_ctc_aligner.py` and its consumers.

### Step 1 — Add `confidence` to `WordAlignment`

```python
@dataclass
class WordAlignment:
    word: str
    start: float   # seconds
    end: float     # seconds
    confidence: float | None = None  # acoustic confidence [0.0, 1.0]; None if unavailable
```

### Step 2 — Configure Parakeet-CTC for confidence in `align()`

After loading the model and before calling `get_forced_aligned_timestamps_with_external_model`, configure decoding with `ConfidenceConfig`:

```python
from nemo.collections.asr.parts.utils.asr_confidence_utils import (
    ConfidenceConfig, ConfidenceMethodConfig,
)
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCBPEDecodingConfig

confidence_cfg = ConfidenceConfig(
    preserve_word_confidence=True,
    aggregation="min",
    method_cfg=ConfidenceMethodConfig(name="max_prob"),
)
decoding_cfg = CTCBPEDecodingConfig(
    preserve_alignments=True,
    compute_timestamps=True,
    confidence_cfg=confidence_cfg,
)
ctc_model.change_decoding_strategy(decoding_cfg)
```

### Step 3 — Run a free transcription pass to get Parakeet's confidence

After the existing forced alignment call (which produces SALM word timestamps), run a second pass:

```python
parakeet_hypotheses = ctc_model.transcribe(
    [str(audio_path)],
    return_hypotheses=True,
)
parakeet_hyp = parakeet_hypotheses[0]
# parakeet_hyp.word_confidence: List[float]
# parakeet_hyp.timestamp['word']: List[{word, start, end}]
```

### Step 4 — Map Parakeet confidence to SALM words by time-window overlap

```python
def _map_confidence_by_time(
    salm_words: list[WordAlignment],
    parakeet_words: list[dict],        # {word, start, end}
    parakeet_confidence: list[float],
) -> None:
    """Mutates salm_words in place, setting confidence from overlapping Parakeet words."""
    for salm_word in salm_words:
        overlapping = [
            conf
            for pw, conf in zip(parakeet_words, parakeet_confidence)
            if pw["end"] > salm_word.start and pw["start"] < salm_word.end
        ]
        salm_word.confidence = min(overlapping) if overlapping else None
```

### Step 5 — Update consumers

`AlignmentResult` needs no changes — `WordAlignment` gains a nullable field with a default.

`AlignAudioCommand` and anything that serialises `word_alignments.json` should include `confidence` in the output. Downstream cross-talk detection reads `confidence` from `WordAlignment` objects and flags contiguous runs below threshold.

### What does NOT need to change

- `CanaryQwenTranscriber` — unchanged
- The forced alignment call itself — unchanged
- VRAM discipline — the second `transcribe()` call reuses the already-loaded model; no extra load/unload cycle needed

### Scope summary

| Change | Location | Size |
|---|---|---|
| Add `confidence` field | `WordAlignment` dataclass | 1 line |
| Configure `ConfidenceConfig` | `align()` | ~10 lines |
| Free transcription pass | `align()` | ~5 lines |
| Time-window mapping helper | `align()` or module-level | ~15 lines |
| Wire it together | `align()` | ~5 lines |
| Update serialisation | `AlignAudioCommand` / consumers | ~5 lines |

Total: ~40 lines of new code, no NeMo internals touched, no new dependencies.

---

## Sources

- [NeMo SALM Models Documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/speechlm2/models.html)
- [nvidia/canary-qwen-2.5b on Hugging Face](https://huggingface.co/nvidia/canary-qwen-2.5b)
- [NeMo ASR Confidence Estimation Tutorial (GitHub)](https://github.com/NVIDIA-NeMo/NeMo/blob/main/tutorials/asr/ASR_Confidence_Estimation.ipynb)
- [Entropy-Based Methods for Word-Level ASR Confidence (NVIDIA Blog)](https://developer.nvidia.com/blog/entropy-based-methods-for-word-level-asr-confidence-estimation/)
- [NeMo Forced Aligner Documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/tools/nemo_forced_aligner.html)
- [HuggingFace Transformers: Generation with output_scores](https://huggingface.co/docs/transformers/main_classes/text_generation)
- NeMo source: `nemo/collections/speechlm2/models/salm.py` — `generate()` signature and HF passthrough
- NeMo source: `nemo/collections/asr/parts/utils/rnnt_utils.py` — `Hypothesis` dataclass with confidence fields
