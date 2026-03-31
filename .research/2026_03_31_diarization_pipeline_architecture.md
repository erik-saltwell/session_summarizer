
# Diarization Pipeline Architecture Plan

This document summarizes the agreed architecture for the diarization + transcription cleanup pipeline,
including model/package selections and processing order.

---

# End-to-End Pipeline Overview

Audio
→ ClearerVoice / MossFormer2 speech enhancement
→ Silence-aware chunking (NeMo VAD)
→ ASR (Whisper Turbo or GPT-4o-transcribe candidate)
→ NeMo diarization
→ Early embedding adjacency merge (ERes2NetV2 embeddings)
→ SA-WER optimizer (custom module)
→ DiarizationLM speaker repair (Google diarizationlm or LLM-based equivalent)
→ Global speaker identification (registration embeddings via ERes2NetV2)
→ Final adjacency merge
→ NeMo punctuation restoration
→ NeMo inverse text normalization (ITN)
→ LLM transcript cleanup pass (GPT‑4o / Gemini / Qwen2.5)
→ Final transcript

---

# Stage-by-Stage Architecture

## 1. Speech Enhancement

Recommended:

- ClearerVoice-Studio
- MossFormer2 enhancement model

Purpose:

- Improve diarization stability
- Improve ASR accuracy
- Reduce overlap noise artifacts

---

## 2. Chunking Strategy

Silence-aware segmentation using:

Recommended tooling:

- pyannote.audio VAD
or
- NeMo VAD

Goals:

- Split at silence boundaries
- Avoid speaker-boundary splits
- Keep chunk sizes GPU-safe

---

## 3. ASR Layer

Recommended models:

Primary candidates:

- Qwen 2.5 Canary
- Alignment with NFA from Nemo/ParakeetCTC
- Score confidence with a transcription pass to Nemo CTCParakeet
Output format:

Segment JSON:

{
  speaker,
  start,
  end,
  confidence,
  text
}

---

## 4. Diarization

Recommended framework:

NVIDIA NeMo

Embedding model:

ERes2NetV2 speaker embeddings

Reasons:

- strong meeting-style diarization
- supports registration matching
- compatible with chunk pipelines

---

## 5. Early Embedding Merge Pass

Light merge stage before optimization.

Rules:

Merge adjacent segments if:

- cosine similarity ≥ 0.88
- silence gap ≤ 0.3 seconds

Uses:

ERes2NetV2 embeddings

Purpose:

Reduce fragmentation before SA-WER optimization

---

## 6. SA-WER Optimizer (Custom Module)

Module name:

sawer_optimizer/

Responsibilities:

- adjacency merge
- embedding-based relabel
- speaker island removal
- turn-length smoothing
- optional known-speaker matching

Inputs:

Segment objects:

{
  segment_id,
  recording_id,
  start,
  end,
  speaker_id,
  text,
  embedding
}

Outputs:

Optimized segments + audit log

---

## 7. DiarizationLM Speaker Repair

Recommended options:

Option A:

Google diarizationlm package

pip install diarizationlm

Option B:

LLM-based repair pass using:

- GPT‑4o
- Gemini 1.5 Pro
- Qwen2.5‑14B‑Instruct

Purpose:

Fix:

- alternating speaker errors
- discourse inconsistencies
- cross-chunk drift

---

## 8. Speaker Registration Matching

Embedding model:

ERes2NetV2

Registry structure:

KnownSpeaker:

{
  speaker_name,
  embedding
}

Purpose:

Assign:

GM
Player1
Player2

to anonymous diarizer labels

---

## 9. Final Merge Pass

After identity assignment:

Merge:

Alice Bob Alice

→

Alice Alice Alice

Purpose:

Improve readability
Improve embedding stability

---

## 10. Punctuation Restoration

Recommended model:

NeMo punctuation_en_bert

Alternative:

Silero punctuation model

Purpose:

Restore:

- sentence boundaries
- commas
- capitalization

Improves downstream LLM reasoning

---

## 11. Inverse Text Normalization (ITN)

Recommended:

NeMo ITN English model

Examples:

twenty three → 23
doctor smith → Dr. Smith

Purpose:

Standardize transcript formatting

---

## 12. Glossary-Aware Cleanup Pass (LLM)

Recommended models:

Cloud:

- GPT‑4o
- Gemini 1.5 Pro

Local:

- Qwen2.5‑14B‑Instruct
- Llama‑3‑70B‑Instruct

Allowed edits:

- punctuation
- capitalization
- spelling
- numeric normalization
- entity correction

Not allowed edits:

- timestamps
- speaker labels
- paraphrasing
- meaning changes

Input format:

Segment JSON

---

# Supporting Libraries

Recommended Python dependencies:

pyannote.audio
nemo_toolkit
diarizationlm
rapidfuzz
numpy
scipy
transformers
torch

Optional:

spaCy (entity correction layer)
jiwer (WER baseline metrics)
meeteval (cpWER / tcpWER scoring)

---

# Evaluation Stack

Metrics:

cpWER
tcpWER
DER
SA-WER surrogate optimizer score

Recommended package:

meeteval

pip install meeteval

---

# Final Pipeline Summary

Audio
→ MossFormer2 enhancement
→ silence-aware chunking
→ Whisper Turbo / GPT‑4o-transcribe
→ NeMo diarization
→ early embedding merge
→ SA-WER optimizer
→ DiarizationLM repair
→ speaker registration matching
→ final adjacency merge
→ NeMo punctuation restoration
→ NeMo ITN
→ glossary-aware LLM cleanup
→ final transcript
