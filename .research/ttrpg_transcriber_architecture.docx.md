
**TTRPG Session Transcriber**

Architectural Design Document

Speaker-Attributed Transcription & Diarization Pipeline

Version 1.0  •  March 23, 2026

Cross-Functional Design Team

**Design Team**

| Role | Focus Area |
| :---- | :---- |
| **Product Manager** | Utility, usability, workflow integration |
| **Architect** | System design, maintainability, extensibility |
| **Engineer** | Implementation, GPU constraints, dependencies |
| **LLM Scientist** | Model selection, accuracy, SOTA techniques |

# **Table of Contents**

# **1\. Product Vision & Requirements**

## **1.1 Problem Statement**

A Game Master running weekly tabletop RPG	 sessions needs to convert 2.5-hour audio recordings into accurate, speaker-identified transcripts. The GM knows exactly who played in each session and has voice samples for every player. The resulting transcripts serve as session notes, campaign reference material, and narrative documentation.

## **1.2 User Profile & Constraints**

The sole user is a technical GM who runs sessions weekly. The application processes one recording per week with no real-time requirements. Hardware available: RTX 3080 (10 GB VRAM) locally, RunPod A40 (48 GB VRAM) in the cloud. Sessions are recorded in English only, either via Discord (potentially multi-track via Craig bot) or a single microphone for in-person play.

## **1.3 Functional Requirements**

**\[PM\]** The application must accept a single audio file (WAV, FLAC, MP3, or OGG) alongside a session manifest specifying which players were present and pointing to their voice enrollment samples. It must output a diarized, speaker-identified transcript in markdown format with timestamps and speaker names. The system must also support a multi-track mode where pre-separated per-speaker audio files (from Craig bot) bypass diarization entirely.

## **1.4 Non-Functional Requirements**

**\[PM\]** Processing must complete within 60 minutes for 2.5 hours of audio on the RTX 3080\. The system must be operable via a single CLI command. Accuracy targets: below 7% Word Error Rate for transcription, below 15% Diarization Error Rate for speaker attribution. The transcript must use real player names, not anonymous labels.

## **1.5 Product Decisions**

**\[PM\]** Several deliberate product-level decisions shape this architecture. First, the system is English-only, which unlocks the use of NVIDIA’s Canary Qwen 2.5B — the current leader on the Open ASR Leaderboard at 5.63% WER — instead of the multilingual but lower-accuracy Whisper family. Second, because the user processes one file per week, wall-clock time is less critical than accuracy; the architecture can afford sequential model execution with aggressive VRAM management. Third, the known-speaker constraint is the application’s most powerful advantage and must be exploited at every pipeline stage.

# **2\. Architectural Overview**

## **2.1 Design Philosophy**

**\[Architect\]** The architecture follows a cascaded pipeline pattern with strict phase boundaries and explicit model lifecycle management. Each phase loads exactly one model family, processes the full audio, writes intermediate results to disk, and then unloads completely before the next phase begins. This “sequential-exclusive” execution model is the key constraint that makes the entire pipeline viable on 10 GB VRAM.

The cascaded approach is deliberately chosen over end-to-end alternatives. While unified models like SpeakerLM (Alibaba, 2025\) theoretically eliminate error propagation between phases, they remain paper-only with no public weights. The cascaded pipeline also provides critical debuggability: when a speaker is misidentified, the GM can inspect intermediate RTTM files, embedding distances, and word-level alignments to diagnose exactly where the error occurred.

## **2.2 Pipeline Phases**

| Phase | Name | Description | VRAM |
| :---- | :---- | :---- | :---- |
| **Phase 0** | Audio Preprocessing | Normalize to 16 kHz mono WAV, speech enhancement via MossFormer2, loudness normalization. Runs on CPU. | \~2 GB |
| **Phase 1** | Speaker Enrollment | Extract embeddings from voice samples using WeSpeaker/TitaNet. Build identity database. | \~1 GB |
| **Phase 2** | Transcription | Chunk audio into 40s segments. Transcribe with Canary Qwen 2.5B (SALM). Produces punctuated English text. | \~8 GB |
| **Phase 3** | Forced Alignment | Align transcript to audio using NeMo Forced Aligner \+ Parakeet CTC. Produces word-level timestamps. | \~2–3 GB |
| **Phase 4** | Speaker Diarization | Segment audio into speaker-homogeneous regions using nemo with oracle speaker count. | \~1.6 GB |
| **Phase 5** | Identity Resolution | Match anonymous clusters to enrolled players via cosine similarity \+ Hungarian algorithm. | \~1 GB |
| **Phase 6** | Fusion & Post-Processing | Merge word timestamps with speaker segments. Optional DiarizationLM semantic correction. | \~8 GB (opt) |
| **Phase 7** | Output Formatting | Render final markdown transcript with timestamps, speaker names, and turn grouping. | CPU only |

# **3\. Detailed Phase Specifications**

## **3.1 Transcription with Canary Qwen 2.5B**

**\[LLM Scientist\]** NVIDIA’s Canary Qwen 2.5B represents the current state of the art for English ASR. It is a Speech-Augmented Language Model (SALM) combining a FastConformer encoder with a Qwen3-1.7B LLM decoder. The model achieves 5.63% WER on the Open ASR Leaderboard, 1.6% on LibriSpeech Clean, and 3.1% on LibriSpeech Other. Trained on 234,000 hours of diverse English speech. Runs at 418x real-time factor. Released under CC-BY-4.0.

### **Architecture Implications**

Canary Qwen uses an attention-based encoder-decoder architecture (AED), which means it does not natively produce reliable word-level timestamps. An NVIDIA engineer confirmed that the Qwen 2.5B variant does not support the timestamp feature available in the smaller Canary-1b-flash. This necessitates the separate forced alignment phase (Phase 3).

### **Chunking Strategy**

The model’s maximum training duration was 40 seconds with a 1024-token sequence length. For 2.5-hour recordings, the audio must be chunked using overlapping windows: 40-second chunks with a 5-second overlap stride. During reassembly, discard the first and last 2.5 seconds of each chunk and stitch at overlap boundaries using text deduplication.

### **Dual-Mode Capability**

After transcription, the same model can operate in LLM mode to summarize the transcript, answer questions about session content, or extract structured data (e.g., “List all NPCs mentioned”). This provides immediate value for post-session GM workflows at no additional VRAM cost.

**\[Engineer\]** At \~8 GB in fp16, Canary Qwen 2.5B nearly saturates the RTX 3080\. Set torch.cuda.memory\_fraction(0.95) and disable gradient computation. After processing all chunks, fully unload before Phase 3\.

## **3.2 Forced Alignment via NeMo Forced Aligner**

**\[LLM Scientist\]** NeMo Forced Aligner (NFA) uses a Parakeet CTC model to generate frame-level character emission probabilities and applies the Viterbi algorithm for optimal alignment. NFA with Parakeet CTC is recommended because it operates within the NeMo ecosystem, produces superior results to WhisperX’s Wav2Vec2 alignment for English, and handles punctuated text correctly. Alignment error: typically 20–50 ms per word boundary for clean speech.

**\[Architect\]** Abstract behind a BaseAligner interface so the system can swap between NFA, WhisperX alignment, and CTC-Forced-Aligner without changing downstream code. Output: standardized AlignedWord objects with word, start, end, and confidence fields.

## **3.3 Speaker Diarization**

**\[LLM Scientist\]** Recommended: pyannote.audio 3.x with speaker-diarization-3.1. PyanNet segmentation (\~1.4M params) for joint VAD, overlap detection, and local speaker assignment, with WeSpeaker ResNet34-LM embeddings (\~6.6M params) and agglomerative clustering. Peak VRAM: \~1.6 GB. Speaker count constrained via min\_speakers/max\_speakers parameters.

### **Oracle Speaker Count**

The known player count is the single most impactful prior. Injecting it eliminates the fragile eigengap estimation in spectral clustering, preventing both speaker hallucination (phantom voices from background noise) and cluster merging (similar-sounding players collapsed into one). For TTRPG sessions with 4–6 players, this alone reduces DER by 20–40% compared to blind estimation.

### **Advanced Alternative: NeMo MSDD**

NeMo’s Multi-Scale Diarization Decoder extracts embeddings at five temporal scales (0.5s–1.5s) with dynamic 1D-CNN scale weighting. With oracle\_num\_speakers=True and emb\_batch\_size=8: 3–5 GB VRAM but 8–16 GB system RAM for clustering. Better accuracy but significantly more complex setup.

## **3.4 Identity Resolution**

**\[LLM Scientist\]** Extract session centroids from diarized segments (filtering out overlaps and segments under 1.0s), compare against enrollment database via cosine similarity, and apply the Hungarian algorithm for globally optimal 1:1 bipartite matching. Report confidence scores; similarities below 0.65 trigger warnings.

## **3.5 Fusion & LLM Post-Processing**

### **Anchor-Point Mapping**

For each aligned word, calculate temporal midpoint: anchor \= (word\_start \+ word\_end) / 2.0. Query the RTTM timeline for the active speaker at that millisecond.

### **DiarizationLM (Recommended Post-Processing)**

**\[LLM Scientist\]** Google’s DiarizationLM provides superior post-processing over statistical mode smoothing. The open-source framework formats the word-level attributed transcript into a prompt for a fine-tuned LLM that corrects speaker attribution errors based on semantic understanding. The DiarizationLM-8b-Fisher-v2 model reduces Word Diarization Error Rate by 55.5%. The TPST algorithm mathematically guarantees the LLM can only modify speaker labels, never the transcript text.

### **SEAL Framework (Future)**

The SEAL framework (Amazon, ICASSP 2025\) represents the theoretical pinnacle: acoustic-conditioned LLM correction with constrained decoding. However, SEAL has no public code or model release. The architecture should adopt it as a drop-in replacement when available.

# **4\. Advanced Architectural Considerations**

## **4.1 TTRPG-Specific Optimizations**

### **Custom Vocabulary (Hotwords)**

TTRPG sessions are dense with proper nouns that ASR models have never seen. Canary Qwen 2.5B does not currently support hotword biasing, but its LLM decoder’s contextual understanding partially compensates. A post-transcription correction pass using a campaign-specific dictionary is recommended.

### **Overlapping Speech**

TTRPG sessions feature significantly more overlap than business meetings. pyannote handles standard 2-speaker overlap well, but 3+ simultaneous speakers produce degraded results. For heavy overlap, consider ClearerVoice-Studio MossFormer2\_SS\_16K as preprocessing (2-speaker mixtures only).

## **4.2 Future: Target-Speaker Voice Activity Detection**

**\[LLM Scientist\]** TS-VAD transforms diarization from unsupervised clustering to supervised per-speaker binary classification, conditioned on enrollment embeddings. The state-of-the-art TS-VAD+ (2025) uses Conformer cross-speaker layers with a Profile Enhancement Module for continuous denoising. DER reduction from 19.53% to 17.97% on DIHARD-III. The architecture includes a BaseDiarizer interface for future TS-VAD integration.

## **4.3 Future: End-to-End SpeakerLM**

SpeakerLM (Alibaba, 2025\) is a single multimodal LLM ingesting raw audio plus enrollment embeddings, outputting fully speaker-attributed transcripts. Improvements of up to 6.60% cpCER over best cascaded systems. No public weights or code exist. Monitor for release.

## **4.4 Extensibility Architecture**

**\[Architect\]** Each phase is defined by an abstract base class (BasePreprocessor, BaseTranscriber, BaseAligner, BaseDiarizer, BaseIdentityResolver, BaseFusionEngine, BaseFormatter). A centralized model registry manages download caching and enforces the one-model-at-a-time VRAM constraint. Full pipeline configuration via a single YAML file.

# **5\. Implementation Guide**

## **5.1 Project Structure**

ttrpg-transcriber/

├── pyproject.toml

├── config/

│   ├── default.yaml

│   └── session\_manifest.yaml

├── src/ttrpg\_transcriber/

│   ├── cli.py             \# CLI entry point

│   ├── pipeline.py        \# Pipeline orchestrator

│   ├── gpu.py             \# VRAM lifecycle manager

│   ├── preprocessing/     \# Phase 0

│   ├── enrollment/        \# Phase 1

│   ├── transcription/     \# Phase 2 (Canary Qwen)

│   ├── alignment/         \# Phase 3 (NFA)

│   ├── diarization/       \# Phase 4 (pyannote)

│   ├── identity/          \# Phase 5

│   ├── fusion/            \# Phase 6

│   └── output/            \# Phase 7

├── voices/                \# Enrollment samples

└── tests/

## **5.2 Implementation Order**

**\[Architect\]** Implement phases incrementally to build confidence:

| Sprint | Scope | Deliverable |
| :---- | :---- | :---- |
| **Sprint 1** | Skeleton \+ Preprocessing \+ Transcription | Build CLI, config, GPU lifecycle manager, Canary chunking. Validate WER. |
| **Sprint 2** | Diarization \+ Output | Add pyannote \+ markdown formatter. Produces anonymous-label transcripts. |
| **Sprint 3** | Enrollment \+ Identity | Add voice enrollment and Hungarian matching. First milestone with real names. |
| **Sprint 4** | Alignment \+ Fusion | Add NFA forced alignment and anchor-point mapping. Word-level precision. |
| **Sprint 5** | Multi-Track \+ Polish | Craig bot support, campaign dictionary, DiarizationLM, JSON sidecar. |

## **5.3 GPU Lifecycle Manager**

**\[Engineer\]** The GPU lifecycle manager is the single most critical infrastructure code. It enforces the sequential-exclusive execution model using Python context managers that guarantee VRAM cleanup on exit via del model, gc.collect(), and torch.cuda.empty\_cache(). It includes a verify\_cleanup method that asserts allocated VRAM is below a threshold between phases.

## **5.4 Testing Strategy**

**\[Engineer\]** Build a test fixture from the first 10 minutes of a real session. Hand-label ground-truth transcript with speaker attributions and word-level timestamps. Measure WER (via jiwer), DER (via pyannote.metrics), and end-to-end Speaker-Attributed WER across sprints. Maintain a regression test that runs after every dependency update.

# **6\. Risk Register**

| Risk | Impact | Mitigation |
| :---- | :---- | :---- |
| Canary Qwen exceeds 10 GB VRAM | High | Reduce chunk size to 30s; batch\_size=1 |
| NeMo trunk breaks between releases | Medium | Pin to specific git commit in pyproject.toml |
| pyannote 4.x VRAM regression | Medium | Stay on pyannote 3.3.2; monitor releases |
| Enrollment drift over time | Low | Re-enroll quarterly; average multiple samples |
| Heavy TTRPG overlap degrades DER | Medium | Accept limitation; Craig bot eliminates this |
| Campaign vocab causes ASR errors | Medium | Campaign dictionary post-correction |
| DiarizationLM hallucination | Low | TPST algorithm guarantees text preservation |

# **7\. Model Selection Summary**

| Component | Model | Key Metric | VRAM |
| :---- | :---- | :---- | :---- |
| **Transcription** | Canary Qwen 2.5B | 5.63% WER, 418 RTFx, English-only | \~8 GB |
| **Forced Alignment** | Parakeet CTC (NFA) | 20–50 ms alignment error | \~2–3 GB |
| **Voice Activity Detection** | pyannote PyanNet | Integrated in diarization pipeline | \~1.6 GB (total) |
| **Speaker Embeddings** | WeSpeaker ResNet34-LM | 0.723% EER, 256-dim | \~0.5 GB |
| **Diarization** | pyannote speaker-diarization-3.1 | Agglomerative clustering \+ oracle count | \~1.6 GB |
| **Speaker Enhancement** | MossFormer2 SE 48K | ClearerVoice-Studio | \~2 GB |
| **Post-Processing** | DiarizationLM-8b-Fisher-v2 | 55.5% WDER reduction | \~8 GB (4-bit) |

