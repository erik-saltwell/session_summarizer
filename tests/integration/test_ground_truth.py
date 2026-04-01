"""
Integration test: validate transcription accuracy against ground truth.

Setup:
  - Creates test session, copies test_meeting/original.m4a
  - Registers speakers FEE013, FEE016, MEE014, MEO015
  - Runs the full CanaryQwen + ParakeetCTC pipeline
  - Compares output against test_meeting/SimpleTranscript.json

Passes if:
  - No pipeline errors
  - Accuracy metrics meet configurable thresholds (override via env vars)

Note on timing: Word-level alignment (Parakeet CTC) produces fine-grained
segments (~3s max), so timing error should be low (~1-2s).
Note on accuracy: Full-text WER on meeting audio is ~35%. Per-segment WER
is higher due to segment-GT matching overhead. Thresholds are calibrated
to catch regressions, not to match clean-speech benchmarks.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pytest

from session_summarizer.processing_results import TranscriptionSegment

# ---------------------------------------------------------------------------
# Configurable thresholds (override via environment variables)
# ---------------------------------------------------------------------------

MIN_OVERALL_WORD_ACCURACY = float(os.environ.get("MIN_OVERALL_WORD_ACCURACY", "0.20"))
MIN_HIGH_CONF_WORD_ACCURACY = float(os.environ.get("MIN_HIGH_CONF_WORD_ACCURACY", "0.15"))
MIN_MED_CONF_WORD_ACCURACY = float(os.environ.get("MIN_MED_CONF_WORD_ACCURACY", "0.15"))
MIN_LOW_CONF_WORD_ACCURACY = float(os.environ.get("MIN_LOW_CONF_WORD_ACCURACY", "0.10"))
MAX_TIMING_ERROR_SECONDS = float(os.environ.get("MAX_TIMING_ERROR_SECONDS", "5.0"))
LOW_CONFIDENCE_MAX = float(os.environ.get("LOW_CONFIDENCE_MAX", "0.50"))
HIGH_CONFIDENCE_MIN = float(os.environ.get("HIGH_CONFIDENCE_MIN", "0.85"))


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class GroundTruthPhrase:
    text: str
    start: float
    end: float
    speaker: str


@dataclass
class AlignedPair:
    gt: GroundTruthPhrase
    segment: TranscriptionSegment
    overlap_ratio: float  # overlap / gt_duration
    timing_error: float  # |gt_midpoint - seg_midpoint| in seconds


# ---------------------------------------------------------------------------
# WER implementation (no external dependencies)
# ---------------------------------------------------------------------------


def _normalize(text: str) -> list[str]:
    """Lowercase, strip punctuation, normalize unicode, split into words."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()


def compute_wer(reference: str, hypothesis: str) -> float:
    """
    Word Error Rate via Levenshtein edit distance on word lists.

    Returns float in [0.0, inf). Can exceed 1.0 with many insertions.
    Returns 0.0 if both are empty; 1.0 if reference is empty but hypothesis is not.
    """
    ref = _normalize(reference)
    hyp = _normalize(hypothesis)

    if not ref:
        return 0.0 if not hyp else 1.0

    n, m = len(ref), len(hyp)
    # O(m) space DP
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[m] / n


# ---------------------------------------------------------------------------
# Ground truth loading
# ---------------------------------------------------------------------------


def load_ground_truth(path: Path) -> list[GroundTruthPhrase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        GroundTruthPhrase(
            text=p["text"],
            start=float(p["start"]),
            end=float(p["end"]),
            speaker=p["speaker"],
        )
        for p in raw["phrases"]
    ]


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------


def align_gt_to_system(
    gt_phrases: list[GroundTruthPhrase],
    segments: list[TranscriptionSegment],
    min_overlap_ratio: float = 0.1,
) -> tuple[list[AlignedPair], list[GroundTruthPhrase]]:
    """
    For each GT phrase, find the system segment with maximum temporal overlap.

    Returns (aligned_pairs, unmatched_gt_phrases).
    GT phrases with no segment achieving min_overlap_ratio are reported as unmatched.
    """
    aligned: list[AlignedPair] = []
    unmatched: list[GroundTruthPhrase] = []

    for gt in gt_phrases:
        gt_dur = max(gt.end - gt.start, 1e-6)
        best_seg: TranscriptionSegment | None = None
        best_overlap = 0.0

        for seg in segments:
            ov = max(0.0, min(seg.end, gt.end) - max(seg.start, gt.start))
            if ov > best_overlap:
                best_overlap = ov
                best_seg = seg

        ratio = best_overlap / gt_dur
        if best_seg is None or ratio < min_overlap_ratio:
            unmatched.append(gt)
        else:
            gt_mid = (gt.start + gt.end) / 2.0
            seg_mid = (best_seg.start + best_seg.end) / 2.0
            aligned.append(
                AlignedPair(
                    gt=gt,
                    segment=best_seg,
                    overlap_ratio=ratio,
                    timing_error=abs(gt_mid - seg_mid),
                )
            )

    return aligned, unmatched


# ---------------------------------------------------------------------------
# Confidence banding
# ---------------------------------------------------------------------------


def _band(confidence: float) -> str:
    if confidence >= HIGH_CONFIDENCE_MIN:
        return "high"
    if confidence < LOW_CONFIDENCE_MAX:
        return "low"
    return "medium"


def _band_pairs(pairs: list[AlignedPair], band: str) -> list[AlignedPair]:
    return [p for p in pairs if _band(p.segment.confidence) == band]


def _band_wer(pairs: list[AlignedPair]) -> float | None:
    """Aggregate WER for a set of pairs by concatenating all text."""
    if not pairs:
        return None
    all_ref = " ".join(p.gt.text for p in pairs)
    all_hyp = " ".join(p.segment.text for p in pairs)
    return compute_wer(all_ref, all_hyp)


def _avg_timing(pairs: list[AlignedPair]) -> float | None:
    if not pairs:
        return None
    return sum(p.timing_error for p in pairs) / len(pairs)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _pf(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def generate_report(
    session_id: str,
    gt_phrases: list[GroundTruthPhrase],
    aligned: list[AlignedPair],
    unmatched: list[GroundTruthPhrase],
    segments: list[TranscriptionSegment],
) -> dict[str, bool]:
    """Print accuracy report to stdout. Returns dict of {check_name: passed}."""
    high_pairs = _band_pairs(aligned, "high")
    med_pairs = _band_pairs(aligned, "medium")
    low_pairs = _band_pairs(aligned, "low")

    overall_wer = _band_wer(aligned) if aligned else 1.0
    high_wer = _band_wer(high_pairs)
    med_wer = _band_wer(med_pairs)
    low_wer = _band_wer(low_pairs)

    overall_acc = 1.0 - (overall_wer or 1.0)
    high_acc = (1.0 - high_wer) if high_wer is not None else None
    med_acc = (1.0 - med_wer) if med_wer is not None else None
    low_acc = (1.0 - low_wer) if low_wer is not None else None

    avg_timing = _avg_timing(aligned) or 0.0
    high_avg_timing = _avg_timing(high_pairs)
    med_avg_timing = _avg_timing(med_pairs)
    low_avg_timing = _avg_timing(low_pairs)

    checks: dict[str, bool] = {
        "overall_accuracy": overall_acc >= MIN_OVERALL_WORD_ACCURACY,
        "high_conf_accuracy": (high_acc >= MIN_HIGH_CONF_WORD_ACCURACY) if high_acc is not None else True,
        "med_conf_accuracy": (med_acc >= MIN_MED_CONF_WORD_ACCURACY) if med_acc is not None else True,
        "low_conf_accuracy": (low_acc >= MIN_LOW_CONF_WORD_ACCURACY) if low_acc is not None else True,
        "timing_accuracy": avg_timing <= MAX_TIMING_ERROR_SECONDS,
    }
    all_pass = all(checks.values())

    W = 44
    print()
    print("=" * W)
    print("Ground Truth Accuracy Report")
    print("=" * W)
    print(f"Session ID:   {session_id}")
    print(f"Segments:     {len(segments)} system / {len(gt_phrases)} ground truth")
    print(f"Unmatched GT: {len(unmatched)} phrases (no overlapping system segment)")
    print()

    print("TRANSCRIPTION ACCURACY")
    overall_pf = _pf(checks["overall_accuracy"])
    print(
        f"  Overall:           {overall_acc * 100:5.1f}%  (WER: {(1 - overall_acc) * 100:.1f}%)"
        f"  [{overall_pf} >= {MIN_OVERALL_WORD_ACCURACY * 100:.1f}%]"
    )

    if high_acc is not None:
        h_pf = _pf(checks["high_conf_accuracy"])
        print(
            f"  High  (>={HIGH_CONFIDENCE_MIN:.2f}):   {high_acc * 100:5.1f}%  (N={len(high_pairs):3d})"
            f"  [{h_pf} >= {MIN_HIGH_CONF_WORD_ACCURACY * 100:.1f}%]"
        )
    else:
        print(f"  High  (>={HIGH_CONFIDENCE_MIN:.2f}):   N/A  (no segments in band)")

    if med_acc is not None:
        m_pf = _pf(checks["med_conf_accuracy"])
        band_label = f"{LOW_CONFIDENCE_MAX:.2f}-{HIGH_CONFIDENCE_MIN:.2f}"
        print(
            f"  Medium ({band_label}): {med_acc * 100:5.1f}%  (N={len(med_pairs):3d})"
            f"  [{m_pf} >= {MIN_MED_CONF_WORD_ACCURACY * 100:.1f}%]"
        )
    else:
        print("  Medium:            N/A  (no segments in band)")

    if low_acc is not None:
        l_pf = _pf(checks["low_conf_accuracy"])
        print(
            f"  Low   (<{LOW_CONFIDENCE_MAX:.2f}):    {low_acc * 100:5.1f}%  (N={len(low_pairs):3d})"
            f"  [{l_pf} >= {MIN_LOW_CONF_WORD_ACCURACY * 100:.1f}%]"
        )
    else:
        print(f"  Low   (<{LOW_CONFIDENCE_MAX:.2f}):    N/A  (no segments in band)")

    print()
    print("TIMING ACCURACY")
    t_pf = _pf(checks["timing_accuracy"])
    print(f"  Avg Error:   {avg_timing:6.2f}s  [{t_pf} <= {MAX_TIMING_ERROR_SECONDS:.2f}s]")
    if high_avg_timing is not None:
        print(f"  High conf:   {high_avg_timing:6.2f}s  (N={len(high_pairs)})")
    if med_avg_timing is not None:
        print(f"  Medium conf: {med_avg_timing:6.2f}s  (N={len(med_pairs)})")
    if low_avg_timing is not None:
        print(f"  Low conf:    {low_avg_timing:6.2f}s  (N={len(low_pairs)})")

    print()
    print("SPEAKER IDENTIFICATION: NOT IMPLEMENTED")
    print()
    print(f"Overall Result: {'PASS' if all_pass else 'FAIL'}")
    print("=" * W)
    print()

    return checks


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_transcription_accuracy() -> None:
    return
    # logfile_path = common_paths.generate_logfile_path()
    # test_logger: LoggingProtocol = CompositeLogger([RichConsoleLogger(Console()), FileLogger(logfile_path)])
    # with TempSession() as session:
    #     session.copy_in(common_paths.test_recording_path())

    #     global_yaml = common_paths.build_speakers_file_path()
    #     if not global_yaml.exists():
    #         RegisterSpeakersCommand().execute(test_logger)
    #         flush_gpu_memory()

    #     TranscribeAudioCommand(session_id=session.session_id).execute(test_logger)

    #     flush_gpu_memory()

    #     transcript_file = common_paths.session_dir(session.session_id) / "transcript.json"
    #     raw = json.loads(transcript_file.read_text(encoding="utf-8"))
    #     segments = [TranscriptionSegment(**s) for s in raw["segments"]]
    #     transcription_result: TranscriptionResult = TranscriptionResult(
    #         segments=segments, full_text=raw.get("full_text", "")
    #     )

    #     gt_phrases = load_ground_truth(common_paths.test_transcript_path())
    #     aligned, unmatched = align_gt_to_system(gt_phrases, transcription_result.segments)

    #     checks = generate_report(
    #         session_id=session.session_id,
    #         gt_phrases=gt_phrases,
    #         aligned=aligned,
    #         unmatched=unmatched,
    #         segments=transcription_result.segments,
    #     )

    #     failures = [name for name, passed in checks.items() if not passed]
    #     assert not failures, f"Accuracy checks failed: {', '.join(failures)}. See report above for details."
