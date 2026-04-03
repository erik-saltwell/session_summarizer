from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import session_summarizer.utils.common_paths as common_paths

from ..evaluation import TranscriptionValidationResult, clean_text_for_evaluation, evaluate_texts
from ..helpers.audio_cleaner import clean_audio
from ..helpers.audio_diarizer import diarize_audio
from ..helpers.audio_segmenter import SegmentSplitResultSet, compute_vad_segments
from ..helpers.audio_transcriber import transcribe_from_cleaned_audio
from ..helpers.confidence_scorer import score_confidence
from ..helpers.transcript_aligner import align_transcript
from ..processing_results import AlignmentResult, SpeechClipSet, TranscriptionResult
from ..settings.session_settings import SessionSettings
from .session_processing_command import SessionProcessingCommand

_METRIC_LABELS: list[str] = [
    "Word Error Rate (WER)",
    "Match Error Rate (MER)",
    "Word Information Loss (WIL)",
    "Word Information Preserved (WIP)",
]


def _format_metric(value: float) -> str:
    return f"{value:.4f}"


def _result_column(result: TranscriptionValidationResult) -> list[str]:
    return [
        _format_metric(result.word_error_rate),
        _format_metric(result.match_error_rate),
        _format_metric(result.word_information_loss),
        _format_metric(result.word_information_preservation),
    ]


@dataclass
class DumpAndCompareTextsCommand(SessionProcessingCommand):
    def name(self) -> str:
        return "Dump and Compare Texts"

    def create_output_path(self, original_path: Path) -> Path:
        return original_path.parent / (original_path.stem + "_fulltext.txt")

    def clean_and_dump_text(self, text: str, original_path: Path) -> str:
        cleaned_text = clean_text_for_evaluation(text)
        saved_text = cleaned_text.replace(" ", "\n")
        with open(self.create_output_path(original_path), "w") as f:
            f.write(saved_text)
        return cleaned_text

    def evaluate_texts(self, ground_truth: str, evaluated_text: str, name: str) -> TranscriptionValidationResult:
        return evaluate_texts(test_name=name, hypothesis=evaluated_text, reference=ground_truth, logger=self.logger)

    def process_session(self, settings: SessionSettings, session_dir: common_paths.Path) -> None:
        self.gpu_logging_enabled = True
        ground_truth: str = clean_text_for_evaluation(TranscriptionResult.load_from_test_meeting().plain_text())

        clean_audio(settings, session_dir, True, self, self.logger)
        segments: SegmentSplitResultSet = compute_vad_segments(settings, session_dir, True, self, self.logger)

        result: TranscriptionResult = transcribe_from_cleaned_audio(
            settings, session_dir, segments, True, self, self.logger
        )
        transcription_text = self.clean_and_dump_text(result.plain_text(), session_dir / settings.transcript_file)
        transcription_eval = self.evaluate_texts(ground_truth, transcription_text, "Transcription")

        alignment: AlignmentResult = align_transcript(settings, session_dir, result, segments, True, self, self.logger)
        aligned_text = self.clean_and_dump_text(alignment.plain_text(), session_dir / settings.aligned_transcript_path)
        align_eval = self.evaluate_texts(transcription_text, aligned_text, "Aligned Transcript")

        alignment = score_confidence(settings, session_dir, alignment, segments, True, self, self.logger)
        scored_text = self.clean_and_dump_text(
            alignment.plain_text(), session_dir / settings.confidence_transcript_path
        )
        scored_eval = self.evaluate_texts(aligned_text, scored_text, "Confidence Scored Transcript")

        clips: SpeechClipSet = diarize_audio(settings, session_dir, alignment, True, self, self.logger)
        clips.sort_clips()
        diarized_text = self.clean_and_dump_text(clips.plain_text(), session_dir / settings.base_diarized_path)
        diarized_eval = self.evaluate_texts(scored_text, diarized_text, "Diarized Transcript")

        results: list[TranscriptionValidationResult] = [transcription_eval, align_eval, scored_eval, diarized_eval]

        # Build table: rows = metrics, columns = transcribers
        # Build table: rows = metrics, columns = transcribers
        transcriber_names = [item.name for item in results]
        headers: list[str] = ["Metric"] + transcriber_names
        rows: list[list[str]] = []
        columns = [_result_column(result) for result in results]
        for i, label in enumerate(_METRIC_LABELS):
            rows.append([label] + [col[i] for col in columns])

        self.logger.report_message("[bold]Validation Results[/bold]")
        self.logger.report_multicolumn_table(headers, rows)
