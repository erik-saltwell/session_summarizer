from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import session_summarizer.utils.common_paths as common_paths

from ..evaluation import TranscriptionValidationResult, clean_text_for_evaluation, evaluate_texts
from ..processing_results import TranscriptionResult
from ..settings.session_settings import SessionSettings
from .add_embeddings import AddEmbeddingsCommand
from .align_transcript import AlignTranscriptCommand
from .diarizationlm_command import DiarizationLMCommand
from .diarize_audio import DiarizeAudioCommand
from .identify_speakers import IdentifySpeakersCommand
from .score_confidence import ScoreConfidenceCommand
from .session_processing_command import SessionProcessingCommand
from .transcribe_audio import TranscribeAudioCommand

_METRIC_LABELS: list[str] = [
    "Word Error Rate (WER)",
    "Match Error Rate (MER)",
    "Word Information Loss (WIL)",
    "Word Information Preserved (WIP)",
]


class PathAndName(NamedTuple):
    filepath: Path
    name: str


def _format_metric(value: float) -> str:
    return f"{value:.4f}"


def _result_column(result: TranscriptionValidationResult) -> list[str]:
    return [
        _format_metric(result.word_error_rate),
        _format_metric(result.match_error_rate),
        _format_metric(result.word_information_loss),
        _format_metric(result.word_information_preservation),
    ]


def _iterate_file_paths(settings: SessionSettings) -> Iterator[PathAndName]:
    yield PathAndName(settings.paths.transcript, "Transcript")
    yield PathAndName(settings.paths.aligned_transcript, "AlignedTranscript")
    yield PathAndName(settings.paths.confidence_transcript, "ScoredTranscript")
    yield PathAndName(settings.paths.base_diarization, "Diarized")
    yield PathAndName(settings.paths.diarizationlm_processed, "DiarizeLM")
    yield PathAndName(settings.paths.clips_with_embeddings, "Embeddings_Added")
    yield PathAndName(settings.paths.identified_speakers, "Speakers Identified")


@dataclass
class CompareFullTextCommand(SessionProcessingCommand):
    def name(self) -> str:
        return "Compare Texts"

    def add_dependencies(self, settings: SessionSettings, session_dir: Path) -> None:
        self.inputs.extend(path_and_name.filepath for path_and_name in _iterate_file_paths(settings))
        self.dependencies.extend(
            [
                IdentifySpeakersCommand(self.session_id, self.tracer),
                AddEmbeddingsCommand(self.session_id, self.tracer),
                DiarizationLMCommand(self.session_id, self.tracer),
                DiarizeAudioCommand(self.session_id, self.tracer),
                ScoreConfidenceCommand(self.session_id, self.tracer),
                AlignTranscriptCommand(self.session_id, self.tracer),
                TranscribeAudioCommand(self.session_id, self.tracer),
            ]
        )

    def evaluate_texts(self, ground_truth: str, evaluated_text: str, name: str) -> TranscriptionValidationResult:
        return evaluate_texts(test_name=name, hypothesis=evaluated_text, reference=ground_truth, logger=self.logger)

    def process_session(self, settings: SessionSettings, session_dir: common_paths.Path) -> None:
        self.gpu_logging_enabled = True
        ground_truth: str = clean_text_for_evaluation(TranscriptionResult.load_from_test_meeting().plain_text())

        results: list[TranscriptionValidationResult] = []

        for path_and_name in _iterate_file_paths(settings):
            full_text_path = self.postpend_text(path_and_name.filepath, "_fulltext", ".txt")
            evaluation_text = full_text_path.read_text()
            results.append(self.evaluate_texts(ground_truth, evaluation_text, path_and_name.name))
            ground_truth = evaluation_text

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
