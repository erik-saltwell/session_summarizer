from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..helpers.audio_cleaner import clean_audio
from ..protocols import SessionSettings, TranscriberProtocol
from ..protocols.transcriber_protocol import TranscriptionResult
from ..transcription import CanaryQwenTranscriber, WhisperTranscriber
from ..transcription.transcription_validator import TranscriptionValidationResult, validate_transcriber
from .session_processing_command import SessionProcessingCommand

# ---------------------------------------------------------------------------
# Transcriber registry
# ---------------------------------------------------------------------------
# Each entry is (display_name, factory).  The factory receives the device
# string from SessionSettings and returns a ready-to-use transcriber.
# To add a new transcriber, append one tuple here — nothing else to change.

TranscriberFactory = Callable[[str], TranscriberProtocol]

TRANSCRIBER_REGISTRY: list[tuple[str, TranscriberFactory]] = [
    ("Canary Qwen", lambda device: CanaryQwenTranscriber(device=device)),
    ("Whisper", lambda device: WhisperTranscriber(device=device)),
]

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
class ValidateTranscribersCommand(SessionProcessingCommand):
    def name(self) -> str:
        return "Validate Transcribers"

    def process_session(self, settings: SessionSettings, session_dir: Path) -> None:
        clean_audio(settings, session_dir, True, self, self.logger)
        audio_path: Path = session_dir / settings.cleaned_audio_file

        results: dict[str, TranscriptionValidationResult] = {}

        failed: list[str] = []

        for transcriber_name, factory in TRANSCRIBER_REGISTRY:
            self.logger.report_message(f"[blue]Running {transcriber_name}...[/blue]")
            try:
                transcriber = factory(settings.device)
                self.report_gpu_usage(f"before {transcriber_name}")

                transcription: TranscriptionResult = transcriber.transcribe(audio_path, self.logger)
                self.report_gpu_usage(f"after {transcriber_name}")

                validation = validate_transcriber(settings, session_dir, transcription, self.logger)
                results[transcriber_name] = validation
            except Exception as exc:
                self.logger.report_error(f"[red]{transcriber_name} failed: {exc}[/red]")
                failed.append(transcriber_name)

        # Build table: rows = metrics, columns = transcribers
        transcriber_names = list(results.keys())
        headers: list[str] = ["Metric"] + transcriber_names
        rows: list[list[str]] = []
        columns = [_result_column(results[name]) for name in transcriber_names]
        for i, label in enumerate(_METRIC_LABELS):
            rows.append([label] + [col[i] for col in columns])

        self.logger.add_break()
        if results:
            self.logger.report_message("[bold]Transcriber Validation Results[/bold]")
            self.logger.report_multicolumn_table(headers, rows)
        if failed:
            self.logger.report_message(f"[red bold]Failed transcribers: {', '.join(failed)}[/red bold]")
