from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import jiwer
from mathspell import analyze_text

from session_summarizer.protocols.logging_protocol import LoggingProtocol
from session_summarizer.protocols.transcriber_protocol import TranscriptionResult
from session_summarizer.settings.session_settings import SessionSettings


@dataclass
class TranscriptionValidationResult:
    word_error_rate: float
    match_error_rate: float
    word_information_loss: float
    word_information_preservation: float


def _get_cleaned_text(result: TranscriptionResult) -> str:
    return cast(str, analyze_text(result.full_text))


def validate_transcriber(
    settings: SessionSettings, session_dir: Path, transcription_result: TranscriptionResult, logger: LoggingProtocol
) -> TranscriptionValidationResult:
    ground_truth: str = _get_cleaned_text(TranscriptionResult.load_from_test_meeting())
    test_text: str = _get_cleaned_text(transcription_result)

    transform: jiwer.Compose = jiwer.Compose(
        [
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.ReduceToListOfListOfWords(),
        ]
    )

    results: jiwer.WordOutput = jiwer.process_words(
        reference=ground_truth, hypothesis=test_text, reference_transform=transform, hypothesis_transform=transform
    )

    return TranscriptionValidationResult(
        word_error_rate=results.wer,
        match_error_rate=results.mer,
        word_information_loss=results.wil,
        word_information_preservation=results.wip,
    )
