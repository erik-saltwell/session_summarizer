from __future__ import annotations

from dataclasses import dataclass

import jiwer

from session_summarizer.protocols.logging_protocol import LoggingProtocol

from .text_cleaner import clean_text_for_evaluation


@dataclass
class TranscriptionValidationResult:
    name: str
    word_error_rate: float
    match_error_rate: float
    word_information_loss: float
    word_information_preservation: float


def validate_transcriber(
    test_name: str, hypothesis: str, reference: str, logger: LoggingProtocol
) -> TranscriptionValidationResult:
    ground_truth: str = clean_text_for_evaluation(reference)
    test_text: str = clean_text_for_evaluation(hypothesis)

    transform: jiwer.Compose = jiwer.Compose(
        [
            jiwer.ReduceToListOfListOfWords(),
        ]
    )

    results: jiwer.WordOutput = jiwer.process_words(
        reference=ground_truth, hypothesis=test_text, reference_transform=transform, hypothesis_transform=transform
    )

    return TranscriptionValidationResult(
        name=test_name,
        word_error_rate=results.wer,
        match_error_rate=results.mer,
        word_information_loss=results.wil,
        word_information_preservation=results.wip,
    )
