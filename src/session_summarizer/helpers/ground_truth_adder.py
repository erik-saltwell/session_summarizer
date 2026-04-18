from __future__ import annotations

from typing import cast

from session_summarizer.processing_results.alignment_result import WordAlignment

from ..evaluation.ground_truth_factory import PhraseBuilderPair, map_phrase_pairs
from ..evaluation.test_meeting_data import TestMeeting, TestPhrase
from ..processing_results import SpeechClipSet
from ..utils import Tracer


def enhance_words_with_ground_truth(clips: SpeechClipSet, tracer: Tracer) -> None:
    test_meeting: TestMeeting = TestMeeting.load_test_meeting()
    matched_pairs: list[PhraseBuilderPair] = map_phrase_pairs(clips, test_meeting)
    tracer.add_context("ground_truth_count", len(matched_pairs))
    tracer.add_context("total_words", len(list(clips.all_words_in_order())))
    for matched_pair in matched_pairs:
        word: WordAlignment = cast(WordAlignment, matched_pair.baseline)
        test_phrase: TestPhrase = cast(TestPhrase, matched_pair.updated)
        word.ground_truth = test_phrase.speaker
