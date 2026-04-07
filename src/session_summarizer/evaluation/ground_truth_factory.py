from __future__ import annotations

from difflib import SequenceMatcher
from typing import NamedTuple

from ..protocols import TextPhraseBuilder, TextPhraseSet


class PhraseBuilderPair(NamedTuple):
    baseline: TextPhraseBuilder
    updated: TextPhraseBuilder


def _collect_words_with_builders(phrase_set: TextPhraseSet) -> list[tuple[str, TextPhraseBuilder]]:
    plain = phrase_set.plain_text()
    sep_len = phrase_set.phrase_separator_length()
    result: list[tuple[str, TextPhraseBuilder]] = []
    current = 0
    for builder in phrase_set.phrase_builders_in_order():
        phrase = builder.build_phrase_data(current)
        text = plain[phrase.start_offset : phrase.end_offset].lower()
        for word in text.split():
            if word:
                result.append((word, builder))
        current = phrase.end_offset + sep_len
    return result


def map_phrase_pairs(baseline: TextPhraseSet, updated: TextPhraseSet) -> list[PhraseBuilderPair]:
    baseline_words = _collect_words_with_builders(baseline)
    updated_words = _collect_words_with_builders(updated)

    baseline_tokens = [w for w, _ in baseline_words]
    updated_tokens = [w for w, _ in updated_words]

    matcher = SequenceMatcher(None, baseline_tokens, updated_tokens, autojunk=False)

    result: list[PhraseBuilderPair] = []
    for block in matcher.get_matching_blocks():
        for i in range(block.size):
            result.append(
                PhraseBuilderPair(
                    baseline_words[block.a + i][1],
                    updated_words[block.b + i][1],
                )
            )

    return result
