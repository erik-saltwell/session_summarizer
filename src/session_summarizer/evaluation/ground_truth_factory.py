from __future__ import annotations

from collections.abc import Iterator
from difflib import SequenceMatcher
from enum import StrEnum
from typing import NamedTuple, cast

from intervaltree import Interval, IntervalTree

from ..protocols import TextPhrase, TextPhraseBuilder, TextPhraseSet


class OpcodeKind(StrEnum):
    EQUAL = "equal"
    REPLACE = "replace"
    DELETE = "delete"
    INSERT = "insert"


class Opcode(NamedTuple):
    kind: OpcodeKind
    baseline_start: int
    baseline_end: int
    updated_start: int
    updated_end: int


class PhraseBuilderPair(NamedTuple):
    baseline: TextPhraseBuilder
    updated: TextPhraseBuilder


class PhraseWithBuilder(NamedTuple):
    phrase: TextPhrase
    builder: TextPhraseBuilder


def iterate_sequence_match_opcodes(baseline: str, updated: str) -> Iterator[Opcode]:
    matcher = SequenceMatcher(None, baseline, updated)
    for raw_opcode in matcher.get_opcodes():
        opcode = Opcode(
            kind=OpcodeKind(raw_opcode[0]),
            baseline_start=raw_opcode[1],
            baseline_end=raw_opcode[2],
            updated_start=raw_opcode[3],
            updated_end=raw_opcode[4],
        )
        yield opcode


def build_interval_tree_from_phraseset(phrases: TextPhraseSet) -> IntervalTree:
    result: IntervalTree = IntervalTree()
    current: int = 0
    for phrase_bldr in phrases.phrase_builders_in_order():
        text_phrase: TextPhrase = phrase_bldr.build_phrase_data(current)
        result.add(Interval(text_phrase.start_offset, text_phrase.end_offset, phrase_bldr))
        current += text_phrase.length
        current += phrases.phrase_seperator_length()
    return result


def find_phrase_from_interval_tree(tree: IntervalTree, mark: int) -> TextPhraseBuilder | None:
    intervals = tree.at(mark)
    if not intervals:
        return None
    return cast(TextPhraseBuilder, next(iter(intervals)).data)


def build_interval_tree_from_opcodes(opcodes: Iterator[Opcode]) -> IntervalTree:
    result: IntervalTree = IntervalTree()
    for opcode in opcodes:
        if opcode.kind == OpcodeKind.EQUAL:
            result.add(Interval(opcode.baseline_start, opcode.baseline_end, opcode))
    return result


def find_opcode_from_interval_tree(tree: IntervalTree, mark: int) -> Opcode | None:
    intervals = tree.at(mark)
    if not intervals:
        return None
    return cast(Opcode, next(iter(intervals)).data)


def map_phrase_pairs(baseline: TextPhraseSet, updated: TextPhraseSet) -> list[PhraseBuilderPair]:
    result: list[PhraseBuilderPair] = []
    baseline_text: str = baseline.plain_text().lower()
    updated_text: str = updated.plain_text().lower()

    updated_tree: IntervalTree = build_interval_tree_from_phraseset(updated)
    opcode_tree: IntervalTree = build_interval_tree_from_opcodes(
        iterate_sequence_match_opcodes(baseline_text, updated_text)
    )
    current: int = 0
    for baseline_phrase_builder in baseline.phrase_builders_in_order():
        baseline_phrase: TextPhrase = baseline_phrase_builder.build_phrase_data(current)
        opcode: Opcode | None = find_opcode_from_interval_tree(opcode_tree, baseline_phrase.start_offset)
        if opcode is not None:
            delta = baseline_phrase.start_offset - opcode.baseline_start
            updated_phase_builder: TextPhraseBuilder | None = find_phrase_from_interval_tree(
                updated_tree, opcode.updated_start + delta
            )
            if updated_phase_builder is not None:
                result.append(PhraseBuilderPair(baseline_phrase_builder, updated_phase_builder))

        current += baseline_phrase.length
        current += baseline.phrase_seperator_length()

    return result
