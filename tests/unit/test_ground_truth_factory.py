from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import cast

from session_summarizer.evaluation.ground_truth_factory import (
    Opcode,
    OpcodeKind,
    PhraseBuilderPair,
    build_interval_tree_from_opcodes,
    build_interval_tree_from_phraseset,
    find_opcode_from_interval_tree,
    find_phrase_from_interval_tree,
    iterate_sequence_match_opcodes,
    map_phrase_pairs,
)
from session_summarizer.protocols.phrase_data import TextPhrase, TextPhraseBuilder, TextPhraseSet


@dataclass(frozen=True, order=True)
class UnitTestPhraseBuilder(TextPhraseBuilder):
    text: str

    def build_phrase_data(self, start_offset: int) -> TextPhrase:
        return TextPhrase(start_offset, len(self.text))


@dataclass(frozen=True, order=True)
class UnitTestPhraseSet(TextPhraseSet):
    phrases: list[UnitTestPhraseBuilder] = field(init=False, compare=False)

    def __init__(self, *strings: str) -> None:
        object.__setattr__(
            self,
            "phrases",
            [UnitTestPhraseBuilder(text=text) for text in strings],
        )

    def plain_text(self) -> str:
        return " ".join(phrase.text for phrase in self.phrases)

    def phrase_builders_in_order(self) -> Iterator[TextPhraseBuilder]:
        yield from self.phrases

    def phrase_seperator_length(self) -> int:
        return 1


# ---------------------------------------------------------------------------
# iterate_sequence_match_opcodes
# ---------------------------------------------------------------------------


class TestIterateSequenceMatchOpcodes:
    def test_identical_strings(self) -> None:
        opcodes = list(iterate_sequence_match_opcodes("abc", "abc"))
        assert len(opcodes) == 1
        assert opcodes[0].kind == OpcodeKind.EQUAL

    def test_completely_different(self) -> None:
        opcodes = list(iterate_sequence_match_opcodes("aaa", "zzz"))
        kinds = {op.kind for op in opcodes}
        assert OpcodeKind.EQUAL not in kinds

    def test_partial_overlap(self) -> None:
        opcodes = list(iterate_sequence_match_opcodes("abcdef", "abcxyz"))
        kinds = [op.kind for op in opcodes]
        assert OpcodeKind.EQUAL in kinds
        assert any(k != OpcodeKind.EQUAL for k in kinds)

    def test_empty_baseline(self) -> None:
        opcodes = list(iterate_sequence_match_opcodes("", "abc"))
        assert all(op.kind == OpcodeKind.INSERT for op in opcodes)

    def test_empty_updated(self) -> None:
        opcodes = list(iterate_sequence_match_opcodes("abc", ""))
        assert all(op.kind == OpcodeKind.DELETE for op in opcodes)

    def test_both_empty(self) -> None:
        opcodes = list(iterate_sequence_match_opcodes("", ""))
        assert opcodes == []


# ---------------------------------------------------------------------------
# build_interval_tree_from_phraseset / find_phrase_from_interval_tree
# ---------------------------------------------------------------------------


class TestPhraseIntervalTree:
    def test_single_phrase(self) -> None:
        ps = UnitTestPhraseSet("hello")
        tree = build_interval_tree_from_phraseset(ps)
        assert len(tree) == 1
        builder = find_phrase_from_interval_tree(tree, 0)
        assert builder == UnitTestPhraseBuilder("hello")

    def test_multiple_phrases_offsets(self) -> None:
        # "one two three" -> offsets: one=[0,3) two=[4,7) three=[8,13)
        ps = UnitTestPhraseSet("one", "two", "three")
        tree = build_interval_tree_from_phraseset(ps)
        assert len(tree) == 3

        assert find_phrase_from_interval_tree(tree, 0) == UnitTestPhraseBuilder("one")
        assert find_phrase_from_interval_tree(tree, 2) == UnitTestPhraseBuilder("one")
        assert find_phrase_from_interval_tree(tree, 4) == UnitTestPhraseBuilder("two")
        assert find_phrase_from_interval_tree(tree, 8) == UnitTestPhraseBuilder("three")
        assert find_phrase_from_interval_tree(tree, 12) == UnitTestPhraseBuilder("three")

    def test_miss_in_separator(self) -> None:
        # offset 3 is the space separator between "one" and "two"
        ps = UnitTestPhraseSet("one", "two")
        tree = build_interval_tree_from_phraseset(ps)
        assert find_phrase_from_interval_tree(tree, 3) is None

    def test_miss_past_end(self) -> None:
        ps = UnitTestPhraseSet("ab")
        tree = build_interval_tree_from_phraseset(ps)
        assert find_phrase_from_interval_tree(tree, 2) is None

    def test_boundary_start_is_hit(self) -> None:
        ps = UnitTestPhraseSet("ab", "cd")
        tree = build_interval_tree_from_phraseset(ps)
        # "cd" starts at offset 3
        assert find_phrase_from_interval_tree(tree, 3) == UnitTestPhraseBuilder("cd")

    def test_boundary_end_is_miss(self) -> None:
        ps = UnitTestPhraseSet("ab", "cd")
        tree = build_interval_tree_from_phraseset(ps)
        # "ab" spans [0,2), so offset 2 is outside
        assert find_phrase_from_interval_tree(tree, 2) is None


# ---------------------------------------------------------------------------
# build_interval_tree_from_opcodes / find_opcode_from_interval_tree
# ---------------------------------------------------------------------------


class TestOpcodeIntervalTree:
    def test_only_equal_opcodes_included(self) -> None:
        opcodes = iter(
            [
                Opcode(OpcodeKind.REPLACE, 0, 3, 0, 3),
                Opcode(OpcodeKind.EQUAL, 3, 6, 3, 6),
                Opcode(OpcodeKind.DELETE, 6, 9, 6, 6),
            ]
        )
        tree = build_interval_tree_from_opcodes(opcodes)
        assert len(tree) == 1

    def test_empty_iterator(self) -> None:
        tree = build_interval_tree_from_opcodes(iter([]))
        assert len(tree) == 0

    def test_find_hit(self) -> None:
        opcodes = iter([Opcode(OpcodeKind.EQUAL, 5, 10, 15, 20)])
        tree = build_interval_tree_from_opcodes(opcodes)
        result = find_opcode_from_interval_tree(tree, 7)
        assert result is not None
        assert result.baseline_start == 5

    def test_find_miss(self) -> None:
        opcodes = iter([Opcode(OpcodeKind.EQUAL, 5, 10, 15, 20)])
        tree = build_interval_tree_from_opcodes(opcodes)
        assert find_opcode_from_interval_tree(tree, 3) is None

    def test_find_boundary_end_is_miss(self) -> None:
        opcodes = iter([Opcode(OpcodeKind.EQUAL, 5, 10, 15, 20)])
        tree = build_interval_tree_from_opcodes(opcodes)
        assert find_opcode_from_interval_tree(tree, 10) is None


# ---------------------------------------------------------------------------
# map_phrase_pairs
# ---------------------------------------------------------------------------


class TestMapPhrasePairs:
    def test_simple(self) -> None:
        baseline = UnitTestPhraseSet("one", "two", "three", "four")
        updated = UnitTestPhraseSet("one", "two", "three", "four")
        pairs = map_phrase_pairs(baseline, updated)
        assert len(pairs) == 4
        assert PhraseBuilderPair(UnitTestPhraseBuilder("one"), UnitTestPhraseBuilder("one")) in pairs
        assert PhraseBuilderPair(UnitTestPhraseBuilder("two"), UnitTestPhraseBuilder("two")) in pairs
        assert PhraseBuilderPair(UnitTestPhraseBuilder("three"), UnitTestPhraseBuilder("three")) in pairs
        assert PhraseBuilderPair(UnitTestPhraseBuilder("four"), UnitTestPhraseBuilder("four")) in pairs

    def test_long_updated(self) -> None:
        baseline = UnitTestPhraseSet("one", "two", "three", "four")
        updated = UnitTestPhraseSet("one two", "three four")
        pairs = map_phrase_pairs(baseline, updated)
        assert len(pairs) == 4
        assert PhraseBuilderPair(UnitTestPhraseBuilder("one"), UnitTestPhraseBuilder("one two")) in pairs
        assert PhraseBuilderPair(UnitTestPhraseBuilder("two"), UnitTestPhraseBuilder("one two")) in pairs
        assert PhraseBuilderPair(UnitTestPhraseBuilder("three"), UnitTestPhraseBuilder("three four")) in pairs
        assert PhraseBuilderPair(UnitTestPhraseBuilder("four"), UnitTestPhraseBuilder("three four")) in pairs

    def test_long_updated_with_missing_baseline(self) -> None:
        baseline = UnitTestPhraseSet("one", "two", "three", "four")
        updated = UnitTestPhraseSet("one two", "four five")
        pairs = map_phrase_pairs(baseline, updated)
        assert len(pairs) == 3
        assert PhraseBuilderPair(UnitTestPhraseBuilder("one"), UnitTestPhraseBuilder("one two")) in pairs
        assert PhraseBuilderPair(UnitTestPhraseBuilder("two"), UnitTestPhraseBuilder("one two")) in pairs
        assert PhraseBuilderPair(UnitTestPhraseBuilder("four"), UnitTestPhraseBuilder("four five")) in pairs

    def test_case_insensitive(self) -> None:
        baseline = UnitTestPhraseSet("Hello", "World")
        updated = UnitTestPhraseSet("hello", "world")
        pairs = map_phrase_pairs(baseline, updated)
        assert len(pairs) == 2
        assert PhraseBuilderPair(UnitTestPhraseBuilder("Hello"), UnitTestPhraseBuilder("hello")) in pairs
        assert PhraseBuilderPair(UnitTestPhraseBuilder("World"), UnitTestPhraseBuilder("world")) in pairs

    def test_no_overlap(self) -> None:
        baseline = UnitTestPhraseSet("aaa", "bbb")
        updated = UnitTestPhraseSet("xxx", "yyy")
        pairs = map_phrase_pairs(baseline, updated)
        assert pairs == []

    def test_single_word(self) -> None:
        baseline = UnitTestPhraseSet("one")
        updated = UnitTestPhraseSet("one")
        pairs = map_phrase_pairs(baseline, updated)
        assert len(pairs) == 1
        assert pairs[0] == PhraseBuilderPair(UnitTestPhraseBuilder("one"), UnitTestPhraseBuilder("one"))

    def test_partial_match_at_end(self) -> None:
        baseline = UnitTestPhraseSet("qqq", "two", "three")
        updated = UnitTestPhraseSet("zzz", "two", "three")
        pairs = map_phrase_pairs(baseline, updated)
        assert PhraseBuilderPair(UnitTestPhraseBuilder("two"), UnitTestPhraseBuilder("two")) in pairs
        assert PhraseBuilderPair(UnitTestPhraseBuilder("three"), UnitTestPhraseBuilder("three")) in pairs
        assert all(cast(UnitTestPhraseBuilder, p.baseline).text != "qqq" for p in pairs)

    def test_partial_match_at_start(self) -> None:
        baseline = UnitTestPhraseSet("one", "two", "three")
        updated = UnitTestPhraseSet("one", "two", "zero")
        pairs = map_phrase_pairs(baseline, updated)
        assert PhraseBuilderPair(UnitTestPhraseBuilder("one"), UnitTestPhraseBuilder("one")) in pairs
        assert PhraseBuilderPair(UnitTestPhraseBuilder("two"), UnitTestPhraseBuilder("two")) in pairs
        assert all(cast(UnitTestPhraseBuilder, p.baseline).text != "three" for p in pairs)

    def test_many_to_one_mapping(self) -> None:
        baseline = UnitTestPhraseSet("a", "b", "c", "d", "e", "f")
        updated = UnitTestPhraseSet("a b c", "d e f")
        pairs = map_phrase_pairs(baseline, updated)
        assert len(pairs) == 6
        for word in ("a", "b", "c"):
            assert PhraseBuilderPair(UnitTestPhraseBuilder(word), UnitTestPhraseBuilder("a b c")) in pairs
        for word in ("d", "e", "f"):
            assert PhraseBuilderPair(UnitTestPhraseBuilder(word), UnitTestPhraseBuilder("d e f")) in pairs

    def test_repeated_words_two_sentences(self) -> None:
        baseline = UnitTestPhraseSet("the", "cat", "the", "dog")
        updated = UnitTestPhraseSet("the cat", "the dog")
        pairs = map_phrase_pairs(baseline, updated)
        assert len(pairs) == 4
        assert PhraseBuilderPair(UnitTestPhraseBuilder("cat"), UnitTestPhraseBuilder("the cat")) in pairs
        assert PhraseBuilderPair(UnitTestPhraseBuilder("dog"), UnitTestPhraseBuilder("the dog")) in pairs

    def test_repeated_words_three_sentences(self) -> None:
        baseline = UnitTestPhraseSet("the", "cat", "the", "dog", "the", "bird")
        updated = UnitTestPhraseSet("the cat", "the dog", "the bird")
        pairs = map_phrase_pairs(baseline, updated)
        assert PhraseBuilderPair(UnitTestPhraseBuilder("cat"), UnitTestPhraseBuilder("the cat")) in pairs
        assert PhraseBuilderPair(UnitTestPhraseBuilder("dog"), UnitTestPhraseBuilder("the dog")) in pairs
        assert PhraseBuilderPair(UnitTestPhraseBuilder("bird"), UnitTestPhraseBuilder("the bird")) in pairs

    def test_repeated_words_identical_texts(self) -> None:
        baseline = UnitTestPhraseSet("go", "go", "go")
        updated = UnitTestPhraseSet("go go go")
        pairs = map_phrase_pairs(baseline, updated)
        assert len(pairs) == 3
        assert all(p.updated == UnitTestPhraseBuilder("go go go") for p in pairs)

    def test_repeated_words_with_unique_anchors(self) -> None:
        # Unique words anchor repeated "is" to the correct sentence
        baseline = UnitTestPhraseSet("he", "is", "tall", "she", "is", "short")
        updated = UnitTestPhraseSet("he is tall", "she is short")
        pairs = map_phrase_pairs(baseline, updated)
        assert len(pairs) == 6
        for word in ("he", "is", "tall"):
            assert PhraseBuilderPair(UnitTestPhraseBuilder(word), UnitTestPhraseBuilder("he is tall")) in pairs
        for word in ("she", "is", "short"):
            assert PhraseBuilderPair(UnitTestPhraseBuilder(word), UnitTestPhraseBuilder("she is short")) in pairs

    def test_repeated_words_some_unmatched(self) -> None:
        baseline = UnitTestPhraseSet("the", "cat", "the", "zzz")
        updated = UnitTestPhraseSet("the cat", "the dog")
        pairs = map_phrase_pairs(baseline, updated)
        assert PhraseBuilderPair(UnitTestPhraseBuilder("cat"), UnitTestPhraseBuilder("the cat")) in pairs
        assert all(cast(UnitTestPhraseBuilder, p.baseline).text != "zzz" for p in pairs)
