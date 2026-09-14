"""
Tests for src/wsc/extract/offsets.py.

Whether a pipeline reads a sentence rightly is kwic's to answer; what is asked here is
the query, the fallback, and the order the ranges come back in.
"""

from itertools import pairwise

from hypothesis import given
from hypothesis import strategies as st
from kwic import POS as UNIVERSAL_POS
from kwic import Locator
from strategies import words

from wsc.extract import build_query, find_word_offsets
from wsc.models import POS

_FORMS = st.lists(words, max_size=4).map(frozenset)

_PARTS_OF_SPEECH = st.sampled_from(POS)

_UNIVERSAL_TAGS = {
    POS.NOUN: UNIVERSAL_POS.NOUN,
    POS.PROPN: UNIVERSAL_POS.PROPN,
    POS.VERB: UNIVERSAL_POS.VERB,
    POS.ADJECTIVE: UNIVERSAL_POS.ADJ,
    POS.ADVERB: UNIVERSAL_POS.ADV,
}


class TestQueries:
    """What one entry is looked for by."""

    @given(words, _PARTS_OF_SPEECH, _FORMS)
    def test_preserves_query_forms(
        self,
        headword: str,
        pos: POS,
        forms: frozenset[str],
    ) -> None:
        """A sentence attests the lemma in whatever form it needs."""
        query = build_query(headword, pos, forms)

        assert query.lemma == headword
        assert query.forms == forms

    @given(words, _PARTS_OF_SPEECH, _FORMS)
    def test_maps_engine_categories(
        self,
        headword: str,
        pos: POS,
        forms: frozenset[str],
    ) -> None:
        """Wiktionary writes adj where Universal Dependencies writes ADJ."""
        assert build_query(headword, pos, forms).pos == _UNIVERSAL_TAGS[pos]


class TestSearches:
    """What comes back for a batch of sentences."""

    @given(st.lists(words, max_size=6))
    def test_returns_sentence_offsets(
        self,
        locator: Locator,
        headwords: list[str],
    ) -> None:
        """A sentence attesting nothing is still a sentence that was read."""
        searches = [
            (f"1 {headword} 2", build_query(headword, POS.NOUN, frozenset({headword})))
            for headword in headwords
        ]

        assert len(list(find_word_offsets(locator, searches))) == len(searches)

    @given(st.lists(words, min_size=1, max_size=6))
    def test_isolates_lemma_queries(
        self,
        locator: Locator,
        headwords: list[str],
    ) -> None:
        """The ranges come back beside the sentence they were read out of."""
        searches = [
            (f"1 {headword} 2", build_query(headword, POS.NOUN, frozenset({headword})))
            for headword in headwords
        ]

        offsets = list(find_word_offsets(locator, searches))

        assert all(
            text[start:end].casefold() == query.lemma.casefold()
            for (text, query), word_offsets in zip(searches, offsets, strict=True)
            for start, end in word_offsets
        )

    @given(words)
    def test_orders_unique_offsets(
        self,
        locator: Locator,
        headword: str,
    ) -> None:
        """A sentence may attest the lemma more than once, and each occurrence once."""
        text = f"{headword} and {headword} again {headword}"

        (offsets,) = find_word_offsets(
            locator,
            [(text, build_query(headword, POS.NOUN, frozenset({headword})))],
        )

        assert all(start < end for start, end in offsets)
        assert all(before[1] <= after[0] for before, after in pairwise(offsets))


class TestFallback:
    """Matching the listed forms where the reading found nothing."""

    @given(words)
    def test_matches_unrecognized_forms(
        self,
        locator: Locator,
        headword: str,
    ) -> None:
        """The engine reads every word as a noun, so a verb is never read off."""
        text = f"1 {headword} 2"

        (offsets,) = find_word_offsets(
            locator,
            [(text, build_query(headword, POS.VERB, frozenset({headword})))],
        )

        assert offsets == ((2, 2 + len(headword)),)

    @given(st.data())
    def test_returns_empty_matches(
        self,
        locator: Locator,
        data: st.DataObject,
    ) -> None:
        """A sentence illustrating a sense need not spell the lemma out."""
        text = data.draw(st.text(alphabet="123 ", max_size=20))
        headword = data.draw(words)

        (offsets,) = find_word_offsets(
            locator,
            [(text, build_query(headword, POS.VERB, frozenset({headword})))],
        )

        assert offsets == ()

    @given(st.data())
    def test_isolates_sentence_fallback(
        self,
        locator: Locator,
        data: st.DataObject,
    ) -> None:
        """Each sentence is answered for itself, whichever found it."""
        recognized_word = data.draw(words)
        fallback_word = data.draw(
            words.filter(lambda word: word.casefold() != recognized_word.casefold()),
        )

        searches = [
            (
                f"1 {recognized_word} 2",
                build_query(recognized_word, POS.NOUN, frozenset({recognized_word})),
            ),
            (
                f"1 {fallback_word} 2",
                build_query(fallback_word, POS.VERB, frozenset({fallback_word})),
            ),
        ]

        assert list(find_word_offsets(locator, searches)) == [
            ((2, 2 + len(recognized_word)),),
            ((2, 2 + len(fallback_word)),),
        ]
