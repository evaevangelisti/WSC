"""
Tests for src/wsc/extract/offsets.py.

Whether a pipeline reads a sentence rightly is kwic's to answer; what is asked
here is the query, the fallback, and the order the ranges come back in.
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

# The tagset every engine reports, stated here as it is stated in the source:
# what a reader is handed is worth writing down twice.
_UNIVERSAL_TAGS = {
    POS.NOUN: UNIVERSAL_POS.NOUN,
    POS.NAME: UNIVERSAL_POS.PROPN,
    POS.VERB: UNIVERSAL_POS.VERB,
    POS.ADJECTIVE: UNIVERSAL_POS.ADJ,
    POS.ADVERB: UNIVERSAL_POS.ADV,
}


class TestQueries:
    """
    What one entry is looked for by.
    """

    @given(words, _PARTS_OF_SPEECH, _FORMS)
    def test_carries_the_headword_and_every_form_collected(
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
    def test_names_the_part_of_speech_in_the_tagset_engines_report(
        self,
        headword: str,
        pos: POS,
        forms: frozenset[str],
    ) -> None:
        """Wiktionary writes adj where Universal Dependencies writes ADJ."""
        assert build_query(headword, pos, forms).pos == _UNIVERSAL_TAGS[pos]


class TestSearches:
    """
    What comes back for a batch of sentences.
    """

    @given(st.lists(words, max_size=6))
    def test_answers_every_sentence_it_was_handed(
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
    def test_reads_each_sentence_for_its_own_lemma(
        self,
        locator: Locator,
        headwords: list[str],
    ) -> None:
        """The ranges come back beside the sentence they were read out of."""
        searches = [
            (f"1 {headword} 2", build_query(headword, POS.NOUN, frozenset({headword})))
            for headword in headwords
        ]

        found = list(find_word_offsets(locator, searches))

        assert all(
            text[start:end].casefold() == query.lemma.casefold()
            for (text, query), word_offsets in zip(searches, found, strict=True)
            for start, end in word_offsets
        )

    @given(words)
    def test_reads_leftmost_first_and_never_twice_over(
        self,
        locator: Locator,
        headword: str,
    ) -> None:
        """A sentence may attest the lemma more than once, and each occurrence once."""
        text = f"{headword} and {headword} again {headword}"

        (found,) = find_word_offsets(
            locator,
            [(text, build_query(headword, POS.NOUN, frozenset({headword})))],
        )

        assert all(start < end for start, end in found)
        assert all(before[1] <= after[0] for before, after in pairwise(found))


class TestFallback:
    """
    Matching the listed forms where the reading found nothing.
    """

    @given(words)
    def test_matches_a_form_the_reading_passed_over(
        self,
        locator: Locator,
        headword: str,
    ) -> None:
        """The engine reads every word as a noun, so a verb is never read off."""
        text = f"1 {headword} 2"

        (found,) = find_word_offsets(
            locator,
            [(text, build_query(headword, POS.VERB, frozenset({headword})))],
        )

        assert found == ((2, 2 + len(headword)),)

    @given(st.data())
    def test_hands_back_nothing_where_neither_finds_it(
        self,
        locator: Locator,
        data: st.DataObject,
    ) -> None:
        """A sentence illustrating a sense need not spell the lemma out."""
        text = data.draw(st.text(alphabet="123 ", max_size=20))
        headword = data.draw(words)

        (found,) = find_word_offsets(
            locator,
            [(text, build_query(headword, POS.VERB, frozenset({headword})))],
        )

        assert found == ()

    @given(st.data())
    def test_falls_back_on_one_sentence_without_touching_the_next(
        self,
        locator: Locator,
        data: st.DataObject,
    ) -> None:
        """Each sentence is answered for itself, whichever found it."""
        read = data.draw(words)
        matched = data.draw(
            words.filter(lambda word: word.casefold() != read.casefold())
        )

        searches = [
            (f"1 {read} 2", build_query(read, POS.NOUN, frozenset({read}))),
            (f"1 {matched} 2", build_query(matched, POS.VERB, frozenset({matched}))),
        ]

        assert list(find_word_offsets(locator, searches)) == [
            ((2, 2 + len(read)),),
            ((2, 2 + len(matched)),),
        ]
