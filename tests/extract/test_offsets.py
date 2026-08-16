"""
Tests for src/wsc/extract/offsets.py.
"""

import string
from itertools import pairwise

from hypothesis import given
from hypothesis import strategies as st
from strategies import words

from wsc.extract import find_word_offsets

_TEXTS = st.text(max_size=60)

_FORMS = st.lists(words, min_size=1, max_size=4)

# ASCII alone, since a text is compared with itself in another case and only
# there does raising a letter leave its length alone.
_ASCII_WORDS = st.text(alphabet=string.ascii_letters, min_size=1, max_size=6)

_ASCII_TEXTS = st.text(alphabet=f"{string.ascii_letters} ", max_size=40)

# A form holding what a pattern would read, and nothing a pattern reads as a
# word, so that the padding around it cannot match.
_METACHARACTERS = st.text(alphabet="ab.*+?[](){}|^$\\", min_size=1, max_size=5)


def _stands_alone(
    text: str,
    start: int,
    end: int,
) -> bool:
    """
    Say whether a range has no word character on either side of it.

    Args:
        text: The sentence the range was read out of.
        start: Where the range opens.
        end: Where it closes, the way Python slices.

    Returns:
        Whether the occurrence is a word of its own.
    """
    neighbours = (text[start - 1] if start else "", text[end : end + 1])

    return not any(character.isalnum() or character == "_" for character in neighbours)


class TestOccurrences:
    """
    What is handed back for one sentence.
    """

    @given(_TEXTS, _FORMS)
    def test_a_range_slices_the_form_back_out(
        self,
        text: str,
        forms: list[str],
    ) -> None:
        """Half-open and in code points, so the text is indexed as Python does."""
        folded = {form.casefold() for form in forms}

        assert all(
            text[start:end].casefold() in folded
            for start, end in find_word_offsets(text, frozenset(forms))
        )

    @given(_TEXTS, _FORMS)
    def test_reads_leftmost_first_and_never_twice_over(
        self,
        text: str,
        forms: list[str],
    ) -> None:
        """A sentence may attest the lemma more than once, and each occurrence once."""
        found = find_word_offsets(text, frozenset(forms))

        assert all(start < end for start, end in found)
        assert all(before[1] <= after[0] for before, after in pairwise(found))

    @given(_TEXTS, _FORMS)
    def test_leaves_a_form_inside_a_longer_word_alone(
        self,
        text: str,
        forms: list[str],
    ) -> None:
        """A banker is not a bank, however the letters run."""
        assert all(
            _stands_alone(text, start, end)
            for start, end in find_word_offsets(text, frozenset(forms))
        )

    @given(st.data())
    def test_locates_every_occurrence_there_is(
        self,
        data: st.DataObject,
    ) -> None:
        """A sentence attesting the lemma five times is evidence five times over."""
        form = data.draw(words)
        others = words.filter(lambda word: word.casefold() != form.casefold())

        tokens = data.draw(st.lists(st.just(form) | others, max_size=6))
        text = " ".join(tokens)

        expected: list[tuple[int, int]] = []
        start = 0

        for token in tokens:
            if token == form:
                expected.append((start, start + len(token)))

            start += len(token) + 1

        assert find_word_offsets(text, frozenset({form})) == tuple(expected)

    @given(st.data())
    def test_hands_back_nothing_when_the_lemma_is_absent(
        self,
        data: st.DataObject,
    ) -> None:
        """A sentence illustrating a sense need not spell the lemma out."""
        text = data.draw(_TEXTS)
        form = data.draw(
            words.filter(lambda form: form.casefold() not in text.casefold())
        )

        assert find_word_offsets(text, frozenset({form})) == ()


class TestForms:
    """
    Which shapes of the lemma are looked for, and how.
    """

    @given(_ASCII_TEXTS, st.lists(_ASCII_WORDS, min_size=1, max_size=3))
    def test_reads_the_lemma_whatever_the_case(
        self,
        text: str,
        forms: list[str],
    ) -> None:
        """A sentence opening on the lemma capitalises it, and it is the lemma."""
        looked_for = frozenset(forms)
        found = find_word_offsets(text, looked_for)

        assert find_word_offsets(text.upper(), looked_for) == found
        assert find_word_offsets(text.lower(), looked_for) == found

    @given(words, words)
    def test_takes_the_longest_form_that_fits(
        self,
        head: str,
        tail: str,
    ) -> None:
        """A phrasal verb is what it is, not the verb that opens it."""
        phrase = f"{head} {tail}"

        assert find_word_offsets(phrase, frozenset({head, phrase})) == (
            (0, len(phrase)),
        )

    @given(words)
    def test_locates_a_form_closing_on_an_apostrophe(
        self,
        word: str,
    ) -> None:
        """A word boundary would fall on the wrong side of the apostrophe."""
        form = f"{word}'"
        text = f"He is {form} fast."

        assert find_word_offsets(text, frozenset({form})) == ((6, 6 + len(form)),)

    @given(_METACHARACTERS)
    def test_reads_a_form_as_text_rather_than_as_a_pattern(
        self,
        form: str,
    ) -> None:
        """A form holding what a regex would read is matched letter for letter."""
        text = f"! {form} !"

        assert find_word_offsets(text, frozenset({form})) == ((2, 2 + len(form)),)

    @given(words, words)
    def test_reads_one_lemma_after_another(
        self,
        first: str,
        second: str,
    ) -> None:
        """Only the last pattern is held on to, so forms must not outlive them."""
        text = f"{first} {second}"

        opening = find_word_offsets(text, frozenset({first}))
        closing = find_word_offsets(text, frozenset({second}))

        assert opening[0] == (0, len(first))
        assert closing[-1] == (len(first) + 1, len(text))
        assert find_word_offsets(text, frozenset({first})) == opening
