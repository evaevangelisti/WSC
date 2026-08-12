"""
Tests for src/wsc/extract/offsets.py.
"""

from wsc.extract import find_word_offsets


class TestOccurrences:
    """
    What is handed back for one sentence.
    """

    def test_locates_the_lemma(
        self,
    ) -> None:
        """The range is where the lemma sits, and nothing else is."""
        offsets = find_word_offsets("He robbed a bank.", frozenset({"bank"}))

        assert offsets == ((12, 16),)

    def test_a_range_slices_the_form_back_out(
        self,
    ) -> None:
        """Half-open and in code points, so the text is indexed as Python does."""
        text = "Città and bank."
        start, end = find_word_offsets(text, frozenset({"bank"}))[0]

        assert text[start:end] == "bank"

    def test_locates_every_occurrence_leftmost_first(
        self,
    ) -> None:
        """A sentence may attest the lemma more than once."""
        offsets = find_word_offsets("A bank beside a bank.", frozenset({"bank"}))

        assert offsets == ((2, 6), (16, 20))

    def test_hands_back_nothing_when_the_lemma_is_absent(
        self,
    ) -> None:
        """A sentence illustrating a sense need not spell the lemma out."""
        assert find_word_offsets("She went there.", frozenset({"bank"})) == ()


class TestForms:
    """
    Which shapes of the lemma are looked for, and how.
    """

    def test_locates_an_inflection(
        self,
    ) -> None:
        """A lemma is attested in whatever form the sentence needs."""
        offsets = find_word_offsets("Two banks closed.", frozenset({"bank", "banks"}))

        assert offsets == ((4, 9),)

    def test_reads_the_lemma_whatever_the_case(
        self,
    ) -> None:
        """A sentence opening on the lemma capitalises it."""
        assert find_word_offsets("Banks closed.", frozenset({"banks"})) == ((0, 5),)

    def test_leaves_a_form_inside_a_longer_word_alone(
        self,
    ) -> None:
        """A banker is not a bank, however the letters run."""
        assert find_word_offsets("The banker left.", frozenset({"bank"})) == ()

    def test_takes_the_longest_form_that_fits(
        self,
    ) -> None:
        """A phrasal verb is what it is, not the verb that opens it."""
        offsets = find_word_offsets("They give up.", frozenset({"give", "give up"}))

        assert offsets == ((5, 12),)

    def test_locates_a_form_closing_on_an_apostrophe(
        self,
    ) -> None:
        """A word boundary would fall on the wrong side of the apostrophe."""
        offsets = find_word_offsets("He is runnin' fast.", frozenset({"runnin'"}))

        assert offsets == ((6, 13),)

    def test_reads_a_form_as_text_rather_than_as_a_pattern(
        self,
    ) -> None:
        """A form holding what a regex would read is matched letter for letter."""
        offsets = find_word_offsets("It is 5 a.m. now.", frozenset({"a.m."}))

        assert offsets == ((8, 12),)

    def test_reads_one_lemma_after_another(
        self,
    ) -> None:
        """Only the last pattern is held on to, so forms must not outlive them."""
        _ = find_word_offsets("A bank.", frozenset({"bank"}))
        _ = find_word_offsets("He ran.", frozenset({"run", "ran"}))

        assert find_word_offsets("A bank.", frozenset({"bank"})) == ((2, 6),)
