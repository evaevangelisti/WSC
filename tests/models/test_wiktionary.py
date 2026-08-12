"""
Tests for src/wsc/models/wiktionary.py.
"""

import pytest

from wsc.models import POS, Example, Quotation, Sense


class TestPOS:
    """
    The parts of speech the collector keeps.
    """

    @pytest.mark.parametrize(
        ("code", "expected"),
        [
            ("noun", POS.NOUN),
            ("verb", POS.VERB),
            ("adj", POS.ADJECTIVE),
            ("adv", POS.ADVERB),
        ],
    )
    def test_converts_from_a_wiktextract_code(
        self,
        code: str,
        expected: POS,
    ) -> None:
        """The values are wiktextract's own, so its codes convert directly."""
        assert POS(code) is expected

    def test_refuses_a_part_of_speech_it_does_not_keep(
        self,
    ) -> None:
        """The refusal is what the extractor reads to skip an entry."""
        with pytest.raises(ValueError, match="not a valid POS"):
            _ = POS("intj")


class TestAttestation:
    """
    The two kinds of sentence a sense carries.
    """

    def test_an_example_is_not_a_quotation(
        self,
    ) -> None:
        """The kinds are told apart by class, so neither passes for the other."""
        assert Example("A sentence.") != Quotation("A sentence.", "1999, A Book")

    def test_a_quotation_may_name_no_year(
        self,
    ) -> None:
        """A reference the year could not be read off is still a quotation."""
        assert Quotation("A sentence.", "A Book").year is None


class TestSense:
    """
    One meaning, read off its gloss chain.
    """

    def test_gloss_is_the_innermost_of_the_chain(
        self,
    ) -> None:
        """A sub-sense means what its own gloss says, not what its parent does."""
        sense = Sense("bank.noun.1.01", ("A financial institution.", "Its building."))

        assert sense.gloss == "Its building."

    def test_definition_joins_the_chain_outermost_first(
        self,
    ) -> None:
        """A sub-sense stands alone only once its parents are read into it."""
        sense = Sense("bank.noun.1.01", ("A financial institution.", "Its building."))

        assert sense.definition == "A financial institution. Its building."

    @pytest.mark.parametrize(
        ("glosses", "expected"),
        [
            (("A financial institution.",), 1),
            (("A financial institution.", "Its building."), 2),
        ],
    )
    def test_depth_counts_the_chain(
        self,
        glosses: tuple[str, ...],
        expected: int,
    ) -> None:
        """Nesting is what the chain records, so its length is the level."""
        assert Sense("bank.noun.1.01", glosses).depth == expected
