"""
Tests for src/wsc/models/resources/wiktionary.py.

A sense reading its own gloss chain is stated here rather than downstream:
nothing else in the collector asks a sense what it means, so nothing else
would notice were it to answer wrongly.
"""

from hypothesis import given
from hypothesis import strategies as st
from strategies import glosses, identifiers

from wsc.models import Sense

_CHAINS = st.lists(glosses, min_size=1, max_size=4)


class TestSense:
    """
    One meaning, read off its gloss chain.
    """

    @given(identifiers, _CHAINS)
    def test_closes_on_its_own_gloss(
        self,
        identifier: str,
        chain: list[str],
    ) -> None:
        """A sub-sense means what its own gloss says, not what its parent does."""
        sense = Sense(identifier, tuple(chain))

        assert sense.gloss == chain[-1]
        assert sense.definition.endswith(sense.gloss)

    @given(identifiers, _CHAINS)
    def test_a_definition_holds_the_whole_chain(
        self,
        identifier: str,
        chain: list[str],
    ) -> None:
        """A sub-sense stands alone only once its parents are read into it."""
        definition = Sense(identifier, tuple(chain)).definition

        assert all(gloss in definition for gloss in chain)

    @given(identifiers, glosses)
    def test_a_chain_of_one_is_its_own_definition(
        self,
        identifier: str,
        gloss: str,
    ) -> None:
        """A top-level sense has no parent to be read into it."""
        sense = Sense(identifier, (gloss,))

        assert sense.definition == sense.gloss

    @given(identifiers, _CHAINS, glosses)
    def test_nesting_one_deeper_counts_one_more(
        self,
        identifier: str,
        chain: list[str],
        gloss: str,
    ) -> None:
        """Nesting is what the chain records, so its length is the level."""
        sense = Sense(identifier, tuple(chain))
        nested = Sense(identifier, (*chain, gloss))

        assert sense.depth == len(chain)
        assert nested.depth == sense.depth + 1
