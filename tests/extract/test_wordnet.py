"""
Tests for src/wsc/extract/wordnet.py.
"""

import gzip
import string
from collections.abc import Callable, Iterable
from pathlib import Path

import pytest
from documents import lexical_entry, lexicon, synset
from hypothesis import given
from hypothesis import strategies as st
from strategies import parts_of_speech

from wsc.extract import WordNetExtractor
from wsc.models import POS, Synset

# WordNet's own codes. A satellite adjective is an adjective all the same.
_POS_CODES = st.sampled_from(["n", "v", "a", "s", "r"])

# A code some other wordnet writes, which this one does not.
_UNKNOWN_POS_CODES = st.text(alphabet=string.ascii_lowercase, max_size=3).filter(
    lambda code: code not in ("n", "v", "a", "s", "r")
)

# What names an entry, a synset or the meaning behind one, spelled the way a
# release spells it.
_IDENTIFIERS = st.text(
    alphabet=f"{string.ascii_lowercase}{string.digits}-_",
    min_size=1,
    max_size=12,
)

# XML holds no control character, and normalises the ends of the lines it
# does hold, so a definition carrying one is not a definition WordNet wrote.
_TEXTS = st.text(
    alphabet=st.characters(
        codec="utf-8",
        exclude_categories=("Cc", "Cs", "Zl", "Zp"),
    ),
    max_size=40,
)

# WN-LMF writes a score of relations, and an alignment reads the one.
_RELATION_TYPES = st.sampled_from(
    ["hypernym", "hyponym", "mero_part", "similar", "also"]
)

# A synset stands on its own unless a test spells out the entries naming it.
_NO_MEMBERS: st.SearchStrategy[tuple[str, ...]] = st.just(())


@st.composite
def _synsets(
    draw: st.DrawFn,
    pos_codes: st.SearchStrategy[str] = _POS_CODES,
    members: st.SearchStrategy[tuple[str, ...]] = _NO_MEMBERS,
) -> str:
    """
    Draw one synset, written as a release writes it.

    Args:
        draw: Turns a strategy into one of its values.
        pos_codes: WordNet's codes for a part of speech.
        members: The lexical entries expressing it, by identifier.

    Returns:
        The element, to be placed after the entries it names.
    """
    return synset(
        identifier=draw(_IDENTIFIERS),
        pos=draw(pos_codes),
        members=draw(members),
        definition=draw(_TEXTS),
        ili=draw(_IDENTIFIERS),
        examples=draw(st.lists(_TEXTS, max_size=3)),
        relations=draw(
            st.lists(st.tuples(_RELATION_TYPES, _IDENTIFIERS), max_size=3),
        ),
    )


@pytest.fixture
def extract(
    workspace: Callable[[], Path],
) -> Callable[..., list[Synset]]:
    """
    Run an extraction over a wordnet a property drew.

    Args:
        workspace: Sets aside a directory for the file being read.

    Returns:
        A runner taking the elements and the filter, and handing back the
        synsets that came through.
    """

    def run(
        elements: Iterable[str] = (),
        allowed_pos: frozenset[POS] | None = None,
        name: str = "wordnet.xml",
    ) -> list[Synset]:
        path = workspace() / name
        text = lexicon(*elements)

        if path.suffix == ".gz":
            _ = path.write_bytes(gzip.compress(text.encode()))
        else:
            _ = path.write_text(text, encoding="utf-8")

        return list(WordNetExtractor(allowed_pos).extract(path))

    return run


class TestOpening:
    """
    Reading the file however it was compressed.
    """

    @given(st.lists(_synsets(), max_size=3))
    def test_reads_the_same_wordnet_whatever_the_suffix_names(
        self,
        extract: Callable[..., list[Synset]],
        elements: list[str],
    ) -> None:
        """The release is gzipped, but a file found elsewhere may be plain."""
        assert extract(elements, name="wordnet.xml.gz") == extract(
            elements,
            name="wordnet.xml",
        )


class TestSynsets:
    """
    What a synset carries over.
    """

    def test_reads_what_the_alignment_will_need(
        self,
        extract: Callable[..., list[Synset]],
    ) -> None:
        """A synset is a meaning, the words expressing it, and what it hangs under."""
        elements = [
            lexical_entry(),
            synset(
                examples=["He went to the bank."],
                relations=[("hypernym", "oewn-08419984-n")],
            ),
        ]

        assert extract(elements) == [
            Synset(
                "oewn-08420278-n",
                "i54321",
                POS.NOUN,
                "a financial institution.",
                ("bank",),
                ("oewn-08419984-n",),
                ("He went to the bank.",),
            )
        ]

    @given(st.data())
    def test_spells_out_the_members_of_a_synset(
        self,
        extract: Callable[..., list[Synset]],
        data: st.DataObject,
    ) -> None:
        """A synset names its members by entry, and an entry holds the word."""
        written_forms = data.draw(
            st.dictionaries(_IDENTIFIERS, _TEXTS, min_size=1, max_size=4)
        )
        members = data.draw(
            st.lists(st.sampled_from(sorted(written_forms)), max_size=4)
        )

        elements = [
            *(
                lexical_entry(identifier, written_form)
                for identifier, written_form in written_forms.items()
            ),
            data.draw(_synsets(members=st.just(tuple(members)))),
        ]

        assert extract(elements)[0].members == tuple(
            written_forms[member] for member in members
        )

    @given(_TEXTS)
    def test_strips_a_definition(
        self,
        extract: Callable[..., list[Synset]],
        definition: str,
    ) -> None:
        """The element is written across lines, and the whitespace is not the gloss."""
        elements = [synset(members=(), definition=f"\n  {definition}\n  ")]

        assert extract(elements)[0].definition == definition.strip()


class TestExamples:
    """
    The sentences a synset is given, which read alongside its definition.
    """

    @given(st.lists(_TEXTS, max_size=4))
    def test_keeps_them_in_the_order_they_were_written(
        self,
        extract: Callable[..., list[Synset]],
        examples: list[str],
    ) -> None:
        """The order is WordNet's, and nothing here has a reason to better it."""
        elements = [synset(members=(), examples=examples)]

        assert extract(elements)[0].examples == tuple(
            example.strip() for example in examples
        )


class TestHypernyms:
    """
    What a synset hangs under, which is how far apart two of them are.
    """

    @given(st.lists(st.tuples(_RELATION_TYPES, _IDENTIFIERS), max_size=5))
    def test_keeps_every_synset_it_is_a_kind_of_and_nothing_else(
        self,
        extract: Callable[..., list[Synset]],
        relations: list[tuple[str, str]],
    ) -> None:
        """A meaning may sit under more than one, WordNet being a lattice."""
        elements = [synset(members=(), relations=relations)]

        assert extract(elements)[0].hypernyms == tuple(
            target for rel_type, target in relations if rel_type == "hypernym"
        )


class TestPartsOfSpeech:
    """
    Reading WordNet's codes onto the ones the collector keeps.
    """

    @pytest.mark.parametrize(
        ("code", "expected"),
        [
            ("n", POS.NOUN),
            ("v", POS.VERB),
            ("a", POS.ADJECTIVE),
            ("s", POS.ADJECTIVE),
            ("r", POS.ADVERB),
        ],
    )
    def test_converts_every_code_wordnet_writes(
        self,
        extract: Callable[..., list[Synset]],
        code: str,
        expected: POS,
    ) -> None:
        """A satellite adjective is an adjective all the same."""
        assert extract([synset(pos=code, members=())])[0].pos is expected

    @given(st.lists(_synsets(), max_size=4), st.data())
    def test_keeps_only_the_parts_of_speech_asked_for(
        self,
        extract: Callable[..., list[Synset]],
        elements: list[str],
        data: st.DataObject,
    ) -> None:
        """The filter narrows the candidates the way it narrows the senses."""
        allowed = data.draw(st.sets(parts_of_speech, min_size=1))

        assert extract(elements, allowed_pos=frozenset(allowed)) == [
            found for found in extract(elements) if found.pos in allowed
        ]

    @given(st.lists(_synsets(), max_size=4), _synsets(pos_codes=_UNKNOWN_POS_CODES))
    def test_reads_past_a_code_it_does_not_know(
        self,
        extract: Callable[..., list[Synset]],
        elements: list[str],
        unknown: str,
    ) -> None:
        """A wordnet of another language may cut its parts of speech elsewhere."""
        assert extract([*elements, unknown]) == extract(elements)
