"""Tests for src/wsc/extract/wordnet.py."""

import gzip
import string
from collections.abc import Callable, Iterable
from pathlib import Path
from xml.etree import ElementTree

import pytest
from documents import lexical_entry, lexicon, synset
from hypothesis import given
from hypothesis import strategies as st
from strategies import parts_of_speech

from wsc.export.formats.jsonl import JSONLWriter
from wsc.extract import WordNetExtractor
from wsc.models import POS, Synset
from wsc.reading import read_synsets

# WordNet satellite adjectives share the adjective part of speech.
_POS_CODES = st.sampled_from(["n", "v", "a", "s", "r"])

_UNKNOWN_POS_CODES = st.text(alphabet=string.ascii_lowercase, max_size=3).filter(
    lambda code: code not in ("n", "v", "a", "s", "r"),
)

_IDENTIFIERS = st.text(
    alphabet=f"{string.ascii_lowercase}{string.digits}-_",
    min_size=1,
    max_size=12,
)

# XML excludes control characters and normalizes line endings.
_TEXTS = st.text(
    alphabet=st.characters(
        codec="utf-8",
        exclude_categories=("Cc", "Cs", "Zl", "Zp"),
    ),
    max_size=40,
)

_DEFINITIONS = _TEXTS.filter(lambda text: bool(text.strip()))

_RELATION_TYPES = st.sampled_from(
    ["hypernym", "hyponym", "mero_part", "similar", "also"],
)

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
        definition=draw(_DEFINITIONS),
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
        A runner extracting synsets under the supplied filter.
    """

    def run(
        elements: Iterable[str] = (),
        allowed_pos: frozenset[POS] | None = None,
        name: str = "wordnet.xml",
    ) -> list[Synset]:
        """
        Extract synsets from a generated WordNet document.

        Args:
            elements: XML elements included in the generated lexicon.
            allowed_pos: Parts of speech to retain, or None for all supported ones.
            name: Source filename selecting the compression format.

        Returns:
            Synsets retained by the configured filters.
        """
        path = workspace() / name
        text = lexicon(*elements)

        if path.suffix == ".gz":
            _ = path.write_bytes(gzip.compress(text.encode()))
        else:
            _ = path.write_text(text, encoding="utf-8")

        return list(WordNetExtractor(allowed_pos).extract(path))

    return run


class TestOpening:
    """Reading the file however it was compressed."""

    @given(st.lists(_synsets(), max_size=3))
    def test_reads_compressed_wordnet(
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
    """What a synset carries over."""

    def test_preserves_synset_fields(
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
            ),
        ]

    @given(st.data())
    def test_resolves_synset_members(
        self,
        extract: Callable[..., list[Synset]],
        data: st.DataObject,
    ) -> None:
        """A synset names its members by entry, and an entry holds the word."""
        written_forms = data.draw(
            st.dictionaries(_IDENTIFIERS, _TEXTS, min_size=1, max_size=4),
        )
        members = data.draw(
            st.lists(st.sampled_from(sorted(written_forms)), max_size=4),
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

    @given(_DEFINITIONS)
    def test_strips_definition(
        self,
        extract: Callable[..., list[Synset]],
        definition: str,
    ) -> None:
        """The element is written across lines, and the whitespace is not the gloss."""
        elements = [synset(members=(), definition=f"\n  {definition}\n  ")]

        assert extract(elements)[0].definition == definition.strip()

    @given(
        st.dictionaries(
            st.sampled_from(["id", "ili", "Definition"]),
            st.none() | st.text(alphabet=" \t\n\r", max_size=4),
            min_size=1,
            max_size=3,
        ),
    )
    def test_discards_incomplete_synsets(
        self,
        extract: Callable[..., list[Synset]],
        workspace: Callable[[], Path],
        missing_fields: dict[str, str | None],
    ) -> None:
        """Incomplete synsets are skipped while retained records remain readable."""
        incomplete = ElementTree.fromstring(synset(members=()))

        for field, value in missing_fields.items():
            if field == "Definition":
                definition = incomplete.find("Definition")

                assert definition is not None

                if value is None:
                    incomplete.remove(definition)
                else:
                    definition.text = value
            elif value is None:
                del incomplete.attrib[field]
            else:
                incomplete.set(field, value)

        records = extract(
            [
                synset(identifier="first", members=()),
                ElementTree.tostring(incomplete, encoding="unicode"),
                synset(identifier="last", members=()),
            ],
        )

        assert [record.id for record in records] == ["first", "last"]

        output_path = workspace() / "synsets.jsonl"

        with JSONLWriter[Synset](output_path) as writer:
            for record in records:
                writer.write(record)

        assert list(read_synsets(output_path)) == records


class TestExamples:
    """The sentences a synset is given, which read alongside its definition."""

    @given(st.lists(_TEXTS, max_size=4))
    def test_preserves_example_order(
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
    """What a synset hangs under, which is how far apart two of them are."""

    @given(st.lists(st.tuples(_RELATION_TYPES, _IDENTIFIERS), max_size=5))
    def test_selects_hypernym_relations(
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
    """Reading WordNet's codes onto the ones the collector keeps."""

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
    def test_maps_wordnet_categories(
        self,
        extract: Callable[..., list[Synset]],
        code: str,
        expected: POS,
    ) -> None:
        """A satellite adjective is an adjective all the same."""
        assert extract([synset(pos=code, members=())])[0].pos is expected

    @given(st.lists(_synsets(), max_size=4), st.data())
    def test_filters_selected_categories(
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
    def test_skips_unknown_categories(
        self,
        extract: Callable[..., list[Synset]],
        elements: list[str],
        unknown: str,
    ) -> None:
        """A wordnet of another language may cut its parts of speech elsewhere."""
        assert extract([*elements, unknown]) == extract(elements)
