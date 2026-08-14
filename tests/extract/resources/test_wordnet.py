"""
Tests for src/wsc/extract/resources/wordnet.py.
"""

import gzip
from collections.abc import Callable, Iterable
from pathlib import Path

import pytest

from wsc.extract import WordNetExtractor
from wsc.models import POS, Synset

# The slice of WN-LMF the extractor reads, laid out as the release lays it.

_WORDNET = """<?xml version="1.0" encoding="UTF-8"?>
<LexicalResource>
  <Lexicon id="oewn" language="en" version="2025">{body}
  </Lexicon>
</LexicalResource>
"""

_ENTRY = """
    <LexicalEntry id="{id}">
      <Lemma writtenForm="{written_form}" partOfSpeech="{pos}"/>
    </LexicalEntry>"""

_SYNSET = """
    <Synset id="{id}" ili="{ili}" partOfSpeech="{pos}" members="{members}">
      <Definition>{definition}</Definition>{body}
    </Synset>"""

_EXAMPLE = """
      <Example>{text}</Example>"""

_RELATION = """
      <SynsetRelation relType="{rel_type}" target="{target}"/>"""


def entry(
    identifier: str = "oewn-bank-n",
    written_form: str = "bank",
    pos: str = "n",
) -> str:
    """
    Write one lexical entry, which is where a written form is spelled out.

    Args:
        identifier: What the synsets naming it as a member will refer to.
        written_form: The word itself.
        pos: WordNet's code for its part of speech.

    Returns:
        The element, to be placed before the synsets.
    """
    return _ENTRY.format(id=identifier, written_form=written_form, pos=pos)


def synset(
    identifier: str = "oewn-08420278-n",
    pos: str = "n",
    members: str = "oewn-bank-n",
    definition: str = "a financial institution.",
    ili: str = "i54321",
    examples: Iterable[str] = (),
    relations: Iterable[tuple[str, str]] = (),
) -> str:
    """
    Write one synset.

    Args:
        identifier: What an alignment will record.
        pos: WordNet's code for its part of speech.
        members: The lexical entries expressing it, separated by spaces.
        definition: The gloss WordNet writes for it.
        ili: The interlingual index naming the same meaning elsewhere.
        examples: The sentences to hang off it.
        relations: The relations to hang off it, each a type and a target.

    Returns:
        The element, to be placed after the entries.
    """
    body = "".join(
        [
            *(
                _RELATION.format(rel_type=rel_type, target=target)
                for rel_type, target in relations
            ),
            *(_EXAMPLE.format(text=text) for text in examples),
        ]
    )

    return _SYNSET.format(
        id=identifier,
        ili=ili,
        pos=pos,
        members=members,
        definition=definition,
        body=body,
    )


@pytest.fixture
def extract(
    tmp_path: Path,
) -> Callable[..., list[Synset]]:
    """
    Run an extraction over a wordnet a test spells out.

    Args:
        tmp_path: The directory pytest set aside for this test.

    Returns:
        A runner taking the elements and the filter, and handing back the
        synsets that came through.
    """

    def run(
        elements: Iterable[str] = (),
        allowed_pos: frozenset[POS] | None = None,
        name: str = "wordnet.xml",
    ) -> list[Synset]:
        path = tmp_path / name
        text = _WORDNET.format(body="".join(elements))

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

    @pytest.mark.parametrize("name", ["wordnet.xml", "wordnet.xml.gz"])
    def test_reads_what_the_suffix_names(
        self,
        extract: Callable[..., list[Synset]],
        name: str,
    ) -> None:
        """The release is gzipped, but a file found elsewhere may be plain."""
        assert extract([entry(), synset()], name=name)


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
            entry(),
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

    def test_spells_out_the_members_of_a_synset(
        self,
        extract: Callable[..., list[Synset]],
    ) -> None:
        """A synset names its members by entry, and an entry holds the word."""
        elements = [
            entry("oewn-bank-n", "bank"),
            entry("oewn-savings_bank-n", "savings bank"),
            synset(members="oewn-bank-n oewn-savings_bank-n"),
        ]

        assert extract(elements)[0].members == ("bank", "savings bank")

    def test_a_synset_may_have_no_members(
        self,
        extract: Callable[..., list[Synset]],
    ) -> None:
        """Members are read off an attribute, which need not be there."""
        assert extract([synset(members="")])[0].members == ()

    def test_strips_a_definition(
        self,
        extract: Callable[..., list[Synset]],
    ) -> None:
        """The element is written across lines, and the whitespace is not the gloss."""
        elements = [entry(), synset(definition="\n        a financial institution.\n")]

        assert extract(elements)[0].definition == "a financial institution."


class TestExamples:
    """
    The sentences a synset is given, which read alongside its definition.
    """

    def test_keeps_them_in_the_order_they_were_written(
        self,
        extract: Callable[..., list[Synset]],
    ) -> None:
        """The order is WordNet's, and nothing here has a reason to better it."""
        elements = [
            entry(),
            synset(examples=["He went to the bank.", "The bank closed."]),
        ]

        assert extract(elements)[0].examples == (
            "He went to the bank.",
            "The bank closed.",
        )

    def test_a_synset_may_have_none(
        self,
        extract: Callable[..., list[Synset]],
    ) -> None:
        """Under half of the synsets are illustrated at all."""
        assert extract([entry(), synset()])[0].examples == ()


class TestHypernyms:
    """
    What a synset hangs under, which is how far apart two of them are.
    """

    def test_keeps_every_synset_it_is_a_kind_of(
        self,
        extract: Callable[..., list[Synset]],
    ) -> None:
        """A meaning may sit under more than one, WordNet being a lattice."""
        elements = [
            entry(),
            synset(
                relations=[
                    ("hypernym", "oewn-08419984-n"),
                    ("hypernym", "oewn-08061042-n"),
                ]
            ),
        ]

        assert extract(elements)[0].hypernyms == (
            "oewn-08419984-n",
            "oewn-08061042-n",
        )

    def test_leaves_every_other_relation_alone(
        self,
        extract: Callable[..., list[Synset]],
    ) -> None:
        """WN-LMF writes a score of relations, and an alignment reads the one."""
        elements = [
            entry(),
            synset(
                relations=[
                    ("hyponym", "oewn-08420746-n"),
                    ("hypernym", "oewn-08419984-n"),
                    ("mero_part", "oewn-08420246-n"),
                ]
            ),
        ]

        assert extract(elements)[0].hypernyms == ("oewn-08419984-n",)

    def test_an_adjective_carries_none(
        self,
        extract: Callable[..., list[Synset]],
    ) -> None:
        """Nouns and verbs alone are arranged that way, so none is all there is."""
        elements = [
            entry("oewn-solvent-a", "solvent", "a"),
            synset("oewn-01234567-a", "a", members="oewn-solvent-a"),
        ]

        assert extract(elements)[0].hypernyms == ()


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
        assert extract([synset(pos=code, members="")])[0].pos is expected

    def test_skips_a_code_it_does_not_know(
        self,
        extract: Callable[..., list[Synset]],
    ) -> None:
        """A wordnet of another language may cut its parts of speech elsewhere."""
        assert extract([synset(pos="x")]) == []

    def test_keeps_only_the_parts_of_speech_asked_for(
        self,
        extract: Callable[..., list[Synset]],
    ) -> None:
        """The filter narrows the candidates the way it narrows the senses."""
        elements = [
            synset("oewn-08420278-n", "n", members=""),
            synset("oewn-02306462-v", "v", members=""),
        ]

        synsets = extract(elements, allowed_pos=frozenset({POS.VERB}))

        assert [found.id for found in synsets] == ["oewn-02306462-v"]
