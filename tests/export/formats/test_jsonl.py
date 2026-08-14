"""
Tests for src/wsc/export/formats/jsonl.py.

What was written is compared whole rather than reached into, so a key that
appears where none was expected is caught along with one that went missing.
"""

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from wsc.export.formats.jsonl import JsonlWriter
from wsc.models import POS, Example, Lemma, Quotation, Sense


@pytest.fixture
def written(
    tmp_path: Path,
) -> Callable[..., list[object]]:
    """
    Write lemmas and read back what landed on disk.

    Args:
        tmp_path: The directory pytest set aside for this test.

    Returns:
        A runner handing back one decoded object per line written.
    """

    def run(
        *lemmas: Lemma,
    ) -> list[object]:
        output_path = tmp_path / "senses.jsonl"

        with JsonlWriter(output_path) as writer:
            for lemma in lemmas:
                writer.write(lemma)

        return [
            json.loads(line)
            for line in output_path.read_text(encoding="utf-8").splitlines()
        ]

    return run


def sense(
    *sentences: Example | Quotation,
) -> Sense:
    """
    Build the plainest sense there could be, carrying the sentences given.

    Args:
        sentences: What illustrates it.

    Returns:
        A sense with one gloss and no labels.
    """
    return Sense("bank.noun.1.01", ("A meaning.",), sentences=list(sentences))


def lemma_of(
    *senses: Sense,
) -> Lemma:
    """
    Build a lemma around senses a test spells out.

    Args:
        senses: Its meanings.

    Returns:
        The lemma to write.
    """
    return Lemma("bank.noun.1", "bank", POS.NOUN, list(senses))


class TestJsonlWriter:
    """
    One JSON object per line.
    """

    def test_writes_one_line_per_lemma(
        self,
        written: Callable[..., list[object]],
    ) -> None:
        """A line at a time is what lets a reader stream the file back."""
        assert len(written(lemma_of(sense()), lemma_of(sense()))) == 2

    def test_writes_the_whole_lemma(
        self,
        written: Callable[..., list[object]],
    ) -> None:
        """Nothing the extractor gathered is dropped on the way out."""
        lemma = Lemma(
            "bank.noun.1",
            "bank",
            POS.NOUN,
            [
                Sense(
                    "bank.noun.1.01",
                    ("A financial institution.",),
                    ("business",),
                    ("countable",),
                    [
                        Example("He went to the bank.", word_offsets=((15, 19),)),
                        Quotation(
                            "A bank stood there.",
                            "1999, A Book",
                            1999,
                            word_offsets=((2, 6),),
                        ),
                    ],
                    ("oewn-08420278-n",),
                )
            ],
        )

        assert written(lemma) == [
            {
                "id": "bank.noun.1",
                "lemma": "bank",
                "pos": "noun",
                "senses": [
                    {
                        "id": "bank.noun.1.01",
                        "glosses": ["A financial institution."],
                        "topics": ["business"],
                        "tags": ["countable"],
                        "sentences": [
                            {
                                "text": "He went to the bank.",
                                "word_offsets": [[15, 19]],
                            },
                            {
                                "text": "A bank stood there.",
                                "word_offsets": [[2, 6]],
                                "reference": "1999, A Book",
                                "year": 1999,
                            },
                        ],
                        "synset_ids": ["oewn-08420278-n"],
                    }
                ],
            }
        ]

    def test_drops_what_holds_nothing(
        self,
        written: Callable[..., list[object]],
    ) -> None:
        """A key holding nothing is left out, an empty table and an empty list alike."""
        assert written(lemma_of(sense())) == [
            {
                "id": "bank.noun.1",
                "lemma": "bank",
                "pos": "noun",
                "senses": [{"id": "bank.noun.1.01", "glosses": ["A meaning."]}],
            }
        ]

    def test_an_example_reads_back_as_one(
        self,
        written: Callable[..., list[object]],
    ) -> None:
        """An example carries its text alone, so it reads back as an example."""
        lemma = lemma_of(sense(Example("He ran.")))

        assert written(lemma) == [
            {
                "id": "bank.noun.1",
                "lemma": "bank",
                "pos": "noun",
                "senses": [
                    {
                        "id": "bank.noun.1.01",
                        "glosses": ["A meaning."],
                        "sentences": [{"text": "He ran."}],
                    }
                ],
            }
        ]

    def test_keeps_an_undated_quotation_a_quotation(
        self,
        written: Callable[..., list[object]],
    ) -> None:
        """
        A year that could not be read is dropped, but the reference is not.

        The reference alone is what tells the two kinds apart on the way back.
        """
        lemma = lemma_of(sense(Quotation("He ran.", "A Book", None)))

        assert written(lemma) == [
            {
                "id": "bank.noun.1",
                "lemma": "bank",
                "pos": "noun",
                "senses": [
                    {
                        "id": "bank.noun.1.01",
                        "glosses": ["A meaning."],
                        "sentences": [{"text": "He ran.", "reference": "A Book"}],
                    }
                ],
            }
        ]

    def test_writes_text_as_it_stands(
        self,
        tmp_path: Path,
    ) -> None:
        """Text is written as it stands, rather than escaped."""
        output_path = tmp_path / "senses.jsonl"

        with JsonlWriter(output_path) as writer:
            writer.write(Lemma("città.noun.1", "città", POS.NOUN))

        assert "città" in output_path.read_text(encoding="utf-8")

    def test_refuses_to_write_before_it_is_entered(
        self,
        tmp_path: Path,
    ) -> None:
        """The file is opened on entry, so there is nowhere to write before it."""
        writer = JsonlWriter(tmp_path / "senses.jsonl")

        with pytest.raises(RuntimeError, match="context manager"):
            writer.write(lemma_of(sense()))

    def test_refuses_to_write_once_the_block_is_left(
        self,
        tmp_path: Path,
    ) -> None:
        """Closing lets go of the file, rather than leaving a closed one behind."""
        writer = JsonlWriter(tmp_path / "senses.jsonl")

        with writer:
            writer.write(lemma_of(sense()))

        with pytest.raises(RuntimeError, match="context manager"):
            writer.write(lemma_of(sense()))
