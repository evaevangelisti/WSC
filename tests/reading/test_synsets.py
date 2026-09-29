"""
Exercise JSON Lines input for sourced synsets.
"""

import json
from pathlib import Path

import pytest

from wsc.models import POS, SynsetResource
from wsc.reading import read_synsets


def test_reads_source_evidence_and_generated_ids(
    tmp_path: Path,
) -> None:
    """
    Each resource keeps its own members, glosses, and examples.

    Args:
        tmp_path: Isolated directory for the source file.
    """
    path = tmp_path / "synsets.jsonl"
    _ = path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "pos": "noun",
                        "sources": {
                            "source-a": {
                                "members": ["word"],
                                "glosses": [
                                    "A lexical concept.",
                                    "Another description.",
                                ],
                                "examples": ["An example."],
                            },
                            "source-b": {
                                "members": ["term"],
                                "glosses": ["A second description."],
                            },
                        },
                    },
                ),
                json.dumps(
                    {
                        "id": "provided",
                        "pos": "verb",
                        "sources": {
                            "lexicon": {
                                "members": ["act", "perform"],
                                "glosses": ["Perform an action."],
                            },
                        },
                    },
                ),
            ],
        )
        + "\n",
        encoding="utf-8",
    )

    first, second = tuple(read_synsets(path))

    assert first.id == "synset-00000001"
    assert first.pos == POS.NOUN
    assert first.resources == {
        "source-a": SynsetResource(
            ("word",),
            ("A lexical concept.", "Another description."),
            ("An example.",),
        ),
        "source-b": SynsetResource(("term",), ("A second description.",)),
    }
    assert second.id == "provided"
    assert second.resources == {
        "lexicon": SynsetResource(("act", "perform"), ("Perform an action.",)),
    }


@pytest.mark.parametrize(
    "sources",
    [
        {},
        {"source": {"members": [], "glosses": ["A meaning."]}},
        {"source": {"members": ["word"], "glosses": []}},
    ],
)
def test_rejects_missing_source_evidence(
    tmp_path: Path,
    sources: dict[str, dict[str, list[str]]],
) -> None:
    """
    Every declared resource must identify and describe its concept.

    Args:
        tmp_path: Isolated directory for the source file.
        sources: Resource mapping missing essential lexical evidence.
    """
    path = tmp_path / "synsets.jsonl"
    _ = path.write_text(json.dumps({"pos": "noun", "sources": sources}) + "\n")

    with pytest.raises(ValueError, match="needs members and glosses"):
        _ = tuple(read_synsets(path))


@pytest.mark.parametrize("source", ["", "remaining"])
def test_rejects_unnamed_or_reserved_resources(
    tmp_path: Path,
    source: str,
) -> None:
    """
    Source names must identify resources without colliding with pass labels.

    Args:
        tmp_path: Isolated directory for the source file.
        source: Empty or reserved resource name.
    """
    path = tmp_path / "synsets.jsonl"
    _ = path.write_text(
        json.dumps(
            {
                "pos": "noun",
                "sources": {
                    source: {"members": ["word"], "glosses": ["A meaning."]},
                },
            },
        )
        + "\n",
    )

    with pytest.raises(ValueError, match="invalid source names"):
        _ = tuple(read_synsets(path))
