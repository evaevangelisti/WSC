"""Exercise JSON Lines input for generic synsets."""

import json
from pathlib import Path

from wsc.models import POS, SynsetMember
from wsc.reading import read_synsets


def test_reads_synsets_with_generated_ids_and_member_sources(
    tmp_path: Path,
) -> None:
    """Absent identifiers and member sources survive input normalization.

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
                        "members": {"source-a": ["word"], "source-b": ["term"]},
                        "glosses": ["A lexical concept."],
                        "examples": ["An example."],
                    },
                ),
                json.dumps(
                    {
                        "id": "provided",
                        "pos": "verb",
                        "members": ["act"],
                        "glosses": ["Perform an action."],
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
    assert first.members == (
        SynsetMember("word", "source-a"),
        SynsetMember("term", "source-b"),
    )
    assert first.examples == ("An example.",)
    assert second.id == "provided"
