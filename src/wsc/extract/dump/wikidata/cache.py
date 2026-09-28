"""
Persist Wikidata IDs associated with Wiktionary senses.
"""

import json
from pathlib import Path
from typing import cast

from ....files import open_compressed

type WikidataIds = dict[str, tuple[str, ...]]
"""
Explicit Wikidata identifiers grouped by Wiktionary sense identifier.
"""


def write_wikidata_ids(
    output_path: Path,
    identifiers: WikidataIds,
) -> None:
    """
    Write the explicit identifiers for later collection.

    Args:
        output_path: Cache file to create.
        identifiers: Wikidata items indexed by sense identifier.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = output_path.with_name(f"{output_path.name}.part")
    with open_compressed(temporary_path, "wt") as stream:
        _ = stream.write(
            json.dumps(dict(sorted(identifiers.items())), ensure_ascii=False)
        )

    _ = temporary_path.replace(output_path)


def read_wikidata_ids(
    input_path: Path,
) -> WikidataIds:
    """
    Read cached identifiers obtained from the dump.

    Args:
        input_path: Cache file created during parsing.

    Returns:
        Wikidata items indexed by sense identifier.
    """
    with open_compressed(input_path, "rt") as stream:
        contents = cast(dict[str, list[str]], json.load(stream))

    return {identifier: tuple(items) for identifier, items in contents.items()}
