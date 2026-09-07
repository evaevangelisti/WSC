"""Walking the entries of a wiktextract file."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import cast

from tqdm import tqdm

from ...files import open_compressed
from .schema import RawEntry


def read_entries(
    input_path: Path,
    description: str,
) -> Iterator[RawEntry]:
    """
    Walk the entries of a wiktextract file, passing over what is not one.

    Args:
        input_path: The wiktextract file to read, compressed or not.
        description: What the progress bar says the pass is doing.

    Yields:
        One entry per line that holds a whole one.
    """
    with open_compressed(input_path, "rt") as file:
        for line in tqdm(file, desc=description, unit=" entry"):
            if not line.startswith("{"):
                continue

            try:
                yield cast(RawEntry, json.loads(line))
            except json.JSONDecodeError:
                continue
