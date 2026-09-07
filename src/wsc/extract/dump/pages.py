"""Walking the pages a Wiktionary dump holds, in the markup they were written in."""

from collections.abc import Iterator
from pathlib import Path
from typing import cast
from xml.etree import ElementTree

from tqdm import tqdm

from ...files import open_compressed

_ARTICLES = 0


def _name(
    tag: str,
) -> str:
    """
    Read an element's name without the namespace a dump declares.

    Args:
        tag: The tag as the parser reports it.

    Returns:
        The name alone.
    """
    return tag.rpartition("}")[2]


def read_pages(
    input_path: Path,
) -> Iterator[tuple[str, str]]:
    """
    Walk the articles of a dump, letting each page go once read.

    Args:
        input_path: The dump to read, compressed or not.

    Yields:
        The title and the markup of every page in the article namespace.
    """
    with open_compressed(input_path, "rb") as file:
        elements = cast(
            Iterator[tuple[str, ElementTree.Element]],
            ElementTree.iterparse(file, events=("end",)),
        )

        for _, element in tqdm(elements, desc=input_path.name, unit=" element"):
            if _name(element.tag) != "page":
                continue

            namespace = element.findtext("{*}ns", "").strip()

            title = element.findtext("{*}title", "").strip()
            markup = element.findtext("{*}revision/{*}text", "")

            element.clear()

            if namespace == str(_ARTICLES) and title and markup:
                yield title, markup
