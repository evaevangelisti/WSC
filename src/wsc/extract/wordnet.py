"""
Extraction of synsets from a wordnet published in WN-LMF.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import cast
from xml.etree import ElementTree

from tqdm import tqdm

from ..files import open_compressed
from ..models import POS, Synset

# WordNet's own codes. A satellite adjective is an adjective all the same.
_POS_BY_CODE = {
    "n": POS.NOUN,
    "v": POS.VERB,
    "a": POS.ADJECTIVE,
    "s": POS.ADJECTIVE,
    "r": POS.ADVERB,
}


class WordNetExtractor:
    """
    Reads synsets out of a wordnet published in WN-LMF.

    The filter is settled once, on the extractor, the way the Wiktionary side
    settles its own.
    """

    def __init__(
        self,
        allowed_pos: frozenset[POS] | None,
    ) -> None:
        """
        Set the filter every extraction will answer to.

        Args:
            allowed_pos: Parts of speech to keep, or None for every one.
        """
        self._allowed_pos: frozenset[POS] | None = allowed_pos

    def _parse_synset(
        self,
        element: ElementTree.Element,
        written_forms: dict[str, str],
    ) -> Synset | None:
        """
        Read one synset off the element holding it.

        Args:
            element: The Synset element, still holding its children.
            written_forms: The form each lexical entry was written in.

        Returns:
            The synset, or None if its part of speech is not one we keep.
        """
        pos = _POS_BY_CODE.get(element.get("partOfSpeech", ""))
        if pos is None:
            return None

        if self._allowed_pos is not None and pos not in self._allowed_pos:
            return None

        return Synset(
            element.get("id", ""),
            element.get("ili", ""),
            pos,
            (element.findtext("Definition") or "").strip(),
            tuple(
                written_forms[member] for member in element.get("members", "").split()
            ),
            tuple(
                relation.get("target", "")
                for relation in element.findall("SynsetRelation")
                if relation.get("relType") == "hypernym"
            ),
            tuple(
                (example.text or "").strip() for example in element.findall("Example")
            ),
        )

    def extract(
        self,
        input_path: Path,
    ) -> Iterator[Synset]:
        """
        Read synsets from a WN-LMF file, one at a time.

        Lexical entries are listed before the synsets naming them as members,
        so one pass is enough. Only the outer elements are cleared, since
        clearing a child would empty it before its parent is read.

        Args:
            input_path: The wordnet to read, compressed or not.

        Yields:
            One synset per meaning of a part of speech we keep.
        """
        written_forms: dict[str, str] = {}

        with open_compressed(input_path, "rb") as file:
            # iterparse is typed loosely; an end event carries the element
            # that ended.
            elements = cast(
                Iterator[tuple[str, ElementTree.Element]],
                ElementTree.iterparse(file, events=("end",)),
            )

            for _, element in tqdm(elements, desc=input_path.name, unit=" element"):
                if element.tag == "LexicalEntry":
                    # WN-LMF gives every entry a lemma, and gives it one only.
                    lemma = cast(ElementTree.Element, element.find("Lemma"))

                    written_forms[element.get("id", "")] = lemma.get("writtenForm", "")

                    element.clear()

                elif element.tag == "Synset":
                    synset = self._parse_synset(element, written_forms)
                    element.clear()

                    if synset is not None:
                        yield synset
