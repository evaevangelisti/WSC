"""
How else a lemma is written, gathered from the entries stating so.
"""

from collections import defaultdict
from collections.abc import Iterable, Iterator

from ....models import POS
from ..schema import RawEntry, RawForm, RawSense

# Tags marking a sense that states a form rather than a meaning. The lemma it
# points at is the one the headword is a way of writing.
_FORM_TAGS = frozenset({"form-of", "alt-of"})

# What an entry tags the forms it lists under Alternative forms with.
_ALTERNATIVE = "alternative"

type Variants = dict[tuple[str, POS], frozenset[str]]
"""The other spellings of each headword, by headword and part of speech."""


def _targets(
    raw_sense: RawSense,
) -> Iterator[str]:
    """
    Read the lemmas one sense states its headword to be a form of.

    Args:
        raw_sense: What wiktextract listed under the entry.

    Yields:
        Each lemma pointed at, named and stripped.
    """
    for target in (*raw_sense.get("form_of", []), *raw_sense.get("alt_of", [])):
        word = target.get("word", "").strip()

        if word:
            yield word


def alternative_forms(
    raw_forms: list[RawForm],
    lemma: str,
) -> frozenset[str]:
    """
    Read the spellings an entry lists for itself under Alternative forms.

    Args:
        raw_forms: What wiktextract listed under the entry.
        lemma: The headword, which is no variant of itself.

    Returns:
        The other spellings the entry names.
    """
    return frozenset(
        form
        for raw_form in raw_forms
        if _ALTERNATIVE in raw_form.get("tags", [])
        and (form := raw_form.get("form", "").strip())
        and form != lemma
    )


def gather_variants(
    entries: Iterable[RawEntry],
    language: str,
) -> Variants:
    """
    Gather every headword that states itself to be a form of another.

    Wiktionary writes an inflection on a page of its own, pointing back at the
    lemma, so the two meet only once the whole file has been read.

    Args:
        entries: The wiktextract file, read whole.
        language: Wiktionary's code for the language to read.

    Returns:
        The other spellings of each headword, by headword and part of speech.
    """
    gathered_variants: defaultdict[tuple[str, POS], set[str]] = defaultdict(set)

    for entry in entries:
        if entry.get("lang_code") != language:
            continue

        variant = entry.get("word", "").strip()
        if not variant:
            continue

        try:
            pos = POS(entry.get("pos", ""))
        except ValueError:
            continue

        for raw_sense in entry.get("senses", []):
            if _FORM_TAGS.isdisjoint(raw_sense.get("tags", [])):
                continue

            for target in _targets(raw_sense):
                if target != variant:
                    gathered_variants[target, pos].add(variant)

    return {
        target: frozenset(variants) for target, variants in gathered_variants.items()
    }
