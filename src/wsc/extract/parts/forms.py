"""
The shapes an occurrence of a headword may take.
"""

from ..schema import RawForm

# Rows describing an inflection table rather than the lemma, and the
# transliterations standing beside a form rather than for it.
_SERVICE_TAGS = frozenset({"inflection-template", "romanization", "table-tags"})

# What an inflection table writes for a cell it leaves empty.
_EMPTY_CELL = "-"


def parse_forms(
    raw_forms: list[RawForm],
    lemma: str,
) -> frozenset[str]:
    """
    Collect the shapes an occurrence of the lemma may take.

    Args:
        raw_forms: What wiktextract listed under the entry.
        lemma: The headword, which is a form of itself.

    Returns:
        The headword and every inflection worth looking for.
    """
    forms = {lemma}

    for raw_form in raw_forms:
        form = raw_form.get("form", "").strip()
        if not form or form == _EMPTY_CELL:
            continue

        if not _SERVICE_TAGS.isdisjoint(raw_form.get("tags", [])):
            continue

        forms.add(form)

    return frozenset(forms)
