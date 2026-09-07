"""
The slice of the wiktextract schema the collector reads.

Every key is optional: this describes someone else's JSON, and an entry carries only
what the page it was made from happened to say.
"""

from collections.abc import Mapping
from typing import TypedDict, cast


class RawForm(TypedDict, total=False):
    """
    One written form of an entry, inflected or otherwise.

    Attributes:
        form: The form itself.
        tags: What it is a form of, and how it was arrived at.
    """

    form: str
    tags: list[str]


class RawExample(TypedDict, total=False):
    """
    One sentence illustrating a sense.

    Attributes:
        text: The sentence.
        ref: The source, when the sentence is quoted from one.
        type: Which kind wiktextract read it as, where it read one.
    """

    text: str
    ref: str
    type: str
    bold_text_offsets: list[list[int]]


class RawSynonym(TypedDict, total=False):
    """
    One word standing for the same meaning as a sense or an entry.

    Attributes:
        word: The synonym itself.
    """

    word: str


class RawTarget(TypedDict, total=False):
    """
    The lemma a sense states itself to be a form or a spelling of.

    Attributes:
        word: The lemma pointed at.
    """

    word: str


class RawSense(TypedDict, total=False):
    """
    One sense of an entry.

    Attributes:
        glosses: The gloss chain, outermost first.
        tags: Labels of grammar and register.
        topics: Subject fields the sense belongs to.
        examples: The sentences illustrating it.
        wikidata: The Wikidata items it was tied to.
        alt_of: The lemma it states itself to be another spelling of.
        synonyms: Other words standing for this sense alone.
    """

    glosses: list[str]
    tags: list[str]
    topics: list[str]
    examples: list[RawExample]
    wikidata: list[str]
    alt_of: list[RawTarget]
    synonyms: list[RawSynonym]


class RawTranslation(TypedDict, total=False):
    """
    One word another language uses for a sense of the entry.

    Attributes:
        word: The translation itself.
        lang_code: The language it belongs to, by code.
        sense: The gloss it translates, as the translation table heads it.
    """

    word: str
    lang_code: str
    sense: str


class RawEntry(TypedDict, total=False):
    """
    One dictionary entry.

    Attributes:
        word: The headword.
        pos: Its part of speech.
        lang_code: The language the headword belongs to.
        etymology_number: Optional etymology number within the page.
        forms: The shapes the headword takes.
        senses: Its meanings.
        translations: What other languages call it.
    """

    word: str
    pos: str
    lang_code: str
    etymology_number: str
    forms: list[RawForm]
    senses: list[RawSense]
    translations: list[RawTranslation]


_ENTRY_KEYS = frozenset(RawEntry.__optional_keys__)
_SENSE_KEYS = frozenset(RawSense.__optional_keys__)
_FORM_KEYS = frozenset(RawForm.__optional_keys__)
_EXAMPLE_KEYS = frozenset(RawExample.__optional_keys__)
_TARGET_KEYS = frozenset(RawTarget.__optional_keys__)
_SYNONYM_KEYS = frozenset(RawSynonym.__optional_keys__)
_TRANSLATION_KEYS = frozenset(RawTranslation.__optional_keys__)

_SENSE_LISTS = {
    "examples": _EXAMPLE_KEYS,
    "alt_of": _TARGET_KEYS,
    "synonyms": _SYNONYM_KEYS,
}


def _narrow_record(
    record: object,
    keys: frozenset[str],
) -> dict[str, object]:
    """
    Cut one record down to the keys named, passing over what is not a record.

    Args:
        record: What the file held under that key.
        keys: The keys to keep.

    Returns:
        The record, holding those keys alone.
    """
    if not isinstance(record, Mapping):
        return {}

    kept_record = cast(Mapping[str, object], record)

    return {key: value for key, value in kept_record.items() if key in keys}


def _narrow_records(
    records: object,
    keys: frozenset[str],
) -> list[dict[str, object]]:
    """
    Cut down every record of a list, passing over what is not a list.

    Args:
        records: What the file held under that key.
        keys: The keys to keep.

    Returns:
        The records, each holding those keys alone.
    """
    if not isinstance(records, list):
        return []

    return [_narrow_record(record, keys) for record in cast(list[object], records)]


def narrow(
    entry: object,
) -> dict[str, object]:
    """
    Cut one entry down to what this module declares.

    An extraction carries every field wiktextract can write; few are read.

    Args:
        entry: One entry, as the extraction wrote it.

    Returns:
        The same entry, holding the declared keys alone.
    """
    narrowed = _narrow_record(entry, _ENTRY_KEYS)

    if "forms" in narrowed:
        narrowed["forms"] = _narrow_records(narrowed["forms"], _FORM_KEYS)

    if "translations" in narrowed:
        narrowed["translations"] = _narrow_records(
            narrowed["translations"], _TRANSLATION_KEYS
        )

    if "senses" in narrowed:
        senses = _narrow_records(narrowed["senses"], _SENSE_KEYS)

        for sense in senses:
            for key, keys in _SENSE_LISTS.items():
                if key in sense:
                    sense[key] = _narrow_records(sense[key], keys)

        narrowed["senses"] = senses

    return narrowed
