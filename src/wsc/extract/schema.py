"""
The slice of the wiktextract schema the collector reads.

Every key is optional: this describes someone else's JSON, and an entry
carries only what the page it was made from happened to say.
"""

from typing import TypedDict


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
        senseid: What Wiktionary names the sense, where it names it.
        wikidata: The Wikidata items it was tied to.
        form_of: The lemma it states itself to be a form of.
        alt_of: The lemma it states itself to be another spelling of.
    """

    glosses: list[str]
    tags: list[str]
    topics: list[str]
    examples: list[RawExample]
    senseid: list[str]
    wikidata: list[str]
    form_of: list[RawTarget]
    alt_of: list[RawTarget]


class RawForm(TypedDict, total=False):
    """
    One written form of an entry, inflected or otherwise.

    Attributes:
        form: The form itself.
        tags: What it is a form of, and how it was arrived at.
    """

    form: str
    tags: list[str]


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
        forms: The shapes the headword takes.
        senses: Its meanings.
        translations: What other languages call it.
    """

    word: str
    pos: str
    lang_code: str
    forms: list[RawForm]
    senses: list[RawSense]
    translations: list[RawTranslation]
