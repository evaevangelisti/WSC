"""
Gathering the entries Wiktionary splits by etymology into one.
"""

from collections.abc import Iterable, Iterator
from dataclasses import replace

from kwic import Query

from ...models import Lemma, Sense, Translations


def add_translations(
    kept_translations: Translations,
    added_translations: Translations,
) -> None:
    """
    Add one set of translation tables to another, gloss by gloss.

    Args:
        kept_translations: What is kept already, added to in place.
        added_translations: What to add to it.
    """
    for gloss, added_words in added_translations.items():
        kept_words = kept_translations.setdefault(gloss, {})

        for language, words in added_words.items():
            kept_words[language] = kept_words.get(language, frozenset()) | words


def _united(
    kept: tuple[str, ...],
    added: tuple[str, ...],
) -> tuple[str, ...]:
    """
    Add what is not carried already, in the order it was first written.

    Args:
        kept: What is carried already.
        added: What to add to it.

    Returns:
        The two, a repeat kept once.
    """
    return (*kept, *(item for item in added if item not in kept))


def merge_senses(
    senses: list[Sense],
) -> list[Sense]:
    """
    Gather the senses of one entry that say the same thing into one.

    Wiktionary writes a gloss twice often enough that a name settled by its
    meaning cannot tell the two apart.

    Args:
        senses: The senses of one entry, in the order they were read.

    Returns:
        One sense per meaning, carrying what each of them carried.
    """
    gathered_senses: dict[str, Sense] = {}

    for sense in senses:
        kept_sense = gathered_senses.get(sense.id)

        if kept_sense is None:
            gathered_senses[sense.id] = sense
            continue

        kept_sense.synonyms = _united(kept_sense.synonyms, sense.synonyms)

        kept_sense.topics = _united(kept_sense.topics, sense.topics)
        kept_sense.tags = _united(kept_sense.tags, sense.tags)

        kept_sense.sentences += sense.sentences

    return list(gathered_senses.values())


def merge_lemmas(
    queried_lemmas: Iterable[tuple[Lemma, Query]],
) -> Iterator[tuple[Lemma, Query]]:
    """
    Gather the entries of one headword and part of speech into one.

    Nothing downstream reads the etymology split.

    Args:
        queried_lemmas: The entries read, each with what to search its
        sentences for.

    Yields:
        One entry per headword and part of speech, in the order the first of
        its etymologies was read.
    """
    lemmas: dict[str, tuple[Lemma, Query]] = {}

    for lemma, query in queried_lemmas:
        gathered_lemma = lemmas.get(lemma.id)

        if gathered_lemma is None:
            lemmas[lemma.id] = lemma, query
            continue

        kept_lemma, kept_query = gathered_lemma

        kept_lemma.variants |= lemma.variants
        kept_lemma.senses += lemma.senses

        add_translations(kept_lemma.translations, lemma.translations)

        # Each etymology inflects the headword its own way.
        lemmas[lemma.id] = (
            kept_lemma,
            replace(kept_query, forms=kept_query.forms | query.forms),
        )

    for lemma, query in lemmas.values():
        lemma.senses = merge_senses(lemma.senses)

        yield lemma, query
