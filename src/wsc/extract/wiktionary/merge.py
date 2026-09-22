"""Gathering the entries Wiktionary splits by etymology into one."""

from collections.abc import Iterable, Iterator
from dataclasses import replace

from kwic import Query

from ...identifiers import translation_table_id
from ...models import (
    Lemma,
    Offset,
    Sense,
    TranslationTable,
    WordOffset,
    WordOffsetSource,
)


def add_translations(
    kept_translations: tuple[TranslationTable, ...],
    added_translations: tuple[TranslationTable, ...],
    lemma_id: str,
) -> tuple[TranslationTable, ...]:
    """
    Add one set of translation tables to another, gloss by gloss.

    Args:
        kept_translations: Previously retained translation tables.
        added_translations: What to add to it.
        lemma_id: The entry identifier used to name merged tables.

    Returns:
        Tables containing the combined words for each gloss and language.
    """
    gathered_translation_tables = {
        table.gloss: {
            language: set(words) for language, words in table.translations.items()
        }
        for table in kept_translations
    }

    for table in added_translations:
        kept_words = gathered_translation_tables.setdefault(table.gloss, {})

        for language, words in table.translations.items():
            kept_words.setdefault(language, set()).update(words)

    return tuple(
        TranslationTable(
            translation_table_id(lemma_id, gloss),
            gloss,
            {language: frozenset(words) for language, words in translated.items()},
        )
        for gloss, translated in gathered_translation_tables.items()
    )


def merge_word_offsets(
    bold_offsets: tuple[WordOffset, ...],
    lemmatizer_offsets: tuple[Offset, ...],
) -> tuple[WordOffset, ...]:
    """
    Merge identical candidates while retaining every source.

    Args:
        bold_offsets: Candidates supplied by Wiktextract.
        lemmatizer_offsets: Candidates supplied by the lemmatizer pipeline.

    Returns:
        Every distinct candidate with its supporting sources.
    """
    offset_sources = {
        word_offset.offset: set(word_offset.sources) for word_offset in bold_offsets
    }

    for offset in lemmatizer_offsets:
        offset_sources.setdefault(offset, set()).add(WordOffsetSource.LEMMATIZER)

    return tuple(
        WordOffset(
            offset,
            tuple(
                source
                for source in WordOffsetSource
                if source in offset_sources[offset]
            ),
        )
        for offset in sorted(offset_sources)
    )


def _merge_items(
    kept: tuple[str, ...],
    added: tuple[str, ...],
) -> tuple[str, ...]:
    """
    Append items not already kept, preserving their order.

    Args:
        kept: Items already retained.
        added: Items to append when absent.

    Returns:
        The retained items followed by new additions.
    """
    return (*kept, *(item for item in added if item not in kept))


def merge_senses(
    senses: list[Sense],
) -> list[Sense]:
    """
    Gather the senses of one entry that say the same thing into one.

    Wiktionary writes a gloss twice often enough that a name settled by its meaning
    cannot tell the two apart.

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

        kept_sense.synonyms = _merge_items(kept_sense.synonyms, sense.synonyms)

        kept_sense.tags = _merge_items(kept_sense.tags, sense.tags)
        kept_sense.topics = _merge_items(kept_sense.topics, sense.topics)

        kept_sense.wikidata_ids = _merge_items(
            kept_sense.wikidata_ids,
            sense.wikidata_ids,
        )

        kept_sense.sentences += sense.sentences

    return list(gathered_senses.values())


def merge_lemmas(
    queried_lemmas: Iterable[tuple[Lemma, Query]],
) -> Iterator[tuple[Lemma, Query]]:
    """
    Gather the entries of one headword and part of speech into one.

    Nothing downstream reads the etymology split.

    Args:
        queried_lemmas: Collected entries and their sentence search queries.

    Yields:
        Merged entries in their first occurrence order.
    """
    gathered_lemmas: dict[str, tuple[Lemma, Query]] = {}

    for lemma, query in queried_lemmas:
        gathered_lemma = gathered_lemmas.get(lemma.id)

        if gathered_lemma is None:
            gathered_lemmas[lemma.id] = lemma, query
            continue

        kept_lemma, kept_query = gathered_lemma

        kept_lemma.variants |= lemma.variants
        kept_lemma.senses += lemma.senses

        kept_lemma.translation_tables = add_translations(
            kept_lemma.translation_tables,
            lemma.translation_tables,
            kept_lemma.id,
        )

        # Etymologies can provide different inflection forms for the same headword.
        gathered_lemmas[lemma.id] = (
            kept_lemma,
            replace(kept_query, forms=kept_query.forms | query.forms),
        )

    for lemma, query in gathered_lemmas.values():
        lemma.senses = merge_senses(lemma.senses)

        yield lemma, query
