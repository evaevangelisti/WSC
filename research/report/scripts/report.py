"""
Reporting the figures of a collected export.

The report measures pipeline output. Annotation studies assess record accuracy.
"""

import argparse
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

from corpus import SENSES, Entry, Sense, WordOffset, read

REPORTS = Path(__file__).resolve().parent.parent / "reports"
"""Where the written reports are kept, one file per pass."""

MARKS = str.maketrans({"’": "'", "‘": "'", "‐": "-", "‑": "-", "–": "-"})
"""
Punctuation Wiktionary sets typographically, unified before a near miss is looked for.
"""

WORD = re.compile(r"\w", re.UNICODE)
"""One word character, which is what an offset may not be flanked by."""

PARTS_OF_SPEECH = {
    "noun": "Noun",
    "name": "Proper noun",
    "verb": "Verb",
    "adj": "Adjective",
    "adv": "Adverb",
}
"""What the collector keeps, in the order a grammar names them."""

LABELS = 10
"""How many of the most common tags and topics the report names."""

EVERY = 1
"""How much of the export is read by default, one entry in this many."""


@dataclass(frozen=True, slots=True)
class Table:
    """
    One table of a section.

    Attributes:
        caption: What the table is about, within what the section is about.
        columns: The heading of each column, the first being the row's name.
        rows: The rows, already written out.
    """

    caption: str
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


@dataclass(frozen=True, slots=True)
class Section:
    """
    One part of the report, holding the tables that speak to one thing.

    Figures on the same subject are read against one another, so they are kept together.

    Attributes:
        title: What the tables have in common.
        tables: The tables, in reading order.
    """

    title: str
    tables: tuple[Table, ...]


@dataclass(slots=True)
class Figures:
    """
    What one pass over an export adds up to.

    Attributes:
        entries: How many entries carry each part of speech.
        senses: How many senses carry each part of speech.
        attested: How many sentences hang off each part of speech.
        senses_per_entry: How many senses each entry carries.
        depths: How deep the gloss chains nest.
        barren: How many senses carry no sentence at all.
        sentences: How many sentences of each kind there are.
        undated: How many quotations name no year.
        years: The years the quotations name.
        occurrences: How many offsets each sentence carries.
        offsets: How many offsets there are in all.
        offset_sources: How many candidates each source combination supports.
        sentence_sources: How the sources relate within each sentence.
        unlocated: How many sentences carry none, by part of speech.
        shapes: What the headword looks like where none were found.
        near_misses: How many of those the headword is spelled in after all.
        broken: How many offsets break each promise the extractor makes.
        inflected: How many offsets fall on a form other than the headword.
        tags: How often each tag is used.
        topics: How often each topic is used.
        tied: How many senses name a Wikidata item.
        varying: How many entries are written another way too.
        variants: How many other spellings there are in all.
        synonymised: How many senses carry a synonym of their own.
        sense_synonyms: How many sense-tied synonyms there are in all.
        translated: How many entries carry a translation table.
        translations: How many translations there are in all.
        glosses_translated: How many glosses a translation table heads.
        languages: How often each language offers a translation.
    """

    entries: Counter[str] = field(default_factory=Counter)
    senses: Counter[str] = field(default_factory=Counter)
    attested: Counter[str] = field(default_factory=Counter)
    senses_per_entry: list[int] = field(default_factory=list)
    depths: Counter[int] = field(default_factory=Counter)
    barren: int = 0
    sentences: Counter[str] = field(default_factory=Counter)
    undated: int = 0
    years: list[int] = field(default_factory=list)
    occurrences: Counter[int] = field(default_factory=Counter)
    offsets: int = 0
    offset_sources: Counter[str] = field(default_factory=Counter)
    sentence_sources: Counter[str] = field(default_factory=Counter)
    unlocated: Counter[str] = field(default_factory=Counter)
    shapes: Counter[str] = field(default_factory=Counter)
    near_misses: int = 0
    broken: Counter[str] = field(default_factory=Counter)
    inflected: int = 0
    tags: Counter[str] = field(default_factory=Counter)
    topics: Counter[str] = field(default_factory=Counter)
    tied: int = 0
    varying: int = 0
    variants: int = 0
    synonymised: int = 0
    sense_synonyms: int = 0
    translated: int = 0
    translations: int = 0
    glosses_translated: int = 0
    languages: Counter[str] = field(default_factory=Counter)


def shape_of(
    lemma: str,
) -> str:
    """
    Say what a headword looks like, where no occurrence of it was found.

    The first class that fits wins, so a headword is counted once. A plain one is the
    interesting case: nothing about it explains the miss.

    Args:
        lemma: The headword.

    Returns:
        The class it falls in.
    """
    if " " in lemma:
        return "Multiword"

    if "'" in lemma or "’" in lemma:
        return "Apostrophe"

    if "-" in lemma:
        return "Hyphenated"

    if not lemma.isascii():
        return "Non-ASCII"

    return "Plain"


def spelled_in(
    lemma: str,
    text: str,
) -> bool:
    """
    Say whether a sentence spells a headword the extractor did not find.

    Case and typographic punctuation are unified first. Word boundaries are kept, so a
    headword inside a longer word still does not count.

    Args:
        lemma: The headword.
        text: The sentence.

    Returns:
        Whether it is in there after all.
    """

    def normalise(
        value: str,
    ) -> str:
        return unicodedata.normalize("NFKC", value.translate(MARKS)).casefold()

    return (
        re.search(rf"(?<!\w){re.escape(normalise(lemma))}(?!\w)", normalise(text))
        is not None
    )


def check(
    text: str,
    offsets: list[WordOffset],
    broken: Counter[str],
) -> None:
    """
    Hold one sentence's offsets against what the extractor promises.

    These checks identify violations of the extractor's offset contract.

    Args:
        text: The sentence the offsets index.
        offsets: The offsets, as the export writes them.
        broken: Where a promise broken is counted.
    """
    read_up_to = {"bold": 0, "lemmatizer": 0}

    for word_offset in offsets:
        start, end = word_offset["offset"]

        for source in word_offset["sources"]:
            if not 0 <= start < end <= len(text):
                broken[f"Out of range ({source})"] += 1
                continue

            if start < read_up_to[source]:
                broken[f"Overlapping or unordered ({source})"] += 1

            read_up_to[source] = end

            if text[start:end] != text[start:end].strip():
                broken[f"Padded with space ({source})"] += 1

            before = text[start - 1] if start else ""
            after = text[end] if end < len(text) else ""

            if WORD.match(before) or WORD.match(after):
                broken[f"Inside a longer word ({source})"] += 1


def source_relation(
    offsets: list[WordOffset],
) -> str:
    """
    Describe how both methods located one sentence.

    Args:
        offsets: Candidate ranges and their supporting methods.

    Returns:
        The relationship between bold and lemmatizer proposals.
    """
    bold = {
        tuple(word_offset["offset"])
        for word_offset in offsets
        if "bold" in word_offset["sources"]
    }
    lemmatizer = {
        tuple(word_offset["offset"])
        for word_offset in offsets
        if "lemmatizer" in word_offset["sources"]
    }

    if not bold:
        return "Lemmatizer only"

    if not lemmatizer:
        return "Bold only"

    if bold == lemmatizer:
        return "Exact agreement"

    if bold & lemmatizer:
        return "Partial agreement"

    if any(
        bold_start < lemmatizer_end and lemmatizer_start < bold_end
        for bold_start, bold_end in bold
        for lemmatizer_start, lemmatizer_end in lemmatizer
    ):
        return "Overlapping proposals"

    return "Disagreement"


def count_entry(
    figures: Figures,
    entry: Entry,
) -> None:
    """
    Add up what one entry carries beside its senses.

    Args:
        figures: What the pass has added up so far.
        entry: The entry to read.
    """
    variants = entry.get("variants", [])

    if variants:
        figures.varying += 1
        figures.variants += len(variants)

    translations = entry.get("translations", {})

    if not translations:
        return

    figures.translated += 1
    figures.glosses_translated += len(translations)

    for offered in translations.values():
        for language, words in offered.items():
            figures.languages[language] += len(words)
            figures.translations += len(words)


def count_sense(
    figures: Figures,
    sense: Sense,
) -> None:
    """
    Add up what one sense carries beside its sentences.

    Args:
        figures: What the pass has added up so far.
        sense: The sense to read.
    """
    figures.tied += bool(sense.get("wikidata_ids"))

    synonyms = sense.get("synonyms", [])

    if synonyms:
        figures.synonymised += 1
        figures.sense_synonyms += len(synonyms)


def tally(
    entries: Iterable[Entry],
    every: int,
) -> Figures:
    """
    Walk an export once, adding up everything the report names.

    Args:
        entries: The export to read.
        every: How much of it to read, one entry in this many.

    Returns:
        The figures.
    """
    figures = Figures()

    for position, entry in enumerate(entries):
        if position % every:
            continue

        lemma, pos = entry["lemma"], entry["pos"]
        senses = entry.get("senses", [])

        figures.entries[pos] += 1
        figures.senses_per_entry.append(len(senses))

        count_entry(figures, entry)

        for sense in senses:
            figures.senses[pos] += 1
            figures.depths[len(sense["glosses"])] += 1
            figures.tags.update(sense.get("tags", []))
            figures.topics.update(sense.get("topics", []))

            count_sense(figures, sense)

            sentences = sense.get("sentences", [])

            if not sentences:
                figures.barren += 1

            for sentence in sentences:
                text = sentence["text"]
                quoted = "reference" in sentence

                figures.attested[pos] += 1
                figures.sentences["Quotation" if quoted else "Example"] += 1

                if quoted:
                    if "year" in sentence:
                        figures.years.append(sentence["year"])
                    else:
                        figures.undated += 1

                offsets = sentence.get("word_offsets", [])
                figures.occurrences[min(len(offsets), 5)] += 1

                if not offsets:
                    figures.unlocated[pos] += 1
                    figures.shapes[shape_of(lemma)] += 1

                    if spelled_in(lemma, text):
                        figures.near_misses += 1

                    continue

                figures.sentence_sources[source_relation(offsets)] += 1

                check(text, offsets, figures.broken)

                figures.offsets += len(offsets)
                figures.offset_sources.update(
                    "+".join(word_offset["sources"]) for word_offset in offsets
                )
                figures.inflected += sum(
                    text[start:end].casefold() != lemma.casefold()
                    for word_offset in offsets
                    for start, end in (word_offset["offset"],)
                )

    return figures


def count(
    part: int,
) -> str:
    """Write a count, grouped in thousands."""
    return f"{part:,}"


def share(
    part: int,
    whole: int,
) -> str:
    """
    Write what part of a whole a count is.

    Args:
        part: The count.
        whole: What it is part of.

    Returns:
        The share, or nothing where there is no whole to take it of.
    """
    return f"{part / whole:.1%}" if whole else ""


def _totals(
    figures: Figures,
) -> Table:
    """Lay out how much of each thing the export holds."""
    per_entry = figures.senses_per_entry or [0]
    senses = sum(figures.senses.values())

    return Table(
        "Totals",
        ("Figure", "Count", "Share"),
        (
            ("Entries", count(sum(figures.entries.values())), ""),
            ("Senses", count(senses), ""),
            ("Sentences", count(sum(figures.sentences.values())), ""),
            (
                "Senses with no sentence",
                count(figures.barren),
                share(figures.barren, senses),
            ),
            (
                "Senses per entry",
                f"median {median(per_entry):.0f}, most {max(per_entry)}",
                "",
            ),
            ("Distinct tags", count(len(figures.tags)), ""),
            ("Distinct topics", count(len(figures.topics)), ""),
        ),
    )


def _depths(
    figures: Figures,
) -> Table:
    """Lay out how deep the gloss chains nest, one row per level."""
    senses = sum(figures.senses.values())

    return Table(
        "Gloss chain depth",
        ("Depth", "Senses", "Share"),
        tuple(
            (str(depth), count(nested), share(nested, senses))
            for depth, nested in sorted(figures.depths.items())
        ),
    )


def _by_pos(
    figures: Figures,
) -> Table:
    """Lay out what each part of speech contributes."""
    return Table(
        "By part of speech",
        ("Part of speech", "Entries", "Senses", "Sentences", "Unlocated"),
        tuple(
            (
                name,
                count(figures.entries[pos]),
                count(figures.senses[pos]),
                count(figures.attested[pos]),
                count(figures.unlocated[pos]),
            )
            for pos, name in PARTS_OF_SPEECH.items()
            if pos in figures.entries
        ),
    )


def _kinds(
    figures: Figures,
) -> Table:
    """Lay out what kind of sentence the export holds."""
    sentences = sum(figures.sentences.values())

    return Table(
        "By kind",
        ("Kind", "Count", "Share"),
        tuple(
            (kind, count(written), share(written, sentences))
            for kind, written in figures.sentences.most_common()
        ),
    )


def _dated(
    figures: Figures,
) -> Table:
    """
    Lay out how many quotations name the year they were written in.

    A quotation either names a year, and so is counted among the years, or names none,
    which is what the two rows divide.
    """
    dated = len(figures.years)
    quotations = dated + figures.undated

    return Table(
        "By date",
        ("Figure", "Count", "Share"),
        (
            ("Dated", count(dated), share(dated, quotations)),
            ("Undated", count(figures.undated), share(figures.undated, quotations)),
        ),
    )


def _years(
    figures: Figures,
) -> Table | None:
    """Lay out the span the dated quotations cover, where any names a year."""
    if not figures.years:
        return None

    return Table(
        "Years",
        ("Figure", "Year"),
        (
            ("Earliest", str(min(figures.years))),
            ("Latest", str(max(figures.years))),
            ("Median", f"{median(figures.years):.0f}"),
        ),
    )


def _found(
    figures: Figures,
) -> Table:
    """Lay out how many sentences the headword was found in."""
    sentences = sum(figures.sentences.values())
    unlocated = sum(figures.unlocated.values())
    located_sentences = sentences - unlocated

    return Table(
        "By sentence",
        ("Figure", "Sentences", "Share"),
        (
            (
                "Headword found",
                count(located_sentences),
                share(located_sentences, sentences),
            ),
            ("Headword not found", count(unlocated), share(unlocated, sentences)),
        ),
    )


def _occurrences(
    figures: Figures,
) -> Table:
    """Lay out how many times over the headword was found in one sentence."""
    sentences = sum(figures.sentences.values())

    return Table(
        "Candidate offsets per sentence",
        ("Occurrences", "Sentences", "Share"),
        tuple(
            (
                "5 and over" if occurrences == 5 else str(occurrences),
                count(attesting),
                share(attesting, sentences),
            )
            for occurrences, attesting in sorted(figures.occurrences.items())
        ),
    )


def _source_relations(
    figures: Figures,
) -> Table:
    """
    Lay out how source proposals relate within sentences.

    Args:
        figures: Counts gathered from the export.

    Returns:
        Source relationships ordered by sentence count.
    """
    located = sum(figures.sentence_sources.values())

    return Table(
        "Sources by sentence",
        ("Figure", "Count", "Share"),
        tuple(
            (
                relation,
                count(sentences),
                share(sentences, located),
            )
            for relation, sentences in figures.sentence_sources.most_common()
        ),
    )


def _offset_sources(
    figures: Figures,
) -> Table:
    """
    Lay out which methods support each candidate offset.

    Args:
        figures: Counts gathered from the export.

    Returns:
        Source combinations ordered by candidate count.
    """
    names = {
        "bold+lemmatizer": "Bold and lemmatizer",
        "bold": "Bold",
        "lemmatizer": "Lemmatizer",
    }

    return Table(
        "Sources by offset",
        ("Figure", "Count", "Share"),
        tuple(
            (
                names[sources],
                count(offsets),
                share(offsets, figures.offsets),
            )
            for sources, offsets in figures.offset_sources.most_common()
        ),
    )


def _landed(
    figures: Figures,
) -> Table:
    """Lay out which form of the headword the offsets fall on."""
    written = figures.offsets - figures.inflected

    return Table(
        "What the offsets landed on",
        ("Form", "Offsets", "Share"),
        (
            (
                "The headword as written",
                count(written),
                share(written, figures.offsets),
            ),
            (
                "An inflection of it",
                count(figures.inflected),
                share(figures.inflected, figures.offsets),
            ),
        ),
    )


def _unlocated(
    figures: Figures,
) -> Table:
    """Lay out what the headwords look like where none was found."""
    unlocated = sum(figures.unlocated.values())

    return Table(
        "Headwords not found",
        ("Shape", "Sentences", "Share"),
        (
            *(
                (shape, count(missed), share(missed, unlocated))
                for shape, missed in figures.shapes.most_common()
            ),
            (
                "Spelled in after all",
                count(figures.near_misses),
                share(figures.near_misses, unlocated),
            ),
        ),
    )


def _broken(
    figures: Figures,
) -> Table | None:
    """Lay out the promises the extractor broke, if it broke any."""
    if not figures.broken:
        return None

    return Table(
        "Promises broken",
        ("Promise", "Offsets"),
        tuple(
            (promise, count(broken)) for promise, broken in figures.broken.most_common()
        ),
    )


def _tied(
    figures: Figures,
) -> Table:
    """Lay out how many senses are tied to something outside Wiktionary."""
    senses = sum(figures.senses.values())

    return Table(
        "Ties",
        ("Figure", "Senses", "Share"),
        (("Tied to Wikidata", count(figures.tied), share(figures.tied, senses)),),
    )


def _variants(
    figures: Figures,
) -> Table:
    """Lay out how many entries are written another way too."""
    entries = sum(figures.entries.values())
    plain = entries - figures.varying

    return Table(
        "By entry",
        ("Figure", "Entries", "Share"),
        (
            (
                "Written another way",
                count(figures.varying),
                share(figures.varying, entries),
            ),
            ("Written one way alone", count(plain), share(plain, entries)),
            (
                "Spellings per entry that has one",
                f"{figures.variants / figures.varying:.1f}" if figures.varying else "",
                "",
            ),
        ),
    )


def _synonyms(
    figures: Figures,
) -> Table:
    """Lay out how many senses Wiktionary offers another word for."""
    senses = sum(figures.senses.values())
    plain = senses - figures.synonymised

    return Table(
        "By sense",
        ("Figure", "Senses", "Share"),
        (
            (
                "With a synonym",
                count(figures.synonymised),
                share(figures.synonymised, senses),
            ),
            ("With none", count(plain), share(plain, senses)),
            (
                "Synonyms per sense that has one",
                f"{figures.sense_synonyms / figures.synonymised:.1f}"
                if figures.synonymised
                else "",
                "",
            ),
        ),
    )


def _translated(
    figures: Figures,
) -> Table:
    """Lay out how many entries Wiktionary hangs a translation table off."""
    entries = sum(figures.entries.values())
    untranslated = entries - figures.translated

    return Table(
        "By entry",
        ("Figure", "Entries", "Share"),
        (
            (
                "Translated",
                count(figures.translated),
                share(figures.translated, entries),
            ),
            ("Untranslated", count(untranslated), share(untranslated, entries)),
        ),
    )


def _translations(
    figures: Figures,
) -> Table:
    """Lay out how much a translated entry carries."""
    translated = figures.translated or 1

    return Table(
        "Totals",
        ("Figure", "Count"),
        (
            ("Translations", count(figures.translations)),
            ("Glosses translated", count(figures.glosses_translated)),
            ("Distinct languages", count(len(figures.languages))),
            (
                "Translations per entry translated",
                f"{figures.translations / translated:.1f}",
            ),
        ),
    )


def _languages(
    figures: Figures,
) -> Table:
    """Lay out the languages offering the most translations."""
    return Table(
        "Most translating languages",
        ("Language", "Translations"),
        tuple(
            (language, count(offered))
            for language, offered in figures.languages.most_common(LABELS)
        ),
    )


def _tags(
    figures: Figures,
) -> Table:
    """Lay out the most common tags."""
    return Table(
        "Most common tags",
        ("Tag", "Uses"),
        tuple((tag, count(uses)) for tag, uses in figures.tags.most_common(LABELS)),
    )


def _topics(
    figures: Figures,
) -> Table:
    """Lay out the most common topics."""
    return Table(
        "Most common topics",
        ("Topic", "Uses"),
        tuple(
            (topic, count(uses)) for topic, uses in figures.topics.most_common(LABELS)
        ),
    )


def _section(
    title: str,
    *tables: Table | None,
) -> Section:
    """
    Gather tables under one title, dropping those with nothing to show.

    Args:
        title: What the tables have in common.
        tables: Tables available in the generated report.

    Returns:
        The section.
    """
    return Section(title, tuple(table for table in tables if table is not None))


def compose(
    figures: Figures,
) -> list[Section]:
    """
    Gather the figures into the sections the report is made of.

    Args:
        figures: What one pass added up to.

    Returns:
        The sections, in reading order.
    """
    return [
        _section(
            "Records",
            _totals(figures),
            _by_pos(figures),
            _depths(figures),
            _tied(figures),
        ),
        _section("Variants", _variants(figures)),
        _section("Synonyms", _synonyms(figures)),
        _section(
            "Translations",
            _translated(figures),
            _translations(figures),
            _languages(figures),
        ),
        _section("Sentences", _kinds(figures)),
        _section("Quotations", _dated(figures), _years(figures)),
        _section(
            "Word offsets",
            _found(figures),
            _source_relations(figures),
            _offset_sources(figures),
            _occurrences(figures),
            _landed(figures),
            _unlocated(figures),
            _broken(figures),
        ),
        _section("Tags and topics", _tags(figures), _topics(figures)),
    ]


def to_markdown(
    sections: Iterable[Section],
    note: str,
) -> str:
    """
    Write the report as Markdown, for a document to take as it stands.

    Args:
        sections: The sections to write.
        note: What to say before them, or nothing.

    Returns:
        The document.
    """
    lines = ["# Report", ""]

    if note:
        lines += [note, ""]

    for section in sections:
        lines += [f"## {section.title}", ""]

        for table in section.tables:
            lines += [
                f"### {table.caption}",
                "",
                f"| {' | '.join(table.columns)} |",
                f"|{'|'.join(' --- ' for _ in table.columns)}|",
            ]
            lines += [f"| {' | '.join(row)} |" for row in table.rows]
            lines.append("")

    return "\n".join(lines)


def write(
    sections: Iterable[Section],
    note: str,
    directory: Path,
    export_path: Path,
) -> Path:
    """
    Write the report to a directory, named after the export it read.

    Each export has one report, which subsequent runs replace.

    Args:
        sections: The sections to write.
        note: What to say before them, or nothing.
        directory: Where the report goes.
        export_path: The export it was made from.

    Returns:
        The document written.
    """
    directory.mkdir(parents=True, exist_ok=True)

    path = directory / f"{export_path.stem}.md"
    _ = path.write_text(to_markdown(sections, note), encoding="utf-8")

    return path


class Arguments(argparse.Namespace):
    """
    What the command line settles.

    Attributes:
        export: Which export to read.
        every: How much of it to read, one entry in this many.
        to: Which directory the report is written to.
    """

    export: Path = SENSES
    every: int = EVERY
    to: Path = REPORTS


def read_arguments() -> Arguments:
    """
    Read what the command line settles, falling back on the defaults above.

    Returns:
        Which export to read, how much of it, and where the report goes.
    """
    parser = argparse.ArgumentParser(description="Report the figures of an export.")

    _ = parser.add_argument(
        "export",
        nargs="?",
        type=Path,
        default=SENSES,
        help=f"the export to read (default: {SENSES})",
    )
    _ = parser.add_argument(
        "--every",
        type=int,
        help=(
            "read one entry in this many, for a quick pass "
            f"(default: {EVERY}, the whole export)"
        ),
    )
    _ = parser.add_argument(
        "--to",
        type=Path,
        help=f"write the report to this directory (default: {REPORTS})",
    )

    return parser.parse_args(namespace=Arguments())


def sampling_note(
    every: int,
) -> str:
    """
    Say what part of the export a pass read, where it read only part of it.

    A sampled report holds the same figures as a whole one, and nothing on its face says
    how much of the export they came from.

    Args:
        every: How much was read, one entry in this many.

    Returns:
        The note, or nothing where the whole export was read.
    """
    if every == 1:
        return ""

    sampled = count(every)

    return f"One entry in every {sampled} was read"


def main() -> None:
    """Walk the export and write what it adds up to."""
    arguments = read_arguments()

    figures = tally(read(arguments.export), arguments.every)
    sections = compose(figures)
    note = sampling_note(arguments.every)

    written = write(sections, note, arguments.to, arguments.export)

    print(f"Wrote {written}")  # noqa: T201


if __name__ == "__main__":
    main()
