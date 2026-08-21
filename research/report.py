"""
Reporting the figures of a collected export.

What the pipeline produced is counted here rather than judged: the report
says how much of each thing there is, and where the offsets fall short of
what the extractor promises. Whether a record is faithful is a question for
the annotation study.

The figures are gathered into sections, which are what the renderers take.
One writes them to a terminal and one to Markdown, so the same pass feeds a
reading and a document.
"""

import argparse
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

from corpus import Entry, read
from rich.console import Console
from rich.table import Table

MARKS = str.maketrans({"’": "'", "‘": "'", "‐": "-", "‑": "-", "–": "-"})
"""Punctuation Wiktionary sets typographically, unified before a near miss is
looked for. The extractor matches the form as it is spelled, so a headword
written with a straight apostrophe misses a sentence written with a curly
one."""

WORD = re.compile(r"\w", re.UNICODE)
"""One word character, which is what an offset may not be flanked by."""

LABELS = 10
"""How many of the most common tags and topics the report names."""

EVERY = 1
"""How much of the export is read by default, one entry in this many."""


@dataclass(frozen=True, slots=True)
class Section:
    """
    One table of the report.

    Attributes:
        title: What the table is about.
        columns: The heading of each column, the first being the row's name.
        rows: The rows, already written out.
    """

    title: str
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


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
        unlocated: How many sentences carry none, by part of speech.
        shapes: What the headword looks like where none were found.
        near_misses: How many of those the headword is spelled in after all.
        broken: How many offsets break each promise the extractor makes.
        inflected: How many offsets fall on a form other than the headword.
        tags: How often each tag is used.
        topics: How often each topic is used.
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
    unlocated: Counter[str] = field(default_factory=Counter)
    shapes: Counter[str] = field(default_factory=Counter)
    near_misses: int = 0
    broken: Counter[str] = field(default_factory=Counter)
    inflected: int = 0
    tags: Counter[str] = field(default_factory=Counter)
    topics: Counter[str] = field(default_factory=Counter)


def shape_of(
    lemma: str,
) -> str:
    """
    Say what a headword looks like, where no occurrence of it was found.

    The classes are tried in turn and the first that fits wins, so that a
    headword is counted once. A plain one is the interesting case: nothing
    about how it is spelled explains the miss.

    Args:
        lemma: The headword.

    Returns:
        The class it falls in.
    """
    if " " in lemma:
        return "multiword"

    if "'" in lemma or "’" in lemma:
        return "apostrophe"

    if "-" in lemma:
        return "hyphenated"

    if not lemma.isascii():
        return "non-ascii"

    return "plain"


def spelled_in(
    lemma: str,
    text: str,
) -> bool:
    """
    Say whether a sentence spells a headword the extractor did not find.

    Case and typographic punctuation are unified first, which is where the
    extractor is strict and Wiktionary is not. Word boundaries are kept, so a
    headword sitting inside a longer word still does not count.

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
    offsets: list[list[int]],
    broken: Counter[str],
) -> None:
    """
    Hold one sentence's offsets against what the extractor promises.

    A failure here is a defect rather than a shortfall: the extractor states
    that offsets are ordered, disjoint, within the text and flanked by no
    word character, and nothing downstream re-checks it.

    Args:
        text: The sentence the offsets index.
        offsets: The offsets, as the export writes them.
        broken: Where a promise broken is counted.
    """
    read_up_to = 0

    for offset in offsets:
        start, end = offset

        if not 0 <= start < end <= len(text):
            broken["out of range"] += 1
            continue

        if start < read_up_to:
            broken["overlapping or unordered"] += 1

        if text[start:end] != text[start:end].strip():
            broken["padded with space"] += 1

        before = text[start - 1] if start else ""
        after = text[end] if end < len(text) else ""

        if WORD.match(before) or WORD.match(after):
            broken["inside a longer word"] += 1

        read_up_to = end


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

        for sense in senses:
            figures.senses[pos] += 1
            figures.depths[len(sense["glosses"])] += 1
            figures.tags.update(sense.get("tags", []))
            figures.topics.update(sense.get("topics", []))

            sentences = sense.get("sentences", [])
            if not sentences:
                figures.barren += 1

            for sentence in sentences:
                text = sentence["text"]
                quoted = "reference" in sentence

                figures.attested[pos] += 1
                figures.sentences["quotation" if quoted else "example"] += 1

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

                check(text, offsets, figures.broken)

                figures.offsets += len(offsets)
                figures.inflected += sum(
                    text[start:end].casefold() != lemma.casefold()
                    for start, end in offsets
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


def _records(
    figures: Figures,
) -> Section:
    """Lay out how much of each thing the export holds."""
    per_entry = figures.senses_per_entry or [0]
    senses = sum(figures.senses.values())

    return Section(
        "Records",
        ("Figure", "Count", "Share"),
        (
            ("entries", count(sum(figures.entries.values())), ""),
            ("senses", count(senses), ""),
            ("sentences", count(sum(figures.sentences.values())), ""),
            (
                "senses with no sentence",
                count(figures.barren),
                share(figures.barren, senses),
            ),
            (
                "senses per entry",
                f"median {median(per_entry):.0f}, most {max(per_entry)}",
                "",
            ),
            ("distinct tags", count(len(figures.tags)), ""),
            ("distinct topics", count(len(figures.topics)), ""),
        ),
    )


def _depths(
    figures: Figures,
) -> Section:
    """Lay out how deep the gloss chains nest, one row per level."""
    senses = sum(figures.senses.values())

    return Section(
        "Gloss chain depth",
        ("Depth", "Senses", "Share"),
        tuple(
            (str(depth), count(held), share(held, senses))
            for depth, held in sorted(figures.depths.items())
        ),
    )


def _by_pos(
    figures: Figures,
) -> Section:
    """Lay out what each part of speech contributes."""
    order = sorted(figures.entries, key=lambda pos: -figures.senses[pos])

    return Section(
        "By part of speech",
        ("Part of speech", "Entries", "Senses", "Sentences", "Unlocated"),
        tuple(
            (
                pos,
                count(figures.entries[pos]),
                count(figures.senses[pos]),
                count(figures.attested[pos]),
                count(figures.unlocated[pos]),
            )
            for pos in order
        ),
    )


def _sentences(
    figures: Figures,
) -> Section:
    """Lay out what kind of sentence the export holds."""
    sentences = sum(figures.sentences.values())

    return Section(
        "Sentences",
        ("Kind", "Count", "Share"),
        tuple(
            (kind, count(held), share(held, sentences))
            for kind, held in figures.sentences.most_common()
        ),
    )


def _quotations(
    figures: Figures,
) -> Section:
    """Lay out what the quotations say about themselves."""
    quotations = figures.sentences["quotation"]

    rows = [
        ("undated", count(figures.undated), share(figures.undated, quotations)),
    ]

    if figures.years:
        rows += [
            ("earliest year", str(min(figures.years)), ""),
            ("latest year", str(max(figures.years)), ""),
            ("median year", f"{median(figures.years):.0f}", ""),
        ]

    return Section("Quotations", ("Figure", "Count", "Share"), tuple(rows))


def _offsets(
    figures: Figures,
) -> Section:
    """Lay out how many sentences the headword was found in."""
    sentences = sum(figures.sentences.values())
    unlocated = sum(figures.unlocated.values())
    located = sentences - unlocated

    return Section(
        "Word offsets",
        ("Figure", "Sentences", "Share"),
        (
            ("headword found", count(located), share(located, sentences)),
            ("headword not found", count(unlocated), share(unlocated, sentences)),
        ),
    )


def _occurrences(
    figures: Figures,
) -> Section:
    """Lay out how many times over the headword was found in one sentence."""
    sentences = sum(figures.sentences.values())

    return Section(
        "Occurrences per sentence",
        ("Occurrences", "Sentences", "Share"),
        tuple(
            (
                "5 and over" if found == 5 else str(found),
                count(held),
                share(held, sentences),
            )
            for found, held in sorted(figures.occurrences.items())
        ),
    )


def _landed(
    figures: Figures,
) -> Section:
    """Lay out which form of the headword the offsets fall on."""
    written = figures.offsets - figures.inflected

    return Section(
        "What the offsets landed on",
        ("Form", "Offsets", "Share"),
        (
            (
                "the headword as written",
                count(written),
                share(written, figures.offsets),
            ),
            (
                "an inflection of it",
                count(figures.inflected),
                share(figures.inflected, figures.offsets),
            ),
        ),
    )


def _unlocated(
    figures: Figures,
) -> Section:
    """Lay out what the headwords look like where none was found."""
    unlocated = sum(figures.unlocated.values())

    return Section(
        "Headwords not found",
        ("Shape", "Sentences", "Share"),
        (
            *(
                (shape, count(held), share(held, unlocated))
                for shape, held in figures.shapes.most_common()
            ),
            (
                "spelled in after all",
                count(figures.near_misses),
                share(figures.near_misses, unlocated),
            ),
        ),
    )


def _broken(
    figures: Figures,
) -> Section | None:
    """Lay out the promises the extractor broke, if it broke any."""
    if not figures.broken:
        return None

    return Section(
        "Promises broken",
        ("Promise", "Offsets"),
        tuple((promise, count(held)) for promise, held in figures.broken.most_common()),
    )


def _tags(
    figures: Figures,
) -> Section:
    """Lay out the most common tags."""
    return Section(
        "Most common tags",
        ("Tag", "Uses"),
        tuple((tag, count(held)) for tag, held in figures.tags.most_common(LABELS)),
    )


def _topics(
    figures: Figures,
) -> Section:
    """Lay out the most common topics."""
    return Section(
        "Most common topics",
        ("Topic", "Uses"),
        tuple(
            (topic, count(held)) for topic, held in figures.topics.most_common(LABELS)
        ),
    )


def compose(
    figures: Figures,
) -> list[Section]:
    """
    Gather the figures into the tables the report is made of.

    A broken promise is a defect rather than a figure, so its table is left
    out of a report that has none to show.

    Args:
        figures: What one pass added up to.

    Returns:
        The sections, in reading order.
    """
    sections = [
        _records(figures),
        _by_pos(figures),
        _depths(figures),
        _sentences(figures),
        _quotations(figures),
        _offsets(figures),
        _occurrences(figures),
        _landed(figures),
        _unlocated(figures),
        _broken(figures),
        _tags(figures),
        _topics(figures),
    ]

    return [section for section in sections if section is not None]


def to_console(
    sections: Iterable[Section],
    note: str,
) -> None:
    """
    Write the report to the terminal.

    Args:
        sections: The tables to write.
        note: What to say before them, or nothing.
    """
    console = Console()

    if note:
        console.print(note, style="yellow")

    for section in sections:
        table = Table(title=section.title, title_justify="left", title_style="bold")

        for position, column in enumerate(section.columns):
            table.add_column(column, justify="left" if position == 0 else "right")

        for row in section.rows:
            table.add_row(*row)

        console.print(table)


def to_markdown(
    sections: Iterable[Section],
    note: str,
) -> str:
    """
    Write the report as Markdown, for a document to take as it stands.

    Args:
        sections: The tables to write.
        note: What to say before them, or nothing.

    Returns:
        The document.
    """
    lines = ["# Figures", ""]

    if note:
        lines += [note, ""]

    for section in sections:
        lines += [
            f"## {section.title}",
            "",
            f"| {' | '.join(section.columns)} |",
            f"|{'|'.join(' --- ' for _ in section.columns)}|",
        ]
        lines += [f"| {' | '.join(row)} |" for row in section.rows]
        lines.append("")

    return "\n".join(lines)


class Arguments(argparse.Namespace):
    """
    What the command line settles.

    Attributes:
        every: How much of the export to read, one entry in this many.
        to: Where to write the report, or None for the terminal.
    """

    every: int = EVERY
    to: Path | None = None


def read_arguments() -> Arguments:
    """
    Read what the command line settles, falling back on the defaults above.

    Returns:
        How much of the export to read, and where the report goes.
    """
    parser = argparse.ArgumentParser(description="Report the figures of an export.")

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
        help="write the report to this file as Markdown, rather than to the terminal",
    )

    return parser.parse_args(namespace=Arguments())


def main() -> None:
    """
    Walk the export and write what it adds up to.
    """
    arguments = read_arguments()

    figures = tally(read(), arguments.every)
    sections = compose(figures)

    note = (
        f"One entry in {arguments.every}; scale the counts."
        if arguments.every > 1
        else ""
    )

    if arguments.to is None:
        to_console(sections, note)
        return

    arguments.to.parent.mkdir(parents=True, exist_ok=True)
    _ = arguments.to.write_text(to_markdown(sections, note), encoding="utf-8")

    print(f"Wrote {arguments.to}")  # noqa: T201


if __name__ == "__main__":
    main()
