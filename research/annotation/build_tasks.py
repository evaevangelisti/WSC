"""
Building the Label Studio tasks for the annotation study.

The study asks whether the occurrences marked in a sentence are the right
ones. The sample it draws is what gets versioned, and the tasks are rebuilt
from it and the seed.
"""

import argparse
import json
import random
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path
from urllib.parse import quote

from corpus import SEPARATOR, Entry, read

SEED = 0
"""What the drawing answers to by default, one study to a number. A rerun on
the same seed draws the same sample."""

COUNT = 200
"""How many sentences the study draws by default."""

MARKERS = ("<b>", "</b>")
"""What an occurrence is wrapped in. Label Studio renders the sentence as
markup, so the sentence itself is escaped before the tags go in."""

ENTRY_URL = "https://en.wiktionary.org/wiki/{lemma}#English"
"""Where the annotator reads the entry a sentence was gathered from."""

type Stratum = tuple[str, bool]
"""A part of speech, and whether anything was found at all. The drawing
balances over these, so the empty case is read as often as the full one
however rare it is. A sample is therefore no estimate of the export: the
report is what says how common each stratum really is."""

type Task = dict[str, dict[str, object]]
"""One task, as Label Studio reads it: everything shown sits under data."""


@dataclass(frozen=True, slots=True)
class Occurrence:
    """
    One sentence put up for judgement on where its headword was found.

    Attributes:
        id: The sense identifier and the place of the sentence under it.
        lemma: The headword that was looked for.
        pos: Its part of speech.
        url: Where the entry it was gathered from is published.
        gloss: The sense the sentence illustrates.
        text: The sentence.
        offsets: Where the headword was found, leftmost first.
    """

    id: str
    lemma: str
    pos: str
    url: str
    gloss: str
    text: str
    offsets: tuple[tuple[int, int], ...]


class Reservoir[Item]:
    """
    Holds a uniform sample of a stream, one reservoir per stratum.

    The export is walked once and never held whole, so what is kept has to be
    settled while it goes past. Every item stands an equal chance of being
    kept, and the memory stays bounded by the strata.
    """

    def __init__(
        self,
        stratum_of: Callable[[Item], Stratum],
        keep: int,
        rng: random.Random,
    ) -> None:
        """
        Set what the drawing answers to.

        Args:
            stratum_of: Where an item belongs.
            keep: How many to hold on to per stratum.
            rng: What the drawing answers to.
        """
        self._stratum_of: Callable[[Item], Stratum] = stratum_of
        self._keep: int = keep
        self._rng: random.Random = rng

        self._kept: defaultdict[Stratum, list[Item]] = defaultdict(list)
        self._seen: defaultdict[Stratum, int] = defaultdict(int)

    def offer(
        self,
        item: Item,
    ) -> None:
        """
        Put one item to the reservoir, which keeps it or lets it go.

        Args:
            item: The item going past.
        """
        stratum = self._stratum_of(item)
        self._seen[stratum] += 1

        kept = self._kept[stratum]

        if len(kept) < self._keep:
            kept.append(item)
            return

        index = self._rng.randrange(self._seen[stratum])
        if index < self._keep:
            kept[index] = item

    def draw(
        self,
        count: int,
    ) -> list[Item]:
        """
        Spread a count as evenly over the strata as they allow.

        What does not divide goes to the first strata in order, and a stratum
        with too little in it hands over what it has, the shortfall being
        left to the report rather than made up elsewhere.

        Args:
            count: How many to draw in all.

        Returns:
            The sample, shuffled so that no stratum is read in a block.
        """
        order = sorted(self._kept)
        quota, remainder = divmod(count, len(order) or 1)

        drawn: list[Item] = []
        for position, stratum in enumerate(order):
            available = self._kept[stratum]
            wanted = quota + (1 if position < remainder else 0)

            drawn.extend(self._rng.sample(available, min(wanted, len(available))))

        self._rng.shuffle(drawn)

        return drawn

    def offered(
        self,
    ) -> int:
        """How many items went past, kept or not."""
        return sum(self._seen.values())

    def strata(
        self,
    ) -> int:
        """How many strata the drawing spread over."""
        return len(self._kept)


def walk(
    entries: Iterable[Entry],
) -> Iterator[Occurrence]:
    """
    Read one item per sentence off a pass over the export.

    Args:
        entries: The export to read.

    Yields:
        Every sentence, whether or not the headword was found in it.
    """
    for entry in entries:
        lemma, pos = entry["lemma"], entry["pos"]
        url = ENTRY_URL.format(lemma=quote(lemma.replace(" ", "_")))

        for sense in entry.get("senses", []):
            gloss = SEPARATOR.join(sense["glosses"])

            for index, sentence in enumerate(sense.get("sentences", [])):
                yield Occurrence(
                    f"{sense['id']}.{index}",
                    lemma,
                    pos,
                    url,
                    gloss,
                    sentence["text"],
                    tuple(
                        (start, end) for start, end in sentence.get("word_offsets", [])
                    ),
                )


def mark(
    occurrence: Occurrence,
) -> str:
    """
    Wrap every occurrence found, so the annotator reads which spans are meant.

    The sentence is escaped piece by piece rather than whole, escaping being
    what changes a length and offsets being read off the text as it stands.

    Args:
        occurrence: The sentence to mark.

    Returns:
        The sentence as markup, its occurrences wrapped.
    """
    opening, closing = MARKERS
    text = occurrence.text

    pieces: list[str] = []
    read_up_to = 0

    for start, end in occurrence.offsets:
        pieces.append(escape(text[read_up_to:start], quote=False))
        pieces.append(f"{opening}{escape(text[start:end], quote=False)}{closing}")
        read_up_to = end

    pieces.append(escape(text[read_up_to:], quote=False))

    return "".join(pieces)


def lay_out(
    occurrences: Iterable[Occurrence],
) -> list[Task]:
    """
    Turn the sample into what Label Studio is handed.

    Args:
        occurrences: The sample to lay out.

    Returns:
        The tasks.
    """
    return [
        {
            "data": {
                "item_id": occurrence.id,
                "headword": f"{occurrence.lemma} ({occurrence.pos})",
                "url": occurrence.url,
                "gloss": occurrence.gloss,
                "marked": mark(occurrence),
                "found": len(occurrence.offsets),
            }
        }
        for occurrence in occurrences
    ]


def write_json(
    path: Path,
    payload: object,
) -> None:
    """
    Write one JSON file, its parent made if need be.

    Args:
        path: Where it goes.
        payload: What to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def write_sample(
    path: Path,
    occurrences: Iterable[Occurrence],
) -> None:
    """
    Write the sample itself, which is what a rerun is checked against.

    Args:
        path: Where it goes.
        occurrences: The sample to write.
    """
    with path.open("w", encoding="utf-8") as file:
        for occurrence in occurrences:
            record = json.dumps(asdict(occurrence), ensure_ascii=False)

            _ = file.write(f"{record}\n")


class Arguments(argparse.Namespace):
    """
    What the command line settles.

    Attributes:
        seed: What the drawing answers to.
        count: How many sentences the study draws.
    """

    seed: int = SEED
    count: int = COUNT


def read_arguments() -> Arguments:
    """
    Read what the command line settles, falling back on the defaults above.

    A namespace carrying the defaults is handed to the parser, which leaves
    alone whatever it already holds, so each default is written once.

    Returns:
        The seed and the size of the sample.
    """
    parser = argparse.ArgumentParser(description="Build the annotation tasks.")

    _ = parser.add_argument(
        "--seed",
        type=int,
        help=f"what the drawing answers to (default: {SEED})",
    )
    _ = parser.add_argument(
        "--count",
        type=int,
        help=f"how many sentences the study draws (default: {COUNT})",
    )

    return parser.parse_args(namespace=Arguments())


def main() -> None:
    """
    Draw the sample and lay it out for Label Studio.
    """
    arguments = read_arguments()

    here = Path(__file__).resolve().parent
    rng = random.Random(arguments.seed)

    reservoir: Reservoir[Occurrence] = Reservoir(
        lambda occurrence: (occurrence.pos, bool(occurrence.offsets)),
        arguments.count,
        rng,
    )

    for occurrence in walk(read()):
        reservoir.offer(occurrence)

    sample = reservoir.draw(arguments.count)

    write_sample(here / "offsets.jsonl", sample)
    write_json(here / "tasks" / "offsets.json", lay_out(sample))

    spread = f"over {reservoir.strata()} strata"

    print(  # noqa: T201
        f"{len(sample)} drawn {spread}, out of {reservoir.offered():,} offered"
    )


if __name__ == "__main__":
    main()
