"""
Building the Label Studio tasks for the word offset study.

One export at a time: the tasks are named after it, so judging a second run
is building a second pass rather than mixing the two.
"""

import argparse
import json
import random
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from html import escape
from pathlib import Path

# Run as a module from research/, where corpus sits beside the packages
# reading it.
from corpus import DATA, Entry, read

HERE = Path(__file__).resolve().parent

SEED = 0
"""What the drawing and the shuffling answer to by default."""

COUNT = 100
"""How many sentences are drawn."""

EXPORT = "spacy.jsonl"
"""Which export is judged unless another is named."""

REPEATED_SHARE = 0.1
"""What part of the pass comes round again, to measure agreement with oneself."""

MARKERS = ("<b>", "</b>")
"""What an occurrence is wrapped in. Label Studio renders the sentence as
markup, so the sentence is escaped before the tags go in."""

PARTS_OF_SPEECH = ("noun", "name", "verb", "adj", "adv")
"""What the collector keeps, in the order a grammar names them."""

type Stratum = tuple[str, bool]
"""A part of speech, and whether the export marked the sentence."""

type Task = dict[str, dict[str, object]]
"""One task, as Label Studio reads it: everything shown sits under data."""


@dataclass(frozen=True, slots=True)
class Sentence:
    """
    One sentence of an export, and where it put the lemma.

    Attributes:
        key: Names the sentence, as bank.noun.3f9c1a2b#0.
        lemma: The headword the sentence was collected under.
        pos: Its part of speech.
        text: The sentence.
        offsets: Where the export marked the lemma, leftmost first.
    """

    key: str
    lemma: str
    pos: str
    text: str
    offsets: tuple[tuple[int, int], ...]


def walk(
    entries: Iterable[Entry],
) -> Iterator[Sentence]:
    """
    Read every sentence of an export, keyed so exports can be lined up.

    Args:
        entries: The export to read.

    Yields:
        One sentence per attestation, in the order the export writes them.
    """
    for entry in entries:
        for sense in entry.get("senses", []):
            for position, sentence in enumerate(sense.get("sentences", [])):
                yield Sentence(
                    key=f"{sense['id']}#{position}",
                    lemma=entry["lemma"],
                    pos=entry["pos"],
                    text=sentence["text"],
                    offsets=tuple(
                        (start, end) for start, end in sentence.get("word_offsets", [])
                    ),
                )


def reserve(
    sentences: Iterable[Sentence],
    keep: int,
    rng: random.Random,
) -> dict[Stratum, list[Sentence]]:
    """
    Draw up to a fixed number per stratum in one pass over an export.

    An export runs to hundreds of megabytes, so it is never held whole, and
    every sentence stands an equal chance of being kept.

    Args:
        sentences: Everything the export offers.
        keep: How many to hold on to per stratum.
        rng: What the drawing answers to.

    Returns:
        What was kept, by stratum.
    """
    reserved: defaultdict[Stratum, list[Sentence]] = defaultdict(list)
    sentences_seen: defaultdict[Stratum, int] = defaultdict(int)

    for sentence in sentences:
        stratum = (sentence.pos, bool(sentence.offsets))
        sentences_seen[stratum] += 1

        if len(reserved[stratum]) < keep:
            reserved[stratum].append(sentence)
            continue

        index = rng.randrange(sentences_seen[stratum])
        if index < keep:
            reserved[stratum][index] = sentence

    return reserved


def balance(
    pools: dict[Stratum, list[Sentence]],
    count: int,
    rng: random.Random,
) -> list[Sentence]:
    """
    Spread a count as evenly as the strata allow.

    A sentence the export left unmarked is rare, so drawing evenly is what
    gives the misses enough weight to be read.

    Args:
        pools: What each stratum has to offer.
        count: How many to draw in all.
        rng: What the drawing answers to.

    Returns:
        The sample, walked stratum by stratum.
    """
    order = sorted(
        pools,
        key=lambda stratum: (PARTS_OF_SPEECH.index(stratum[0]), stratum[1]),
    )
    quota, remainder = divmod(count, len(order) or 1)

    drawn: list[Sentence] = []
    for position, stratum in enumerate(order):
        wanted = quota + (1 if position < remainder else 0)

        drawn.extend(rng.sample(pools[stratum], min(wanted, len(pools[stratum]))))

    return drawn


def mark(
    sentence: Sentence,
) -> str:
    """
    Wrap every occurrence the export found, so the reader sees what it marked.

    The sentence is escaped piece by piece rather than whole, escaping being
    what changes a length and offsets being read off the text as it stands.

    Args:
        sentence: The sentence and where its lemma was put.

    Returns:
        The sentence as markup, its occurrences wrapped.
    """
    opening, closing = MARKERS

    pieces: list[str] = []
    read_up_to = 0

    for start, end in sentence.offsets:
        pieces.append(escape(sentence.text[read_up_to:start], quote=False))
        pieces.append(
            f"{opening}{escape(sentence.text[start:end], quote=False)}{closing}"
        )
        read_up_to = end

    pieces.append(escape(sentence.text[read_up_to:], quote=False))

    return "".join(pieces)


def lay_out(
    drawn: list[Sentence],
    source: str,
    rng: random.Random,
) -> list[Task]:
    """
    Turn the sample into what the pass is handed.

    The source travels with each task and is never shown, so that several
    passes may sit in one project without saying which run made which mark.

    Args:
        drawn: The sentences drawn, in the order they were drawn.
        source: The export they were drawn from.
        rng: What the shuffling answers to.

    Returns:
        The tasks, shuffled, a tenth of them coming round again.
    """
    tasks: list[Task] = [
        {
            "data": {
                "item_id": sentence.key,
                "source": source,
                "lemma": sentence.lemma,
                "pos": sentence.pos,
                "target": f"{sentence.lemma} ({sentence.pos})",
                "marked": mark(sentence),
            }
        }
        for sentence in drawn
    ]

    tasks += rng.sample(tasks, int(len(tasks) * REPEATED_SHARE))

    rng.shuffle(tasks)

    return tasks


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


class Arguments(argparse.Namespace):
    """
    What the command line settles.

    Attributes:
        export: Which export the sentences are drawn from.
        seed: What the drawing and the shuffling answer to.
        count: How many sentences are drawn.
    """

    export: Path = DATA / EXPORT
    seed: int = SEED
    count: int = COUNT


def read_arguments() -> Arguments:
    """
    Read what the command line settles, falling back on the defaults above.

    Returns:
        The export to draw from, the seed, and the size of the sample.
    """
    parser = argparse.ArgumentParser(description="Build the word offset tasks.")

    _ = parser.add_argument(
        "export",
        nargs="?",
        type=Path,
        default=DATA / EXPORT,
        help=f"the export to draw from (default: {DATA / EXPORT})",
    )
    _ = parser.add_argument(
        "--seed",
        type=int,
        help=f"what the drawing and the shuffling answer to (default: {SEED})",
    )
    _ = parser.add_argument(
        "--count",
        type=int,
        help=f"how many sentences are drawn (default: {COUNT})",
    )

    return parser.parse_args(namespace=Arguments())


def main() -> None:
    """
    Draw the sentences of one export and lay them out as a pass.
    """
    arguments = read_arguments()

    rng = random.Random(arguments.seed)

    keep = -(-arguments.count * 2 // (len(PARTS_OF_SPEECH) * 2))
    pools = reserve(walk(read(arguments.export)), keep, rng)

    drawn = balance(pools, arguments.count, rng)
    tasks = lay_out(drawn, arguments.export.stem, rng)

    written = HERE / "tasks" / f"{arguments.export.stem}.json"
    write_json(written, tasks)

    print(f"{len(drawn)} sentences, {len(tasks)} tasks -> {written}")  # noqa: T201


if __name__ == "__main__":
    main()
