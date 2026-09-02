"""
Building the Label Studio tasks for the word offset study.

The same sentences are drawn once and shown as each export marked them, all
shuffled into one pass, so that no reading can tell which export it is judging.
"""

import argparse
import json
import random
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
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
"""How many sentences are drawn, each shown once per export."""

EXPORTS = ("spacy.jsonl", "stanza.jsonl", "lemminflect.jsonl")
"""Which exports are judged, the first of them settling the strata."""

REPEATED_SHARE = 0.1
"""What part of the pass comes round again, to measure agreement with oneself."""

MARKERS = ("<b>", "</b>")
"""What an occurrence is wrapped in. Label Studio renders the sentence as
markup, so the sentence is escaped before the tags go in."""

PARTS_OF_SPEECH = ("noun", "name", "verb", "adj", "adv")
"""What the collector keeps, in the order a grammar names them."""

type Stratum = tuple[str, bool]
"""A part of speech, and whether the reference export marked the sentence."""

type Task = dict[str, dict[str, object]]
"""One task, as Label Studio reads it: everything shown sits under data."""


@dataclass(frozen=True, slots=True)
class Sentence:
    """
    One sentence of one export, and where that export put the lemma.

    Attributes:
        key: Names the sentence across exports, as bank.noun.1.02#0.
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

    A sentence the reference export left unmarked is rare, so drawing evenly
    is what gives the misses enough weight to be read.

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


def collect(
    paths: list[Path],
    keys: set[str],
) -> dict[str, dict[str, Sentence]]:
    """
    Read back how each export marked the sentences drawn.

    Args:
        paths: The exports to read.
        keys: The sentences to pick up.

    Returns:
        The sentence each export holds under each key.
    """
    return {
        path.name: {
            sentence.key: sentence
            for sentence in walk(read(path))
            if sentence.key in keys
        }
        for path in paths
    }


def lay_out(
    drawn: list[Sentence],
    marked: dict[str, dict[str, Sentence]],
    rng: random.Random,
) -> tuple[list[Task], dict[str, str]]:
    """
    Turn the sample into what the pass is handed, and into the key.

    Args:
        drawn: The sentences drawn, in the order they were drawn.
        marked: How each export marked them.
        rng: What the shuffling answers to.

    Returns:
        The tasks, shuffled, and which export each one came from.
    """
    tasks: list[Task] = []
    key: dict[str, str] = {}

    for name, sentences in marked.items():
        for sentence in drawn:
            marked_sentence = sentences.get(sentence.key)
            if marked_sentence is None:
                continue

            item_id = f"{name}:{marked_sentence.key}"

            tasks.append(
                {
                    "data": {
                        "item_id": item_id,
                        "lemma": marked_sentence.lemma,
                        "pos": marked_sentence.pos,
                        "marked": mark(marked_sentence),
                    }
                }
            )
            key[item_id] = name

    repeated = rng.sample(tasks, int(len(tasks) * REPEATED_SHARE))
    tasks += repeated

    rng.shuffle(tasks)

    return tasks, key


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
        seed: What the drawing and the shuffling answer to.
        count: How many sentences are drawn.
        exports: Which exports are judged, the first settling the strata.
    """

    seed: int = SEED
    count: int = COUNT
    exports: Sequence[str] = EXPORTS


def read_arguments() -> Arguments:
    """
    Read what the command line settles, falling back on the defaults above.

    Returns:
        The seed, the size of the sample, and the exports to judge.
    """
    parser = argparse.ArgumentParser(description="Build the word offset tasks.")

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
    _ = parser.add_argument(
        "--exports",
        nargs="+",
        help=f"which exports to judge (default: {' '.join(EXPORTS)})",
    )

    return parser.parse_args(namespace=Arguments())


def main() -> None:
    """
    Draw the sentences, read how each export marked them, and lay out the pass.
    """
    arguments = read_arguments()

    rng = random.Random(arguments.seed)
    paths = [DATA / name for name in arguments.exports]

    keep = -(-arguments.count * 2 // (len(PARTS_OF_SPEECH) * 2))
    pools = reserve(walk(read(paths[0])), keep, rng)

    drawn = balance(pools, arguments.count, rng)
    marked = collect(paths, {sentence.key for sentence in drawn})

    tasks, key = lay_out(drawn, marked, rng)

    write_json(HERE / "offsets.json", tasks)
    write_json(HERE / "key.json", key)

    print(f"{len(drawn)} sentences, {len(tasks)} tasks over {len(paths)} exports")  # noqa: T201


if __name__ == "__main__":
    main()
