"""Build offset judgements for identical sentences across extraction engines."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from html import escape
from pathlib import Path

from corpus import Entry, read

from .sampling import EXPORTS, repeat_items, sample_items, write_json


@dataclass(frozen=True, slots=True)
class Sentence:
    """
    Shared sentence identity and one engine's candidate offsets.

    Attributes:
        key: Sense identifier followed by attestation position.
        lemma: Collected headword.
        pos: Part of speech.
        text: Unmodified sentence text.
        offsets: Candidate ranges with their supporting sources.
    """

    key: str
    lemma: str
    pos: str
    text: str
    offsets: tuple[tuple[int, int, tuple[str, ...]], ...]


def walk(entries: Iterable[Entry]) -> Iterator[Sentence]:
    """
    Enumerate attestations with identities shared across extraction engines.

    Args:
        entries: Collected corpus entries.

    Yields:
        Sentence text and source-aware candidate ranges.
    """
    for entry in entries:
        for sense in entry.get("senses", []):
            for position, sentence in enumerate(sense.get("sentences", [])):
                yield Sentence(
                    f"{sense['id']}#{position}",
                    entry["lemma"],
                    entry["pos"],
                    sentence["text"],
                    tuple(
                        (item["offset"][0], item["offset"][1], tuple(item["sources"]))
                        for item in sentence.get("word_offsets", [])
                    ),
                )


def mark(sentence: Sentence) -> str:
    """
    Highlight the union of proposals without duplicating overlapping text.

    Args:
        sentence: Text and possibly overlapping offset proposals.

    Returns:
        Escaped HTML with the proposed character union in bold.
    """
    ranges: list[tuple[int, int]] = []

    for start, end, _ in sorted(sentence.offsets):
        if ranges and start <= ranges[-1][1]:
            ranges[-1] = (ranges[-1][0], max(ranges[-1][1], end))
        else:
            ranges.append((start, end))

    pieces: list[str] = []
    position = 0

    for start, end in ranges:
        pieces.extend(
            (
                escape(sentence.text[position:start]),
                f"<b>{escape(sentence.text[start:end])}</b>",
            )
        )
        position = end

    pieces.append(escape(sentence.text[position:]))

    return "".join(pieces)


def build_offsets(data_dir: Path, output_dir: Path, count: int, seed: int) -> None:
    """
    Write common offset samples and separate second-annotator assignments.

    Args:
        data_dir: Directory containing all three collected engine exports.
        output_dir: Directory receiving engine-specific task files.
        count: Number of shared sentences.
        seed: Sampling and duplicate-selection seed.

    Raises:
        ValueError: If an engine lacks a sampled sentence or changes its identity.
    """
    reference = sample_items(walk(read(data_dir / EXPORTS[0])), count, seed)
    identities = {sentence.key: sentence for sentence in reference}

    for export in EXPORTS:
        selected = (
            identities
            if export == EXPORTS[0]
            else {
                sentence.key: sentence
                for sentence in walk(read(data_dir / export))
                if sentence.key in identities
            }
        )

        if selected.keys() != identities.keys():
            raise ValueError(f"{export} lacks sampled sentences")

        tasks: list[dict[str, object]] = []

        for original in reference:
            sentence = selected[original.key]

            if (sentence.text, sentence.lemma, sentence.pos) != (
                original.text,
                original.lemma,
                original.pos,
            ):
                raise ValueError(
                    f"Sentence identity differs in {export}: {sentence.key}"
                )

            tasks.append(
                {
                    "data": {
                        "task": "offsets",
                        "item_id": sentence.key,
                        "source": Path(export).stem,
                        "target": f"{sentence.lemma} ({sentence.pos})",
                        "text": sentence.text,
                        "marked": mark(sentence),
                        "word_offsets": [
                            {"offset": [start, end], "sources": list(sources)}
                            for start, end, sources in sentence.offsets
                        ],
                        "seed": seed,
                    }
                }
            )

        stem = Path(export).stem
        write_json(output_dir / f"{stem}.json", tasks)
        write_json(output_dir / f"{stem}.second.json", repeat_items(tasks, seed))
