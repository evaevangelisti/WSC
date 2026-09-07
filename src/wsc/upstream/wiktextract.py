"""Parsing of a Wiktionary dump with wiktextract, and taking a parse already made."""

import json
import subprocess
from collections.abc import Callable, Iterable
from compression import zstd
from pathlib import Path
from typing import IO, cast

from tqdm import tqdm

from ..constants import COMPRESSION_LEVEL, LANGUAGE
from ..files import open_compressed, partial_file

_READING = "Reading the dump"
_NARROWING = "Narrowing the extraction"

type Narrow = Callable[[object], object]
"""Cuts one entry down to the fields that will be read of it."""


def _write(
    lines: Iterable[bytes],
    partial_path: Path,
    narrow: Narrow,
    description: str,
) -> tuple[int, int]:
    """
    Cut every entry down and write one JSON object per line.

    Args:
        lines: The extraction, one line at a time.
        partial_path: Where the compressed JSONL is written.
        narrow: Cuts one entry down to the fields that will be read of it.
        description: What the progress bar says the pass is doing.

    Returns:
        How many lines were set aside, and how many entries were written.
    """
    skipped_lines = 0
    written_entries = 0

    with (
        zstd.open(partial_path, "wb", level=COMPRESSION_LEVEL) as file,
        tqdm(desc=description, unit=" entry") as pbar,
    ):
        for line in lines:
            # Wiktextract interleaves reporting messages with JSON entries.
            if not line.startswith(b"{"):
                skipped_lines += 1
                continue

            try:
                entry = cast(object, json.loads(line))
            except json.JSONDecodeError:
                # Interrupted extraction can leave an incomplete JSON entry.
                skipped_lines += 1
                continue

            written = json.dumps(narrow(entry), ensure_ascii=False)
            _ = file.write(f"{written}\n".encode())

            written_entries += 1

            _ = pbar.update(1)

    return skipped_lines, written_entries


def parse(
    dump_path: Path,
    output_path: Path,
    processes: int,
    narrow: Narrow,
    database_path: Path | None = None,
) -> int:
    """
    Turn a Wiktionary dump into the compressed JSONL wiktextract makes of it.

    Args:
        dump_path: The Wiktionary dump to read.
        output_path: Where the compressed JSONL is placed.
        processes: How many processes wiktextract may run, at 4 GB each.
        narrow: Cuts one entry down to the fields that will be read of it.
        database_path: Existing page database or None for a temporary database.

    Returns:
        Number of discarded wiktextract reporting lines.

    Raises:
        subprocess.CalledProcessError: If wiktextract answers with an error.
        RuntimeError: If its output cannot be read, or holds no entry.
    """
    command = [
        "wiktwords",
        "--out",
        "-",
        "--edition",
        LANGUAGE,
        "--language-code",
        LANGUAGE,
        "--examples",
        "--translations",
        "--linkages",
        "--etymologies",
        "--quiet",
        "--num-processes",
        str(processes),
    ]

    if database_path is not None:
        command += ["--db-path", str(database_path)]

    command.append(str(dump_path))

    with (
        partial_file(output_path) as partial_path,
        subprocess.Popen(command, stdout=subprocess.PIPE) as process,
    ):
        if process.stdout is None:
            raise RuntimeError("wiktextract offered no output to read")

        stdout: IO[bytes] = process.stdout

        skipped_lines, written_entries = _write(
            stdout,
            partial_path,
            narrow,
            _READING,
        )

        return_code = process.wait()

        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)

        if not written_entries:
            raise RuntimeError("wiktextract wrote no entry")

    return skipped_lines


def narrow_file(
    input_path: Path,
    output_path: Path,
    narrow: Narrow,
) -> int:
    """
    Cut a published extraction down to the fields the collector reads.

    Args:
        input_path: The published extraction, compressed or not.
        output_path: Where the compressed JSONL is placed.
        narrow: Cuts one entry down to the fields that will be read of it.

    Returns:
        How many lines were set aside as holding no entry.

    Raises:
        RuntimeError: If the extraction holds no entry.
    """
    with partial_file(output_path) as partial_path:
        with open_compressed(input_path, "rb") as file:
            skipped_lines, written_entries = _write(
                file,
                partial_path,
                narrow,
                _NARROWING,
            )

        if not written_entries:
            raise RuntimeError(f"No entry in {input_path}")

    return skipped_lines
