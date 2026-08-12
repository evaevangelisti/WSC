"""
Parsing of a Wiktionary dump with wiktextract.
"""

import subprocess
from compression import zstd
from pathlib import Path
from typing import IO

from tqdm import tqdm

from ..constants import COMPRESSION_LEVEL


def parse(
    dump_path: Path,
    output_path: Path,
    language: str,
    processes: int,
) -> int:
    """
    Turn a Wiktionary dump into the compressed JSONL wiktextract makes of it.

    Entries go to a sibling .part file, renamed into place once parsing
    completes, so a run cut short leaves nothing that passes for finished.
    Far more than a few hundred lines set aside means something went wrong.

    Args:
        dump_path: The Wiktionary dump to read.
        output_path: Where the compressed JSONL is placed.
        language: Wiktionary's code for the edition and the language to keep.
        processes: How many processes wiktextract may run, at 4 GB each.

    Returns:
        How many lines of wiktextract's own reporting were set aside.

    Raises:
        subprocess.CalledProcessError: If wiktextract answers with an error.
        RuntimeError: If its output cannot be read.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_name(f"{output_path.name}.part")

    command = [
        "wiktwords",
        "--out",
        "-",
        "--edition",
        language,
        "--language-code",
        language,
        "--examples",
        "--num-processes",
        str(processes),
        str(dump_path),
    ]

    skipped_lines = 0

    try:
        with (
            zstd.open(partial_path, "wb", level=COMPRESSION_LEVEL) as file,
            subprocess.Popen(command, stdout=subprocess.PIPE) as process,
        ):
            if process.stdout is None:
                raise RuntimeError("wiktextract offered no output to read")

            # Popen carries bytes, but typeshed leaves its streams loosely typed.
            stdout: IO[bytes] = process.stdout

            with tqdm(desc=dump_path.name, unit=" entry") as pbar:
                for line in stdout:
                    # wiktextract reports itself down the same stream as the entries,
                    # so what does not open an object is not one.
                    if not line.startswith(b"{"):
                        skipped_lines += 1
                        continue

                    _ = file.write(line)
                    _ = pbar.update(1)

            return_code = process.wait()
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, command)
    except BaseException:
        partial_path.unlink(missing_ok=True)
        raise

    _ = partial_path.replace(output_path)

    return skipped_lines
