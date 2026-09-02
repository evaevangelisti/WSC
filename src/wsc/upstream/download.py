"""
Retrieval of remote files.
"""

from collections.abc import Iterator
from pathlib import Path

import requests
from tqdm import tqdm


def download(
    url: str,
    output_path: Path,
    user_agent: str,
    timeout: tuple[int, int],
    chunk_size: int,
) -> None:
    """
    Download a file, resuming a previous attempt when one is found.

    Bytes go to a sibling .part file, renamed into place once the transfer
    completes. A failed attempt leaves it behind: that is what resumes.

    Args:
        url: The file to download.
        output_path: Where the finished file is placed.
        user_agent: How the client names itself to the server.
        timeout: Connect and read timeouts, in seconds.
        chunk_size: How much of the response to read at a time.

    Raises:
        requests.RequestException: If the transfer fails.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_name(f"{output_path.name}.part")

    downloaded_bytes = partial_path.stat().st_size if partial_path.exists() else 0

    headers = {"User-Agent": user_agent}
    if downloaded_bytes:
        headers["Range"] = f"bytes={downloaded_bytes}-"

    with requests.get(url, headers=headers, timeout=timeout, stream=True) as response:
        response.raise_for_status()

        # 200 rather than 206 means the range was ignored.
        if response.status_code == 200:
            downloaded_bytes = 0

        # A 206 reports the requested range alone, so the whole file is what
        # came before plus it.
        content_length = int(response.headers.get("content-length") or 0)
        total_bytes = downloaded_bytes + content_length if content_length else None

        with (
            partial_path.open("ab" if downloaded_bytes else "wb") as file,
            tqdm(
                desc=output_path.name,
                total=total_bytes,
                unit="B",
                unit_scale=True,
                initial=downloaded_bytes,
            ) as pbar,
        ):
            # iter_content is typed loosely.
            chunks: Iterator[bytes] = response.iter_content(chunk_size=chunk_size)

            for chunk in chunks:
                _ = pbar.update(file.write(chunk))

    _ = partial_path.replace(output_path)
