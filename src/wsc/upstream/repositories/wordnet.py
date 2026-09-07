"""What the wordnet's repository holds, and where."""

import re

import requests

from ...constants import WORDNET_INDEX_URL, WORDNET_URL

_VERSION_PATTERN = re.compile(r'href="[^"]*english-wordnet-(\d{4})\.xml\.gz"')


def url(
    version: str,
) -> str:
    """
    Name the file one edition of the wordnet is published in.

    Args:
        version: The edition, as 2025.

    Returns:
        The address to download it from.
    """
    return WORDNET_URL.format(version=version)


def latest_version(
    user_agent: str,
    timeout: tuple[int, int],
) -> str:
    """
    Find the most recent edition of the wordnet.

    Published editions contain complete WordNet releases.

    Args:
        user_agent: How the client names itself to the server.
        timeout: Connect and read timeouts, in seconds.

    Returns:
        The edition, as 2025.

    Raises:
        requests.RequestException: If the index cannot be read.
        RuntimeError: If it lists no edition.
    """
    response = requests.get(
        WORDNET_INDEX_URL,
        headers={"User-Agent": user_agent},
        timeout=timeout,
    )
    response.raise_for_status()

    versions: list[str] = _VERSION_PATTERN.findall(response.text)

    if not versions:
        raise RuntimeError("No wordnet edition to be found")

    return max(versions)
