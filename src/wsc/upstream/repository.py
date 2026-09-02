"""
What Wikimedia's dump repository holds, and where.
"""

import re
from typing import TypedDict, cast

import requests

from ..constants import DUMP_INDEX_URL, DUMP_STATUS_URL, DUMP_URL

# The index lists one directory per dump, named after the day it began.
_DATE_PATTERN = re.compile(r'href="(\d{8})/"')

# The job recombining the split page files into the single archive we want.
_JOB = "articlesdumprecombine"


# The two classes below name the slice of the status file this module reads.
# Every key is optional, since it describes someone else's JSON.


class _Job(TypedDict, total=False):
    """
    One step of building a dump.

    Attributes:
        status: How far along that step is.
    """

    status: str


class _Status(TypedDict, total=False):
    """
    What a dump reports about its own building.

    Attributes:
        jobs: Every step of the build, by name.
    """

    jobs: dict[str, _Job]


def url(
    language: str,
    date: str,
) -> str:
    """
    Name the archive of pages for one dump.

    Args:
        language: Wiktionary's code for the edition, such as en.
        date: The day the dump began, as 20260801.

    Returns:
        The address to download it from.
    """
    return DUMP_URL.format(language=language, date=date)


def latest_date(
    language: str,
    user_agent: str,
    timeout: tuple[int, int],
) -> str:
    """
    Find the most recent dump that has finished being built.

    The newest directory is not the answer: a dump still under way has one
    too. The one to take is the newest reporting its pages as done.

    Args:
        language: Wiktionary's code for the edition, such as en.
        user_agent: How the client names itself to the server.
        timeout: Connect and read timeouts, in seconds.

    Returns:
        The day that dump began, as 20260801.

    Raises:
        requests.RequestException: If the index cannot be read.
        RuntimeError: If no dump has finished being built.
    """
    headers = {"User-Agent": user_agent}

    response = requests.get(
        DUMP_INDEX_URL.format(language=language),
        headers=headers,
        timeout=timeout,
    )
    response.raise_for_status()

    # findall is typed loosely; one group means one string per match.
    dates: list[str] = _DATE_PATTERN.findall(response.text)

    for date in sorted(set(dates), reverse=True):
        status = requests.get(
            DUMP_STATUS_URL.format(language=language, date=date),
            headers=headers,
            timeout=timeout,
        )
        if not status.ok:
            continue

        try:
            report = cast(_Status, status.json())
        except ValueError:
            # An error page answered with a 200 is not a report.
            continue

        if report.get("jobs", {}).get(_JOB, {}).get("status") == "done":
            return date

    raise RuntimeError(f"No finished {language}wiktionary dump to be found")
