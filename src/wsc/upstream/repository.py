"""What Wikimedia's dump repository holds, and where."""

import re
from typing import TypedDict, cast

import requests

from ..constants import DUMP_INDEX_URL, DUMP_STATUS_URL, DUMP_URL

_DATE_PATTERN = re.compile(r'href="(\d{8})/"')

_JOB = "articlesdumprecombine"


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
    date: str,
) -> str:
    """
    Name the archive of pages for one dump.

    Args:
        date: The day the dump began, as 20260801.

    Returns:
        The address to download it from.
    """
    return DUMP_URL.format(date=date)


def latest_date(
    user_agent: str,
    timeout: tuple[int, int],
) -> str:
    """
    Find the most recent dump that has finished being built.

    Choose the newest dump whose page archive is complete.

    Args:
        user_agent: How the client names itself to the server.
        timeout: Connect and read timeouts, in seconds.

    Returns:
        The day that dump began, as 20260801.

    Raises:
        requests.RequestException: If the index cannot be read.
        RuntimeError: If no dump has finished being built.
    """
    headers = {"User-Agent": user_agent}

    response = requests.get(DUMP_INDEX_URL, headers=headers, timeout=timeout)
    response.raise_for_status()

    dates: list[str] = _DATE_PATTERN.findall(response.text)

    for date in sorted(set(dates), reverse=True):
        status = requests.get(
            DUMP_STATUS_URL.format(date=date),
            headers=headers,
            timeout=timeout,
        )

        if not status.ok:
            continue

        try:
            report = cast(_Status, status.json())
        except ValueError:
            # Servers can return an HTML error page with status 200.
            continue

        if report.get("jobs", {}).get(_JOB, {}).get("status") == "done":
            return date

    raise RuntimeError("No finished dump to be found")
