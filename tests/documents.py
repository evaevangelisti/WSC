"""
What Wikimedia publishes, written the way it writes it.

A document is laid out in full rather than cut down, so a pattern that fits
a shorter page is caught by the page it will meet.
"""

import json

_DUMP_INDEX = """<html>
<head><title>Index of /enwiktionary/</title></head>
<body><pre>
{links}</pre></body>
</html>
"""

_DUMP_LINK = '<a href="{date}/">{date}/</a>      01-Aug-2026 09:12       -\n'


def dump_index(
    *dates: str,
) -> str:
    """
    Write the listing Wikimedia serves for one edition.

    Args:
        dates: The days the dumps it holds began, one directory apiece.

    Returns:
        The page, as it is served.
    """
    links = "".join(_DUMP_LINK.format(date=date) for date in dates)

    return _DUMP_INDEX.format(links=links)


def dump_status(
    state: str,
) -> str:
    """
    Write what one dump reports about the job the collector waits on.

    Args:
        state: How far along that job is.

    Returns:
        The status file, as JSON.
    """
    return json.dumps(
        {
            "version": "0.8",
            "jobs": {
                "articlesdump": {"status": "done"},
                "articlesdumprecombine": {"status": state},
            },
        }
    )
