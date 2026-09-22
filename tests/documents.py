"""
What each source publishes, written the way that source writes it.

Complete source documents exercise realistic parsing boundaries.
"""

import json
from xml.sax.saxutils import escape

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
        },
    )


_DUMP = """<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/" version="0.11">
  <siteinfo>
    <sitename>Wiktionary</sitename>
  </siteinfo>
{pages}</mediawiki>
"""

_DUMP_PAGE = """  <page>
    <title>{title}</title>
    <ns>{namespace}</ns>
    <id>{identifier}</id>
    <revision>
      <id>{identifier}</id>
      <text xml:space="preserve">{markup}</text>
    </revision>
  </page>
"""


def page(
    title: str,
    markup: str,
    namespace: int = 0,
    identifier: int = 1,
) -> str:
    """
    Write one page of a dump, in the markup an editor wrote it in.

    Args:
        title: What the page is called.
        markup: What it says.
        namespace: Which namespace holds it, 0 being the articles.
        identifier: What the wiki numbers it.

    Returns:
        The page element.
    """
    return _DUMP_PAGE.format(
        title=escape(title),
        namespace=namespace,
        identifier=identifier,
        markup=escape(markup),
    )


def dump(
    *pages: str,
) -> str:
    """
    Write a dump holding the pages given.

    Args:
        pages: The page elements, as page writes them.

    Returns:
        The document, as Wikimedia publishes one.
    """
    return _DUMP.format(pages="".join(pages))
