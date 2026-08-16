"""
What each source publishes, written the way that source writes it.

A document is laid out here in full rather than cut down to the part being
read, so that a pattern which happens to fit a shorter page is caught by the
page it will actually meet. Whatever is written into one is escaped on the
way in, an attribute carrying its own quotes, so that a value holding markup
lands as a value rather than as markup.
"""

import json
from collections.abc import Iterable
from xml.sax.saxutils import escape, quoteattr

_DUMP_INDEX = """<html>
<head><title>Index of /enwiktionary/</title></head>
<body><pre>
{links}</pre></body>
</html>
"""

_DUMP_LINK = '<a href="{date}/">{date}/</a>      01-Aug-2026 09:12       -\n'

_WORDNET_INDEX = """<html>
<body>
<ul>
{links}</ul>
</body>
</html>
"""

_WORDNET_LINK = (
    '<li><a href="/downloads/english-wordnet-{version}.{suffix}">{suffix}</a></li>\n'
)

# Every edition is offered in each of these, and one of them is read.
_WORDNET_SUFFIXES = ("ttl.gz", "xml.gz", "zip")

_LEXICON = """<?xml version="1.0" encoding="UTF-8"?>
<LexicalResource>
  <Lexicon id="oewn" label="Open English WordNet" language="en" version="2025">{body}
  </Lexicon>
</LexicalResource>
"""

_LEXICAL_ENTRY = """
    <LexicalEntry id={id}>
      <Lemma writtenForm={written_form} partOfSpeech={pos}/>
    </LexicalEntry>"""

_SYNSET = """
    <Synset id={id} ili={ili} partOfSpeech={pos} members={members}>
      <Definition>{definition}</Definition>{body}
    </Synset>"""

_EXAMPLE = """
      <Example>{text}</Example>"""

_RELATION = """
      <SynsetRelation relType={rel_type} target={target}/>"""


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


def wordnet_index(
    *versions: str,
) -> str:
    """
    Write the listing the wordnet serves for its editions.

    Args:
        versions: The editions it holds, each offered in every format.

    Returns:
        The page, as it is served.
    """
    links = "".join(
        _WORDNET_LINK.format(version=version, suffix=suffix)
        for version in versions
        for suffix in _WORDNET_SUFFIXES
    )

    return _WORDNET_INDEX.format(links=links)


def lexical_entry(
    identifier: str = "oewn-bank-n",
    written_form: str = "bank",
    pos: str = "n",
) -> str:
    """
    Write one lexical entry, which is where a written form is spelled out.

    Args:
        identifier: What the synsets naming it as a member refer to.
        written_form: The word itself.
        pos: WordNet's code for its part of speech.

    Returns:
        The element, to be placed before the synsets.
    """
    return _LEXICAL_ENTRY.format(
        id=quoteattr(identifier),
        written_form=quoteattr(written_form),
        pos=quoteattr(pos),
    )


def synset(
    identifier: str = "oewn-08420278-n",
    pos: str = "n",
    members: Iterable[str] = ("oewn-bank-n",),
    definition: str = "a financial institution.",
    ili: str = "i54321",
    examples: Iterable[str] = (),
    relations: Iterable[tuple[str, str]] = (),
) -> str:
    """
    Write one synset.

    Args:
        identifier: What an alignment will record.
        pos: WordNet's code for its part of speech.
        members: The lexical entries expressing it, by identifier.
        definition: The gloss WordNet writes for it.
        ili: The interlingual index naming the same meaning elsewhere.
        examples: The sentences to hang off it.
        relations: The relations to hang off it, each a type and a target.

    Returns:
        The element, to be placed after the entries it names.
    """
    body = "".join(
        [
            *(
                _RELATION.format(
                    rel_type=quoteattr(rel_type),
                    target=quoteattr(target),
                )
                for rel_type, target in relations
            ),
            *(_EXAMPLE.format(text=escape(text)) for text in examples),
        ]
    )

    return _SYNSET.format(
        id=quoteattr(identifier),
        ili=quoteattr(ili),
        pos=quoteattr(pos),
        members=quoteattr(" ".join(members)),
        definition=escape(definition),
        body=body,
    )


def lexicon(
    *elements: str,
) -> str:
    """
    Write a wordnet around the elements it holds, in WN-LMF.

    Args:
        elements: The entries and synsets, entries first.

    Returns:
        The document, as the release publishes it.
    """
    return _LEXICON.format(body="".join(elements))
