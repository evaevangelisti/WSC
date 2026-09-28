"""
Retain mathematical source and score context lost during Wiktextract parsing.
"""

import json
import re
import sys
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from functools import partial
from io import StringIO
from pathlib import Path
from typing import cast

from wiktextract.clean import to_math

from ...files import open_compressed
from ...models import Attestation
from ..offsets import substitute
from .markup import plain, section_lines

_MATH = re.compile(r"<math\b[^>]*>(.*?)</math\s*>", re.IGNORECASE | re.DOTALL)
_SCORE = re.compile(r"<score\b", re.IGNORECASE)
_EXAMPLE_MARKER = re.compile(r"^\s*#+[:*]\s*")
_LIST_MARKER = re.compile(r"^\s*#+[:*]?\s*")
_HEADING_TEMPLATE = re.compile(
    r"\{\{(?:trans-top(?:-also|-see)?|trans-see)\|",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class MathSource:
    """
    Retain one formula and its surrounding source text.

    Attributes:
        rendered: Display text produced by Wiktextract.
        source: Original TeX between math tags.
        prefix: Text before the formula in its source line.
        suffix: Text after the formula in its source line.
    """

    rendered: str
    source: str
    prefix: str
    suffix: str


@dataclass(frozen=True, slots=True)
class PageMarkup:
    """
    Describe source fragments that Wiktextract does not retain.

    Attributes:
        mathematics: Original formulae and their source contexts.
        score_prefixes: Text preceding score tags in example lines.
    """

    mathematics: tuple[MathSource, ...]
    score_prefixes: tuple[str, ...]


type MarkupIndex = dict[str, PageMarkup]


def unwrap_mathematics(
    text: str,
) -> tuple[str, tuple[str, ...]]:
    """
    Keep original TeX while removing math tags from dump text.

    Args:
        text: Wikitext containing zero or more math tags.

    Returns:
        Text with bare TeX and the original formulae it contains.
    """
    sources = tuple(match[1].strip() for match in _MATH.finditer(text))

    return _MATH.sub(lambda match: match[1].strip(), text), sources


def _render_mathematics(
    source: str,
) -> str:
    """
    Discard Wiktextract's spurious diagnostic for binomial notation.

    Args:
        source: Original TeX between math tags.

    Returns:
        Display text produced by Wiktextract.
    """
    if r"\binom" not in source:
        return to_math(source).strip()

    diagnostics = StringIO()

    with redirect_stdout(diagnostics):
        rendered = to_math(source).strip()

    unexpected = diagnostics.getvalue().replace(
        "MATH FRAC/BINOM ERROR: '\\\\binom'\n",
        "",
    )

    if unexpected:
        _ = sys.stdout.write(unexpected)

    return rendered


def read_page_markup(
    markup: str,
    language_section: str,
) -> PageMarkup | None:
    """
    Read mathematical source and score contexts from one language.

    Args:
        markup: Original page wikitext.
        language_section: Language heading to inspect.

    Returns:
        Source fragments, or None if the page has no relevant tags.
    """
    lines = [line for pos, line in section_lines(markup, language_section) if pos]
    text = "\n".join(lines)

    formulae: list[MathSource] = []

    for match in _MATH.finditer(text):
        source = match[1].strip()
        rendered = _render_mathematics(source)

        if source and rendered:
            line_start = text.rfind("\n", 0, match.start()) + 1

            line_end = text.find("\n", match.end())
            line_end = len(text) if line_end < 0 else line_end

            before = text[line_start : match.start()]
            after = text[match.end() : line_end]

            heading = next(iter(_HEADING_TEMPLATE.finditer(before)), None)

            if heading is not None:
                before = before[heading.end() :]
                after = after.split("|", 1)[0].split("}}", 1)[0]

            prefix = plain(_LIST_MARKER.sub("", before)).strip()
            suffix = plain(after).strip()

            formulae.append(MathSource(rendered, source, prefix, suffix))

    mathematics = tuple(dict.fromkeys(formulae))

    score_prefixes = tuple(
        dict.fromkeys(
            prefix
            for line in lines
            if (found := _SCORE.search(line)) is not None
            and _EXAMPLE_MARKER.match(line)
            if (prefix := plain(_EXAMPLE_MARKER.sub("", line[: found.start()])))
        )
    )

    if not mathematics and not score_prefixes:
        return None

    return PageMarkup(mathematics, score_prefixes)


def _recover_formula(
    match: re.Match[str],
    *,
    formulae: tuple[MathSource, ...],
) -> str:
    """
    Restore a formula only when its source context identifies it.

    Args:
        match: Candidate rendered formula in parsed text.
        formulae: Source candidates for that rendering.

    Returns:
        Unique matching TeX source, or unchanged display text.
    """
    preceding = match.string[: match.start()].rstrip()
    following = match.string[match.end() :].strip()

    sources = {
        formula.source
        for formula in formulae
        if (
            preceding == formula.prefix
            if formula.prefix
            else not preceding and following == formula.suffix
        )
    }

    return next(iter(sources)) if len(sources) == 1 else match[0]


def _restore_adjacent_mathematics(
    value: Attestation,
    mathematics: tuple[MathSource, ...],
) -> Attestation:
    """
    Separate a formula joined to the same surrounding words in the dump.

    Args:
        value: Parsed text and any known word offsets.
        mathematics: Source formulae with their surrounding text.

    Returns:
        Text with confirmed formula boundaries and original TeX.
    """
    for formula in mathematics:
        if "=" not in formula.source:
            continue

        before = re.search(r"[A-Za-z]{2,}$", formula.prefix)
        after = re.match(r"[A-Za-z]{2,}", formula.suffix)

        if before is None or after is None:
            continue

        prefix = re.escape(before[0])
        suffix = re.escape(after[0])
        expression = re.compile(
            rf"(?<={prefix}){re.escape(formula.rendered)}(?={suffix})"
        )
        source = formula.source.replace("\\", "\\\\")
        value = substitute(value, expression, f" {source} ")

    return value


def restore_mathematics(
    value: Attestation,
    mathematics: tuple[MathSource, ...],
) -> Attestation:
    """
    Restore unambiguous dump formulae in Wiktextract's display text.

    Args:
        value: Parsed text and any known word offsets.
        mathematics: Original formulae with their source contexts.

    Returns:
        Text with original LaTeX and relocated offsets.
    """
    value = _restore_adjacent_mathematics(value, mathematics)

    grouped: dict[str, list[MathSource]] = {}

    for formula in mathematics:
        grouped.setdefault(formula.rendered, []).append(formula)

    for rendered, formulae in sorted(grouped.items(), key=lambda item: -len(item[0])):
        expression = re.compile(rf"(?<!\w){re.escape(rendered)}(?!\w)")
        value = substitute(
            value,
            expression,
            partial(_recover_formula, formulae=tuple(formulae)),
        )

    return value


def write_markup_index(
    output_path: Path,
    index: MarkupIndex,
) -> None:
    """
    Cache source fragments needed by collection.

    Args:
        output_path: Cache file to create.
        index: Relevant page fragments indexed by title.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f"{output_path.name}.part")

    with open_compressed(temporary_path, "wt") as stream:
        json.dump(
            {
                title: {
                    "mathematics": [asdict(formula) for formula in page.mathematics],
                    "score_prefixes": page.score_prefixes,
                }
                for title, page in sorted(index.items())
            },
            stream,
            ensure_ascii=False,
        )

    _ = temporary_path.replace(output_path)


def read_markup_index(
    input_path: Path,
) -> MarkupIndex:
    """
    Read source fragments captured during dump parsing.

    Args:
        input_path: Cache file created during parsing.

    Returns:
        Relevant page fragments indexed by title.
    """
    with open_compressed(input_path, "rt") as stream:
        contents = cast(dict[str, dict[str, object]], json.load(stream))

    return {
        title: PageMarkup(
            tuple(
                MathSource(**formula)
                for formula in cast(list[dict[str, str]], page["mathematics"])
            ),
            tuple(cast(list[str], page["score_prefixes"])),
        )
        for title, page in contents.items()
    }
