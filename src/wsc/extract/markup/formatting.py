"""
Normalize visible Wikitext and mathematical notation.
"""

import re
from html import unescape

from wiktextract.clean import math_map, to_subscript, to_superscript

from ...constants.extraction import MOJIBAKE_REPLACEMENTS
from ...models import Attestation
from ..mathematics import KNOWN_MATH_COMMANDS, latexize
from ..offsets import substitute

_ENTITY = re.compile(r"&(?:#[xX][0-9a-fA-F]+|#\d+|[A-Za-z][A-Za-z0-9]+);")
_NUMERIC_ENTITY = re.compile(r"(?<!&)#(?:[xX][0-9a-fA-F]{2,6}|\d{2,7});")
_SCRIPT = re.compile(r"<(sup|sub)\b[^>]*>([^<>]*)</\1>", re.IGNORECASE)
_TAG = re.compile(r"</?(?:b|i|em|strong|small|span)\b[^>]*>", re.IGNORECASE)
_REFERENCE = re.compile(r"<ref\b[^>]*(?:/>|>.*?</ref>)", re.IGNORECASE | re.DOTALL)
_BREAK = re.compile(r"<br\s*/?>", re.IGNORECASE)
_LINK = re.compile(r"(?<!\[)\[\[([^\[\]|]+)(?:\|([^\[\]]*))?\]\](?!\])")
_TRIPLE_LINK = re.compile(r"\[\[\[([^\[\]]+)\]\]\]")
_EDITORIAL_FOOTNOTE = re.compile(r"[ \t]*\^\(\[(?:sic|wikipedia)\]\)", re.IGNORECASE)
_EMPHASIS = re.compile(r"'{2,}")
_LAYOUT = re.compile("[\u00ad\u200b\u2060\ufeff]")
_EDGES = re.compile(r"^\s+|\s+$")
_SPACES = re.compile(r"[ \t]{2,}")
_MOJIBAKE = re.compile("|".join(re.escape(value) for value in MOJIBAKE_REPLACEMENTS))
_LATEX_COMMAND = re.compile(r"\\([A-Za-z]+)")
_LEADING_MATH = re.compile(
    r"^[a-z](?=\s*(?:[=+<>\u00d7\u00f7\u2212]|\\(?:in|leq|geq|times|approx|subseteq?)\b))"
)
_GROUPED_COMMANDS = frozenset(
    {
        "mathbb",
        "mathcal",
        "mathfrak",
        "mathbf",
        "mathit",
        "frac",
        "dfrac",
        "tfrac",
        "sqrt",
        "operatorname",
        "text",
    }
)
_SYMBOL_COMMANDS = frozenset({"surd"})

_BROKEN_MATH = re.compile(
    r"(?<!\\)\b(?:mathbf|mathbb|mathcal|mathfrak|dfrac|tfrac|operatorname|displaystyle)\b(?=[_{(])"
)

_WIKI_FRAGMENT = re.compile(r"(?<!\[)\[\[[A-Za-z][^]\n]*\]?(?!\])")
_TEMPLATE_ERROR = re.compile(r"(?:^|[\s:])Template:|^(?:Lua error|Script error)\b")

_LITERAL = re.compile(
    r"\b(?:HTML|Python|JavaScript|template|tags?|filter|code)\b|`", re.IGNORECASE
)


def is_literal_markup(
    text: str,
) -> bool:
    """
    Recognize examples explicitly discussing code or markup.

    Args:
        text: Text whose surrounding prose may identify a code example.

    Returns:
        Whether markup should be preserved as literal content.
    """
    prose = re.sub(r"https?://\S+", "", text)
    prose = _LINK.sub("", prose)
    prose = re.sub(r"\{\{[^{}]*\}\}", "", prose)

    return bool(
        "`" in prose
        or re.search(r"<(?:table|div)\b", prose)
        or (_LITERAL.search(prose) and re.search(r"<[^>]+>|\{\{|\\|'{2,}", prose))
    )


def is_unrecoverable(
    text: str,
    *,
    mathematical_sources: tuple[str, ...] = (),
) -> bool:
    """
    Recognize mathematical and template fragments requiring their source.

    Args:
        text: Display text after supported formatting has been normalized.
        mathematical_sources: Complete formulae recovered from source markup.

    Returns:
        Whether damaged notation or unresolved markup remains.
    """
    for source in mathematical_sources:
        text = text.replace(source, "")

    if (
        _BROKEN_MATH.search(text)
        or _TEMPLATE_ERROR.search(text)
        or _WIKI_FRAGMENT.search(text)
        or "\ufffd" in text
        or "{{" in text
        or "}}" in text
    ):
        return True

    return any(
        (
            match[1] not in math_map
            and match[1] not in _GROUPED_COMMANDS
            and match[1] not in _SYMBOL_COMMANDS
            and match[1] not in KNOWN_MATH_COMMANDS
        )
        or (
            match[1] in _GROUPED_COMMANDS and text[match.end() : match.end() + 1] != "{"
        )
        for match in _LATEX_COMMAND.finditer(text)
    )


def _script(
    match: re.Match[str],
) -> str:
    """
    Render a script while keeping exponents and chemical indices distinct.

    Args:
        match: An HTML superscript or subscript element and its content.

    Returns:
        The content rendered with Unicode script characters.
    """
    render = to_superscript if match[1].casefold() == "sup" else to_subscript

    return render(match[2])


def normalize_formatting(
    value: Attestation,
    *,
    preserve_markup: bool = False,
    preserve_joiners: bool = False,
) -> Attestation:
    """
    Normalize display fragments and track their exact positional changes.

    Args:
        value: Text and any previously known word offsets.
        preserve_markup: Whether markup is literal example content.
        preserve_joiners: Whether script-specific invisible characters are meaningful.

    Returns:
        Normalized text with relocated word offsets.
    """
    if not preserve_joiners:
        value = substitute(value, _LAYOUT, "")
        value = substitute(
            value, _MOJIBAKE, lambda match: MOJIBAKE_REPLACEMENTS[match[0]]
        )

    if not preserve_markup:
        value = substitute(value, _EDITORIAL_FOOTNOTE, "")
        value = substitute(value, _ENTITY, lambda match: unescape(match[0]))

        value = substitute(
            value,
            _NUMERIC_ENTITY,
            lambda match: unescape("&" + match[0]),
        )

        value = substitute(value, _REFERENCE, "")
        value = substitute(value, _SCRIPT, _script)
        value = substitute(value, _TAG, "")
        value = substitute(value, _BREAK, "\n")
        value = substitute(value, _TRIPLE_LINK, lambda match: f"[{match[1]}]")

        value = substitute(
            value,
            _LINK,
            lambda match: (
                match[2] if match[2] is not None else match[1].split("#", 1)[0]
            ),
        )

        value = substitute(value, _EMPHASIS, "")

        value = latexize(value)

    if not preserve_markup:
        value = substitute(value, _SPACES, " ")

    return substitute(value, _EDGES, "")


_SENTENCE_ENDINGS = frozenset(".?!…‽")


def _trim_unmatched_closing_parentheses(
    text: str,
) -> str:
    """
    Remove extra closing parentheses after otherwise balanced prose.

    Args:
        text: Definition or reference before sentence punctuation is normalized.

    Returns:
        Text without surplus closing parentheses at its end.
    """
    while (body := text.rstrip(".?!…‽")).endswith(")") and "(" in body:
        balance = 0

        for character in body[:-1]:
            balance += (character == "(") - (character == ")")

            if balance < 0:
                return text

        if balance != 0:
            return text

        text = body[:-1] + text[len(body) :]

    return text


def normalize_statement(
    text: str,
) -> str:
    """
    Normalize capitalization and terminal punctuation in prose.

    Args:
        text: A definition or reference.

    Returns:
        Prose beginning with a capital and ending with sentence punctuation.
    """
    text = text.strip()

    if not any(character.isalnum() for character in text):
        return ""

    text = _trim_unmatched_closing_parentheses(text)

    command = _LATEX_COMMAND.search(text)
    mathematical_lead = _LEADING_MATH.match(text) is not None

    for index, character in enumerate(text):
        if character.isdigit():
            break

        if character.isalpha():
            if not mathematical_lead and (
                command is None or not command.start() <= index < command.end()
            ):
                text = f"{text[:index]}{character.upper()}{text[index + 1 :]}"

            break

    if text[-1] in ",;:":
        return f"{re.sub(r'[\s,;:]+$', '', text)}."

    if text[-1] in _SENTENCE_ENDINGS:
        return text

    content = text.rstrip("'\"’”)]}")

    return text if content and content[-1] in _SENTENCE_ENDINGS else f"{text}."
