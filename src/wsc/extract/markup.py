"""Normalize shared display markup and bounded editorial references."""

import re
from html import unescape

from wiktextract.clean import (
    math_map,
    mathbb_fn,
    mathcal_fn,
    mathfrak_fn,
    to_subscript,
    to_superscript,
)

from ..constants.extraction import MOJIBAKE_REPLACEMENTS
from ..models import Attestation
from .offsets import substitute

_ENTITY = re.compile(r"&(?:#[xX][0-9a-fA-F]+|#\d+|[A-Za-z][A-Za-z0-9]+);")
_NUMERIC_ENTITY = re.compile(r"(?<!&)#(?:[xX][0-9a-fA-F]{2,6}|\d{2,7});")
_SCRIPT = re.compile(r"<(sup|sub)\b[^>]*>([^<>]*)</\1>", re.IGNORECASE)
_TAG = re.compile(r"</?(?:b|i|em|strong|small|span)\b[^>]*>", re.IGNORECASE)
_REFERENCE = re.compile(r"<ref\b[^>]*(?:/>|>.*?</ref>)", re.IGNORECASE | re.DOTALL)
_BREAK = re.compile(r"<br\s*/?>", re.IGNORECASE)
_LINK = re.compile(r"(?<!\[)\[\[([^\[\]|]+)(?:\|([^\[\]]*))?\]\](?!\])")
_EMPHASIS = re.compile(r"'{2,}")
_LAYOUT = re.compile("[\u00ad\u200b\u2060\ufeff]")
_EDGES = re.compile(r"^\s+|\s+$")
_SPACES = re.compile(r"[ \t]{2,}")
_MOJIBAKE = re.compile("|".join(re.escape(value) for value in MOJIBAKE_REPLACEMENTS))
_MATH_ALPHABET = re.compile(r"\\(mathbb|mathcal|mathfrak)\{([A-Za-z0-9]+)\}")
_MATH_SYMBOL = re.compile(r"\\([A-Za-z]+)\b")

_BROKEN_MATH = re.compile(
    r"\\[A-Za-z]+|\b(?:mathbf|mathbb|mathcal|dfrac|tfrac|operatorname|displaystyle)\b(?=[_{(])"
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
) -> bool:
    """
    Recognize mathematical and template fragments requiring their source.

    Args:
        text: Display text after supported formatting has been normalized.

    Returns:
        Whether damaged notation or unresolved markup remains.
    """
    return bool(
        _BROKEN_MATH.search(text)
        or _TEMPLATE_ERROR.search(text)
        or _WIKI_FRAGMENT.search(text)
        or "\ufffd" in text
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


def _math_alphabet(
    match: re.Match[str],
) -> str:
    """
    Render complete mathematical alphabet groups without guessing operators.

    Args:
        match: A supported mathematical alphabet command and its complete argument.

    Returns:
        The argument rendered in the requested mathematical alphabet.
    """
    renderers = {"mathbb": mathbb_fn, "mathcal": mathcal_fn, "mathfrak": mathfrak_fn}

    return renderers[match[1]](match[2])


def _math_symbol(
    match: re.Match[str],
) -> str:
    """
    Render known single-symbol commands without guessing missing boundaries.

    Args:
        match: A complete mathematical command name.

    Returns:
        A known single Unicode symbol, or the unchanged command.
    """
    symbol = math_map.get(match[1], "")

    return symbol if len(symbol) == 1 else match[0]


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
        value = substitute(value, _ENTITY, lambda match: unescape(match[0]))

        value = substitute(
            value,
            _NUMERIC_ENTITY,
            lambda match: unescape("&" + match[0]),
        )

        value = substitute(value, _MATH_ALPHABET, _math_alphabet)
        value = substitute(value, _MATH_SYMBOL, _math_symbol)
        value = substitute(value, _REFERENCE, "")
        value = substitute(value, _SCRIPT, _script)
        value = substitute(value, _TAG, "")
        value = substitute(value, _BREAK, "\n")

        value = substitute(
            value,
            _LINK,
            lambda match: (
                match[2] if match[2] is not None else match[1].split("#", 1)[0]
            ),
        )

        value = substitute(value, _EMPHASIS, "")

    if not preserve_markup:
        value = substitute(value, _SPACES, " ")

    return substitute(value, _EDGES, "")


METADATA = re.compile(
    r"^(?:near[- ]synonyms?|meronyms?|holonyms?|usage notes?):", re.IGNORECASE
)

BIBLIOGRAPHY = re.compile(
    r"^(?:c\.|ca\.|circa)?\s*[12]\d{3}(?:\s*[-–—]\s*\d{2,4})?"
    + r"(?:\s+[A-Z][a-z]+\.?(?:\s+\d{1,2})?)?\s*[,:]\s*[\"“'‘(]?[A-Z]"
)

NAVIGATION = re.compile(
    r"^(?:[*•#-]\s*)?(?:[.;]\s*(?:compare|cf\.)\s+|"
    + r"(?:but(?:\s+also)?|also)\s+see(?:\s*:\s*|\s+)|see\s*:\s*|"
    + r"see\s+(?:also\b|(?:Citations|Thesaurus|Appendix|Wikipedia):|"
    + r"(?:the\s+)?(?:quotations?|usage notes?|translations?)\b)|"
    + r"for\s+(?:examples?|quotations?)\b[^\n]*\bsee\b)",
    re.IGNORECASE,
)

_SEE = (
    r"(?:(?:but(?:\s+also)?|also)\s+)?"
    + r"see(?:\s*:\s*|,\s*for example,\s*|\s+(?:also\s+)?)"
)
_PURPOSE = r"(?:for\s+[^();\n]*?,\s*|for\s+[^();\n]*?\bforms\s+|more formally,\s*)"
_DIRECTIVE = rf"(?:{_PURPOSE})?(?:{_SEE}|cf\.\s+|compare(?:\s+with)?\s+)"
_BODY = r"(?:[^.\n()]|\.(?!\s|$)|\((?:[^()]|\([^()]*\))*\))+"

_START = re.compile(rf"^(?:[*•#-]\s*)?{_SEE}", re.IGNORECASE)
_PURPOSE_START = re.compile(rf"^{_PURPOSE}{_SEE}", re.IGNORECASE)
_DEFINITION = re.compile(r"^see\s+[^\n]+?,\s*for:\s*", re.IGNORECASE)

_PARENTHESES = re.compile(
    rf"[ \t]*\(\s*{_DIRECTIVE}[^()]*(?:\([^()]*\)[^()]*)*\)", re.IGNORECASE
)

_UNCLOSED_REFERENCE = re.compile(
    rf"[ \t]*\(\s*{_DIRECTIVE}[^()]*(?:\([^()]*\)[^()]*)*$", re.IGNORECASE
)

_CLAUSE = re.compile(
    rf"(?P<before>[.;:,]|[ \t]*[—–-])\s*{_DIRECTIVE}"
    + _BODY
    + r"(?:\.(?=\s|$)|(?=\)|$))",
    re.IGNORECASE,
)

_LEXICAL_OBJECT = re.compile(
    rf"^,\s*{_SEE}(?:it|them|him|her|us|me|you|what|how|whether|if|that)\b",
    re.IGNORECASE,
)

_UNPUNCTUATED_COMPARISON = re.compile(r"[ \t]+cf\.\s+" + _BODY + r"\.?$")

_QUOTED_TARGET = re.compile(
    r"(?<=[\"”'])\s+see:\s*" + _BODY + r"(?=\)|$)", re.IGNORECASE
)

_TABLE_TAIL = re.compile(
    r"\s+[—–-]\s+(?:otherwise\s+see|see\s+also)\b.*$", re.IGNORECASE | re.DOTALL
)

_SOURCE = re.compile(
    r"\s*\((?:Source:\s*)?https?://[^\s()]+\)|\s*\(Source:[^()]+https?://[^()]+\)",
    re.IGNORECASE,
)

_URL = re.compile(r"[ \t]*https?://[^\s<>]+(?:[ \t]+https?://[^\s<>]+)*")

_FURTHER_REFERENCE = re.compile(
    r"[ \t]*For (?:more|details|examples)\b[^()\n]*?\bsee\s+https?://[^\s()]+",
    re.IGNORECASE,
)

_DEMONSTRATION = re.compile(
    r"\s*For video demonstration, click here:\s*https?://\S+", re.IGNORECASE
)

_EXPLICIT = re.compile(
    r"\b(?:compare\s+|(?:but(?:\s+also)?|also)\s+see\s*:|"
    + r"see\s*:\s*|see\s+(?:also\b|"
    + r"(?:Citations|Thesaurus|Wikipedia|Appendix):|usage notes?\b))",
    re.IGNORECASE,
)

_EMPTY = re.compile(r"\([ \t]*\)|[ \t]+([,.;:])")


def normalize_definition_punctuation(text: str) -> str:
    """
    Replace a definition's final colon with a period.

    Args:
        text: A sense gloss or translation-table heading.

    Returns:
        The definition with normalized terminal punctuation.
    """
    return f"{text[:-1]}." if text.endswith(":") else text


def remove_references(
    value: Attestation,
    *,
    explicit: bool = False,
) -> Attestation:
    """
    Remove bounded references, requiring explicit editorial cues in examples.

    Args:
        value: Text and known word offsets before reference removal.
        explicit: Whether each removed reference must contain an editorial cue.

    Returns:
        Text with bounded references removed and surviving offsets relocated.
    """
    original = value

    for pattern in (_PARENTHESES, _UNCLOSED_REFERENCE):
        value = substitute(
            value,
            pattern,
            lambda match: (
                match[0] if explicit and not _EXPLICIT.search(match[0]) else ""
            ),
        )

    value = substitute(
        value,
        _CLAUSE,
        lambda match: (
            match[0]
            if (explicit and not _EXPLICIT.search(match[0]))
            or _LEXICAL_OBJECT.match(match[0])
            else "."
            if match["before"] == "."
            else ""
        ),
    )

    if value.text != original.text:
        value = substitute(value, _EMPTY, lambda match: match[1] or "")

    return value


def clean_definition_references(
    text: str,
    *,
    table: bool = False,
) -> str:
    """
    Keep definitions while removing navigation and editorial source URLs.

    Args:
        text: A definition with possible navigation or source references.
        table: Whether the definition heads a translation table.

    Returns:
        The retained definition, or an empty string for standalone navigation.
    """
    if text.casefold().startswith(("see;", "see.")):
        return text

    value = substitute(Attestation(text), _DEFINITION, "")

    if (
        NAVIGATION.match(value.text)
        or _PURPOSE_START.match(value.text)
        or (not table and _START.match(value.text))
    ):
        return ""

    if table:
        value = substitute(value, _TABLE_TAIL, "")

    value = remove_references(value, explicit=text.casefold().startswith("to see "))
    value = substitute(value, _UNPUNCTUATED_COMPARISON, "")
    value = substitute(value, _QUOTED_TARGET, "")

    value = substitute(value, _DEMONSTRATION, "")
    value = substitute(value, _FURTHER_REFERENCE, "")
    value = substitute(value, _SOURCE, "")

    value = substitute(
        value,
        _URL,
        lambda match: (
            "."
            if match[0].endswith(".")
            and not match.string[: match.start()].rstrip().endswith((".", "!", "?"))
            else ""
        ),
    )

    return value.text.strip()
