"""
Normalize shared display markup and bounded editorial references.
"""

import re
from html import unescape

from wiktextract.clean import math_map, to_subscript, to_superscript

from ..constants.extraction import MOJIBAKE_REPLACEMENTS
from ..models import Attestation
from .mathematics import KNOWN_MATH_COMMANDS, latexize
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
    + r"for\s+(?:examples?|quotations?)\b[^\n]*\bsee\b(?=\s+\S))",
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


_SENTENCE_ENDINGS = frozenset(".?!…‽")


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

    if not text:
        return ""

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
