"""
Render mathematical notation as LaTeX while preserving word offsets.
"""

import re
from collections.abc import Callable
from importlib import import_module
from typing import cast
from unicodedata import category, name, normalize

from wiktextract.clean import math_map, mathbb_fn, mathcal_fn, mathfrak_fn

from ..models import Attestation
from .offsets import substitute

_SCRIPT_DIGITS = "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱ"
_SUBSCRIPT_DIGITS = "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ᵢₙ"

_SUPERSCRIPT = re.compile(rf"(?<=\w)[{re.escape(_SCRIPT_DIGITS)}]+")
_SUBSCRIPT = re.compile(rf"(?<=\w)[{re.escape(_SUBSCRIPT_DIGITS)}]+")
_ROOT = re.compile(r"√(?:\{([^{}]+)\}|(\d+))")
_MINUS = re.compile("\u2212")

_ALPHABET_RENDERERS = {
    "mathbb": mathbb_fn,
    "mathcal": mathcal_fn,
    "mathfrak": mathfrak_fn,
}

_ALPHABET = {
    renderer(letter): rf"\{command}{{{letter}}}"
    for command, renderer in _ALPHABET_RENDERERS.items()
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    if renderer(letter) != letter
}

_SYMBOL_COMMANDS: dict[str, str] = {}

for command, symbol in math_map.items():
    if (
        re.fullmatch(r"[A-Za-z]+", command)
        and len(symbol) == 1
        and not symbol.isascii()
        and category(symbol) == "Sm"
    ):
        current = _SYMBOL_COMMANDS.get(symbol)

        if current is None or (len(command), command) < (len(current), current):
            _SYMBOL_COMMANDS[symbol] = command

_SYMBOL_COMMANDS.update({"∅": "emptyset", "∆": "Delta", "√": "surd"})

_GET_UNICODE_LATEX = cast(
    Callable[[], dict[int, str]],
    import_module("pylatexenc.latexencode").get_builtin_uni2latex_dict,
)
_UNICODE_LATEX = _GET_UNICODE_LATEX()

for codepoint, latex in _UNICODE_LATEX.items():
    symbol = chr(codepoint)
    command = re.fullmatch(r"\\ensuremath\{\\([A-Za-z]+)\}", latex)

    if category(symbol) == "Sm" and command is not None:
        _ = _SYMBOL_COMMANDS.setdefault(symbol, command[1])

KNOWN_MATH_COMMANDS = frozenset(_SYMBOL_COMMANDS.values())

_SYMBOL = re.compile("[" + re.escape("".join(_SYMBOL_COMMANDS)) + "]")
_ALPHABET_SYMBOL = re.compile("[" + re.escape("".join(_ALPHABET)) + "]")
_STYLED_LETTER = re.compile("[\U0001d400-\U0001d7ff]")
_GREEK_MATH = re.compile(r"(?<!\w)[\u0370-\u03ff](?=[/+=<>\d])")


def _root(
    match: re.Match[str],
) -> str:
    """
    Give a square root its explicit LaTeX argument.

    Args:
        match: Root symbol and its adjacent expression.

    Returns:
        LaTeX for the same square root.
    """
    return rf"\sqrt{{{match[1] or match[2]}}}"


def _script(
    match: re.Match[str],
    marker: str,
) -> str:
    """
    Group adjacent script characters into one LaTeX argument.

    Args:
        match: Superscript or subscript characters.
        marker: LaTeX exponent or index marker.

    Returns:
        Grouped LaTeX script.
    """
    argument = "".join(normalize("NFKC", character) for character in match[0])
    argument = argument.replace("\u2212", "-")

    return f"{marker}{{{argument}}}"


def _symbol(
    match: re.Match[str],
) -> str:
    """
    Convert one mathematical operator to its LaTeX command.

    Args:
        match: Unicode mathematical operator.

    Returns:
        The command with a separator before following letters.
    """
    command = _SYMBOL_COMMANDS[match[0]]
    following = match.string[match.end() : match.end() + 1]

    return f"\\{command}{' ' if following.isalpha() else ''}"


def _styled_letter(
    match: re.Match[str],
) -> str:
    """
    Restore the LaTeX alphabet of one mathematical letter.

    Args:
        match: Unicode mathematical alphabet character.

    Returns:
        LaTeX alphabet command, or the original non-letter symbol.
    """
    character = match[0]

    character_name = name(character, "")
    plain = normalize("NFKC", character)

    if len(plain) != 1 or not plain.isascii() or not plain.isalnum():
        return character

    if "DOUBLE-STRUCK" in character_name:
        command = "mathbb"
    elif "SCRIPT" in character_name:
        command = "mathcal"
    elif "FRAKTUR" in character_name:
        command = "mathfrak"
    elif "BOLD" in character_name:
        command = "mathbf"
    elif "ITALIC" in character_name:
        command = "mathit"
    else:
        return character

    return rf"\{command}{{{plain}}}"


def _greek_math(
    match: re.Match[str],
) -> str:
    """
    Convert a Greek letter only when adjoining formula syntax identifies it.

    Args:
        match: Greek character immediately before a mathematical operator.

    Returns:
        Its LaTeX command, or the unchanged character if unavailable.
    """
    latex = _UNICODE_LATEX.get(ord(match[0]), "")
    command = re.fullmatch(r"\\ensuremath\{(\\[A-Za-z]+)\}", latex)

    return command[1] if command else match[0]


def latexize(
    value: Attestation,
) -> Attestation:
    """
    Render recognizable mathematical symbols as LaTeX text.

    Args:
        value: Display text and offsets before mathematical normalization.

    Returns:
        LaTeX text with relocated word offsets.
    """
    value = substitute(value, _ROOT, _root)
    value = substitute(value, _ALPHABET_SYMBOL, lambda match: _ALPHABET[match[0]])
    value = substitute(value, _STYLED_LETTER, _styled_letter)
    value = substitute(value, _MINUS, "-")
    value = substitute(value, _SYMBOL, _symbol)
    value = substitute(value, _GREEK_MATH, _greek_math)
    value = substitute(value, _SUPERSCRIPT, lambda match: _script(match, "^"))

    return substitute(value, _SUBSCRIPT, lambda match: _script(match, "_"))
