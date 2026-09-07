"""What one entry is made of, one module apiece."""

from .forms import parse_forms
from .senses import parse_senses
from .sentences import parse_sentences, parse_year, read_source
from .synonyms import parse_synonyms
from .translations import parse_translations
from .variants import Variants, gather_variants

__all__ = [
    "Variants",
    "gather_variants",
    "parse_forms",
    "parse_senses",
    "parse_sentences",
    "parse_synonyms",
    "parse_translations",
    "parse_year",
    "read_source",
]
