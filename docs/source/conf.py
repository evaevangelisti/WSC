"""
Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import sys
from pathlib import Path

sys.path.insert(0, (Path(__file__).parents[2] / "src").as_posix())

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "WSC"
project_copyright = "2026, Eva Evangelisti"
author = "Eva Evangelisti"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

exclude_patterns = []

# -- Docstrings --------------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html

# Google is the style the docstrings are written in; NumPy's is not read, so
# that one written in it fails rather than being half understood.
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# An Attributes section describes the fields autodoc has already found, so it
# is rendered beside them rather than as a second entry for the same name.
napoleon_use_ivar = True

# The order of a module is the order it was written in, and the annotations
# are already on every signature: they read better beside what they describe.
autodoc_member_order = "bysource"
autodoc_typehints = "description"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_title = f"{project} {release}"
