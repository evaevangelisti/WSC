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

# README excerpts start at second-level headings, which Docutils renders as titles.
suppress_warnings = ["myst.header"]

# -- Docstrings --------------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html

# Parse the project's Google-style docstrings.
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# Render attribute descriptions beside the fields discovered by autodoc.
napoleon_use_ivar = True

# Preserve source order and display annotations beside their descriptions.
autodoc_member_order = "bysource"
autodoc_typehints = "description"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "shibuya"

html_static_path = ["_static"]
html_favicon = "_static/favicon.svg"
html_theme_options = {
    "light_logo": "_static/logo-light.svg",
    "dark_logo": "_static/logo-dark.svg",
    "github_url": "https://github.com/evaevangelisti/WSC",
}
