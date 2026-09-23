# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath("../src"))
import qstn

project = "QSTN"
copyright = "2025, Maximilian Kreutner"
author = "Maximilian Kreutner"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


def _latest_stable_tag() -> str | None:
    """Return the most recent stable tag (optionally prefixed with 'v')."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    try:
        output = subprocess.check_output(
            ["git", "tag", "--sort=-v:refname"],
            cwd=repo_root,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

    for raw_tag in output.splitlines():
        tag = raw_tag.strip()
        if not tag:
            continue
        normalized = tag[1:] if tag.startswith("v") else tag
        parts = normalized.split(".")
        if len(parts) == 3 and all(part.isdigit() for part in parts):
            return normalized
    return None


release = _latest_stable_tag() or qstn.__version__

extensions = [
    "sphinx.ext.autodoc",  # Core library to pull documentation from docstrings
    "sphinx.ext.napoleon",  # Support for Google and NumPy style docstrings
    "sphinx.ext.viewcode",  # Add links to highlighted source code
    "myst_nb",  # To write documentation in Markdown instead of reStructuredText
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Tell Sphinx to use MyST for docstrings
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "private-members": False,
}

autosummary_generate = True

autodoc_mock_imports = ["vllm", "torch", "transformers"]

# Use the description for typehints, not the signature
autodoc_typehints = "description"

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}

nb_execution_mode = "off"

root_doc = "index"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
