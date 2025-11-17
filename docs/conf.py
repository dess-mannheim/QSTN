# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
# Make sure the path to your project's source code is on the Python path
# This is crucial for Sphinx's autodoc extension to find your modules.
# Replace '../src' with the actual path to your source code directory.
sys.path.insert(0, os.path.abspath('../src'))

project = 'SurveyGen'
copyright = '2025, Maximilian Kreutner'
author = 'Maximilian Kreutner'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',      # Core library to pull documentation from docstrings
    'sphinx.ext.napoleon',     # Support for Google and NumPy style docstrings
    'sphinx.ext.viewcode',     # Add links to highlighted source code
    'myst_parser',             # To write documentation in Markdown instead of reStructuredText
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

root_doc = 'index'


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
