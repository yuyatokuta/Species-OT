project = "speciesot"
author = "Yuya Tokuta"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "nbsphinx",
    "myst_parser",
    "sphinx.ext.mathjax",
    "sphinx_rtd_theme",
]

# Generate autosummary stub pages automatically
autosummary_generate = True

templates_path = ["_templates"]

autosummary_imported_members = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "inherited-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
    # もし非公開も出したければ:
    # "private-members": True,
}


# Do not execute notebooks during the build
nbsphinx_execute = "never"

# Mock heavy dependencies during docs build (keep minimal)
autodoc_mock_imports = ["jax", "ott"]

# Theme (set here rather than via extensions)
html_theme = "sphinx_rtd_theme"

# Add project root so 'speciesot' can be imported by autodoc
import os, sys
sys.path.insert(0, os.path.abspath(".."))
