# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import pathlib
import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../../'))

# The readme that already exists
readme_path = pathlib.Path(__file__).parent.resolve().parent.parent / "README.md"
# We copy a modified version here
readme_target = pathlib.Path(__file__).parent / "readme.md"

with readme_target.open("w") as outf:
    # Change the title to "Readme"
    outf.write(
        "\n".join(
            [
                "Readme",
                "======",
            ]
        )
    )
    lines = []
    for line in readme_path.read_text().split("\n"):
        if line.startswith("# "):
            # Skip title, because we now use "Readme"
            # Could also simply exclude first line for the same effect
            continue
        lines.append(line)
    outf.write("\n".join(lines))


# -- Project information -----------------------------------------------------

project = 'improveai'
copyright = '2022, Improve AI'
author = 'Improve AI'

# The short X.Y version
version = '7.2.2'

# The full version, including alpha/beta/rc tags
release = '7.2.2'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.napoleon',
    'recommonmark',
    'sphinx_automodapi.automodapi',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx_autodoc_typehints',
    'enum_tools.autoenum'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
source_suffix = ['.rst', '.md']

autodoc_default_flags = ['members']
autosummary_generate = True
add_module_names = False
autodoc_typehints = 'both'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
autodoc_class_signature = 'separated'
