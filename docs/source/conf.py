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
import sys
from pathlib import Path

# sys.path.insert(0, os.path.abspath('../ashdisperse'))
sys.path.insert(0, str(Path('..','ashdisperse').resolve()))
from distutils.util import convert_path

abspath = os.path.abspath(__file__)
filedir = os.path.dirname(abspath)
parentdir = os.path.dirname(filedir)
vpath = os.path.abspath(os.path.join(parentdir,'../ashdisperse/version.py'))

# -- Project information -----------------------------------------------------

project = 'AshDisperse'
copyright = '2022, Mark J. Woodhouse'
author = 'Mark J. Woodhouse'

main_ns = {}
ver_path = convert_path(vpath)
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

# The full version, including alpha/beta/rc tags
release = main_ns['__version__']


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx_click',
    'sphinx_rtd_theme',
    'sphinx_copybutton',
    'sphinx_gallery.gen_gallery',
    'sphinx.ext.autosectionlabel',
    'sphinxcontrib.bibtex',
    'notfound.extension',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']