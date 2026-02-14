# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

# RST substitutions removed - use full text directly in documents
# rst_prolog = ""

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'lwdid'
copyright = '2025, Xuanyu Cai, Wenli Xu'
author = 'Xuanyu Cai, Wenli Xu'
release = '0.2.1'
version = '0.2.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',           # Auto-generate API docs from docstrings
    'sphinx.ext.autosummary',       # Auto-generate API summaries
    'sphinx.ext.napoleon',          # Support NumPy and Google style docstrings
    'sphinx.ext.viewcode',          # Add links to source code
    'sphinx.ext.intersphinx',       # Link to other project documentation
    'sphinx.ext.mathjax',           # Math formula support
    'sphinx_autodoc_typehints',     # Type hints support
    'sphinx_copybutton',            # Code block copy button
    'myst_parser',                  # Markdown support
]

# Auto-generate API summaries
autosummary_generate = True
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Avoid duplicate documentation of properties
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
autodoc_typehints_description_target = 'documented'

# Prevent single-letter parameter names from being mistakenly identified as cross-references
suppress_warnings = ['ref.python']

# Napoleon settings - support NumPy style docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Intersphinx mapping - link to other Python package documentation
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'statsmodels': ('https://www.statsmodels.org/stable/', None),
}

# Template and static file paths
templates_path = ['_templates']
exclude_patterns = []

# Supported file formats
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# ReadTheDocs theme configuration
html_theme_options = {
    'analytics_id': '',
    'logo_only': False,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# Sidebar configuration
html_sidebars = {
    '**': [
        'globaltoc.html',
        'relations.html',
        'sourcelink.html',
        'searchbox.html',
    ]
}

# Code highlighting
pygments_style = 'sphinx'

# Document title
html_title = f'{project} {version}'
html_short_title = project

# Show source file links
html_show_sourcelink = True
html_show_sphinx = True
html_show_copyright = True

# Markdown-related settings
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]