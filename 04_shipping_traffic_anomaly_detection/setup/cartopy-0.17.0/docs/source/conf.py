# (C) British Crown Copyright 2011 - 2018, Met Office
#
# This file is part of cartopy.
#
# cartopy is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# cartopy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with cartopy.  If not, see <https://www.gnu.org/licenses/>.


# -*- coding: utf-8 -*-
#
# cartopy documentation build configuration file, created by
# sphinx-quickstart on Thu Aug 16 09:41:05 2012.
#
# This file is execfile()d with the current directory set to its containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

import sys, os

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# -- General configuration -----------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx = '1.6'

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
              'cartopy.sphinxext.summarise_package',
              'sphinx.ext.autodoc',
              'sphinx.ext.doctest',
              'sphinx.ext.intersphinx',
              'sphinx.ext.coverage',
              'sphinx.ext.viewcode',
              'sphinx.ext.extlinks',
              # 'sphinx.ext.autosummary',
              # 'sphinxcontrib.napoleon', (Needs work before this can
              # be enabled.)
              # 'matplotlib.sphinxext.plot_directive'
              # We use a local copy of the plot_directive until
              # https://github.com/matplotlib/matplotlib/pull/6213 is
              # available in order
              # to benefit from cached rebuilds of plots.
              'sphinxext.plot_directive',
              # Monkey-patch sphinx_gallery to handle cartopy's __tags__
              # example convention.
              'sphinxext.pre_sphinx_gallery',
              'sphinx_gallery.gen_gallery',
              'sphinx.ext.napoleon'
              ]

import matplotlib
matplotlib.use('Agg')

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = '.rst'

# The encoding of source files.
# source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = u'cartopy'
copyright = u'2011 - 2018 British Crown Copyright'  # the template will need
# updating if this is changed

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
import cartopy
version = cartopy.__version__
# The full version, including alpha/beta/rc tags.
release = cartopy.__version__

# Sphinx gallery configuration
from sphinx_gallery.sorting import ExampleTitleSortKey
sphinx_gallery_conf = {
    'examples_dirs': ['../../lib/cartopy/examples'],
    'filename_pattern': '^((?!sgskip).)*$',
    'gallery_dirs': ['gallery'],
    'within_subsection_order': ExampleTitleSortKey,
    'doc_module': ('cartopy',),
    'reference_url': {'cartopy': None},
    'backreferences_dir': '../build/backrefs',
}

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
# language = None

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
# today = ''
# Else, today_fmt is used as the format for a strftime call.
# today_fmt = '%B %d, %Y'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = []

# The reST default role (used for this markup: `text`) to use for all documents.
default_role = 'py:obj'

# If true, '()' will be appended to :func: etc. cross-reference text.
# add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
# add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
# show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# A list of ignored prefixes for module index sorting.
# modindex_common_prefix = []


# -- Options for HTML output ---------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'sphinxdoc'

# Add any paths that contain custom themes here, relative to this directory.
# html_theme_path = []

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
# html_title = None

# A shorter title for the navigation bar.  Default is the same as html_title.
# html_short_title = None

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
# html_logo = None

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
# html_favicon = None

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
# html_last_updated_fmt = '%b %d, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
# html_use_smartypants = True

# Custom sidebar templates, maps document names to template names.
# html_sidebars = {}

# Additional templates that should be rendered to pages, maps page names to
# template names.
# html_additional_pages = {}

# If false, no module index is generated.
# html_domain_indices = True

# If false, no index is generated.
# html_use_index = True

# If true, the index is split into individual pages for each letter.
# html_split_index = False

# If true, links to the reST sources are added to the pages.
# html_show_sourcelink = True

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
# html_show_copyright = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
# html_use_opensearch = ''

# This is the file name suffix for HTML files (e.g. ".xhtml").
# html_file_suffix = None

# Output file base name for HTML help builder.
htmlhelp_basename = 'cartopydoc'

html_context = {'rellinks': [('genindex', 'General Index', 'I', 'index'),
                             ('cartopy_outline', 'Module outline', 'O',
                              'outline')]}


# -- Options for LaTeX output --------------------------------------------------

latex_elements = {
                    # The paper size ('letterpaper' or 'a4paper').
                    # 'papersize': 'letterpaper',

                    # The font size ('10pt', '11pt' or '12pt').
                    # 'pointsize': '10pt',

                    # Additional stuff for the LaTeX preamble.
                    # 'preamble': '',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual]).
latex_documents = [
  ('index', 'cartopy.tex', u'Cartopy Introduction',
   u'Philip Elson, Richard Hattersley', 'manual', False),
  ('introductory_examples/index', 'cartopy_examples.tex', u'Cartopy examples',
   u'Philip Elson, Richard Hattersley', 'manual', True)
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
# latex_logo = None

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
# latex_use_parts = False

# If true, show page references after internal links.
# latex_show_pagerefs = False

# If true, show URL addresses after external links.
# latex_show_urls = False

# Documents to append as an appendix to all manuals.
# latex_appendices = []

# If false, no module index is generated.
# latex_domain_indices = True


# -- Options for manual page output --------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    ('index', 'cartopy', u'cartopy Documentation',
     [u'Philip Elson, Richard Hattersley'], 1)
]

# If true, show URL addresses after external links.
# man_show_urls = False


# -- Options for Texinfo output ------------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
  ('index', 'cartopy', u'cartopy Documentation',
   u'Philip Elson, Richard Hattersley', 'cartopy',
   'One line description of project.',
   'Miscellaneous'),
]

# Documents to append as an appendix to all manuals.
# texinfo_appendices = []

# If false, no module index is generated.
# texinfo_domain_indices = True

# How to display URL addresses: 'footnote', 'no', or 'inline'.
# texinfo_show_urls = 'footnote'


# -- Options for Epub output ---------------------------------------------------

# Bibliographic Dublin Core info.
epub_title = u'cartopy'
epub_author = u'Philip Elson, Richard Hattersley'
epub_publisher = u'Philip Elson, Richard Hattersley'
epub_copyright = u'2012, Philip Elson, Richard Hattersley'

# The language of the text. It defaults to the language option
# or en if the language is not set.
# epub_language = ''

# The scheme of the identifier. Typical schemes are ISBN or URL.
# epub_scheme = ''

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
# epub_identifier = ''

# A unique identification for the text.
# epub_uid = ''

# A tuple containing the cover image and cover page html template filenames.
# epub_cover = ()

# HTML files that should be inserted before the pages created by sphinx.
# The format is a list of tuples containing the path and title.
# epub_pre_files = []

# HTML files shat should be inserted after the pages created by sphinx.
# The format is a list of tuples containing the path and title.
# epub_post_files = []

# A list of files that should not be packed into the epub file.
# epub_exclude_files = []

# The depth of the table of contents in toc.ncx.
# epub_tocdepth = 3

# Allow duplicate toc entries.
# epub_tocdup = True


# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {'python': ('https://docs.python.org/3', None),
                       'matplotlib': ('https://matplotlib.org', None),
                       'numpy': ('https://docs.scipy.org/doc/numpy/', None),
                       'shapely': ('http://toblerity.org/shapely', None), }





############ extlinks extension ############
extlinks = {'issues': ('https://github.com/SciTools/cartopy/labels/%s',
                      'issues labeled with '),
            'issue': ('https://github.com/SciTools/cartopy/issues/%s',
                      'Issue #'),
            'pull': ('https://github.com/SciTools/cartopy/pull/%s', 'PR #'),
            }



############ package summary extension ###########

summarise_package_names = ['cartopy']
summarise_package_exclude_directories = [['tests', 'examples', 'sphinxext']]
summarise_package_fnames = ['cartopy_outline.rst']


############ plot directive ##############

plot_html_show_formats = False
# plot_rcparams = {'figure.autolayout': True}
plot_rcparams = {'figure.subplot.bottom': 0.04,
                 'figure.subplot.top': 0.96,
                 'figure.subplot.left': 0.04,
                 'figure.subplot.right': 0.96}
plot_formats = [('thumb.png', 20),
                'png',
                'pdf'
                ]


############ autodoc config ##############

autoclass_content = 'both'

# Napoleon settings
# napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = False

# # Numpydoc settings
# numpydoc_show_class_members = True
# numpydoc_show_inherited_class_members = False
# numpydoc_class_members_toctree = False
# numpydoc_use_blockquotes = True