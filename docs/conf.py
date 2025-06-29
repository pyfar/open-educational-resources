# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
import urllib3
import shutil
sys.path.insert(0, os.path.abspath('..'))

# import open_educational_resources  # noqa

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'matplotlib.sphinxext.plot_directive',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'autodocsumm',
    'sphinx_design',
    'sphinx_favicon',
    'sphinx_reredirects',
    'sphinx_mdinclude',
    'nbsphinx',
]

# show tocs for classes and functions of modules using the autodocsumm
# package
autodoc_default_options = {'autosummary': True}

# show the code of plots that follows the command .. plot:: based on the
# package matplotlib.sphinxext.plot_directive
plot_include_source = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = 'open_educational_resources'
copyright = "2025, The pyfar developers"
author = "The pyfar developers"

# The version info for the project you're documenting, acts as replacement
# for |version| and |release|, also used in various other places throughout
# the built documents.
#
# The short X.Y version.
# version = open_educational_resources.__version__
# The full version, including alpha/beta/rc tags.
# release = open_educational_resources.__version__

# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'en'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use (Not defining
# uses the default style of the html_theme).
# pygments_style = 'sphinx'

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# default language for highlighting in source code
highlight_language = "python3"

# intersphinx mapping
intersphinx_mapping = {
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'pyfar': ('https://pyfar.readthedocs.io/en/stable/', None),
    }

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_css_files = ['css/custom.css']
html_logo = 'resources/logos/pyfar_logos_fixed_size_open_educational_resources.png'
html_title = "open_educational_resources"
html_favicon = '_static/favicon.ico'

# -- HTML theme options
# https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/layout.html

html_theme_options = {
    "navbar_start": ["navbar-logo"],
    "navbar_end": ["navbar-icon-links", "theme-switcher"],
    "navbar_align": "content",
    "header_links_before_dropdown": None,  # will be automatically set later based on headers.rst
    "header_dropdown_text": "Packages",  # Change dropdown name from "More" to "Packages"
    "icon_links": [
        {
          "name": "GitHub",
          "url": "https://github.com/pyfar",
          "icon": "fa-brands fa-square-github",
          "type": "fontawesome",
        },
        {
            "name": "CC-BY",
            "url": "https://creativecommons.org/licenses/by/4.0/deed.de",
            "icon": "fa-brands fa-creative-commons-by",
            "type": "fontawesome",
        }
    ],
    # Configure secondary (right) side bar
    "show_toc_level": 3,  # Show all subsections of notebooks
    "show_nav_level": 2,
    "secondary_sidebar_items": ["page-toc"],  # Omit 'show source' link that that shows notebook in json format
    "navigation_with_keys": True,
    # Configure navigation depth for section navigation
    "navigation_depth": 1,
}

html_context = {
   "default_mode": "light"
}

# redirect index to pyfar.html
redirects = {
     "index": "open_educational_resources.html"
}

# -- download navbar and style files from gallery -----------------------------
branch = 'main'
link = f'https://github.com/pyfar/gallery/raw/{branch}/docs/'
folders_in = [
    '_static/css/custom.css',
    '_static/favicon.ico',
    '_static/header.rst',
    'resources/logos/pyfar_logos_fixed_size_open_educational_resources.png',
    ]

def download_files_from_gallery(link, folders_in):
    c = urllib3.PoolManager()
    for file in folders_in:
        url = link + file
        filename = file
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with c.request('GET', url, preload_content=False) as res:
            if res.status == 200:
                with open(filename, 'wb') as out_file:
                    shutil.copyfileobj(res, out_file)

download_files_from_gallery(link, folders_in)
# if logo does not exist, use pyfar logo
if not os.path.exists(html_logo):
    download_files_from_gallery(
        link, ['resources/logos/pyfar_logos_fixed_size_pyfar.png'])
    shutil.copyfile(
        'resources/logos/pyfar_logos_fixed_size_pyfar.png', html_logo)


# -- modify downloaded header file from the gallery to   ----------------------
# -- aline with the local toctree ---------------------------------------------

# read the header file from the gallery
with open("_static/header.rst", "rt") as fin:
    lines = [line for line in fin]

# replace readthedocs link with internal link to this documentation
lines_mod = [
    line.replace(f'https://pyfar-oer.readthedocs.io', project) for line in lines]

# write the modified header file
# to the doc\header.rst folder, so that it can be used in the documentation
with open("header.rst", "wt") as fout:
    fout.writelines(lines_mod)


# -- find position of pyfar package in toctree --------------------------------
# -- this is required to define the dropdown of Packages in the header --------

# find line where pyfar package is mentioned, to determine the start of 
# the packages list in the header
n_line_pyfar = 0
for i, line in enumerate(lines):
    if 'https://pyfar.readthedocs.io' in line:
        n_line_pyfar = i
        break

# the first 4 lines of the header file are defining the settings and a empty
# line of the toctree, therefore we need to subtract 4 from the line number
# of the pyfar package to get the correct position in the toctree
n_toctree_pyfar = n_line_pyfar - 4

if n_toctree_pyfar < 1:	
    raise ValueError(
        "Could not find the line where pyfar package is mentioned. "
        "Please check the header.rst file in the gallery."
    )

# set dropdown header at pyfar appearance, so that pyfar is the first item in
# the dropdown called Packages
html_theme_options['header_links_before_dropdown'] = n_toctree_pyfar

# -- Options for nbsphinx -------------------------------------------------
nbsphinx_prolog = r"""
{% set docname = 'docs/' + env.doc2path(env.docname, base=None)|string %}

.. raw:: html

    <div class="admonition note">
      {% if "open-educational-resources/no_binder/" in docname %}
      This example must be run locally, please
      {% else %}
      Open an interactive online version by clicking the badge
      <span style="white-space: nowrap;"><a href="https://mybinder.org/v2/gh/pyfar/open-educational-resources/main?labpath={{ docname|e }}"><img alt="Binder badge" src="https://mybinder.org/badge_logo.svg" style="vertical-align:text-bottom"></a></span>
      or
      {% endif %}
      <a href="{{ env.docname.split('/')|last|e + '.ipynb' }}" class="reference download internal" download>download</a>
      the notebook.
      <script>
        if (document.location.host) {
          let nbviewer_link = document.createElement('a');
          nbviewer_link.setAttribute('href',
            'https://nbviewer.org/url' +
            (window.location.protocol == 'https:' ? 's/' : '/') +
            window.location.host +
            window.location.pathname.slice(0, -4) +
            'ipynb');
          nbviewer_link.innerHTML = 'Or view it on <em>nbviewer</em>';
          nbviewer_link.classList.add('reference');
          nbviewer_link.classList.add('external');
          document.currentScript.replaceWith(nbviewer_link, '.');
        }
      </script>
    </div>

"""

# --manage thumbnails --------------------------------------------------------
# must be located in 'docs/_static'
nbsphinx_thumbnails = {
    'oer/courses/pyfar/pyfar_challenge/pyfar_challenge': \
        '_static/pyfar_pf_transparent.png',
    ('oer/courses/Virtual_Acoustic_Reality_TUB/'
    'hrtfs_reading_and_inspecting/HRTF_basics'): \
        '_static/head_mesh.png',
    ('oer/courses/Virtual_Acoustic_Reality_TUB/'
    'hrfts_local_interpolation/hrtf_local_interpolation'): \
        '_static/head_mesh.png',
    ('oer/courses/Virtual_Acoustic_Reality_TUB/'
    'room_simulation_ism/image_source_model'): \
        '_static/image_source_model.png',
    ('oer/courses/Virtual_Acoustics_Lab_FAU/'
    'Assignment1/Assignment1'): \
        '_static/Artificial-Audio-Logo.jpg',
}

nbsphinx_execute = 'never'
