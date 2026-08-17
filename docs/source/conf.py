"""Sphinx configuration for the bayesplain documentation."""

import inspect
import os
import sys

from packaging.version import Version

import bayesplain

# -- Project information -----------------------------------------------------

project = "bayesplain"
copyright = "2026-, Eli Knaap"  # noqa: A001 - shadowing a Python builtin
author = "Eli Knaap"

version = Version(bayesplain.__version__).public  # remove commit hash
release = version

language = "en"
html_title = project

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinxcontrib.bibtex",
    "sphinx_copybutton",
    "sphinx_immaterial",
]

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]

bibtex_bibfiles = ["_static/references.bib"]
bibtex_reference_style = "author_year"

master_doc = "index"
templates_path = ["_templates"]
exclude_patterns = []

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- autodoc / autosummary ---------------------------------------------------

autosummary_generate = True
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
# Render "Attributes" as :ivar: fields on the class rather than as separate
# object descriptions. Without this, the dataclasses get every attribute
# documented twice: once from the docstring and once from autodoc's discovery
# of the annotated fields.
napoleon_use_ivar = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "inherited-members": True,
    # The classes document their constructor arguments in the class docstring,
    # which is where a reader looks for them. Pulling __init__ in as well makes
    # autodoc emit every parameter twice.
    "exclude-members": "__init__",
}
autoclass_content = "class"
autodoc_typehints = "none"
suppress_warnings = ["ref.ref"]

# -- notebooks ---------------------------------------------------------------

nb_execution_mode = "force"
nb_execution_timeout = -1
nb_kernel_rgx_aliases = {".*": "python3"}
nb_merge_streams = True
nb_execution_raise_on_error = True
nb_execution_show_tb = True

# -- HTML output -------------------------------------------------------------

html_theme = "sphinx_immaterial"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme_options = {
    "icon": {"repo": "fontawesome/brands/github", "edit": "material/file-code"},
    "site_url": "https://knaaptime.github.io/bayesplain",
    "repo_url": "https://github.com/knaaptime/bayesplain/",
    "edit_uri": "blob/main/docs",
    "repo_name": "knaaptime/bayesplain",
    "features": [
        "navigation.sections",
        "navigation.top",
        "search.share",
        "search.suggest",
        "toc.follow",
        "toc.sticky",
        "content.code.copy",
        "content.action.edit",
    ],
    "palette": [
        {
            "media": "(prefers-color-scheme)",
            "toggle": {
                "icon": "material/brightness-auto",
                "name": "Switch to light mode",
            },
        },
        {
            "media": "(prefers-color-scheme: light)",
            "scheme": "default",
            "primary": "black",
            "accent": "red",
            "toggle": {
                "icon": "material/lightbulb",
                "name": "Switch to dark mode",
            },
        },
        {
            "media": "(prefers-color-scheme: dark)",
            "scheme": "slate",
            "primary": "black",
            "accent": "red",
            "toggle": {
                "icon": "material/lightbulb-outline",
                "name": "Switch to system preference",
            },
        },
    ],
}


def linkcode_resolve(domain, info):
    """Point the source links at the file and lines on GitHub."""

    def find_source():
        obj = sys.modules[info["module"]]
        for part in info["fullname"].split("."):
            obj = getattr(obj, part)
        filename = inspect.getsourcefile(obj)
        filename = os.path.relpath(filename, start=os.path.dirname(bayesplain.__file__))
        source, lineno = inspect.getsourcelines(obj)
        return filename, lineno, lineno + len(source) - 1

    if domain != "py" or not info["module"]:
        return None
    try:
        target = "bayesplain/%s#L%d-L%d" % find_source()  # noqa: UP031
    except Exception:
        target = info["module"].replace(".", "/") + ".py"
    return f"https://github.com/knaaptime/bayesplain/blob/main/{target}"
