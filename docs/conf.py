from recommonmark.transform import AutoStructify

project = "ubc"
version = "0.0.4"
copyright = "2019, Joaquin"
author = "Joaquin"

master_doc = "index"
html_theme = "sphinx_rtd_theme"

source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}

html_static_path = ["_static"]
htmlhelp_basename = project

extensions = [
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_markdown_tables",
    "sphinx.ext.doctest",
    "recommonmark",
    "sphinx_autodoc_typehints",
]

# Order members by source
autodoc_member_order = "bysource"

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    "build",
    "extra/**",
    "report/**",
]

napoleon_use_param = True


def setup(app):
    app.add_config_value(
        "recommonmark_config",
        {"auto_toc_tree_section": "Contents", "enable_eval_rst": True},
        True,
    )
    app.add_transform(AutoStructify)
