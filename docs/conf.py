import detoxai

project = "DetoxAI"
copyright = "2025, Ignacy Stepka, Lukasz Sztukiewicz, Michal Wilinski"
author = "Ignacy Stepka, Lukasz Sztukiewicz, Michal Wilinski"
release = detoxai.__version__
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # For Google docstring support
    "sphinx.ext.viewcode",
]
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"

autodoc_member_order = "bysource"
