import tomllib

with open("../pyproject.toml", "rb") as f:
    pyproject = tomllib.load(f)
    
    
# Copy examples folder to docs
import shutil
import os

notebooks_source = "../examples"
notebooks_dir = "examples"
if os.path.exists(notebooks_dir):
    shutil.rmtree(notebooks_dir)
shutil.copytree(notebooks_source, notebooks_dir)
    

release = pyproject["project"]["version"]

project = "DetoxAI"
copyright = "2025, Ignacy Stepka, Lukasz Sztukiewicz, Michal Wilinski"
author = "Ignacy Stepka, Lukasz Sztukiewicz, Michal Wilinski"
release = pyproject["project"]["version"]
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # For Google docstring support
    "sphinx.ext.viewcode",
    'sphinx.ext.mathjax',
    'sphinx.ext.autodoc',
    "nbsphinx",
]
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"

autodoc_member_order = "bysource"
nbsphinx_execute = 'never'
