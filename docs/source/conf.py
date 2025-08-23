from __future__ import annotations
import os
import sys

# -- Path setup: rendre le package importable par Sphinx ---------------------
PROJECT_ROOT = os.path.abspath(os.path.join(__file__, "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

# -- Project information -----------------------------------------------------
project = "coconut-tools"
author = "Luis Linan"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",         # Google-style docstrings
    "sphinx.ext.autosummary",      # tableaux d’API automatiques
    "sphinx.ext.viewcode",         # lien vers le code source
    "sphinx_copybutton",           # bouton copier les blocs de code
    "sphinx_autodoc_typehints",    # jolis hints de types
]

autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_attr_annotations = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- HTML theme --------------------------------------------------------------
html_theme = "furo"
html_title = f"{project} {release}"
html_static_path = ["_static"]
html_last_updated_fmt = "%Y-%m-%d"

# -- Autodoc options ---------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

# Évite d'installer des dépendances lourdes/non disponibles sur RTD
autodoc_mock_imports = [
    "vtk",
    "pyvista",
    "pyevtk",
    "sunpy",
    "astropy",
    "sklearn",        # alias courant; scikit-learn sera aussi mocké
    "scikit-learn",
    "cmocean",
    "solarmach",
    "natsort",
    "h5py",
    "matplotlib",
    "pandas",
    "numpy",
    "bs4",
    "requests",
]