from __future__ import annotations
import os
import sys
from pathlib import Path
import importlib.metadata as ilm

# -- Path setup: rendre le package importable par Sphinx ---------------------
# Ce fichier est dans docs/source/conf.py -> on remonte à la racine du repo
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# -- Project information -----------------------------------------------------
project = "coconut-tools"
author = "Luis Linan"

# Tente de récupérer la version du paquet installé (si dispo), sinon fallback
def _detect_version() -> str:
    for dist_name in ("coconut-tools", "coconut_tools"):
        try:
            return ilm.version(dist_name)
        except Exception:
            pass
    return os.environ.get("PROJECT_VERSION", "0.1.0")

release = _detect_version()
version = release

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
# Optionnel mais utile : afficher les hints dans la description
autodoc_typehints = "description"

if os.environ.get("READTHEDOCS") == "True":
    autodoc_mock_imports = [
        "numpy", "scipy", "matplotlib", "pyvista", "vtk",
        "pandas", "cmocean", "sunpy", "astropy", "skimage",
    ]