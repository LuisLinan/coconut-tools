# src/coconut_tools/__init__.py

# Évite tout import de sous-modules ici (pas de: import rotation_angle, etc.)
# Laisse le package s'importer "à vide" pour RTD.

from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("coconut-tools")
except PackageNotFoundError:
    __version__ = "0.0.0"