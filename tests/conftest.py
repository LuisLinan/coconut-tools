# tests/conftest.py
from __future__ import annotations
from pathlib import Path
import pytest
import pooch
import shutil

DATASET = "LuisLinan/coconut-tools-testdata"

URLS = {
    "vtu": f"https://huggingface.co/datasets/LuisLinan/coconut-tools-testdata/resolve/main/corona-mhd_0.vtu",
    "cfmesh": f"https://huggingface.co/datasets/LuisLinan/coconut-tools-testdata/resolve/main/corona.CFmesh",
}

# (Optionnel) remplace "unverified" par les vrais SHA256 si tu veux sécuriser les téléchargements
HASHES = {
    "vtu": "A53AEC75A921F454E4DC7BD7CAD7056804F43A5397684662999EDB9327EFF9F2",
    "cfmesh": "DE039B0CB873976735A5CBE592C74064C11E5475794CA57102CB4F7DAF04F06A",
}

def _cache_dir() -> Path:
    """Retourne le dossier local de cache des gros fichiers."""
    data = Path(__file__).parent / "_bigdata_cache"
    data.mkdir(exist_ok=True)
    return data

def _fetch(name: str) -> Path:
    url = URLS[name]
    return Path(
        pooch.retrieve(
            url=url,
            known_hash=HASHES[name],
            fname=Path(url).name,
            path=_cache_dir(),
            downloader=pooch.HTTPDownloader(progressbar=True),
        )
    )

def pytest_addoption(parser):
    parser.addoption(
        "--prefetch-bigdata",
        action="store_true",
        default=False,
        help="Télécharge au lancement de pytest les gros fichiers (VTU + CFmesh).",
    )
    parser.addoption(
        "--cleanup-bigdata",
        action="store_true",
        default=False,
        help="Supprime les gros fichiers après la session pytest.",
    )

@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart(session):
    """Si --prefetch-bigdata est passé, télécharge les fichiers au début."""
    if session.config.getoption("--prefetch-bigdata"):
        for key in ("vtu", "cfmesh"):
            path = _fetch(key)
            print(f"[prefetch] Downloaded: {path} ({path.stat().st_size/1e6:.1f} MB)")

@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    """Si --cleanup-bigdata est passé, supprime le cache après les tests."""
    if session.config.getoption("--cleanup-bigdata"):
        cache_dir = _cache_dir()
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            print(f"[cleanup] Removed bigdata cache: {cache_dir}")
    # 2. Always remove the outputs folder (tests/_outputs)
    outputs_dir = Path(__file__).parent / "_outputs"
    """
    if outputs_dir.exists():
        shutil.rmtree(outputs_dir)
        print(f"[cleanup] Removed test outputs: {outputs_dir}")
    """


@pytest.fixture(scope="session")
def big_vtu_path() -> Path:
    return _fetch("vtu")

@pytest.fixture(scope="session")
def big_cfmesh_path() -> Path:
    return _fetch("cfmesh")