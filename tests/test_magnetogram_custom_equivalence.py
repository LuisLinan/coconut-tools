"""Opt-in end-to-end comparison of downloaded and custom magnetograms.

Run explicitly with::

    $env:COCONUT_RUN_MAGNETOGRAM_EQUIVALENCE = "1"
    $env:COCONUT_JSOC_EMAIL = "registered@example.org"
    python -m pytest tests/test_magnetogram_custom_equivalence.py -s

The test downloads every supported FITS product except WSO, executes the SPH
pipeline once through its normal product path, renames the downloaded FITS,
and executes it again through ``custom_magnetogram``.  Artifacts are retained
under ``tests/_outputs/magnetogram_custom_equivalence_20260429``.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil

# The pipeline imports pyplot transitively, so select the non-interactive
# backend before importing it.  Setting this inside the test is too late.
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest

from coconut_tools.magnetogram.sph_filtering import process_config


# CR 2310 midpoint.  This is also the exact filename time of the integral GONG
# CR 2310 maps, keeping every dynamically selected product as close as possible
# to the nominal center date of the static Carrington products.
TARGET_DATE = "2026-04-29T07:01:00"
MAP_TYPES = (
    "GONG_mrzqs",
    "GONG_mrbqs",
    "GONG_mrbqj",
    "GONG_mrmqs",
    "GONG_mrnqs",
    "ADAPT",
    "HMI_small",
    "HMI_polfil",
    "HMI_SYNC",
    "HMI_hourly",
    "HMI_fdt",
)
OUTPUT_ROOT = (
    Path(__file__).parent
    / "_outputs"
    / "magnetogram_custom_equivalence_20260429"
)


def _enabled_map_types() -> set[str]:
    requested = os.environ.get("COCONUT_MAGNETOGRAM_TYPES", "").strip()
    if not requested:
        return set(MAP_TYPES)
    return {item.strip() for item in requested.split(",") if item.strip()}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _renamed_fits_path(source: Path, directory: Path) -> Path:
    lower_name = source.name.casefold()
    if lower_name.endswith(".fits.gz"):
        suffix = ".fits.gz"
    elif lower_name.endswith(".fts.gz"):
        suffix = ".fts.gz"
    else:
        suffix = source.suffix or ".fits"
    return directory / f"renamed_input{suffix}"


def _pipeline_config(
    map_type: str,
    output_dir: Path,
    figure_path: Path,
    date: str = TARGET_DATE,
) -> dict[str, object]:
    config: dict[str, object] = {
        "map_type": map_type,
        "custom_magnetogram": None,
        "lmax": 20,
        "amp": 1,
        "write_map": True,
        "show_map": True,
        "visu_type": "sinlat",
        "alpha": 3 * 10 ** (-6),
        "rotate_to_stonyhurst": True,
        "interpolation": False,
        "interpolation_order": 2,
        "resize": True,
        "flux_correct": False,
        "date": date,
        "adapt_map": 6,
        "output_dir": str(output_dir),
        "download_dir": str(output_dir),
        "output_path_fig": str(figure_path),
    }
    jsoc_email = os.environ.get("COCONUT_JSOC_EMAIL")
    if jsoc_email:
        config["drms_email"] = jsoc_email
    return config


@pytest.mark.bigdata
@pytest.mark.skipif(
    os.environ.get("COCONUT_RUN_MAGNETOGRAM_EQUIVALENCE") != "1",
    reason="Set COCONUT_RUN_MAGNETOGRAM_EQUIVALENCE=1 to run remote comparisons.",
)
@pytest.mark.parametrize("map_type", MAP_TYPES)
def test_downloaded_and_renamed_custom_magnetograms_are_identical(map_type):
    """Compare complete normal/custom SPH products for one downloaded FITS."""
    if map_type not in _enabled_map_types():
        pytest.skip("Product not selected by COCONUT_MAGNETOGRAM_TYPES.")

    slug = map_type.casefold()
    product_root = OUTPUT_ROOT / slug
    normal_dir = product_root / "normal"
    custom_dir = product_root / "custom"
    renamed_dir = product_root / "renamed"
    image_dir = OUTPUT_ROOT / "images"
    for directory in (normal_dir, custom_dir, renamed_dir, image_dir):
        directory.mkdir(parents=True, exist_ok=True)

    normal_config = _pipeline_config(
        map_type,
        normal_dir,
        image_dir / f"{slug}_normal.png",
    )
    normal_result = process_config(normal_config, method_used="sph")[0]
    source = Path(normal_result["local_file"])
    assert source.is_file()

    renamed_source = _renamed_fits_path(source, renamed_dir)
    shutil.copy2(source, renamed_source)

    custom_config = _pipeline_config(
        map_type,
        custom_dir,
        image_dir / f"{slug}_custom.png",
        date=normal_result["effective_date"].isoformat(),
    )
    custom_config["custom_magnetogram"] = str(renamed_source)
    custom_result = process_config(custom_config, method_used="sph")[0]

    normal_dat = Path(normal_result["output_name"])
    custom_dat = Path(custom_result["output_name"])
    normal_hash = _sha256(normal_dat)
    custom_hash = _sha256(custom_dat)
    same_coefficients = np.array_equal(
        normal_result["coefbr"],
        custom_result["coefbr"],
        equal_nan=True,
    )
    same_rotation = normal_result["rotation_angle"] == pytest.approx(
        custom_result["rotation_angle"],
        abs=1.0e-12,
        rel=0.0,
    )
    report = {
        "map_type": map_type,
        "target_date": TARGET_DATE,
        "custom_config_date": custom_config["date"],
        "downloaded_fits": str(source.resolve()),
        "renamed_custom_fits": str(renamed_source.resolve()),
        "normal_dat": str(normal_dat.resolve()),
        "custom_dat": str(custom_dat.resolve()),
        "normal_figure": str(Path(normal_result["figure_path"]).resolve()),
        "custom_figure": str(Path(custom_result["figure_path"]).resolve()),
        "normal_effective_date": normal_result["effective_date"].isoformat(),
        "custom_effective_date": custom_result["effective_date"].isoformat(),
        "effective_date_equal": (
            normal_result["effective_date"] == custom_result["effective_date"]
        ),
        "normal_rotation_angle": normal_result["rotation_angle"],
        "custom_rotation_angle": custom_result["rotation_angle"],
        "dat_sha256_normal": normal_hash,
        "dat_sha256_custom": custom_hash,
        "dat_exactly_equal": normal_hash == custom_hash,
        "coefficients_exactly_equal": bool(same_coefficients),
        "rotation_equal": bool(same_rotation),
    }
    (product_root / "comparison.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    assert normal_result["effective_date"] == custom_result["effective_date"]
    assert same_rotation
    assert same_coefficients
    assert normal_hash == custom_hash
