from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest


def _sha256sum(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_create_example_config(tmp_path: Path):
    from coconut_tools.CIT.cfmesh_spheromak import SpheromakInsertionConfig, create_example_config

    config_path = tmp_path / "example.ini"
    create_example_config(config_path)
    config = SpheromakInsertionConfig.from_ini(config_path)

    assert config.case_name == "spheromak_test"
    assert config.center_radius_rsun == 10.0
    assert config.radius_rsun == 5.0
    assert config.mass_density_kg_m3 is None
    assert config.temperature_k is None


@pytest.mark.bigdata
def test_apply_spheromak_to_cached_cfmesh(big_cfmesh_path: Path, tmp_path: Path):
    from coconut_tools.CIT.cfmesh_spheromak import (
        apply_spheromak_to_cfmesh,
        scan_cfmesh_sections,
    )

    config_path = tmp_path / "config.ini"
    output_dir = tmp_path / "outputs"

    config_path.write_text(
        "\n".join(
            [
                "[Paths]",
                f"input_cfmesh = {big_cfmesh_path}",
                f"output_dir = {output_dir}",
                "case_name = big_case",
                "",
                "[Spheromak]",
                "lat_deg = 0.0",
                "lon_deg = 0.0",
                "radius_rsun = 5.0",
                "speed_km_s = 480.248",
                "mass_density_kg_m3 = auto",
                "temperature_k = auto",
                "helicity_sign = 1",
                "tilt_deg = 0.0",
                "toroidal_flux_wb = 1.0e12",
                "",
                "[Placement]",
                "center_radius_rsun = 10.0",
                "",
                "[Plasma]",
                "density_factor = 1.5",
                "temperature_factor = 2.0",
                "",
                "[Visualization]",
                "write_vtu_before = false",
                "write_vtu_after = false",
                "write_vts_before = false",
                "write_vts_after = false",
                "vts_nb_r = 3",
                "vts_nb_theta = 4",
                "vts_nb_phi = 5",
                "vts_eps = 0.05",
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )

    result = apply_spheromak_to_cfmesh(config_path)

    assert result.output_cfmesh.exists()
    assert result.before_vtu is None
    assert result.after_vtu is None
    assert result.before_vts is None
    assert result.after_vts is None
    assert result.modified_cell_count > 0

    input_sections = scan_cfmesh_sections(big_cfmesh_path)
    output_sections = scan_cfmesh_sections(result.output_cfmesh)
    assert output_sections == input_sections

    assert result.output_cfmesh.parent == output_dir / "big_case"
    assert _sha256sum(result.output_cfmesh) != _sha256sum(big_cfmesh_path)
