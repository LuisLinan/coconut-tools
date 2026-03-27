from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv


def _build_prism_nodes(center_x: float, start_index: int) -> tuple[list[tuple[float, float, float]], str]:
    """Create a small prism centered on the x-axis."""
    yz_points = [
        (0.0, 0.2),
        (0.17320508075688773, -0.1),
        (-0.17320508075688773, -0.1),
    ]
    nodes = []
    for x_shift in (-0.1, 0.1):
        for y_coord, z_coord in yz_points:
            nodes.append((center_x + x_shift, y_coord, z_coord))

    connectivity = " ".join(str(start_index + i) for i in range(6))
    connectivity += f" {start_index // 6}"
    return nodes, connectivity


def _write_mini_cfmesh(cfmesh_path: Path) -> list[str]:
    """Write a tiny CFmesh file with four prism cells."""
    centers_x = [10.5, 13.0, 16.0, 25.0]
    all_nodes: list[tuple[float, float, float]] = []
    connectivity_rows: list[str] = []
    for cell_index, center_x in enumerate(centers_x):
        nodes, connectivity = _build_prism_nodes(center_x, cell_index * 6)
        all_nodes.extend(nodes)
        connectivity_rows.append(connectivity)

    state_lines = [
        "1.0000000000000000e+00 1.0000000000000001e-01 0.0000000000000000e+00 0.0000000000000000e+00 1.0000000000000000e-02 0.0000000000000000e+00 0.0000000000000000e+00 5.0000000000000003e-02 9.0000000000000000e+00\n",
        "1.1000000000000001e+00 1.5000000000000000e-01 1.0000000000000001e-02 0.0000000000000000e+00 2.0000000000000000e-02 1.0000000000000000e-03 0.0000000000000000e+00 6.0000000000000005e-02 8.0000000000000000e+00\n",
        "2.0000000000000000e+00 2.0000000000000001e-01 2.0000000000000000e-02 0.0000000000000000e+00 3.0000000000000002e-02 2.0000000000000000e-03 0.0000000000000000e+00 1.0000000000000001e-01 7.0000000000000000e+00\n",
        "4.0000000000000000e+00 3.0000000000000004e-01 3.0000000000000002e-02 0.0000000000000000e+00 4.0000000000000001e-02 3.0000000000000001e-03 0.0000000000000000e+00 2.0000000000000001e-01 6.0000000000000000e+00\n",
    ]

    lines = [
        "!COOLFLUID_VERSION 2013.9\n",
        "!CFMESH_FORMAT_VERSION 1.3\n",
        "!NB_DIM 3\n",
        "!NB_EQ 9\n",
        f"!NB_NODES {len(all_nodes)} 0\n",
        "!NB_STATES 4 0\n",
        "!NB_ELEM 4\n",
        "!NB_ELEM_TYPES 1\n",
        "!GEOM_POLYORDER 1\n",
        "!SOL_POLYORDER 0\n",
        "!ELEM_TYPES Prism \n",
        "!NB_ELEM_PER_TYPE 4\n",
        "!NB_NODES_PER_TYPE 6\n",
        "!NB_STATES_PER_TYPE 1\n",
        "!LIST_ELEM \n",
    ]
    lines.extend(f"{row}\n" for row in connectivity_rows)
    lines.extend(
        [
            "!NB_TRSs 2\n",
            "!TRS_NAME Inlet\n",
            "!NB_TRs 1\n",
            "!NB_GEOM_ENTS 1\n",
            "!GEOM_TYPE Face\n",
            "!LIST_GEOM_ENT\n",
            "0\n",
            "!TRS_NAME Outlet\n",
            "!NB_TRs 1\n",
            "!NB_GEOM_ENTS 1\n",
            "!GEOM_TYPE Face\n",
            "!LIST_GEOM_ENT\n",
            "1\n",
            "!EXTRA_VARS \n",
            "!LIST_NODE \n",
        ]
    )
    lines.extend(f"{x:.16e} {y:.16e} {z:.16e}\n" for x, y, z in all_nodes)
    lines.append("!LIST_STATE 1\n")
    lines.extend(state_lines)
    lines.extend(["!END\n"] * 6)

    cfmesh_path.write_text("".join(lines), encoding="utf-8", newline="\n")
    return state_lines


def _read_state_lines(cfmesh_path: Path) -> tuple[int, int, list[str], dict[str, int]]:
    """Read state lines and key counters from a mini CFmesh."""
    lines = cfmesh_path.read_text(encoding="utf-8").splitlines(keepends=True)
    state_header_idx = lines.index("!LIST_STATE 1\n")
    state_start = state_header_idx + 1
    state_end = state_start
    while state_end < len(lines) and not lines[state_end].startswith("!END"):
        state_end += 1

    headers = {}
    for key in ("!NB_NODES", "!NB_STATES", "!NB_ELEM", "!NB_EQ"):
        header_line = next(line for line in lines if line.startswith(key))
        headers[key] = int(header_line.split()[1])
    return state_start, state_end, lines, headers


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


def test_apply_spheromak_to_cfmesh_preserves_structure_and_units(tmp_path: Path):
    from coconut_tools.CIT.cfmesh_spheromak import (
        apply_spheromak_to_cfmesh,
        code_pressure_and_density_to_temperature,
    )

    input_cfmesh = tmp_path / "mini.CFmesh"
    original_state_lines = _write_mini_cfmesh(input_cfmesh)
    config_path = tmp_path / "config.ini"
    output_dir = tmp_path / "outputs"

    config_path.write_text(
        "\n".join(
            [
                "[Paths]",
                f"input_cfmesh = {input_cfmesh}",
                f"output_dir = {output_dir}",
                "case_name = mini_case",
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
                "write_vtu_before = true",
                "write_vtu_after = true",
                "write_vts_before = true",
                "write_vts_after = true",
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
    assert result.before_vtu is not None and result.before_vtu.exists()
    assert result.after_vtu is not None and result.after_vtu.exists()
    assert result.before_vts is not None and result.before_vts.exists()
    assert result.after_vts is not None and result.after_vts.exists()
    assert result.modified_cell_count == 2

    original_state_start, original_state_end, original_lines, original_headers = _read_state_lines(input_cfmesh)
    output_state_start, output_state_end, output_lines, output_headers = _read_state_lines(result.output_cfmesh)

    assert len(original_lines) == len(output_lines)
    assert original_state_start == output_state_start
    assert original_state_end == output_state_end
    assert original_headers == output_headers

    for idx, (src_line, dst_line) in enumerate(zip(original_lines, output_lines)):
        if idx in (original_state_start, original_state_start + 1):
            continue
        assert src_line == dst_line

    output_state_rows = [
        np.fromstring(output_lines[original_state_start + offset].strip(), sep=" ")
        for offset in range(4)
    ]
    output_state_rows = np.vstack(output_state_rows)

    # Cells 0 and 1 are inside the sphere, 2 and 3 must stay untouched.
    assert np.array_equal(output_state_rows[2], np.fromstring(original_state_lines[2].strip(), sep=" "))
    assert np.array_equal(output_state_rows[3], np.fromstring(original_state_lines[3].strip(), sep=" "))

    assert np.isclose(output_state_rows[0, 0], 3.0)
    assert np.isclose(output_state_rows[1, 0], 3.0)
    assert np.isclose(output_state_rows[0, 1], 1.0)
    assert np.isclose(output_state_rows[1, 1], 1.0)
    assert np.isclose(output_state_rows[0, 2], 0.0)
    assert np.isclose(output_state_rows[1, 2], 0.0)
    assert np.isclose(output_state_rows[0, 3], 0.0)
    assert np.isclose(output_state_rows[1, 3], 0.0)
    assert np.isclose(output_state_rows[0, 7], 0.3)
    assert np.isclose(output_state_rows[1, 7], 0.3)
    assert np.isclose(output_state_rows[0, 8], 9.0)
    assert np.isclose(output_state_rows[1, 8], 8.0)

    expected_temperature = code_pressure_and_density_to_temperature(
        np.array([3.0]),
        np.array([0.3]),
    )[0]
    ambient_temperature = code_pressure_and_density_to_temperature(
        np.array([2.0]),
        np.array([0.1]),
    )[0]
    assert np.isclose(expected_temperature, 2.0 * ambient_temperature)

    assert not np.isclose(output_state_rows[0, 4], 0.01)
    assert not np.isclose(output_state_rows[1, 4], 0.02)

    before_mesh = pv.read(result.before_vtu)
    after_mesh = pv.read(result.after_vtu)
    assert "inside_spheromak" in before_mesh.cell_data
    assert "Bx" in after_mesh.cell_data
    assert before_mesh.n_cells == 4
    assert after_mesh.n_cells == 4
    assert int(after_mesh.cell_data["inside_spheromak"].sum()) == 2
    assert np.isclose(after_mesh.bounds[0], 10.4)

    sliced = after_mesh.slice(normal="x", origin=(10.5, 0.0, 0.0))
    assert sliced.n_cells > 0
