from pathlib import Path
import os
import pytest

@pytest.mark.bigdata
def test_pyvista_slice_saves_png(big_vtu_path: Path, monkeypatch):
    """
    Integration test for the pyvista slice pipeline:
    - read -> convert_units -> convert_to_spherical -> visualize
    - uses the cached VTU from _bigdata_cache
    - renders off-screen and writes a PNG
    - asserts the PNG exists and is non-empty
    """
    # Ensure off-screen rendering for headless environments/CI
    os.environ["PYVISTA_OFF_SCREEN"] = "true"
    try:
        import pyvista as pv
        pv.global_theme.off_screen = True
    except Exception:
        # If pyvista is not available or fails to import, let the test fail clearly later
        pass

    # Persistent output directory (will be cleaned by pytest_sessionfinish as you configured)
    outdir = Path(__file__).parent / "_outputs"
    outdir.mkdir(parents=True, exist_ok=True)
    out_png = outdir / "pyvista_slice.png"

    # Import the pipeline pieces
    from coconut_tools.pyvista_slice import (
        read_mesh,
        convert_units,
        convert_to_spherical,
        visualize,
    )

    # Run the pipeline
    mesh = read_mesh(str(big_vtu_path))
    mesh = convert_units(mesh)
    mesh = convert_to_spherical(mesh)

    # Save the figure
    visualize(
        mesh,
        slice_normal="y",
        save_path=str(out_png),
        show=False,           # important for CI/headless
    )

    # Assertions
    assert out_png.exists(), "PyVista slice PNG was not created"
    assert out_png.stat().st_size > 0, "PyVista slice PNG is empty"
