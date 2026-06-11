from pathlib import Path
from datetime import datetime
import pytest

@pytest.mark.bigdata
def test_plots():
    """
    End-to-end test:
      1) Plot profile from HDF -> test_plot_hdf.png
      2) Plot 2D surface (one time) from DAT -> test_surface_plot.png
    All outputs are written under tests/_outputs/
    """

    # --- Persistent output base (your conftest.py will remove tests/_outputs/ at session end) ---
    out_base = Path(__file__).parent / "_outputs"
    out_base.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------------------------------------
    # 1) Plot profile from HDF -> test_plot_hdf.png
    # --------------------------------------------------------------------------------------------
    from coconut_tools.plot.plot import plot_boundary_profil, Surface_2D_onetime

    plot_png = out_base / "test_plot_hdf.png"

    label_dict = {
        # key is the filename in the input directory
        "CFData_CFMESH_A.hdf5": "CME1",
    }
    color_map = {
        "CME1": "blue",
    }

    intputdir=str(out_base)+"/"

    plot_boundary_profil(
        inputdir=intputdir,
        outputfile=str(plot_png),
        label_dict=label_dict,
        color_map=color_map,
    )

    assert plot_png.exists(), "HDF profile plot PNG was not created"
    assert plot_png.stat().st_size > 0, "HDF profile plot PNG is empty"

    # --------------------------------------------------------------------------------------------
    # 4) Plot 2D surface (one time) from DAT -> test_surface_plot.png
    # --------------------------------------------------------------------------------------------
    surf_png = out_base / "test_surface_plot.png"
    dat_path = out_base / "test.dat"


    Surface_2D_onetime(
        inputfile=str(dat_path),
        outputfile=str(surf_png),
        mode="all",
    )

    assert surf_png.exists(), "Surface 2D plot PNG was not created"
    assert surf_png.stat().st_size > 0, "Surface 2D plot PNG is empty"
