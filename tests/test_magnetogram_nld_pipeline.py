from pathlib import Path
import os
import numpy as np
import pytest

@pytest.mark.bigdata  # optional: remove if you don't want it filtered with -m
def test_magnetogram_nld_pipeline(tmp_path):
    """
    End-to-end test for the NLD implicit method pipeline:
      1) Generate output and magnetogram local file path
      2) Read magnetogram (map)
      3) Filter Br field with nonlinear diffusion
      4) Write boundary condition .dat
      5) Plot maps to a PNG
    All artifacts are written under tests/_outputs/magnetogram/.
    """
    # Ensure headless plotting (no GUI needed)
    os.environ.setdefault("MPLBACKEND", "Agg")

    # Persistent output directory (your conftest.py will clean tests/_outputs/ at session end)
    outdir = Path(__file__).parent / "_outputs" / "magnetogram"
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Configuration (single case) ---
    date = "2011-09-04T12:00:00"
    map_type = "HMI_small"
    lmax = None
    r_st = 1.0
    adapt_map = 6
    tau = 5
    iterations = 7

    # Where to save the diagnostic figure
    output_path_fig = outdir / "hmi_20110904T120000.png"

    # Import the pipeline functions
    from coconut_tools.magnetogram.sph_filtering import (
        read_magnetogram,
        generate_output_and_map_names,
        write_bc_file,
        plot_maps,
    )
    from coconut_tools.magnetogram.NLD_implicit_method import filter_radial_field

    outdir_str = outdir.as_posix() + "/"
    # 1) Resolve output naming and the local magnetogram file path
    #    generate_output_and_map_names may download the magnetogram if needed.
    output_name, local_file = generate_output_and_map_names(
        date, map_type, outdir_str, lmax, "NLD"
    )



    # 2) Read the magnetogram (Br, Theta, Phi grids)
    Br, Theta, Phi = read_magnetogram(local_file, map_type, adapt_map)

    # Sanity on shapes (helps debugging if the source changes)
    assert Br.ndim == 2 and Theta.ndim == 2 and Phi.ndim == 2, "Magnetogram arrays must be 2D"
    assert Br.shape == Theta.shape == Phi.shape, "Br, Theta, Phi must have identical shapes"

    # 3) Nonlinear diffusion filtering (use 1D theta/phi vectors from the 2D mesh)
    Br_filtered, timestep = filter_radial_field(
        Br, Phi[0, :], Theta[:, 0],
        iterations=iterations,
        tau=tau,
        apply_gaussian=True,
        gaussian_sigma=1.0,
        dx_override=1.0,
        dy_override=1.0,
    )

    # Optional clipping as in your script
    threshold = 45.0
    max_val = float(np.max(np.abs(Br_filtered)))
    if max_val > threshold:
        Br_filtered = Br_filtered * (threshold / max_val)

    # 4) Write boundary condition .dat (output_name is typically a stem/path without extension)
    write_bc_file(output_name, Br_filtered, Theta[:, 0], Phi[0, :], r_st)

    # Locate a produced .dat in outdir (write_bc_file may append the extension internally)
    dat_files = list(outdir.glob("*.dat")) + list(outdir.rglob("*.dat"))
    assert dat_files, "No .dat file was produced."
    for df in dat_files:
        assert df.exists(), f"Missing .dat file: {df}"
        assert df.stat().st_size > 0, f"Produced .dat is empty: {df}"

    # 5) Plot maps to PNG
    plot_maps(Br, Br_filtered, Theta[:, 0], Phi[0, :], map_type, "sinlat", output_path=str(output_path_fig))

    assert output_path_fig.exists(), "PNG figure was not created."
    assert output_path_fig.stat().st_size > 0, "PNG figure is empty."
