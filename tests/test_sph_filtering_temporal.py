import os
from pathlib import Path

import pytest


@pytest.mark.bigdata
def test_sph_filtering_temporal_pipeline(tmp_path: Path):
    """End-to-end temporal GONG pipeline for spherical harmonic filtering."""
    os.environ.setdefault("MPLBACKEND", "Agg")

    outdir = tmp_path / "sph"
    rawdir = tmp_path / "raw"
    outdir.mkdir(parents=True, exist_ok=True)
    rawdir.mkdir(parents=True, exist_ok=True)

    date = "2011-09-04T12:00:00"
    map_type = "GONG"
    lmax = 3
    output_path_fig = outdir / "gong_sph_20110904T120000.png"

    from coconut_tools.magnetogram.sph_filtering import (
        correct_net_flux,
        generate_output_and_interpolation_map_names,
        plot_maps,
        project_and_reconstruct,
        read_interpolated_magnetogram,
        write_bc_file,
    )

    output_name, local_files, selection = generate_output_and_interpolation_map_names(
        date,
        map_type,
        outdir.as_posix() + "/",
        lmax,
        method_used="sph",
        download_dir=rawdir.as_posix() + "/",
    )

    assert len(local_files) == 4
    for local_file in local_files:
        assert Path(local_file).exists()
        assert Path(local_file).stat().st_size > 0

    Br, Theta, Phi, Br_linear = read_interpolated_magnetogram(
        local_files,
        map_type,
        selection,
        interpolation_order=2,
    )

    assert Br.ndim == 2 and Theta.ndim == 2 and Phi.ndim == 2
    assert Br.shape == Theta.shape == Phi.shape == Br_linear.shape

    Br = correct_net_flux(Br, Theta[:, 0])
    Br_mode, coefbr = project_and_reconstruct(Br, Theta, Phi, lmax, amp=1, alpha=0)

    assert Br_mode.shape == Br.shape
    assert coefbr.size > 0

    write_bc_file(output_name, Br_mode, Theta[:, 0], Phi[0, :], r_st=1.0)
    output_file = Path(output_name)
    assert output_file.exists()
    assert output_file.stat().st_size > 0

    plot_maps(
        Br,
        Br_mode,
        Theta[:, 0],
        Phi[0, :],
        map_type,
        "sinlat",
        output_path=str(output_path_fig),
        date=date,
    )

    assert output_path_fig.exists()
    assert output_path_fig.stat().st_size > 0
