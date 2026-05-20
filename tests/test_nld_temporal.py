import os
from pathlib import Path

import pytest


def _center_crop(Br, Theta, Phi, n_theta=12, n_phi=12):
    i0 = Br.shape[0] // 2 - n_theta // 2
    i1 = i0 + n_theta
    return Br[i0:i1, :n_phi], Theta[i0:i1, :n_phi], Phi[i0:i1, :n_phi]


@pytest.mark.bigdata
def test_nld_temporal_pipeline(tmp_path: Path):
    """End-to-end temporal GONG pipeline for the nonlinear diffusion filter."""
    os.environ.setdefault("MPLBACKEND", "Agg")

    outdir = tmp_path / "nld"
    rawdir = tmp_path / "raw"
    outdir.mkdir(parents=True, exist_ok=True)
    rawdir.mkdir(parents=True, exist_ok=True)

    date = "2011-09-04T12:00:00"
    map_type = "GONG"
    lmax = 3
    output_path_fig = outdir / "gong_nld_20110904T120000.png"

    from coconut_tools.magnetogram.NLD_implicit_method import filter_radial_field
    from coconut_tools.magnetogram.sph_filtering import (
        correct_net_flux,
        generate_output_and_interpolation_map_names,
        plot_maps,
        read_interpolated_magnetogram,
        write_bc_file,
    )

    output_name, local_files, selection = generate_output_and_interpolation_map_names(
        date,
        map_type,
        outdir.as_posix() + "/",
        lmax,
        method_used="NLD",
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
    assert Br.shape == Theta.shape == Phi.shape == Br_linear.shape

    Br, Theta, Phi = _center_crop(Br, Theta, Phi)
    Br = correct_net_flux(Br, Theta[:, 0])

    Br_filtered, timestep = filter_radial_field(
        Br,
        Phi[0, :],
        Theta[:, 0],
        iterations=1,
        tau=0.5,
        apply_gaussian=True,
        gaussian_sigma=1.0,
        dx_override=1.0,
        dy_override=1.0,
    )

    assert Br_filtered.shape == Br.shape
    assert timestep == 0.5

    write_bc_file(output_name, Br_filtered, Theta[:, 0], Phi[0, :], r_st=1.0)
    output_file = Path(output_name)
    assert output_file.exists()
    assert output_file.stat().st_size > 0

    plot_maps(
        Br,
        Br_filtered,
        Theta[:, 0],
        Phi[0, :],
        map_type,
        "sinlat",
        output_path=str(output_path_fig),
        date=date,
    )

    assert output_path_fig.exists()
    assert output_path_fig.stat().st_size > 0
