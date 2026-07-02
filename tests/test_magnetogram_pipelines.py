import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest


DATE = "2011-09-04T12:00:00"
MAP_TYPE = "GONG"


def _center_crop(Br, Theta, Phi, n_theta=8, n_phi=8):
    i0 = Br.shape[0] // 2 - n_theta // 2
    i1 = i0 + n_theta
    return Br[i0:i1, :n_phi], Theta[i0:i1, :n_phi], Phi[i0:i1, :n_phi]


def _assert_artifact(path: Path):
    assert path.exists(), f"Missing output: {path}"
    assert path.stat().st_size > 0, f"Empty output: {path}"


def test_yaroslavsky_spacing_uses_radian_arc_length(monkeypatch):
    from coconut_tools.magnetogram import Yaroslavsky_filter

    captured = {}

    def fake_filter3(image, dx, dy, alpha, Rn):
        captured.update({"dx": dx, "dy": dy, "alpha": alpha, "Rn": Rn})
        return image.copy()

    monkeypatch.setattr(Yaroslavsky_filter, "filter3", fake_filter3)

    Br = np.ones((4, 5))
    theta = np.array([0.0, 0.2, 0.5, 0.9])
    phi = np.linspace(0.0, 2.0 * np.pi, Br.shape[1], endpoint=False)

    result = Yaroslavsky_filter.filter_radial_field_weighted(
        Br,
        phi,
        theta,
        alpha_factor=1.4,
        Rn=2,
        sig=0.0,
    )

    expected_delta = 696.34e6 * max(np.median(np.diff(theta)), np.median(np.diff(phi)))
    assert np.array_equal(result, Br)
    assert captured["dx"] == pytest.approx(expected_delta)
    assert captured["dy"] == pytest.approx(expected_delta)
    assert captured["alpha"] == 1.4
    assert captured["Rn"] == 2


def _write_dat_and_png(outdir, name, Br_input, Br_output, Theta, Phi):
    from coconut_tools.magnetogram.sph_filtering import plot_maps, write_bc_file

    dat_path = outdir / f"{name}.dat"
    png_path = outdir / f"{name}.png"

    write_bc_file(str(dat_path), Br_output, Theta[:, 0], Phi[0, :], r_st=1.0)
    plot_maps(
        Br_input,
        Br_output,
        Theta[:, 0],
        Phi[0, :],
        MAP_TYPE,
        "sinlat",
        output_path=str(png_path),
        date=DATE,
    )

    _assert_artifact(dat_path)
    _assert_artifact(png_path)


def _run_three_filters(outdir, prefix, Br, Theta, Phi):
    from coconut_tools.magnetogram.NLD_implicit_method import filter_radial_field
    from coconut_tools.magnetogram.Yaroslavsky_filter import filter_radial_field_weighted
    from coconut_tools.magnetogram.sph_filtering import (
        correct_net_flux,
        project_and_reconstruct,
    )

    Br, Theta, Phi = _center_crop(Br, Theta, Phi)
    Br = correct_net_flux(Br, Theta[:, 0], Phi[0, :], method="surface_mean")

    Br_sph, coefbr = project_and_reconstruct(Br, Theta, Phi, lmax=3, amp=1, alpha=0)
    assert Br_sph.shape == Br.shape
    assert coefbr.size > 0
    _write_dat_and_png(outdir, f"{prefix}_sph", Br, Br_sph, Theta, Phi)

    Br_nld, timestep = filter_radial_field(
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
    assert Br_nld.shape == Br.shape
    assert timestep == 0.5
    _write_dat_and_png(outdir, f"{prefix}_nld", Br, Br_nld, Theta, Phi)

    Br_yaroslavsky = filter_radial_field_weighted(
        Br,
        Phi[0, :],
        Theta[:, 0],
        alpha_factor=1.1,
        Rn=1,
        sig=0.0,
    )
    assert Br_yaroslavsky.shape == Br.shape
    _write_dat_and_png(outdir, f"{prefix}_yaroslavsky", Br, Br_yaroslavsky, Theta, Phi)


@pytest.mark.bigdata
def test_real_gong_single_magnetogram_filters_and_outputs():
    os.environ.setdefault("MPLBACKEND", "Agg")

    outdir = Path(__file__).parent / "_outputs"
    workdir = outdir / "magnetogram" / "single"
    workdir.mkdir(parents=True, exist_ok=True)

    from coconut_tools.magnetogram.sph_filtering import (
        apply_configured_longitude_rotation,
        generate_output_and_map_names,
        magnetogram_display_date,
        magnetogram_effective_date,
        read_magnetogram,
    )

    output_name, local_file = generate_output_and_map_names(
        DATE,
        MAP_TYPE,
        str(workdir),
        method_used="sph",
    )

    assert output_name == str(workdir / "map_gong_sph.dat")
    _assert_artifact(Path(local_file))

    Br, Theta, Phi = read_magnetogram(local_file, MAP_TYPE)
    assert Br.ndim == 2
    assert Br.shape == Theta.shape == Phi.shape
    assert np.isfinite(Br).all()

    effective_date = magnetogram_effective_date(local_file, MAP_TYPE, DATE)
    assert effective_date == magnetogram_display_date(local_file, MAP_TYPE, DATE)

    Br, _, rotation_angle = apply_configured_longitude_rotation(
        Br,
        None,
        local_file,
        MAP_TYPE,
        DATE,
        use_interpolation=False,
        rotate_to_stonyhurst=True,
        effective_date=effective_date,
    )
    assert rotation_angle is not None

    _run_three_filters(workdir, "real_gong_single", Br, Theta, Phi)


@pytest.mark.bigdata
def test_real_gong_temporal_interpolation_filters_and_outputs():
    os.environ.setdefault("MPLBACKEND", "Agg")

    outdir = Path(__file__).parent / "_outputs"
    workdir = outdir / "magnetogram" / "temporal"
    rawdir = outdir / "magnetogram" / "raw"
    workdir.mkdir(parents=True, exist_ok=True)
    rawdir.mkdir(parents=True, exist_ok=True)

    from coconut_tools.magnetogram.sph_filtering import (
        apply_configured_longitude_rotation,
        generate_output_and_interpolation_map_names,
        magnetogram_effective_date,
        read_interpolated_magnetogram,
    )

    output_name, local_files, selection = generate_output_and_interpolation_map_names(
        DATE,
        MAP_TYPE,
        str(workdir),
        method_used="sph",
        download_dir=str(rawdir),
    )

    assert output_name == str(workdir / "map_gong_sph.dat")
    assert len(local_files) == 4
    for local_file in local_files:
        _assert_artifact(Path(local_file))

    assert selection.target_date == datetime.fromisoformat(DATE)
    assert selection.coef_before + selection.coef_after == pytest.approx(1.0)

    Br, Theta, Phi, Br_linear = read_interpolated_magnetogram(
        local_files,
        MAP_TYPE,
        selection,
        interpolation_order=2,
    )
    assert Br.shape == Theta.shape == Phi.shape == Br_linear.shape
    assert np.isfinite(Br).all()
    assert magnetogram_effective_date(
        local_files[0],
        MAP_TYPE,
        DATE,
        interpolated=True,
    ) == datetime.fromisoformat(DATE)

    Br, Br_linear, rotation_angle = apply_configured_longitude_rotation(
        Br,
        Br_linear,
        local_files,
        MAP_TYPE,
        DATE,
        use_interpolation=True,
        rotate_to_stonyhurst=True,
        effective_date=DATE,
    )
    assert Br_linear.shape == Br.shape
    assert rotation_angle is not None

    _run_three_filters(workdir, "real_gong_temporal", Br, Theta, Phi)
