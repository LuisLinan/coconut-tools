import numpy as np
import pytest
from astropy.io import fits

from coconut_tools.magnetogram import sph_filtering
from coconut_tools.magnetogram.play_with_the_frame import fits_latitude_axis
from coconut_tools.magnetogram.sph_filtering import (
    build_theta_phi,
    correct_net_flux,
    project_and_reconstruct,
    read_fits_theta_axis,
    read_interpolated_magnetogram,
    read_magnetogram,
    spherical_pixel_areas,
    theta_cell_edges,
    write_bc_file,
)


def _write_map(path, *, size=4, latitude_mode=None, cdelt2=0.5):
    data = np.arange(size * 4.0).reshape(size, 4)
    hdu = fits.PrimaryHDU(data=data)
    hdu.header["CRPIX1"] = 1.0
    hdu.header["CRVAL1"] = 0.0
    hdu.header["CDELT1"] = 90.0
    hdu.header["CRPIX2"] = size / 2.0 + 0.5
    hdu.header["CRVAL2"] = 0.0
    hdu.header["CDELT2"] = cdelt2
    if latitude_mode == "explicit_sine":
        hdu.header["CTYPE2"] = "CRLT-CEA"
        hdu.header["CUNIT2"] = "Sine Latitude"
    elif latitude_mode == "gong_cea":
        hdu.header["CTYPE2"] = "CRLT-CEA"
    elif latitude_mode == "standard_cea":
        hdu.header["CTYPE1"] = "CRLN-CEA"
        hdu.header["CTYPE2"] = "CRLT-CEA"
        hdu.header["CUNIT1"] = "deg"
        hdu.header["CUNIT2"] = "deg"
        hdu.header["PV2_1"] = 1.0
    elif latitude_mode == "latitude":
        hdu.header["LATTYPE"] = 0
        hdu.header["CUNIT2"] = "deg"
    hdu.writeto(path)
    return data


@pytest.mark.parametrize("latitude_mode", ["explicit_sine", "gong_cea"])
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_sine_latitude_headers_produce_physical_centers(
    tmp_path,
    latitude_mode,
    sign,
):
    path = tmp_path / f"{latitude_mode}_{sign:+.0f}.fits"
    data = _write_map(
        path,
        latitude_mode=latitude_mode,
        cdelt2=sign * 0.5,
    )

    theta, flip_rows = read_fits_theta_axis(str(path), "GONG")

    expected_mu = np.array([0.75, 0.25, -0.25, -0.75])
    np.testing.assert_allclose(theta, np.arccos(expected_mu))
    assert flip_rows is (sign > 0.0)
    assert not np.isclose(theta[0], 0.0)
    assert not np.isclose(theta[-1], np.pi)

    field, Theta, _ = read_magnetogram(str(path), "GONG")
    expected_field = data[::-1, :] if sign > 0.0 else data
    np.testing.assert_array_equal(field, expected_field)
    np.testing.assert_allclose(Theta[:, 0], theta)


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_standard_angular_cea_header_is_inverted_by_wcs(tmp_path, sign):
    path = tmp_path / f"hmi_cea_{sign:+.0f}.fits"
    _write_map(
        path,
        latitude_mode="standard_cea",
        cdelt2=sign * np.degrees(0.5),
    )

    theta, flip_rows = read_fits_theta_axis(str(path), "HMI_small")

    np.testing.assert_allclose(
        np.cos(theta),
        [0.75, 0.25, -0.25, -0.75],
        atol=1.0e-12,
    )
    assert flip_rows is (sign > 0.0)


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_adapt_linear_latitude_header_keeps_centered_latitudes(tmp_path, sign):
    path = tmp_path / f"adapt_{sign:+.0f}.fits"
    _write_map(path, latitude_mode="latitude", cdelt2=sign * 45.0)

    theta, flip_rows = read_fits_theta_axis(str(path), "ADAPT")

    expected = (np.arange(4, dtype=float) + 0.5) * np.pi / 4.0
    np.testing.assert_allclose(theta, expected)
    assert flip_rows is (sign > 0.0)


def test_cd2_2_is_accepted_for_linear_latitude(tmp_path):
    path = tmp_path / "adapt_cd_matrix.fits"
    _write_map(path, latitude_mode="latitude", cdelt2=45.0)
    with fits.open(path, mode="update") as hdul:
        del hdul[0].header["CDELT2"]
        hdul[0].header["CD2_2"] = 45.0

    theta, _ = read_fits_theta_axis(str(path), "ADAPT")

    np.testing.assert_allclose(
        theta,
        (np.arange(4, dtype=float) + 0.5) * np.pi / 4.0,
    )


def test_missing_metadata_uses_warned_centered_product_fallback(tmp_path, monkeypatch):
    path = tmp_path / "map_without_latitude_wcs.fits"
    fits.PrimaryHDU(np.zeros((180, 4))).writeto(path)
    warnings = []
    monkeypatch.setattr(
        sph_filtering.logger,
        "warning",
        lambda message, *args: warnings.append(message % args),
    )

    gong_theta, gong_flip = read_fits_theta_axis(str(path), "GONG")
    adapt_theta, adapt_flip = read_fits_theta_axis(str(path), "ADAPT")

    assert len(warnings) == 2
    assert all("Incomplete FITS latitude metadata" in warning for warning in warnings)
    assert gong_flip and adapt_flip
    assert np.degrees(gong_theta[0]) == pytest.approx(
        np.degrees(np.arccos(1.0 - 1.0 / 180.0))
    )
    assert np.degrees(adapt_theta[0]) == pytest.approx(0.5)
    assert not np.isclose(gong_theta[0], 0.0)
    assert not np.isclose(adapt_theta[0], 0.0)


def test_invalid_explicit_header_is_rejected_instead_of_falling_back(tmp_path):
    path = tmp_path / "invalid_sine.fits"
    _write_map(path, latitude_mode="explicit_sine", cdelt2=1.0)

    with pytest.raises(ValueError, match=r"outside \[-1, 1\]"):
        read_fits_theta_axis(str(path), "GONG")


@pytest.mark.parametrize(
    ("latitude_mode", "map_type", "expected_first"),
    [
        ("explicit_sine", "HMI_small", np.arccos(0.875)),
        ("latitude", "ADAPT", np.pi / 16.0),
    ],
)
def test_resize_preserves_source_edges_and_native_coordinate(
    tmp_path,
    latitude_mode,
    map_type,
    expected_first,
):
    path = tmp_path / f"resize_{latitude_mode}.fits"
    step = 0.5 if latitude_mode == "explicit_sine" else 45.0
    _write_map(path, latitude_mode=latitude_mode, cdelt2=step)

    source, _ = read_fits_theta_axis(str(path), map_type)
    resized, _ = read_fits_theta_axis(str(path), map_type, target_size=8)

    assert theta_cell_edges(resized)[0] == pytest.approx(theta_cell_edges(source)[0])
    assert theta_cell_edges(resized)[-1] == pytest.approx(theta_cell_edges(source)[-1])
    assert resized[0] == pytest.approx(expected_first)
    assert not np.isclose(resized[0], 0.0)


def test_temporal_interpolation_rejects_different_physical_latitude_axes(tmp_path):
    paths = []
    for index, step in enumerate([0.5, 0.5, 0.5, 0.4]):
        path = tmp_path / f"hmi_hourly_{index}.fits"
        _write_map(path, latitude_mode="explicit_sine", cdelt2=step)
        paths.append(str(path))

    with pytest.raises(RuntimeError, match="identical physical latitude grids"):
        read_interpolated_magnetogram(paths, "HMI_hourly", selection=None)


@pytest.mark.parametrize("coordinate", ["sine", "latitude"])
def test_complete_centered_grids_have_four_pi_solid_angle(coordinate):
    size = 180
    if coordinate == "sine":
        mu = 1.0 - (np.arange(size, dtype=float) + 0.5) * 2.0 / size
        theta = np.arccos(mu)
    else:
        theta = (np.arange(size, dtype=float) + 0.5) * np.pi / size
    phi = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)

    areas = spherical_pixel_areas(theta, phi)

    assert np.sum(areas) == pytest.approx(4.0 * np.pi, abs=1.0e-12)


@pytest.mark.parametrize("coordinate", ["sine", "latitude"])
@pytest.mark.parametrize("method", ["surface_mean", "polarity_scaling"])
def test_flux_corrections_use_the_same_exact_pixel_areas(coordinate, method):
    size = 18
    if coordinate == "sine":
        theta = np.arccos(
            1.0 - (np.arange(size, dtype=float) + 0.5) * 2.0 / size
        )
    else:
        theta = (np.arange(size, dtype=float) + 0.5) * np.pi / size
    phi = np.linspace(0.0, 2.0 * np.pi, 36, endpoint=False)
    Theta, Phi = build_theta_phi(theta, phi)
    field = np.cos(Theta) + 0.2 * np.cos(Phi) + 0.17
    areas = spherical_pixel_areas(theta, phi)

    corrected = correct_net_flux(field, theta, phi, method=method)

    assert np.sum(corrected * areas) == pytest.approx(0.0, abs=2.0e-14)


@pytest.mark.parametrize("coordinate", ["sine", "latitude"])
def test_dipole_projection_is_consistent_on_both_native_grids(coordinate):
    size = 90
    if coordinate == "sine":
        theta = np.arccos(
            1.0 - (np.arange(size, dtype=float) + 0.5) * 2.0 / size
        )
    else:
        theta = (np.arange(size, dtype=float) + 0.5) * np.pi / size
    phi = np.linspace(0.0, 2.0 * np.pi, 180, endpoint=False)
    Theta, Phi = build_theta_phi(theta, phi)
    dipole = np.cos(Theta)

    reconstructed, coefficients = project_and_reconstruct(
        dipole,
        Theta,
        Phi,
        lmax=1,
    )

    assert coefficients.shape == (2,)
    np.testing.assert_allclose(reconstructed, dipole / 2.2, atol=8.0e-5)


@pytest.mark.parametrize(
    ("theta", "phi", "expected_count"),
    [
        (np.array([0.2, 1.0, 2.5]), np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False), 12),
        (np.array([1.0e-10, 1.0, 2.5]), np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False), 12),
        (np.array([0.0, np.pi / 2.0, np.pi]), np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False), 6),
        (np.array([0.0, 1.0, 2.5]), np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False), 9),
    ],
)
def test_bc_writer_counts_only_true_poles_and_preserves_radius(
    tmp_path,
    theta,
    phi,
    expected_count,
):
    path = tmp_path / "boundary.dat"
    field = np.arange(theta.size * phi.size, dtype=float).reshape(theta.size, phi.size)

    write_bc_file(str(path), field, theta, phi, r_st=2.5)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert int(lines[1].split()[1]) == expected_count
    assert len(lines[2:]) == expected_count
    coordinates = np.loadtxt(lines[2:], usecols=(0, 1, 2))
    np.testing.assert_allclose(np.sum(coordinates**2, axis=1), 2.5**2, atol=1.0e-12)


def test_shared_frame_helper_prioritizes_explicit_sine_unit():
    header = fits.Header()
    header["NAXIS2"] = 4
    header["CRPIX2"] = 2.5
    header["CRVAL2"] = 0.0
    header["CDELT2"] = 0.5
    header["CTYPE2"] = "CRLT-CEA"
    header["CUNIT2"] = "Sine Latitude"

    latitude, _, metadata = fits_latitude_axis(header)

    np.testing.assert_allclose(
        latitude,
        np.degrees(np.arcsin([-0.75, -0.25, 0.25, 0.75])),
    )
    assert metadata["detected_mode"] == "sine_to_degrees"
