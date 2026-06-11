from datetime import datetime

import numpy as np
import pytest
from astropy.io import fits

from coconut_tools.magnetogram import sph_filtering


def test_read_magnetogram_flips_decreasing_longitude(tmp_path):
    file_path = tmp_path / "hmi.Synoptic_Mr_small.2238.fits"
    data = np.array([[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]])
    hdu = fits.PrimaryHDU(data)
    hdu.header["CDELT1"] = -1.0
    hdu.writeto(file_path)

    Br, _, _ = sph_filtering.read_magnetogram(str(file_path), "HMI_small")

    np.testing.assert_array_equal(Br, data[::-1, ::-1])


def test_rotate_longitude_to_stonyhurst_uses_negative_shift():
    Br = np.arange(8).reshape(1, 8)

    rotated = sph_filtering.rotate_longitude_to_stonyhurst(Br, 90.0)

    np.testing.assert_array_equal(rotated, np.roll(Br, -2, axis=1))


def test_rotate_wso_longitude_preserves_duplicate_endpoint():
    Br = np.array([[0, 1, 2, 3, 0]])

    rotated = sph_filtering.rotate_longitude_to_stonyhurst(
        Br,
        90.0,
        has_duplicate_endpoint=True,
    )

    np.testing.assert_array_equal(rotated, [[1, 2, 3, 0, 1]])


def test_closest_longitude_column_uses_physical_cell_centers():
    longitude = np.arange(360, dtype=float) + 0.5

    index, residual = sph_filtering.closest_longitude_column(longitude, 241.766)

    assert index == 241
    assert residual == pytest.approx(-0.266)


def test_regular_phi_starts_at_zero_without_duplicate_endpoint():
    Br = np.zeros((2, 360))

    _, phi = sph_filtering.build_regular_theta_phi(Br, "GONG")

    assert phi[0] == 0.0
    assert phi[-1] == pytest.approx(np.deg2rad(359.0))


def test_temporal_gong_processed_axis_matches_preprocessing_shift(tmp_path):
    file_path = tmp_path / "mrzqs201207t1504c2238_181.fits"
    hdu = fits.PrimaryHDU(np.zeros((2, 360)))
    hdu.header["CRPIX1"] = 180.5
    hdu.header["CRVAL1"] = 1.0
    hdu.header["CDELT1"] = 1.0
    hdu.writeto(file_path)

    longitude = sph_filtering.processed_longitude_axis(
        str(file_path),
        "GONG",
        temporal=True,
    )
    index, residual = sph_filtering.closest_longitude_column(longitude, 241.18)

    assert longitude[0] % 360.0 == pytest.approx(1.5)
    assert index == 240
    assert residual == pytest.approx(0.32)


def test_configured_longitude_rotation_can_be_disabled():
    Br = np.arange(8).reshape(1, 8)
    Br_linear = Br.copy()

    result_br, result_linear, angle = sph_filtering.apply_configured_longitude_rotation(
        Br,
        Br_linear,
        "magnetogram.fits",
        "GONG",
        datetime(2020, 12, 7, 15, 0),
        use_interpolation=False,
        rotate_to_stonyhurst=False,
    )

    assert result_br is Br
    assert result_linear is Br_linear
    assert angle is None


def test_interpolated_gong_uses_target_date_central_meridian(monkeypatch, tmp_path):
    Br = np.arange(8, dtype=float).reshape(1, 8)
    captured = {}

    monkeypatch.setattr(
        sph_filtering,
        "generate_output_and_interpolation_map_names",
        lambda *args, **kwargs: (
            str(tmp_path / "boundary.dat"),
            ["map0", "map1", "map2", "map3"],
            object(),
        ),
    )
    monkeypatch.setattr(
        sph_filtering,
        "read_interpolated_magnetogram",
        lambda *args, **kwargs: (
            Br.copy(),
            np.zeros_like(Br),
            np.zeros_like(Br),
            Br.copy(),
        ),
    )
    monkeypatch.setattr(
        sph_filtering,
        "compute_carrington_central_meridian",
        lambda date: 90.0,
    )
    monkeypatch.setattr(
        sph_filtering,
        "processed_longitude_axis",
        lambda *args, **kwargs: np.arange(8) * 45.0,
    )
    monkeypatch.setattr(
        sph_filtering,
        "compute_rotation_angle",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Native GONG angle must not be reused after interpolation.")
        ),
    )

    def fake_project(projected_br, *args, **kwargs):
        captured["Br"] = projected_br
        return projected_br, np.array([])

    monkeypatch.setattr(sph_filtering, "project_and_reconstruct", fake_project)

    result = sph_filtering.process_magnetogram_date(
        {
            "date": "2020-12-07T15:00:00",
            "map_type": "GONG",
            "interpolation": True,
            "rotate_to_stonyhurst": True,
            "write_map": False,
            "show_map": False,
            "output_dir": str(tmp_path),
        },
        datetime(2020, 12, 7, 15, 0),
    )

    np.testing.assert_array_equal(captured["Br"], np.roll(Br, -2, axis=1))
    np.testing.assert_array_equal(result["Br_linear"], np.roll(Br, -2, axis=1))
    assert result["rotation_angle"] == 90.0
