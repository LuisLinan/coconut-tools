from datetime import datetime
from pathlib import Path
import sys
import types

import numpy as np
import pytest
from astropy.io import fits

from coconut_tools.magnetogram.magnetogram_download import (
    build_output_name,
    default_figure_path,
    magnetogram_display_date,
    magnetogram_effective_date,
    normalize_map_type,
    resolve_figure_path,
)
from coconut_tools.magnetogram.sph_filtering import (
    apply_configured_longitude_rotation,
    closest_longitude_column,
    correct_net_flux,
    read_magnetogram,
    resize_processed_longitude_axis,
    rotate_longitude_to_stonyhurst,
)
from coconut_tools.tools.rotation_angle import is_br_longitude_increasing


TARGET_DATE = datetime(2020, 12, 7, 15, 0)


def test_magnetogram_dates_and_paths_are_resolved_consistently():
    outdir = Path(__file__).parent / "_outputs"

    gong_file = "mrzqs201207t1504c2238_181.fits.gz"
    adapt_file = "adapt40311_044012_202012071400_i00012600n1.fts.gz"

    assert magnetogram_effective_date(gong_file, "GONG", TARGET_DATE) == datetime(
        2020, 12, 7, 15, 4
    )
    assert magnetogram_display_date(adapt_file, "ADAPT", TARGET_DATE) == datetime(
        2020, 12, 7, 14, 0
    )
    assert magnetogram_effective_date(
        "hmi.mrdailysynframe_720s_nrt.20260701_062400_TAI.data.fits",
        "HMI_SYNC",
        datetime(2020, 12, 7, 15, 30, 45),
    ) == datetime(2026, 7, 1, 6, 24)
    assert magnetogram_effective_date(
        "magnetogram.fits",
        "GONG_mrbqs",
        TARGET_DATE,
        interpolated=True,
    ) == TARGET_DATE

    assert build_output_name("GONG_mrbqs", str(outdir), "sph") == str(
        outdir / "map_gong_mrbqs_sph.dat"
    )
    assert default_figure_path(str(outdir), "GONG", TARGET_DATE) == str(
        outdir / "gong_20201207150000.png"
    )
    assert resolve_figure_path(str(outdir), "../", "GONG", TARGET_DATE) == str(
        outdir / "gong_20201207150000.png"
    )
    assert resolve_figure_path(
        str(outdir / "gong.png"),
        "../",
        "GONG",
        TARGET_DATE,
        use_unique_name=True,
    ) == str(outdir / "gong_20201207150000.png")


def test_map_type_normalization_accepts_case_variants():
    outdir = Path(__file__).parent / "_outputs"

    assert normalize_map_type("hmi_sync") == "HMI_SYNC"
    assert normalize_map_type("HMI_SYNC") == "HMI_SYNC"
    assert normalize_map_type("Hmi_Small") == "HMI_small"
    assert normalize_map_type("gong_MRBQS") == "GONG_mrbqs"
    assert normalize_map_type(" adapt ") == "ADAPT"

    assert magnetogram_effective_date(
        "hmi.mrdailysynframe_720s_nrt.20260701_062400_TAI.data.fits",
        "hmi_sync",
        datetime(2020, 12, 7, 15, 30, 45),
    ) == datetime(2026, 7, 1, 6, 24)
    assert build_output_name("hmi_sync", str(outdir), "sph") == str(
        outdir / "map_hmi_sync_sph.dat"
    )
    assert build_output_name("gong_MRBQS", str(outdir), "sph") == str(
        outdir / "map_gong_mrbqs_sph.dat"
    )


def test_local_rotation_helpers_use_the_expected_longitude_convention():
    outdir = Path(__file__).parent / "_outputs" / "magnetogram_helpers"
    outdir.mkdir(parents=True, exist_ok=True)

    file_path = outdir / "mrzqs201207t1504c2238_181.fits"
    data = np.array([[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]])
    hdu = fits.PrimaryHDU(data)
    hdu.header["CRPIX1"] = 1.0
    hdu.header["CRVAL1"] = 0.0
    hdu.header["CDELT1"] = -1.0
    hdu.writeto(file_path, overwrite=True)

    assert is_br_longitude_increasing(str(file_path)) is False

    Br, _, _ = read_magnetogram(str(file_path), "GONG")
    np.testing.assert_array_equal(Br, data[::-1, ::-1])

    Br_lowercase, _, _ = read_magnetogram(str(file_path), "gong")
    np.testing.assert_array_equal(Br_lowercase, data[::-1, ::-1])

    rotated = rotate_longitude_to_stonyhurst(np.arange(8).reshape(1, 8), 90.0)
    np.testing.assert_array_equal(rotated, np.array([[2, 3, 4, 5, 6, 7, 0, 1]]))

    periodic = rotate_longitude_to_stonyhurst(
        np.array([[0, 1, 2, 3, 0]]),
        90.0,
        has_duplicate_endpoint=True,
    )
    np.testing.assert_array_equal(periodic, np.array([[1, 2, 3, 0, 1]]))

    longitude = np.arange(360, dtype=float) + 0.5
    index, residual = closest_longitude_column(longitude, 241.766)
    assert index == 241
    assert residual == pytest.approx(-0.266)


def test_resize_processed_longitude_axis_preserves_origin():
    longitude_original = np.array([181.0, 271.0, 361.0, 451.0])

    longitude_resized = resize_processed_longitude_axis(longitude_original, 8)

    np.testing.assert_allclose(
        longitude_resized,
        np.array([181.0, 226.0, 271.0, 316.0, 361.0, 406.0, 451.0, 496.0]),
    )
    assert resize_processed_longitude_axis(longitude_original, 4) is longitude_original


def test_stonyhurst_rotation_uses_original_or_resized_longitude_axis(monkeypatch):
    from coconut_tools.magnetogram import sph_filtering

    longitude_original = np.array([10.0, 100.0, 190.0, 280.0])
    Br = np.arange(8).reshape(1, 8)

    monkeypatch.setattr(
        sph_filtering,
        "compute_rotation_angle",
        lambda *args, **kwargs: (100.0, TARGET_DATE),
    )
    monkeypatch.setattr(
        sph_filtering,
        "processed_longitude_axis",
        lambda *args, **kwargs: longitude_original,
    )

    Br_original_axis, _, _ = apply_configured_longitude_rotation(
        Br,
        None,
        "dummy_hmi.fits",
        "HMI_small",
        TARGET_DATE,
        use_interpolation=False,
        rotate_to_stonyhurst=True,
        effective_date=TARGET_DATE,
        resize=False,
    )
    Br_resized_axis, _, _ = apply_configured_longitude_rotation(
        Br,
        None,
        "dummy_hmi.fits",
        "HMI_small",
        TARGET_DATE,
        use_interpolation=False,
        rotate_to_stonyhurst=True,
        effective_date=TARGET_DATE,
        resize=True,
    )

    np.testing.assert_array_equal(Br_original_axis, np.array([[1, 2, 3, 4, 5, 6, 7, 0]]))
    np.testing.assert_array_equal(Br_resized_axis, np.array([[2, 3, 4, 5, 6, 7, 0, 1]]))


def test_read_magnetogram_can_resize_after_longitude_normalization(tmp_path, monkeypatch):
    file_path = tmp_path / "mrzqs201207t1504c2238_181.fits"
    data = np.array([[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]])
    hdu = fits.PrimaryHDU(data)
    hdu.header["CRPIX1"] = 1.0
    hdu.header["CRVAL1"] = 0.0
    hdu.header["CDELT1"] = -1.0
    hdu.writeto(file_path, overwrite=True)

    captured = {}
    fake_skimage = types.ModuleType("skimage")
    fake_transform = types.ModuleType("skimage.transform")

    def fake_resize(image, output_shape, **kwargs):
        captured["image"] = image.copy()
        captured["output_shape"] = output_shape
        captured["kwargs"] = kwargs
        return np.full(output_shape, image[0, 0], dtype=float)

    fake_transform.resize = fake_resize
    monkeypatch.setitem(sys.modules, "skimage", fake_skimage)
    monkeypatch.setitem(sys.modules, "skimage.transform", fake_transform)

    Br, Theta, Phi = read_magnetogram(str(file_path), "GONG", resize=True)

    np.testing.assert_array_equal(captured["image"], data[::-1, ::-1])
    assert captured["output_shape"] == (360, 720)
    assert captured["kwargs"] == {
        "preserve_range": True,
        "mode": "edge",
        "clip": False,
        "anti_aliasing": True,
    }
    assert Br.shape == (360, 720)
    assert Theta.shape == Br.shape
    assert Phi.shape == Br.shape


def test_flux_correction_balances_a_small_signed_map():
    Br = np.array([[2.0, -1.0]])
    theta = np.array([np.pi / 2.0])
    phi = np.array([0.0, np.pi])

    balanced = correct_net_flux(
        Br,
        theta,
        phi,
        method="polarity_scaling",
    )

    np.testing.assert_allclose(balanced, np.array([[np.sqrt(2.0), -np.sqrt(2.0)]]))
