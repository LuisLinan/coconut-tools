from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from coconut_tools.magnetogram.magnetogram_download import (
    build_output_name,
    default_figure_path,
    magnetogram_display_date,
    magnetogram_effective_date,
    resolve_figure_path,
)
from coconut_tools.magnetogram.sph_filtering import (
    closest_longitude_column,
    correct_net_flux,
    read_magnetogram,
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
        "hmi_sync.fits",
        "HMI_SYNC",
        datetime(2020, 12, 7, 15, 30, 45),
    ) == datetime(2020, 12, 7, 12, 0)
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
