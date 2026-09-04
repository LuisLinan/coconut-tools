from datetime import datetime, timedelta
from pathlib import Path
import shutil
import sys
import types

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from coconut_tools.magnetogram.io.downloads import (
    InterpolationSelection,
    MagnetogramCandidate,
    build_output_name,
    default_figure_path,
    download_interpolation_magnetograms,
    download_hmi_hourly_magnetogram,
    ensure_hmi_sync_wcs,
    list_gong_diachronic_candidates,
    list_hmi_candidates,
    magnetogram_display_date,
    magnetogram_effective_date,
    normalize_map_type,
    parse_hmi_hourly_filename_date,
    resolve_figure_path,
    select_nearest_candidate,
)
from coconut_tools.magnetogram.io.readers import (
    read_interpolated_magnetogram,
    read_magnetogram,
    read_temporal_br_map,
)
from coconut_tools.magnetogram.io.metadata import (
    infer_known_fits_map_type,
    read_fits_effective_time,
    read_fits_longitude_axis,
)
from coconut_tools.magnetogram.processing.flux_balance import correct_net_flux
from coconut_tools.magnetogram.processing.longitude import (
    apply_configured_longitude_rotation,
    closest_longitude_column,
    processed_longitude_axis,
    resize_processed_longitude_axis,
    rotate_longitude_to_stonyhurst,
)
from coconut_tools.tools.rotation_angle import (
    compute_rotation_angle,
    is_br_longitude_increasing,
)


TARGET_DATE = datetime(2020, 12, 7, 15, 0)


def _write_hmi_fdt_map(file_path, map_time, base_map):
    """Write a small fixed-Carrington HMI-FDT/ADAPT realization cube."""
    data = np.stack([base_map + 100.0 * index for index in range(12)])
    hdu = fits.PrimaryHDU(data=data)
    hdu.header["MAPDATA"] = "HMI_FDTL"
    hdu.header["MAPTIME"] = map_time.isoformat()
    hdu.header["LNGTYPE"] = 0
    hdu.header["LATTYPE"] = 0
    hdu.header["CRLNGEDG"] = 0.0
    hdu.header["CRPIX1"] = 2.5
    hdu.header["CRVAL1"] = 180.0
    hdu.header["CDELT1"] = 90.0
    hdu.header["CRPIX2"] = 2.5
    hdu.header["CRVAL2"] = 0.0
    hdu.header["CDELT2"] = 45.0
    hdu.writeto(file_path)


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
        "hmi.synoptic_hourly_20260701_070000.fits",
        "HMI_hourly",
        datetime(2020, 12, 7, 15, 30, 45),
    ) == datetime(2026, 7, 1, 7, 0)
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


def test_custom_display_and_rotation_date_uses_config_even_with_t_rec(tmp_path):
    path = tmp_path / "custom_time.fits"
    hdu = fits.PrimaryHDU(np.zeros((2, 4)))
    hdu.header["T_REC"] = "2026.08.01_16:24:00_TAI"
    hdu.header["T_OBS"] = "2026.08.01_16:06:07_TAI"
    hdu.header["DATE-OBS"] = "2026-08-01T16:06:07"
    hdu.writeto(path)

    metadata = read_fits_effective_time(str(path))
    expected = datetime(2026, 8, 1, 16, 23, 23)

    assert metadata.keyword == "T_REC"
    assert metadata.source_scale == "tai"
    assert metadata.value == expected
    assert magnetogram_effective_date(
        str(path),
        "custom",
        TARGET_DATE,
    ) == TARGET_DATE
    assert magnetogram_display_date(
        str(path),
        "custom",
        TARGET_DATE,
    ) == TARGET_DATE


def test_gong_diachronic_search_includes_adjacent_months(monkeypatch):
    from coconut_tools.magnetogram.io import downloads

    pages = {
        "https://gong.nso.edu/data/magmap/QR/mqs/202604/": (
            '<a href="mrmqs260429/">mrmqs260429/</a>'
        ),
        "https://gong.nso.edu/data/magmap/QR/mqs/202604/mrmqs260429/": (
            '<a href="mrmqs260429t0702c2310_000.fits.gz">map</a>'
        ),
        "https://gong.nso.edu/data/magmap/QR/mqs/202605/": (
            '<a href="mrmqs260526/">mrmqs260526/</a>'
        ),
        "https://gong.nso.edu/data/magmap/QR/mqs/202605/mrmqs260526/": (
            '<a href="mrmqs260526t1227c2311_000.fits.gz">map</a>'
        ),
        "https://gong.nso.edu/data/magmap/QR/mqs/202606/": "",
    }

    class Response:
        def __init__(self, text):
            self.text = text

    monkeypatch.setattr(
        downloads.requests,
        "get",
        lambda url: Response(pages[url]),
    )

    target = datetime(2026, 5, 9, 1, 47, 5)
    candidates = list_gong_diachronic_candidates(target, "mrmqs")
    selected = select_nearest_candidate(candidates, target)

    assert [candidate.date for candidate in candidates] == [
        datetime(2026, 4, 29, 7, 2),
        datetime(2026, 5, 26, 12, 27),
    ]
    assert selected.date == datetime(2026, 4, 29, 7, 2)


def test_custom_effective_date_falls_back_to_valid_t_obs(tmp_path):
    path = tmp_path / "custom_t_obs.fits"
    hdu = fits.PrimaryHDU(np.zeros((2, 4)))
    hdu.header["T_REC"] = "invalid"
    hdu.header["T_OBS"] = "2026.08.01_16:06:07_UTC"
    hdu.header["DATE-OBS"] = "2026-08-01T15:00:00"
    hdu.writeto(path)

    metadata = read_fits_effective_time(str(path))

    assert metadata.keyword == "T_OBS"
    assert metadata.source_scale == "utc"
    assert metadata.value == datetime(2026, 8, 1, 16, 6, 7)


def test_custom_effective_date_uses_date_obs_as_utc_fallback(tmp_path):
    path = tmp_path / "custom_date_obs.fits"
    hdu = fits.PrimaryHDU(np.zeros((2, 4)))
    hdu.header["DATE-OBS"] = "2026-08-01T16:06:07"
    hdu.writeto(path)

    metadata = read_fits_effective_time(str(path))

    assert metadata.keyword == "DATE-OBS"
    assert metadata.source_scale == "utc"
    assert metadata.value == datetime(2026, 8, 1, 16, 6, 7)


def test_custom_effective_date_combines_gong_mapdate_and_maptime(tmp_path):
    path = tmp_path / "renamed_gong_map.fits"
    hdu = fits.PrimaryHDU(np.zeros((2, 4)))
    hdu.header["MAPDATE"] = "2011-09-08"
    hdu.header["MAPTIME"] = "23:54"
    hdu.writeto(path)

    metadata = read_fits_effective_time(str(path))

    assert metadata.keyword == "MAPTIME"
    assert metadata.raw_value == "23:54"
    assert metadata.value == datetime(2011, 9, 8, 23, 54)


def test_custom_gong_is_inferred_and_matches_explicit_gong_processing(
    tmp_path,
    monkeypatch,
):
    from coconut_tools.magnetogram.processing import longitude

    # The deliberately misleading name proves that only FITS content matters.
    path = tmp_path / "wso.fake.fits.gz"
    data = np.arange(16.0).reshape(4, 4)
    hdu = fits.PrimaryHDU(data)
    hdu.header["ORIGIN"] = "National Solar Observatory -- GONG"
    hdu.header["TELESCOP"] = "NSO-GONG"
    hdu.header["CTYPE1"] = "CRLN-CEA"
    hdu.header["CTYPE2"] = "CRLT-CEA"
    hdu.header["CRPIX1"] = 2.5
    hdu.header["CRPIX2"] = 2.5
    hdu.header["CRVAL1"] = 301.0
    hdu.header["CRVAL2"] = 0.0
    hdu.header["CDELT1"] = 90.0
    hdu.header["CDELT2"] = 0.5
    hdu.header["MAPDATE"] = "2011-09-08"
    hdu.header["MAPTIME"] = "23:54"
    hdu.writeto(path)

    assert infer_known_fits_map_type(str(path)) == "GONG"
    Br_gong, Theta_gong, Phi_gong = read_magnetogram(str(path), "GONG_mrzqs")
    Br_custom, Theta_custom, Phi_custom = read_magnetogram(str(path))

    np.testing.assert_array_equal(Br_custom, Br_gong)
    np.testing.assert_array_equal(Theta_custom, Theta_gong)
    np.testing.assert_array_equal(Phi_custom, Phi_gong)
    assert magnetogram_effective_date(
        str(path),
        "custom",
        TARGET_DATE,
    ) == TARGET_DATE

    monkeypatch.setattr(
        longitude,
        "compute_rotation_angle",
        lambda *args, **kwargs: (60.0, TARGET_DATE),
    )
    monkeypatch.setattr(
        longitude,
        "compute_carrington_central_meridian",
        lambda date: 181.0,
    )
    rotated_gong, _, angle_gong = apply_configured_longitude_rotation(
        Br_gong,
        None,
        str(path),
        "GONG_mrzqs",
        TARGET_DATE,
        use_interpolation=False,
        rotate_to_stonyhurst=True,
        effective_date=TARGET_DATE,
    )
    rotated_custom, _, angle_custom = apply_configured_longitude_rotation(
        Br_custom,
        None,
        str(path),
        "custom",
        TARGET_DATE,
        use_interpolation=False,
        rotate_to_stonyhurst=True,
        effective_date=TARGET_DATE,
    )

    np.testing.assert_array_equal(rotated_custom, rotated_gong)
    assert angle_custom == angle_gong


def test_custom_jsoc_hmi_synoptic_matches_explicit_hmi_processing(
    tmp_path,
    monkeypatch,
):
    from coconut_tools.magnetogram.processing import longitude

    # A deliberately unrelated filename verifies that only FITS metadata are
    # used to recognize the JSOC HMI convention.
    path = tmp_path / "renamed_input.fits"
    data = np.array(
        [
            [180.0, 90.0, 0.0, 270.0],
            [1180.0, 1090.0, 1000.0, 1270.0],
        ]
    )
    hdu = fits.CompImageHDU(data=data)
    hdu.header["ORIGIN"] = "SDO/JSOC-SDP"
    hdu.header["TELESCOP"] = "SDO/HMI"
    hdu.header["INSTRUME"] = "HMI_COMBINED"
    hdu.header["CONTENT"] = "Update Synoptic MAP (Mr)"
    hdu.header["CTYPE1"] = "CRLN-CEA"
    hdu.header["CTYPE2"] = "CRLT-CEA"
    hdu.header["CRVAL1"] = 90.0
    hdu.header["CRPIX1"] = 2.0
    hdu.header["CDELT1"] = -90.0
    hdu.writeto(path)

    assert infer_known_fits_map_type(str(path)) == "HMI_hourly"
    Br_explicit, Theta_explicit, Phi_explicit = read_magnetogram(
        str(path),
        "HMI_hourly",
    )
    Br_custom, Theta_custom, Phi_custom = read_magnetogram(str(path))

    # The HMI convention uses abs(CDELT1): only the Carrington-origin roll is
    # applied, never a left-right reflection of the stored field.
    expected = np.roll(data[::-1, :], 2, axis=1)
    np.testing.assert_array_equal(Br_custom, expected)
    np.testing.assert_array_equal(Br_custom, Br_explicit)
    np.testing.assert_array_equal(Theta_custom, Theta_explicit)
    np.testing.assert_array_equal(Phi_custom, Phi_explicit)

    monkeypatch.setattr(
        longitude,
        "compute_rotation_angle",
        lambda *args, **kwargs: pytest.fail(
            "A header-identified custom HMI must not parse its filename"
        ),
    )
    monkeypatch.setattr(
        longitude,
        "compute_carrington_central_meridian",
        lambda date: 90.0,
    )
    rotated_custom, _, angle_custom = apply_configured_longitude_rotation(
        Br_custom,
        None,
        str(path),
        "custom",
        TARGET_DATE,
        use_interpolation=False,
        rotate_to_stonyhurst=True,
        effective_date=TARGET_DATE,
    )

    np.testing.assert_array_equal(rotated_custom, np.roll(Br_custom, -1, axis=1))
    assert angle_custom == 90.0


@pytest.mark.parametrize(
    ("method", "expected_map_type"),
    [(None, "HMI_small"), ("TEMP_SPAT", "HMI_polfil")],
)
def test_custom_static_hmi_synoptic_is_identified_without_origin(
    tmp_path,
    method,
    expected_map_type,
):
    path = tmp_path / "renamed_static_hmi.fits"
    data = np.arange(8.0).reshape(2, 4)
    hdu = fits.PrimaryHDU(data)
    hdu.header["TELESCOP"] = "SDO/HMI"
    hdu.header["INSTRUME"] = "HMI_SIDE1"
    hdu.header["CONTENT"] = "Carrington Synoptic Chart Of Br Field"
    hdu.header["CTYPE1"] = "CRLN-CEA"
    hdu.header["CTYPE2"] = "CRLT-CEA"
    hdu.header["CUNIT1"] = "degree"
    hdu.header["CUNIT2"] = "Sine Latitude"
    hdu.header["CRPIX1"] = 2.0
    hdu.header["CRPIX2"] = 1.5
    hdu.header["CRVAL1"] = 360.0
    hdu.header["CRVAL2"] = 0.0
    hdu.header["CDELT1"] = -90.0
    hdu.header["CDELT2"] = 1.0
    hdu.header["T_OBS"] = "2026.04.29_07:08:43_TAI"
    if method is not None:
        hdu.header["METHOD"] = method
    hdu.writeto(path)

    assert infer_known_fits_map_type(str(path)) == expected_map_type
    assert magnetogram_effective_date(
        str(path),
        "custom",
        TARGET_DATE,
    ) == TARGET_DATE
    Br_explicit, Theta_explicit, Phi_explicit = read_magnetogram(
        str(path),
        expected_map_type,
    )
    Br_custom, Theta_custom, Phi_custom = read_magnetogram(str(path))

    np.testing.assert_array_equal(Br_custom, Br_explicit)
    np.testing.assert_array_equal(Theta_custom, Theta_explicit)
    np.testing.assert_array_equal(Phi_custom, Phi_explicit)


def test_custom_non_synoptic_hmi_fits_still_obeys_negative_wcs(tmp_path):
    path = tmp_path / "generic_hmi_observation.fits"
    _write_custom_frame_map(path, longitude_start=90.0)
    with fits.open(path, mode="update") as hdul:
        header = hdul[0].header
        header["ORIGIN"] = "SDO/JSOC-SDP"
        header["TELESCOP"] = "SDO/HMI"
        header["INSTRUME"] = "HMI_FRONT2"
        header["CONTENT"] = "Line-of-sight observation"
        header["CTYPE1"] = "CRLN-CAR"
        header["CTYPE2"] = "HGLT-CAR"
        header["CDELT1"] = -90.0

    assert infer_known_fits_map_type(str(path)) is None
    Br, _, _ = read_magnetogram(str(path))

    original = np.arange(8.0).reshape(2, 4)
    np.testing.assert_array_equal(Br, np.roll(original[::-1, ::-1], -2, axis=1))


@pytest.mark.parametrize(
    ("map_data", "expected_map_type"),
    [("GONG", "ADAPT"), ("HMI-FDT", "HMI_fdt")],
)
def test_custom_adapt_ensemble_is_identified_from_headers(
    tmp_path,
    map_data,
    expected_map_type,
):
    path = tmp_path / "renamed_ensemble.fts"
    data = np.arange(3 * 2 * 4.0).reshape(3, 2, 4)
    hdu = fits.PrimaryHDU(data)
    hdu.header["MODEL"] = "ADAPT"
    hdu.header["MAPDATA"] = map_data
    hdu.header["NREAL"] = 3
    hdu.header["MAPTIME"] = "2026-05-09T02:00:00"
    hdu.header["LNGTYPE"] = 0
    hdu.header["LATTYPE"] = 0
    hdu.header["CTYPE1"] = "Long"
    hdu.header["CTYPE2"] = "Lat"
    hdu.header["CUNIT1"] = "deg"
    hdu.header["CUNIT2"] = "deg"
    hdu.header["CRPIX1"] = 2.5
    hdu.header["CRPIX2"] = 1.5
    hdu.header["CRVAL1"] = 180.0
    hdu.header["CRVAL2"] = 0.0
    hdu.header["CDELT1"] = 90.0
    hdu.header["CDELT2"] = 90.0
    hdu.writeto(path)

    assert infer_known_fits_map_type(str(path)) == expected_map_type
    Br_explicit, Theta_explicit, Phi_explicit = read_magnetogram(
        str(path),
        expected_map_type,
        adapt_map=1,
    )
    Br_custom, Theta_custom, Phi_custom = read_magnetogram(
        str(path),
        adapt_map=1,
    )

    np.testing.assert_array_equal(Br_custom, Br_explicit)
    np.testing.assert_array_equal(Theta_custom, Theta_explicit)
    np.testing.assert_array_equal(Phi_custom, Phi_explicit)


def test_custom_solar_longitude_can_infer_missing_cunit1(tmp_path):
    path = tmp_path / "generic_crln_without_unit.fits"
    _write_custom_frame_map(path)
    with fits.open(path, mode="update") as hdul:
        del hdul[0].header["CUNIT1"]

    with pytest.warns(RuntimeWarning, match="CUNIT1 is missing"):
        geometry = read_fits_longitude_axis(str(path))

    np.testing.assert_allclose(geometry.centers_degrees, [0.0, 90.0, 180.0, 270.0])


def _write_custom_frame_map(
    path,
    *,
    ctype1="CRLN-CAR",
    longitude_start=0.0,
    central_meridian=None,
    observer_stonyhurst=None,
):
    hdu = fits.PrimaryHDU(np.arange(8.0).reshape(2, 4))
    hdu.header["CTYPE1"] = ctype1
    hdu.header["CUNIT1"] = "deg"
    hdu.header["CRPIX1"] = 1.0
    hdu.header["CRVAL1"] = longitude_start
    hdu.header["CDELT1"] = 90.0
    hdu.header["CTYPE2"] = "HGLT-CAR"
    hdu.header["CUNIT2"] = "deg"
    hdu.header["CRPIX2"] = 1.5
    hdu.header["CRVAL2"] = 0.0
    hdu.header["CDELT2"] = 90.0
    if central_meridian is not None:
        hdu.header["CRLN_OBS"] = central_meridian
    if observer_stonyhurst is not None:
        hdu.header["HGLN_OBS"] = observer_stonyhurst
    hdu.writeto(path)


def test_custom_carrington_rotation_uses_config_date_not_header_longitude(
    tmp_path,
    monkeypatch,
):
    from coconut_tools.magnetogram.processing import longitude

    path = tmp_path / "custom_carrington.fits"
    _write_custom_frame_map(path, central_meridian=90.0)
    monkeypatch.setattr(
        longitude,
        "compute_carrington_central_meridian",
        lambda date: 90.25,
    )

    Br, _, _ = read_magnetogram(str(path))
    rotated, _, angle = apply_configured_longitude_rotation(
        Br,
        None,
        str(path),
        "custom",
        TARGET_DATE,
        use_interpolation=False,
        rotate_to_stonyhurst=True,
        effective_date=TARGET_DATE,
    )

    np.testing.assert_array_equal(rotated, np.roll(Br, -1, axis=1))
    assert angle == pytest.approx(90.25)


def test_custom_carrington_rotation_ignores_header_observer_for_config_date(
    tmp_path,
    monkeypatch,
):
    from coconut_tools.magnetogram.processing import longitude

    path = tmp_path / "custom_non_earth_observer.fits"
    _write_custom_frame_map(
        path,
        central_meridian=120.0,
        observer_stonyhurst=30.0,
    )
    monkeypatch.setattr(
        longitude,
        "compute_carrington_central_meridian",
        lambda date: 90.0,
    )

    Br, _, _ = read_magnetogram(str(path))
    rotated, _, angle = apply_configured_longitude_rotation(
        Br,
        None,
        str(path),
        "custom",
        TARGET_DATE,
        use_interpolation=False,
        rotate_to_stonyhurst=True,
        effective_date=TARGET_DATE,
    )

    np.testing.assert_array_equal(rotated, np.roll(Br, -1, axis=1))
    assert angle == pytest.approx(90.0)


def test_custom_carrington_rotation_uses_effective_date_without_crln_obs(
    tmp_path,
    monkeypatch,
):
    from coconut_tools.magnetogram.processing import longitude

    path = tmp_path / "custom_carrington_without_observer.fits"
    _write_custom_frame_map(path)
    dates = []

    def fake_central_meridian(date):
        dates.append(date)
        return 180.0

    monkeypatch.setattr(
        longitude,
        "compute_carrington_central_meridian",
        fake_central_meridian,
    )
    Br, _, _ = read_magnetogram(str(path))
    rotated, _, angle = apply_configured_longitude_rotation(
        Br,
        None,
        str(path),
        "custom",
        TARGET_DATE,
        use_interpolation=False,
        rotate_to_stonyhurst=True,
        effective_date=TARGET_DATE,
    )

    np.testing.assert_array_equal(rotated, np.roll(Br, -2, axis=1))
    assert angle == pytest.approx(180.0)
    assert dates == [TARGET_DATE]


def test_custom_stonyhurst_axis_is_not_rotated_again(tmp_path, monkeypatch):
    from coconut_tools.magnetogram.processing import longitude

    path = tmp_path / "custom_stonyhurst.fits"
    _write_custom_frame_map(path, ctype1="HGLN-CAR", longitude_start=-135.0)
    monkeypatch.setattr(
        longitude,
        "compute_carrington_central_meridian",
        lambda date: pytest.fail("A native Stonyhurst map needs no ephemeris roll"),
    )

    Br, _, Phi = read_magnetogram(str(path))
    rotated, _, angle = apply_configured_longitude_rotation(
        Br,
        None,
        str(path),
        "custom",
        TARGET_DATE,
        use_interpolation=False,
        rotate_to_stonyhurst=True,
        effective_date=TARGET_DATE,
    )

    np.testing.assert_array_equal(rotated, Br)
    np.testing.assert_allclose(np.degrees(Phi[0]), [45.0, 135.0, 225.0, 315.0])
    assert angle == pytest.approx(0.0)


def test_custom_rotation_preserves_nonzero_first_longitude_center(
    tmp_path,
    monkeypatch,
):
    from coconut_tools.magnetogram.processing import longitude

    path = tmp_path / "custom_offset_centers.fits"
    _write_custom_frame_map(
        path,
        longitude_start=45.0,
        central_meridian=90.0,
    )
    monkeypatch.setattr(
        longitude,
        "compute_carrington_central_meridian",
        lambda date: 90.0,
    )

    Br, _, Phi = read_magnetogram(str(path))
    rotated, _, _ = apply_configured_longitude_rotation(
        Br,
        None,
        str(path),
        "custom",
        TARGET_DATE,
        use_interpolation=False,
        rotate_to_stonyhurst=True,
        effective_date=TARGET_DATE,
    )

    # Output phi[0]=45 deg samples source Carrington longitude L0+45=135 deg.
    np.testing.assert_allclose(np.degrees(Phi[0]), [45.0, 135.0, 225.0, 315.0])
    np.testing.assert_array_equal(rotated, np.roll(Br, -1, axis=1))


def test_custom_resized_rotation_uses_the_resized_physical_centers(
    tmp_path,
    monkeypatch,
):
    from coconut_tools.magnetogram.processing import longitude

    path = tmp_path / "custom_resized_longitude.fits"
    _write_custom_frame_map(
        path,
        longitude_start=45.0,
        central_meridian=90.0,
    )
    monkeypatch.setattr(
        longitude,
        "compute_carrington_central_meridian",
        lambda date: 90.0,
    )

    Br, _, Phi = read_magnetogram(str(path), resize=True)
    rotated, _, _ = apply_configured_longitude_rotation(
        Br,
        None,
        str(path),
        "custom",
        TARGET_DATE,
        use_interpolation=False,
        rotate_to_stonyhurst=True,
        effective_date=TARGET_DATE,
        resize=True,
    )

    assert Br.shape == (360, 720)
    assert np.degrees(Phi[0, 0]) == pytest.approx(0.25)
    np.testing.assert_array_equal(rotated, np.roll(Br, -180, axis=1))


def test_custom_rotation_rejects_projection_suffix_as_frame_evidence(tmp_path):
    path = tmp_path / "custom_ambiguous_frame.fits"
    _write_custom_frame_map(path, ctype1="LON-CAR")

    geometry = read_fits_longitude_axis(str(path))
    assert geometry.frame == "unknown"
    Br, _, _ = read_magnetogram(str(path))
    with pytest.raises(ValueError, match="projection suffix.*does not identify"):
        apply_configured_longitude_rotation(
            Br,
            None,
            str(path),
            "custom",
            TARGET_DATE,
            use_interpolation=False,
            rotate_to_stonyhurst=True,
            effective_date=TARGET_DATE,
        )


def test_map_type_normalization_accepts_case_variants():
    outdir = Path(__file__).parent / "_outputs"

    assert normalize_map_type("hmi_sync") == "HMI_SYNC"
    assert normalize_map_type("HMI_SYNC") == "HMI_SYNC"
    assert normalize_map_type("HMI_HOURLY") == "HMI_hourly"
    assert normalize_map_type("Hmi_Small") == "HMI_small"
    assert normalize_map_type("gong_MRBQS") == "GONG_mrbqs"
    assert normalize_map_type(" adapt ") == "ADAPT"
    assert normalize_map_type("CUSTOM") == "custom"
    assert build_output_name("custom", str(outdir), "sph") == str(
        outdir / "map_custom_sph.dat"
    )

    assert magnetogram_effective_date(
        "hmi.mrdailysynframe_720s_nrt.20260701_062400_TAI.data.fits",
        "hmi_sync",
        datetime(2020, 12, 7, 15, 30, 45),
    ) == datetime(2026, 7, 1, 6, 24)
    assert build_output_name("hmi_sync", str(outdir), "sph") == str(
        outdir / "map_hmi_sync_sph.dat"
    )
    assert build_output_name("hmi_hourly", str(outdir), "sph") == str(
        outdir / "map_hmi_hourly_sph.dat"
    )
    assert build_output_name("gong_MRBQS", str(outdir), "sph") == str(
        outdir / "map_gong_mrbqs_sph.dat"
    )


def test_hmi_fdt_normalization_output_and_effective_date():
    outdir = Path(__file__).parent / "_outputs"
    map_name = "adapt40i11_044012_202608192000_i00011200n1.fts.gz"
    target = datetime(2026, 8, 20, 1, 30)

    assert normalize_map_type("hmi_fdt") == "HMI_fdt"
    assert normalize_map_type("HMI_FDT") == "HMI_fdt"
    assert build_output_name("hmi_fdt", str(outdir), "sph") == str(
        outdir / "map_hmi_fdt_sph.dat"
    )
    assert magnetogram_effective_date(map_name, "hmi_fdt", target) == datetime(
        2026, 8, 19, 20, 0
    )
    assert magnetogram_effective_date(
        map_name,
        "hmi_fdt",
        target,
        interpolated=True,
    ) == target


def test_hmi_fdt_interpolation_lists_only_fixed_carrington_maps_and_downloads_four(
    tmp_path,
    monkeypatch,
):
    from coconut_tools.magnetogram.io import downloads as magnetogram_download

    remote_dir = "https://gong.nso.edu/adapt/maps/hmi-fdtl/"
    dates = [datetime(2026, 8, 18, 8) + timedelta(hours=12 * index) for index in range(6)]
    remote_names = []
    for date in dates:
        timestamp = date.strftime("%Y%m%d%H%M")
        remote_names.extend(
            [
                f"adapt41i11_044012_{timestamp}_i00010000n1.fts.gz",
                f"adapt40i11_044012_{timestamp}_i00010000n1.fts.gz",
            ]
        )

    listing_calls = []

    def fake_fetch(listing_url, token):
        listing_calls.append((listing_url, token))
        return [name for name in remote_names if token in name]

    downloaded = []

    def fake_download(candidate, output_dir):
        downloaded.append(candidate)
        return str(Path(output_dir) / candidate.name)

    monkeypatch.setattr(magnetogram_download, "fetch_remote_names", fake_fetch)
    monkeypatch.setattr(magnetogram_download, "download_candidate", fake_download)

    target = dates[2] + timedelta(hours=6)
    local_files, selection = download_interpolation_magnetograms(
        target,
        "HMI_fdt",
        str(tmp_path),
    )

    assert listing_calls == [(remote_dir, "adapt40i11")]
    assert [candidate.date for candidate in downloaded] == dates[1:5]
    assert all(candidate.name.startswith("adapt40i11_") for candidate in downloaded)
    assert all(candidate.remote_url == remote_dir + candidate.name for candidate in downloaded)
    assert [Path(path).name for path in local_files] == [
        candidate.name for candidate in downloaded
    ]
    assert selection.before.date == dates[2]
    assert selection.after.date == dates[3]
    assert selection.target_date == target


def test_hmi_hourly_candidates_use_timestamps_encoded_in_their_names(monkeypatch):
    captured = {}
    keys = pd.DataFrame(
        {
            "T_REC": [
                "2026.07.01_08:00:00_TAI",
                "2026.07.01_06:00:00_TAI",
                "2026.07.01_07:00:00_TAI",
            ]
        }
    )
    segments = pd.DataFrame(
        {
            "Mr_polfil": [
                "/SUM97/D123/map_0800.fits",
                "https://example.test/map_0600.fits",
                "NoDataDirectory",
            ]
        }
    )

    class FakeClient:
        def query(self, recordset, key, seg):
            captured.update({"recordset": recordset, "key": key, "seg": seg})
            return keys, segments

    fake_drms = types.SimpleNamespace(Client=FakeClient)
    monkeypatch.setitem(sys.modules, "drms", fake_drms)

    candidates = list_hmi_candidates(datetime(2026, 7, 1, 7, 15))

    assert [candidate.name for candidate in candidates] == [
        "hmi.synoptic_hourly_20260701_060000.fits",
        "hmi.synoptic_hourly_20260701_070000.fits",
        "hmi.synoptic_hourly_20260701_080000.fits",
    ]
    assert [candidate.date for candidate in candidates] == [
        datetime(2026, 7, 1, 6),
        datetime(2026, 7, 1, 7),
        datetime(2026, 7, 1, 8),
    ]
    assert candidates[0].remote_url == "https://example.test/map_0600.fits"
    assert candidates[1].remote_url is None
    assert candidates[2].remote_url == (
        "http://jsoc.stanford.edu/SUM97/D123/map_0800.fits"
    )
    assert captured == {
        "recordset": (
            "hmi.mrdailysynframe_polfil_720s_nrt"
            "[2026.06.30_07:15:00_TAI-2026.07.02_07:15:00_TAI]"
        ),
        "key": "T_REC",
        "seg": "Mr_polfil",
    }


def test_hmi_hourly_download_uses_filename_time_for_jsoc_metadata(
    tmp_path,
    monkeypatch,
):
    source_file = tmp_path / "remote.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(data=np.array([[1.0, np.nan], [-2.0, 3.0]])),
        ]
    ).writeto(source_file)

    captured = {}
    metadata = pd.DataFrame(
        [
            {
                "T_OBS": "2026.07.01_07:00:00_TAI",
                "T_REC_epoch": "1993.01.01_00:00:00_TAI",
                "T_REC_step": 720.0,
                "T_REC_unit": "secs",
                "CSYSER1": "unused",
            }
        ]
    )

    class FakeClient:
        def query(self, recordset, key):
            captured.update({"recordset": recordset, "key": key})
            return metadata

    fake_drms = types.SimpleNamespace(
        Client=FakeClient,
        JsocInfoConstants=types.SimpleNamespace(all="ALL_JSOC_KEYS"),
    )
    monkeypatch.setitem(sys.modules, "drms", fake_drms)

    import urllib.request

    monkeypatch.setattr(
        urllib.request,
        "urlretrieve",
        lambda _url, destination: (shutil.copyfile(source_file, destination), None),
    )

    candidate = MagnetogramCandidate(
        name="hmi.synoptic_hourly_20260701_070000.fits",
        date=datetime(2000, 1, 1),
        remote_url="https://example.test/remote.fits",
    )
    local_file, map_name = download_hmi_hourly_magnetogram(candidate, str(tmp_path))

    assert map_name == candidate.name
    assert local_file == str(tmp_path / candidate.name)
    assert captured == {
        "recordset": (
            "hmi.mrdailysynframe_polfil_720s_nrt"
            "[2026.07.01_07:00:00_TAI]"
        ),
        "key": "ALL_JSOC_KEYS",
    }
    assert not Path(local_file + ".temp").exists()
    np.testing.assert_array_equal(
        fits.getdata(local_file),
        np.array([[1.0, 0.0], [-2.0, 3.0]]),
    )
    header = fits.getheader(local_file, ext=1)
    assert header["DATE-OBS"] == "2026-07-01T07:00:00"
    assert header["CUNIT1"] == "deg"
    assert header["BUNIT"] == "gauss"
    assert header["TRECEPOC"] == "1993.01.01_00:00:00_TAI"
    assert "T_REC_epoch" not in header
    assert parse_hmi_hourly_filename_date(map_name) == datetime(2026, 7, 1, 7)


def test_hmi_hourly_offline_segment_is_staged_through_drms(tmp_path, monkeypatch):
    source_file = tmp_path / "archived_source.fits"
    fits.PrimaryHDU(data=np.array([[1.0, np.nan], [-2.0, 3.0]])).writeto(
        source_file
    )
    captured = {}
    metadata = pd.DataFrame([{"T_OBS": "2025.09.09_01:24:00_TAI"}])

    class FakeExport:
        def download(self, destination):
            exported = Path(destination) / "Mr_polfil.fits"
            shutil.copyfile(source_file, exported)
            return types.SimpleNamespace(download=[str(exported)])

    class FakeClient:
        def __init__(self, email=None):
            captured["email"] = email

        def export(self, recordset, protocol):
            captured.update(
                {"export_recordset": recordset, "protocol": protocol}
            )
            return FakeExport()

        def query(self, recordset, key):
            captured.update({"metadata_recordset": recordset, "key": key})
            return metadata

    fake_drms = types.SimpleNamespace(
        Client=FakeClient,
        JsocInfoConstants=types.SimpleNamespace(all="ALL_JSOC_KEYS"),
    )
    monkeypatch.setitem(sys.modules, "drms", fake_drms)

    candidate = MagnetogramCandidate(
        name="hmi.synoptic_hourly_20250909_012400.fits",
        date=datetime(2025, 9, 9, 1, 24),
        remote_url=None,
    )
    local_file, map_name = download_hmi_hourly_magnetogram(
        candidate,
        str(tmp_path),
        drms_email="registered@example.test",
    )

    assert map_name == candidate.name
    assert local_file == str(tmp_path / candidate.name)
    assert captured == {
        "email": "registered@example.test",
        "export_recordset": (
            "hmi.mrdailysynframe_polfil_720s_nrt"
            "[2025.09.09_01:24:00_TAI]{Mr_polfil}"
        ),
        "protocol": "fits",
        "metadata_recordset": (
            "hmi.mrdailysynframe_polfil_720s_nrt"
            "[2025.09.09_01:24:00_TAI]"
        ),
        "key": "ALL_JSOC_KEYS",
    }
    np.testing.assert_array_equal(
        fits.getdata(local_file, ext=1),
        np.array([[1.0, 0.0], [-2.0, 3.0]]),
    )


def test_hmi_hourly_offline_segment_requires_drms_email(tmp_path, monkeypatch):
    class FakeClient:
        pass

    fake_drms = types.SimpleNamespace(Client=FakeClient)
    monkeypatch.setitem(sys.modules, "drms", fake_drms)
    candidate = MagnetogramCandidate(
        name="hmi.synoptic_hourly_20250909_012400.fits",
        date=datetime(2025, 9, 9, 1, 24),
        remote_url=None,
    )

    with pytest.raises(ValueError, match=r"config\['drms_email'\]"):
        download_hmi_hourly_magnetogram(candidate, str(tmp_path))


@pytest.mark.parametrize(
    ("native_longitude", "crval1", "crpix1", "cdelt1", "expected_shift"),
    [
        ([180.0, 90.0, 0.0, 270.0], 90.0, 2.0, -90.0, 2),
        ([90.0, 180.0, 270.0, 0.0], 180.0, 2.0, 90.0, 1),
    ],
)
def test_hmi_hourly_reader_rolls_without_reflecting_hmi_longitude(
    tmp_path,
    native_longitude,
    crval1,
    crpix1,
    cdelt1,
    expected_shift,
):
    file_path = tmp_path / "hmi.synoptic_hourly_20260701_070000.fits"
    native_longitude = np.asarray(native_longitude)
    data = np.vstack((native_longitude, native_longitude + 1000.0))
    hdu = fits.CompImageHDU(data=data)
    hdu.header["CRVAL1"] = crval1
    hdu.header["CRPIX1"] = crpix1
    hdu.header["CDELT1"] = cdelt1
    hdu.writeto(file_path)

    Br, _, Phi = read_magnetogram(str(file_path), "HMI_hourly")
    temporal_Br = read_temporal_br_map(str(file_path), "HMI_hourly")
    longitude = processed_longitude_axis(str(file_path), "HMI_hourly")

    expected_longitude = np.array([0.0, 90.0, 180.0, 270.0])
    expected_Br = np.roll(data[::-1, :], expected_shift, axis=1)
    np.testing.assert_array_equal(Br, expected_Br)
    np.testing.assert_array_equal(temporal_Br, expected_Br)
    np.testing.assert_allclose(longitude, expected_longitude)
    np.testing.assert_allclose(Phi[0], np.deg2rad(expected_longitude))


def test_hmi_sync_uses_jsoc_wcs_for_the_same_origin_roll(tmp_path):
    file_path = (
        tmp_path
        / "hmi.mrdailysynframe_720s_nrt.20260701_072400_TAI.data.fits"
    )
    data = np.array(
        [
            [180.0, 90.0, 0.0, 270.0],
            [1180.0, 1090.0, 1000.0, 1270.0],
        ]
    )
    hdu = fits.CompImageHDU(data=data)
    hdu.header["CRVAL1"] = 0.0
    hdu.header["CRPIX1"] = 2.5
    hdu.header["CDELT1"] = -90.0
    hdu.writeto(file_path)

    captured = {}
    metadata = pd.DataFrame(
        [
            {
                "CRVAL1": 90.0,
                "CRPIX1": 2.0,
                "CDELT1": -90.0,
                "CTYPE1": "CRLN-CEA",
                "CUNIT1": "degree",
            }
        ]
    )

    class FakeClient:
        def query(self, recordset, key):
            captured.update({"recordset": recordset, "key": key})
            return metadata

    t_rec = "2026.07.01_07:24:00_TAI"
    ensure_hmi_sync_wcs(FakeClient(), t_rec, str(file_path))
    Br, _, Phi = read_magnetogram(str(file_path), "HMI_SYNC")
    longitude = processed_longitude_axis(str(file_path), "HMI_SYNC")

    expected_longitude = np.array([0.0, 90.0, 180.0, 270.0])
    expected_Br = np.roll(data[::-1, :], 2, axis=1)
    np.testing.assert_array_equal(Br, expected_Br)
    np.testing.assert_allclose(longitude, expected_longitude)
    np.testing.assert_allclose(Phi[0], np.deg2rad(expected_longitude))
    assert captured == {
        "recordset": (
            "hmi.Mrdailysynframe_720s_nrt[2026.07.01_07:24:00_TAI]"
        ),
        "key": "CRVAL1,CRPIX1,CDELT1,CTYPE1,CUNIT1",
    }
    header = fits.getheader(file_path, ext=1)
    assert header["CRVAL1"] == 90.0
    assert header["CRPIX1"] == 2.0
    assert header["CDELT1"] == -90.0
    assert header["CTYPE1"] == "CRLN-CEA"
    assert header["CUNIT1"] == "degree"


def test_hmi_hourly_interpolation_downloads_each_map_with_jsoc_metadata(
    tmp_path,
    monkeypatch,
):
    from coconut_tools.magnetogram.io import downloads as magnetogram_download

    dates = [datetime(2026, 7, 1, hour) for hour in range(4)]
    candidates = [
        MagnetogramCandidate(
            name=f"hmi.synoptic_hourly_{date.strftime('%Y%m%d_%H%M%S')}.fits",
            date=date,
            remote_url=f"https://example.test/{date.hour}.fits",
        )
        for date in dates
    ]
    downloaded = []

    monkeypatch.setattr(
        magnetogram_download,
        "list_remote_candidates",
        lambda *_args: candidates,
    )

    def fake_download(candidate, output_dir):
        downloaded.append(candidate)
        return str(Path(output_dir) / candidate.name), candidate.name

    monkeypatch.setattr(
        magnetogram_download,
        "download_hmi_hourly_magnetogram",
        fake_download,
    )
    monkeypatch.setattr(
        magnetogram_download,
        "download_candidate",
        lambda *_args: pytest.fail("Generic downloader must not handle HMI_hourly"),
    )

    target = datetime(2026, 7, 1, 1, 30)
    local_files, selection = download_interpolation_magnetograms(
        target,
        "HMI_hourly",
        str(tmp_path),
    )

    assert downloaded == candidates
    assert [Path(path).name for path in local_files] == [
        candidate.name for candidate in candidates
    ]
    assert selection.before.date == dates[1]
    assert selection.after.date == dates[2]
    assert selection.target_date == target


def test_hmi_hourly_interpolation_aligns_each_wcs_and_uses_target_time(tmp_path):
    dates = [datetime(2026, 7, 1, hour) for hour in range(4)]
    candidates = []
    local_files = []
    base = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [10.0, 11.0, 12.0, 13.0],
        ]
    )

    for index, date in enumerate(dates):
        name = f"hmi.synoptic_hourly_{date.strftime('%Y%m%d_%H%M%S')}.fits"
        file_path = tmp_path / name
        desired_map = base + 10.0 * index
        shift = index
        native_data = np.roll(desired_map, -shift, axis=1)[::-1, :]
        hdu = fits.CompImageHDU(data=native_data)
        hdu.header["CRVAL1"] = -90.0 * shift
        hdu.header["CRPIX1"] = 1.0
        hdu.header["CDELT1"] = -90.0
        hdu.writeto(file_path)
        candidates.append(MagnetogramCandidate(name, date, "unused"))
        local_files.append(str(file_path))

    target = datetime(2026, 7, 1, 1, 30)
    selection = InterpolationSelection(
        before_previous=candidates[0],
        before=candidates[1],
        after=candidates[2],
        after_next=candidates[3],
        coef_before=0.5,
        coef_after=0.5,
        interval_seconds=3600.0,
        previous_interval_seconds=3600.0,
        next_interval_seconds=3600.0,
        target_date=target,
    )

    Br, Theta, Phi, Br_linear = read_interpolated_magnetogram(
        local_files,
        "HMI_hourly",
        selection,
        interpolation_order=2,
    )

    expected = base + 15.0
    np.testing.assert_allclose(Br, expected)
    np.testing.assert_allclose(Br_linear, expected)
    assert Br.shape == Theta.shape == Phi.shape
    assert magnetogram_effective_date(
        local_files[0],
        "HMI_hourly",
        target,
        interpolated=True,
    ) == target


def test_hmi_hourly_interpolation_resizes_aligned_maps_before_interpolation(
    tmp_path,
    monkeypatch,
):
    dates = [datetime(2026, 7, 1, hour) for hour in range(4)]
    candidates = []
    local_files = []
    base = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [10.0, 11.0, 12.0, 13.0],
        ]
    )

    for index, date in enumerate(dates):
        name = f"hmi.synoptic_hourly_{date.strftime('%Y%m%d_%H%M%S')}.fits"
        file_path = tmp_path / name
        desired_map = base + 10.0 * index
        native_data = np.roll(desired_map, -index, axis=1)[::-1, :]
        hdu = fits.CompImageHDU(data=native_data)
        hdu.header["CRVAL1"] = -90.0 * index
        hdu.header["CRPIX1"] = 1.0
        hdu.header["CDELT1"] = -90.0
        hdu.writeto(file_path)
        candidates.append(MagnetogramCandidate(name, date, "unused"))
        local_files.append(str(file_path))

    target = datetime(2026, 7, 1, 1, 30)
    selection = InterpolationSelection(
        before_previous=candidates[0],
        before=candidates[1],
        after=candidates[2],
        after_next=candidates[3],
        coef_before=0.5,
        coef_after=0.5,
        interval_seconds=3600.0,
        previous_interval_seconds=3600.0,
        next_interval_seconds=3600.0,
        target_date=target,
    )

    resized_inputs = []
    fake_skimage = types.ModuleType("skimage")
    fake_transform = types.ModuleType("skimage.transform")

    def fake_resize(image, output_shape, **kwargs):
        resized_inputs.append(image.copy())
        return np.full(output_shape, image[0, 0], dtype=float)

    fake_transform.resize = fake_resize
    monkeypatch.setitem(sys.modules, "skimage", fake_skimage)
    monkeypatch.setitem(sys.modules, "skimage.transform", fake_transform)

    Br, Theta, Phi, Br_linear = read_interpolated_magnetogram(
        local_files,
        "HMI_hourly",
        selection,
        interpolation_order=2,
        resize=True,
    )

    assert len(resized_inputs) == 4
    for index, resize_input in enumerate(resized_inputs):
        np.testing.assert_allclose(resize_input, base + 10.0 * index)
    assert Br.shape == Theta.shape == Phi.shape == Br_linear.shape == (360, 720)
    np.testing.assert_allclose(Br, 15.0)
    np.testing.assert_allclose(Br_linear, 15.0)


def test_hmi_fdt_reader_selects_realization_on_fixed_carrington_grid(tmp_path):
    map_time = datetime(2026, 8, 19, 20)
    file_path = (
        tmp_path
        / "adapt40i11_044012_202608192000_i00011200n1.fts"
    )
    base = np.arange(16.0).reshape(4, 4)
    _write_hmi_fdt_map(file_path, map_time, base)

    Br, Theta, Phi = read_magnetogram(
        str(file_path),
        "HMI_fdt",
        adapt_map=3,
    )
    temporal_Br = read_temporal_br_map(
        str(file_path),
        "HMI_fdt",
        adapt_map=3,
    )
    longitude = processed_longitude_axis(str(file_path), "HMI_fdt")

    expected = (base + 300.0)[::-1, :]
    np.testing.assert_array_equal(Br, expected)
    np.testing.assert_array_equal(temporal_Br, expected)
    np.testing.assert_allclose(
        Theta[:, 0],
        (np.arange(4, dtype=float) + 0.5) * np.pi / 4.0,
    )
    np.testing.assert_allclose(Phi[0], np.deg2rad([0.0, 90.0, 180.0, 270.0]))
    np.testing.assert_allclose(longitude, [45.0, 135.0, 225.0, 315.0])


def test_hmi_fdt_interpolation_resizes_normalized_carrington_cubes(
    tmp_path,
    monkeypatch,
):
    dates = [datetime(2026, 8, 18, 8) + timedelta(hours=12 * index) for index in range(4)]
    candidates = []
    local_files = []
    base = np.arange(16.0).reshape(4, 4)

    for index, date in enumerate(dates):
        name = (
            f"adapt40i11_044012_{date.strftime('%Y%m%d%H%M')}_"
            "i00010000n1.fts"
        )
        file_path = tmp_path / name
        _write_hmi_fdt_map(file_path, date, base + 10.0 * index)
        candidates.append(MagnetogramCandidate(name, date, "unused"))
        local_files.append(str(file_path))

    target = dates[1] + timedelta(hours=6)
    selection = InterpolationSelection(
        before_previous=candidates[0],
        before=candidates[1],
        after=candidates[2],
        after_next=candidates[3],
        coef_before=0.5,
        coef_after=0.5,
        interval_seconds=12.0 * 3600.0,
        previous_interval_seconds=12.0 * 3600.0,
        next_interval_seconds=12.0 * 3600.0,
        target_date=target,
    )

    resized_inputs = []
    fake_skimage = types.ModuleType("skimage")
    fake_transform = types.ModuleType("skimage.transform")

    def fake_resize(image, output_shape, **kwargs):
        resized_inputs.append(image.copy())
        return np.full(output_shape, image[0, 0], dtype=float)

    fake_transform.resize = fake_resize
    monkeypatch.setitem(sys.modules, "skimage", fake_skimage)
    monkeypatch.setitem(sys.modules, "skimage.transform", fake_transform)

    Br, Theta, Phi, Br_linear = read_interpolated_magnetogram(
        local_files,
        "HMI_fdt",
        selection,
        adapt_map=3,
        interpolation_order=2,
        resize=True,
    )

    assert len(resized_inputs) == 4
    for index, resize_input in enumerate(resized_inputs):
        expected = (base + 10.0 * index + 300.0)[::-1, :]
        np.testing.assert_array_equal(resize_input, expected)
    assert Br.shape == Theta.shape == Phi.shape == Br_linear.shape == (360, 720)
    np.testing.assert_allclose(Br, 327.0)
    np.testing.assert_allclose(Br_linear, 327.0)
    np.testing.assert_allclose(
        Theta[:, 0],
        (np.arange(360, dtype=float) + 0.5) * np.pi / 360.0,
    )

    with fits.open(local_files[-1], mode="update") as hdul:
        hdul[0].header["CRVAL1"] = 181.0
    from coconut_tools.magnetogram.io import readers

    monkeypatch.setattr(
        readers,
        "interpolate_br_maps",
        lambda *args, **kwargs: pytest.fail(
            "Longitude frames must be checked before temporal interpolation"
        ),
    )
    with pytest.raises(RuntimeError, match="fixed Carrington longitude grid"):
        read_interpolated_magnetogram(
            local_files,
            "HMI_fdt",
            selection,
            adapt_map=3,
        )


def test_interpolated_hmi_fdt_rotation_uses_target_time_and_carrington_axis(
    tmp_path,
    monkeypatch,
):
    from coconut_tools.magnetogram.processing import longitude

    source_date = datetime(2026, 8, 18, 8)
    target = datetime(2026, 8, 19, 2)
    source_file = (
        tmp_path
        / "adapt40i11_044012_202608180800_i00010000n1.fts"
    )
    _write_hmi_fdt_map(source_file, source_date, np.zeros((4, 4)))

    rotation_dates = []

    def fake_central_meridian(date):
        rotation_dates.append(date)
        return 67.5

    monkeypatch.setattr(
        longitude,
        "compute_carrington_central_meridian",
        fake_central_meridian,
    )

    Br = np.arange(16).reshape(2, 8)
    Br_linear = Br + 10
    Br_rotated, Br_linear_rotated, rotation_angle = apply_configured_longitude_rotation(
        Br,
        Br_linear,
        [str(source_file)] * 4,
        "HMI_fdt",
        target,
        use_interpolation=True,
        rotate_to_stonyhurst=True,
        resize=True,
    )

    np.testing.assert_array_equal(Br_rotated, np.roll(Br, -1, axis=1))
    np.testing.assert_array_equal(Br_linear_rotated, np.roll(Br_linear, -1, axis=1))
    assert rotation_angle == pytest.approx(67.5)
    assert rotation_dates == [target]


def test_single_hmi_fdt_rotation_uses_the_file_effective_time(
    tmp_path,
    monkeypatch,
):
    from coconut_tools.tools import rotation_angle

    source_date = datetime(2026, 8, 19, 20)
    target = datetime(2026, 8, 20, 1, 30)
    source_file = (
        tmp_path
        / "adapt40i11_044012_202608192000_i00011200n1.fts"
    )
    _write_hmi_fdt_map(source_file, source_date, np.zeros((4, 4)))

    rotation_dates = []

    def fake_central_meridian(date):
        rotation_dates.append(date)
        return 131.25

    monkeypatch.setattr(
        rotation_angle,
        "compute_carrington_central_meridian",
        fake_central_meridian,
    )

    angle, effective_date = compute_rotation_angle(
        str(source_file),
        date_hmi=target.isoformat(),
        map_type="HMI_fdt",
    )

    assert angle == pytest.approx(131.25)
    assert effective_date == source_date
    assert rotation_dates == [source_date]


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
    from coconut_tools.magnetogram.processing import longitude

    longitude_original = np.array([10.0, 100.0, 190.0, 280.0])
    Br = np.arange(8).reshape(1, 8)

    monkeypatch.setattr(
        longitude,
        "compute_rotation_angle",
        lambda *args, **kwargs: (100.0, TARGET_DATE),
    )
    monkeypatch.setattr(
        longitude,
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


def test_interpolated_hmi_hourly_rotation_uses_resized_longitude_axis(monkeypatch):
    from coconut_tools.magnetogram.processing import longitude

    Br = np.arange(8).reshape(1, 8)
    Br_linear = Br + 10

    monkeypatch.setattr(
        longitude,
        "compute_carrington_central_meridian",
        lambda *args, **kwargs: 90.0,
    )
    monkeypatch.setattr(
        longitude,
        "processed_longitude_axis",
        lambda *args, **kwargs: np.array([0.0, 90.0, 180.0, 270.0]),
    )

    Br_rotated, Br_linear_rotated, rotation_angle = apply_configured_longitude_rotation(
        Br,
        Br_linear,
        ["dummy_hmi_hourly.fits"] * 4,
        "HMI_hourly",
        TARGET_DATE,
        use_interpolation=True,
        rotate_to_stonyhurst=True,
        effective_date=TARGET_DATE,
        resize=True,
    )

    np.testing.assert_array_equal(Br_rotated, np.roll(Br, -2, axis=1))
    np.testing.assert_array_equal(Br_linear_rotated, np.roll(Br_linear, -2, axis=1))
    assert rotation_angle == pytest.approx(90.0)


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
