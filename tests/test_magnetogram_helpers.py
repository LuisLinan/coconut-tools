from datetime import datetime
from pathlib import Path
import shutil
import sys
import types

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from coconut_tools.magnetogram.magnetogram_download import (
    InterpolationSelection,
    MagnetogramCandidate,
    build_output_name,
    default_figure_path,
    download_interpolation_magnetograms,
    download_hmi_hourly_magnetogram,
    ensure_hmi_sync_wcs,
    list_hmi_candidates,
    magnetogram_display_date,
    magnetogram_effective_date,
    normalize_map_type,
    parse_hmi_hourly_filename_date,
    resolve_figure_path,
)
from coconut_tools.magnetogram.sph_filtering import (
    apply_configured_longitude_rotation,
    closest_longitude_column,
    correct_net_flux,
    processed_longitude_axis,
    read_magnetogram,
    read_interpolated_magnetogram,
    read_temporal_br_map,
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


def test_map_type_normalization_accepts_case_variants():
    outdir = Path(__file__).parent / "_outputs"

    assert normalize_map_type("hmi_sync") == "HMI_SYNC"
    assert normalize_map_type("HMI_SYNC") == "HMI_SYNC"
    assert normalize_map_type("HMI_HOURLY") == "HMI_hourly"
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
    assert build_output_name("hmi_hourly", str(outdir), "sph") == str(
        outdir / "map_hmi_hourly_sph.dat"
    )
    assert build_output_name("gong_MRBQS", str(outdir), "sph") == str(
        outdir / "map_gong_mrbqs_sph.dat"
    )


def test_hmi_hourly_candidates_use_timestamps_encoded_in_their_names(monkeypatch):
    captured = {}
    keys = pd.DataFrame(
        {
            "T_REC": [
                "2026.07.01_08:00:00_TAI",
                "2026.07.01_06:00:00_TAI",
            ]
        }
    )
    segments = pd.DataFrame(
        {
            "Mr_polfil": [
                "/SUM97/D123/map_0800.fits",
                "https://example.test/map_0600.fits",
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
        "hmi.synoptic_hourly_20260701_080000.fits",
    ]
    assert [candidate.date for candidate in candidates] == [
        datetime(2026, 7, 1, 6),
        datetime(2026, 7, 1, 8),
    ]
    assert candidates[0].remote_url == "https://example.test/map_0600.fits"
    assert candidates[1].remote_url == (
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
    from coconut_tools.magnetogram import magnetogram_download

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
