from datetime import datetime
from types import SimpleNamespace

import pytest

from coconut_tools.magnetogram import magnetogram_download


def test_hmi_sync_requires_drms_email(tmp_path, monkeypatch):
    monkeypatch.setattr(
        magnetogram_download.sunpy.coordinates.sun,
        "carrington_rotation_number",
        lambda date: 1234,
    )

    with pytest.raises(ValueError, match="DRMS email"):
        magnetogram_download.generate_output_and_map_names(
            datetime(2020, 12, 7, 15, 0),
            "HMI_SYNC",
            str(tmp_path),
            method_used="sph",
        )


def test_generate_output_and_map_names_accepts_legacy_method_position(tmp_path, monkeypatch):
    monkeypatch.setattr(
        magnetogram_download.sunpy.coordinates.sun,
        "carrington_rotation_number",
        lambda date: 1234,
    )
    (tmp_path / "WSO.1234.txt").write_text("existing map", encoding="utf-8")

    output_name, local_file = magnetogram_download.generate_output_and_map_names(
        datetime(2020, 12, 7, 15, 0),
        "WSO",
        str(tmp_path),
        "NLD",
    )

    assert output_name == str(tmp_path / "map_wso_NLD.dat")
    assert local_file == str(tmp_path / "WSO.1234.txt")


def test_hmi_sync_downloads_with_drms(tmp_path, monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, email):
            captured["email"] = email

        def export(self, series, protocol):
            captured["series"] = series
            captured["protocol"] = protocol
            return SimpleNamespace(
                download=lambda output_dir: SimpleNamespace(
                    download=[str(tmp_path / "hmi_sync.fits")]
                )
            )

    monkeypatch.setattr(
        magnetogram_download.sunpy.coordinates.sun,
        "carrington_rotation_number",
        lambda date: 1234,
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "drms",
        SimpleNamespace(Client=FakeClient),
    )
    monkeypatch.setattr(
        magnetogram_download.sunpy.util.net,
        "download_file",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("HMI_SYNC must use DRMS downloads.")
        ),
    )

    output_name, local_file = magnetogram_download.generate_output_and_map_names(
        datetime(2020, 12, 7, 15, 0),
        "HMI_SYNC",
        str(tmp_path),
        method_used="sph",
        drms_email="user@example.com",
    )

    assert output_name == str(tmp_path / "map_hmi_sync_sph.dat")
    assert local_file == str(tmp_path / "hmi_sync.fits")
    assert captured == {
        "email": "user@example.com",
        "series": "hmi.Mrdailysynframe_720s[2020.12.07_12:00:00_TAI]",
        "protocol": "fits",
    }


def test_hmi_sync_display_date_uses_daily_noon():
    assert magnetogram_download.magnetogram_display_date(
        "hmi_sync.fits",
        "HMI_SYNC",
        datetime(2020, 12, 7, 15, 30, 45),
    ) == datetime(2020, 12, 7, 12, 0)


@pytest.mark.parametrize("file_id", ["mrbqs", "mrbqj"])
def test_gong_variant_uses_file_id_for_download_selection(tmp_path, monkeypatch, file_id):
    captured = []
    map_name = f"{file_id}201207t1504c2238_181.fits.gz"
    (tmp_path / map_name).write_text("existing map", encoding="utf-8")

    monkeypatch.setattr(
        magnetogram_download.sunpy.coordinates.sun,
        "carrington_rotation_number",
        lambda date: 2238,
    )

    def fake_fetch_remote_names(remote_dir, file_id):
        captured.append((remote_dir, file_id))
        if file_id == captured_file_id and "202012" in remote_dir:
            return [map_name]
        return []

    captured_file_id = file_id
    monkeypatch.setattr(
        magnetogram_download,
        "fetch_remote_names",
        fake_fetch_remote_names,
    )

    output_name, local_file = magnetogram_download.generate_output_and_map_names(
        datetime(2020, 12, 7, 15, 0),
        f"GONG_{file_id}",
        str(tmp_path),
        method_used="sph",
    )

    assert output_name == str(tmp_path / f"map_gong_{file_id}_sph.dat")
    assert local_file == str(tmp_path / map_name)
    assert {observed_file_id for _, observed_file_id in captured} == {file_id}
    assert all(f"/{file_id[2:]}/" in remote_dir for remote_dir, _ in captured)


@pytest.mark.parametrize("file_id", ["mrbqs", "mrbqj"])
def test_gong_variant_display_date_uses_variant_file_id(file_id):
    assert magnetogram_download.magnetogram_display_date(
        f"{file_id}201207t1504c2238_181.fits.gz",
        f"GONG_{file_id}",
        datetime(2020, 12, 7, 15, 0),
    ) == datetime(2020, 12, 7, 15, 4)


@pytest.mark.parametrize("file_id", ["mrmqs", "mrnqs"])
def test_gong_diachronic_download_selects_nearest_folder_and_file(
    tmp_path,
    monkeypatch,
    file_id,
):
    map_name = f"{file_id}201207t1104c2238.fits.gz"
    (tmp_path / map_name).write_text("existing map", encoding="utf-8")
    requested_urls = []

    monkeypatch.setattr(
        magnetogram_download.sunpy.coordinates.sun,
        "carrington_rotation_number",
        lambda date: 2238,
    )

    def fake_get(url):
        requested_urls.append(url)
        if url.endswith("/202012/"):
            return SimpleNamespace(
                text=(
                    f'<a href="{file_id}201206/">{file_id}201206/</a>'
                    f'<a href="{file_id}201207/">{file_id}201207/</a>'
                    f'<a href="{file_id}201208/">{file_id}201208/</a>'
                )
            )
        return SimpleNamespace(
            text=(
                f'<a href="{file_id}201207t0904c2238.fits.gz">early</a>'
                f'<a href="{map_name}">nearest</a>'
            )
        )

    monkeypatch.setattr(magnetogram_download.requests, "get", fake_get)

    output_name, local_file = magnetogram_download.generate_output_and_map_names(
        datetime(2020, 12, 7, 11, 0),
        f"GONG_{file_id}",
        str(tmp_path),
        method_used="sph",
    )

    assert output_name == str(tmp_path / f"map_gong_{file_id}_sph.dat")
    assert local_file == str(tmp_path / map_name)
    assert requested_urls == [
        f"https://gong.nso.edu/data/magmap/QR/{file_id[2:]}/202012/",
        f"https://gong.nso.edu/data/magmap/QR/{file_id[2:]}/202012/{file_id}201207/",
    ]
