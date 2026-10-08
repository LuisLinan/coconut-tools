import ast
import logging
import os
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest
from parfive import Results

from coconut_tools.magnetogram.io import downloads


TARGET_DATE = datetime(2026, 10, 1)


def _download_map(entry_point, output_dir):
    if entry_point == "candidate":
        candidate = downloads.MagnetogramCandidate(
            "map.fits", TARGET_DATE, "https://example.com/map.fits"
        )
        return downloads.download_candidate(candidate, str(output_dir))
    _, local_file = downloads.generate_output_and_map_names(
        TARGET_DATE, entry_point, str(output_dir)
    )
    return local_file


@pytest.mark.parametrize("entry_point", ["candidate", "HMI_polfil", "HMI_small"])
def test_download_returns_file_and_reuses_local_cache(tmp_path, monkeypatch, entry_point):
    monkeypatch.setattr(downloads.sunpy.coordinates.sun, "carrington_rotation_number", lambda _: 2316)
    filename = "map.fits" if entry_point == "candidate" else f"hmi.Synoptic_Mr_{entry_point[4:]}.2316.fits"
    destination = tmp_path / filename
    calls = []

    def fake_download(urls, *, path, overwrite):
        calls.append(urls)
        assert Path(path) == tmp_path
        assert overwrite is True
        destination.write_bytes(b"downloaded map")
        return Results([destination])

    monkeypatch.setattr(downloads.Downloader, "simple_download", fake_download)
    assert _download_map(entry_point, tmp_path) == str(destination)
    assert _download_map(entry_point, tmp_path) == str(destination)
    assert len(calls) == 1
    if entry_point != "candidate":
        assert calls[0] == [f"https://jsoc1.stanford.edu/data/hmi/synoptic/{filename}"]


@pytest.mark.parametrize("entry_point", ["candidate", "HMI_polfil", "HMI_small"])
def test_http_404_reports_source_without_returning_a_missing_file(tmp_path, monkeypatch, entry_point):
    monkeypatch.setattr(downloads.sunpy.coordinates.sun, "carrington_rotation_number", lambda _: 2316)
    errors = []

    def fake_download(urls, **kwargs):
        error = HTTPError(urls[0], 404, "Not Found", None, None)
        errors.append(error)
        results = Results()
        results.add_error(None, urls[0], error)
        return results

    monkeypatch.setattr(downloads.Downloader, "simple_download", fake_download)
    with pytest.raises(RuntimeError, match="404") as caught:
        _download_map(entry_point, tmp_path)
    assert errors[0].url in str(caught.value)
    assert caught.value.__cause__ is errors[0]
    assert not list(tmp_path.iterdir())


def test_empty_download_result_is_an_explicit_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(downloads.Downloader, "simple_download", lambda *args, **kwargs: Results())
    with pytest.raises(RuntimeError, match="No file downloaded"):
        _download_map("candidate", tmp_path)


@pytest.mark.parametrize('suffix', ['.fits', '.fits.1', '.fits.2'])
def test_hmi_sync_repeated_download_keeps_returned_path_and_updates_wcs(tmp_path, suffix):
    import numpy as np
    import pandas as pd
    from astropy.io import fits

    destination = tmp_path / ('map' + suffix)
    hdu = fits.PrimaryHDU(np.ones((2, 4)))
    hdu.writeto(destination)
    original = tmp_path / 'existing.fits'
    original.write_bytes(b'keep existing download')
    client = SimpleNamespace(
        export=lambda *args, **kwargs: SimpleNamespace(
            download=lambda directory: SimpleNamespace(download=pd.Series([str(destination)]))),
        query=lambda *args, **kwargs: pd.DataFrame([dict(
            CRVAL1=832949.899988, CRPIX1=1800., CDELT1=-.1,
            CTYPE1='CRLN-CEA', CUNIT1='degree')]),
    )
    result = downloads.try_download_hmi_sync_record(client, '2026.09.01_00:24:00_TAI', str(tmp_path))
    assert result == (str(destination), destination.name)
    with fits.open(destination) as hdus:
        assert hdus[0].header['CRVAL1'] == 832949.899988
        np.testing.assert_array_equal(hdus[0].data, np.ones((2, 4)))
    assert original.read_bytes() == b'keep existing download'


def test_hmi_sync_rejects_non_fits_download(tmp_path):
    client = SimpleNamespace(export=lambda *args, **kwargs: SimpleNamespace(
        download=lambda directory: SimpleNamespace(download=[str(tmp_path / 'error.html')])) )
    with pytest.raises(RuntimeError, match='did not return a FITS file'):
        downloads.try_download_hmi_sync_record(client, '2026.09.01_00:24:00_TAI', str(tmp_path))


def test_reference_batch_continues_after_download_failure(caplog):
    source = Path(__file__).parents[1] / "src/coconut_tools/magnetogram/sph_filtering.py"
    main = next(node for node in ast.parse(source.read_text(encoding="utf-8")).body if isinstance(node, ast.If))
    calls = []

    def fake_process(config, **kwargs):
        calls.append(config.copy())
        if config["map_type"] == "HMI_polfil":
            raise RuntimeError("HTTP 404: Not Found")

    namespace = {
        "__name__": "__main__",
        "os": os,
        "logger": logging.getLogger(__name__),
        "process_config": fake_process,
    }
    with caplog.at_level(logging.INFO):
        exec(compile(ast.Module(body=[main], type_ignores=[]), str(source), "exec"), namespace)
    expected_types = set(downloads.MAP_TYPE_ALIASES.values()) - {"WSO", "custom"}
    expected_types |= {"GONG"} | {f"GONG_{item}" for item in downloads.GONG_FILE_IDS}
    assert {config["map_type"] for config in calls} == expected_types
    assert len({config["date"] for config in calls}) == 1
    assert namespace["failed_map_types"] == ["HMI_polfil"]
    assert len(namespace["completed_map_types"]) == len(expected_types) - 1
    assert "Failed map types: HMI_polfil" in caplog.text
    assert "11/12 map types completed" in caplog.text
