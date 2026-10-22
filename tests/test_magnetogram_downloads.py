import ast
import logging
import os
import sys
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
@pytest.mark.parametrize('series_name', [downloads.HMI_SYNC_SERIES, downloads.HMI_SYNC_DEFINITIVE_SERIES])
def test_hmi_sync_repeated_download_keeps_returned_path_and_updates_wcs(tmp_path, suffix, series_name):
    import numpy as np
    import pandas as pd
    from astropy.io import fits

    destination = tmp_path / ('map' + suffix)
    hdu = fits.PrimaryHDU(np.ones((2, 4)))
    hdu.writeto(destination)
    original = tmp_path / 'existing.fits'
    original.write_bytes(b'keep existing download')
    t_rec = '2026.09.01_00:24:00_TAI'
    expected_recordset = f'{series_name}[{t_rec}]'

    def export(recordset, protocol):
        assert recordset == expected_recordset
        assert protocol == 'fits'
        return SimpleNamespace(download=lambda directory: SimpleNamespace(
            download=pd.Series([str(destination)])))

    def query(recordset, key):
        assert recordset == expected_recordset
        return pd.DataFrame([dict(
            CRVAL1=832949.899988, CRPIX1=1800., CDELT1=-.1,
            CTYPE1='CRLN-CEA', CUNIT1='degree')])

    client = SimpleNamespace(export=export, query=query)
    result = downloads.try_download_hmi_sync_record(
        client, t_rec, str(tmp_path), series_name=series_name
    )
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


def test_hmi_sync_empty_export_allows_trying_another_record(tmp_path, caplog, monkeypatch):
    monkeypatch.setattr(downloads.logger, 'propagate', True)
    client = SimpleNamespace(export=lambda *args, **kwargs: SimpleNamespace(
        download=lambda directory: SimpleNamespace(download=[])))
    assert downloads.try_download_hmi_sync_record(
        client, '2018.03.18_11:12:00_TAI', str(tmp_path)
    ) is None
    assert 'JSOC export returned no files' in caplog.text


@pytest.mark.parametrize('error', [
    'The requested FITS files no longer exist. [status=25]',
    'No FITS files were exported.',
    'Data are offline',
])
def test_hmi_sync_logs_original_unavailable_error(tmp_path, caplog, monkeypatch, error):
    monkeypatch.setattr(downloads.logger, 'propagate', True)
    def export(*args, **kwargs):
        raise RuntimeError(error)

    assert downloads.try_download_hmi_sync_record(
        SimpleNamespace(export=export), '2018.03.18_11:12:00_TAI', str(tmp_path)
    ) is None
    assert error in caplog.text


def test_hmi_sync_does_not_hide_unexpected_export_errors(tmp_path):
    error = RuntimeError('JSOC email is not registered [status=4]')

    def export(*args, **kwargs):
        raise error

    with pytest.raises(RuntimeError) as caught:
        downloads.try_download_hmi_sync_record(
            SimpleNamespace(export=export), '2018.03.18_11:12:00_TAI', str(tmp_path)
        )
    assert caught.value is error


@pytest.mark.parametrize('definitive_state', ['available', 'missing_metadata', 'offline'])
def test_hmi_sync_prefers_definitive_and_falls_back_to_nrt(
    tmp_path, monkeypatch, definitive_state
):
    import numpy as np
    import pandas as pd
    from astropy.io import fits

    t_rec = '2018.03.18_11:12:00_TAI'
    definitive = downloads.HMI_SYNC_DEFINITIVE_SERIES
    nrt = downloads.HMI_SYNC_SERIES
    exports = []
    wcs_queries = []

    class FakeClient:
        def query(self, recordset, key):
            series_name = recordset.split('[')[0]
            if key == 'T_REC':
                if series_name == definitive and definitive_state == 'missing_metadata':
                    return pd.DataFrame(columns=['T_REC'])
                # Deliberately unsorted: the successful export must be the nearest.
                return pd.DataFrame({'T_REC': ['2018.03.18_11:48:00_TAI', t_rec]})
            wcs_queries.append(recordset)
            return pd.DataFrame([dict(CRVAL1=90., CRPIX1=1., CDELT1=-.1,
                                      CTYPE1='CRLN-CEA', CUNIT1='degree')])

        def export(self, recordset, protocol):
            exports.append(recordset)
            if recordset.startswith(definitive + '[') and definitive_state == 'offline':
                raise RuntimeError('Data are offline [status=25]')

            def download(directory):
                destination = Path(directory) / 'map.fits'
                fits.PrimaryHDU(np.ones((2, 4))).writeto(destination)
                return pd.DataFrame({'download': [str(destination)]})

            return SimpleNamespace(download=download)

    monkeypatch.setitem(sys.modules, 'drms', SimpleNamespace(Client=lambda **kwargs: FakeClient()))
    result = downloads.download_hmi_sync_magnetogram(
        datetime(2018, 3, 18), str(tmp_path), 'registered@example.test'
    )
    selected_series = definitive if definitive_state == 'available' else nrt
    selected_record = f'{selected_series}[{t_rec}]'
    assert result == (str(tmp_path / 'map.fits'), 'map.fits')
    assert exports[-1] == selected_record
    assert wcs_queries == [selected_record]
    assert fits.getheader(result[0])['CRVAL1'] == 90.
    if definitive_state == 'available':
        assert exports == [selected_record]
    elif definitive_state == 'offline':
        assert exports[0] == f'{definitive}[{t_rec}]'


def test_hmi_sync_does_not_repeat_failed_records_across_windows(tmp_path, monkeypatch):
    import pandas as pd

    exports = []
    t_rec = '2018.03.18_11:12:00_TAI'

    class FakeClient:
        def query(self, recordset, key):
            return pd.DataFrame({'T_REC': [t_rec]})

        def export(self, recordset, protocol):
            exports.append(recordset)
            raise RuntimeError('No FITS files were exported.')

    monkeypatch.setitem(sys.modules, 'drms', SimpleNamespace(Client=lambda **kwargs: FakeClient()))
    with pytest.raises(RuntimeError, match='Tried 2 distinct metadata records'):
        downloads.download_hmi_sync_magnetogram(
            datetime(2018, 3, 18), str(tmp_path), 'registered@example.test'
        )
    assert exports == [
        f'{downloads.HMI_SYNC_DEFINITIVE_SERIES}[{t_rec}]',
        f'{downloads.HMI_SYNC_SERIES}[{t_rec}]',
    ]


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
