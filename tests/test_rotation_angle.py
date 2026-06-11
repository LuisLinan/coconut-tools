from datetime import datetime

import numpy as np
import pytest
from astropy.io import fits

from coconut_tools.tools.rotation_angle import compute_rotation_angle, is_br_longitude_increasing


def _write_magnetogram(path, extension=False, **header_values):
    hdu = fits.PrimaryHDU(np.zeros((4, 8)))
    for key, value in header_values.items():
        hdu.header[key] = value
    if extension:
        image_hdu = fits.ImageHDU(hdu.data, header=hdu.header)
        fits.HDUList([fits.PrimaryHDU(), image_hdu]).writeto(path)
    else:
        hdu.writeto(path)


@pytest.mark.parametrize(
    ("header_values", "expected"),
    [
        ({"CDELT1": 1.0}, True),
        ({"CDELT1": -1.0}, False),
        ({"CDELT1": -1.0, "PC1_1": -1.0}, True),
        ({"CD1_1": -0.5, "CDELT1": 1.0}, False),
    ],
)
def test_is_br_longitude_increasing(tmp_path, header_values, expected):
    magnetogram = tmp_path / "magnetogram.fits"
    _write_magnetogram(magnetogram, **header_values)

    assert is_br_longitude_increasing(str(magnetogram)) is expected


def test_is_br_longitude_increasing_requires_longitude_step(tmp_path):
    magnetogram = tmp_path / "magnetogram.fits"
    _write_magnetogram(magnetogram)

    with pytest.raises(ValueError, match="neither CD1_1 nor CDELT1"):
        is_br_longitude_increasing(str(magnetogram))


def test_is_br_longitude_increasing_reads_image_extension(tmp_path):
    magnetogram = tmp_path / "magnetogram.fits"
    _write_magnetogram(magnetogram, extension=True, CDELT1=-0.1)

    assert is_br_longitude_increasing(str(magnetogram)) is False


def test_compute_rotation_angle_supports_wso_without_fits_header(monkeypatch, tmp_path):
    magnetogram = tmp_path / "WSO.2238.txt"
    magnetogram.write_text("WSO text magnetogram")
    monkeypatch.setattr(
        "coconut_tools.tools.rotation_angle.compute_carrington_central_meridian",
        lambda date: 241.25,
    )

    angle, date = compute_rotation_angle(
        str(magnetogram),
        date_hmi="2020-12-07T15:00:00",
    )

    assert angle == 241.25
    assert date == datetime(2020, 12, 7, 15, 0)
