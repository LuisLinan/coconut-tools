"""
Compute Carrington to Stonyhurst rotation angle from magnetogram filenames.

This module parses magnetogram names from different providers (GONG, ADAPT, HMI)
and computes the longitudinal rotation needed to align their maps with the
Stonyhurst frame used by heliospheric models like EUHFORIA.

Supports:
- GONG synoptic and ADAPT (CM/CAR modes),
- HMI (with explicit observation time),
- Angle computation based on the central meridian difference.

Used during preprocessing to rotate coronal boundary data into the appropriate HEEQ frame.
"""

from sunpy.coordinates import frames
from astropy.coordinates import SkyCoord
import astropy.units as u
from datetime import datetime
import os
from coconut_tools.tools.logger_config import setup_logger
from typing import Tuple
from astropy.io import fits
import numpy as np


logger = setup_logger(__name__)


def compute_carrington_central_meridian(date: str | datetime) -> float:
    """Return the Carrington longitude of Earth's central meridian."""
    if isinstance(date, str):
        date = datetime.fromisoformat(date)
    central_meridian_stonyhurst = SkyCoord(
        0 * u.deg,
        0 * u.deg,
        frame=frames.HeliographicStonyhurst,
        obstime=date,
        observer="earth",
    )
    central_meridian_carrington = central_meridian_stonyhurst.transform_to(
        frames.HeliographicCarrington(observer="earth", obstime=date)
    )
    return float(central_meridian_carrington.lon.value % 360)


def _magnetogram_header(path: str) -> fits.Header:
    """Return the header of the first HDU containing magnetogram image data."""
    with fits.open(path) as hdul:
        for hdu in hdul:
            if hdu.data is not None and hdu.data.ndim >= 2:
                return hdu.header.copy()
    raise ValueError(f"No magnetogram image HDU found in FITS file: {path}")


def _lon_grid_from_fits_1d(path: str) -> np.ndarray:
    """Reconstruct the 1D longitude grid from a FITS header.

    This function rebuilds the longitude axis using standard FITS WCS
    keywords (CRPIX1, CRVAL1, CDELT1). It is mainly used to determine the
    longitude convention of ADAPT magnetograms, in particular for
    central-meridian-centered products (ADAPT 413).

    Args:
        path (str): Path to the FITS magnetogram file.

    Returns:
        np.ndarray: One-dimensional array of longitudes in degrees,
        following the native convention of the FITS file.
    """
    hdr = _magnetogram_header(path)

    n = int(hdr["NAXIS1"])
    crpix1 = float(hdr.get("CRPIX1", 1.0))
    crval1 = float(hdr.get("CRVAL1", 0.0))
    cdelt1 = float(hdr.get("CDELT1", 1.0))

    i = np.arange(1, n + 1, dtype=float)
    lon = crval1 + (i - crpix1) * cdelt1
    return lon


def increasing_longitude_axis(path: str) -> np.ndarray:
    """Return the native longitude axis ordered like a normalized Br map."""
    lon = _lon_grid_from_fits_1d(path)
    if not is_br_longitude_increasing(path):
        lon = lon[::-1]
    return lon


def is_br_longitude_increasing(path: str) -> bool:
    """Return whether Br columns follow increasing native longitude.

    The longitude axis is reconstructed without wrapping it into [0, 360), so
    crossing the Carrington origin does not affect the direction check.

    Args:
        path (str): Path to the FITS magnetogram file.

    Returns:
        bool: True when longitude increases with the Br column index.

    Raises:
        ValueError: If the longitude direction cannot be determined.
    """
    hdr = _magnetogram_header(path)

    if "CD1_1" in hdr:
        longitude_step = float(hdr["CD1_1"])
    elif "CDELT1" in hdr:
        longitude_step = float(hdr["CDELT1"]) * float(hdr.get("PC1_1", 1.0))
    else:
        raise ValueError(
            f"Could not determine the longitude direction for magnetogram: {path}. "
            "The FITS header has neither CD1_1 nor CDELT1."
        )

    if not np.isfinite(longitude_step) or np.isclose(longitude_step, 0.0):
        raise ValueError(
            f"Could not determine the longitude direction for magnetogram: {path}"
        )

    is_increasing = longitude_step > 0.0
    logger.info(
        "Br longitude increases with column index: %s "
        "(native longitude step: %.6g)",
        is_increasing,
        longitude_step,
    )
    return is_increasing


def compute_rotation_angle(mag_name_path: str, date_hmi: str = None) -> Tuple[float, datetime]:
    """Compute the rotation angle from magnetogram filename.

    Supports GONG, ADAPT (CM and CAR), and HMI.

    Args:
        mag_name_path (str): Path to the magnetogram file.
        date_hmi (str, optional): ISO datetime string used for HMI (e.g. '2024-07-01T00:00:00')

    Returns:
        float: Angle alpha in degrees such that lon_heeq = lon_mag - alpha (mod 360)
        datetime: Observation date of the magnetogram
    """
    mag_name = os.path.basename(mag_name_path)
    logger.info(f"The magnetogram name is: {mag_name}")

    prefix = mag_name[:5].lower()
    is_wso = mag_name.lower().startswith("wso.")
    logger.info(f"Prefix is: {prefix}")
    if not is_wso:
        is_br_longitude_increasing(mag_name_path)

    # GONG
    if prefix in ['mrbqs', 'mrzqs']:
        logger.info("The magnetogram is GONG (synoptic)")
        date = datetime.strptime(mag_name[5:16], '%y%m%dt%H%M')
        start_lon = int(mag_name[22:25])

        CM_CAR_value = compute_carrington_central_meridian(date)
        angle = CM_CAR_value - start_lon
        return angle % 360 , date

    # ADAPT
    elif prefix == 'adapt':
        mode = mag_name[5:8]
        date = datetime.strptime(mag_name[18:30], '%Y%m%d%H%M')

        if mode == '403':
            logger.info("The magnetogram is GONG ADAPT in CAR frame")
            CM_CAR_value = compute_carrington_central_meridian(date)
            return (CM_CAR_value + 0) % 360 , date
        elif mode == '413':
            logger.info("The magnetogram is GONG ADAPT in CM centered frame")

            lon = _lon_grid_from_fits_1d(mag_name_path)
            lon_min = float(np.nanmin(lon))
            lon_max = float(np.nanmax(lon))

            # Cas 1: grille type CMD dans [-180, 180], déjà centrée sur CM
            if lon_min < 0 and lon_max <= 180.1:
                return 0.0, date

            # Cas 2: grille en [0, 360] avec CM au centre par convention (souvent 180)
            if lon_min >= 0 and lon_max > 180.0:
                return 180.0, date

            logger.error(f"Unrecognized ADAPT413 longitude range: [{lon_min:.2f}, {lon_max:.2f}]")
            raise ValueError("Unrecognized ADAPT413 longitude convention.")


    # HMI
    elif 'hmi' in prefix or 'mr' in prefix:
        logger.info("The magnetogram is HMI in CAR frame")
        if not date_hmi:
            raise ValueError("You must provide 'date_hmi' for HMI magnetograms.")
        date = datetime.strptime(date_hmi, '%Y-%m-%dT%H:%M:%S')

        CM_CAR_value = compute_carrington_central_meridian(date)
        return (CM_CAR_value + 0) % 360, date
    elif is_wso:
        logger.info("The magnetogram is WSO in CAR frame")
        if not date_hmi:
            raise ValueError("You must provide a date for WSO magnetograms.")
        date = datetime.fromisoformat(date_hmi)
        return compute_carrington_central_meridian(date), date
    else:
        logger.error("Magnetogram filename format not recognized: %s", mag_name)
        raise ValueError("Magnetogram filename format not recognized.")

if __name__ == "__main__":
    base_dir = r"C:\Users\luisl\Desktop\testmagnetogram"

    test_cases = [
        {
            "desc": "HMI small test",
            "path": os.path.join(base_dir, "hmi.Synoptic_Mr_small.2238.fits"),
            "time": "2020-12-07T15:00:00"
        },
        {
            "desc": "HMI polar-filled test",
            "path": os.path.join(base_dir, "hmi.Synoptic_Mr_polfil.2238.fits"),
            "time": "2020-12-07T15:00:00"
        },
        {
            "desc": "GONG zero-point-corrected synoptic test",
            "path": os.path.join(base_dir, "mrzqs201207t1504c2238_181.fits.gz"),
            "time": None
        },
        {
            "desc": "ADAPT Carrington test",
            "path": os.path.join(base_dir, "adapt40311_044012_202012071400_i00012600n1.fts.gz"),
            "time": None
        },
    ]

    for test in test_cases:
        logger.info(f"\n====== {test['desc']} ======")
        if not os.path.exists(test["path"]):
            logger.warning("Skipping missing magnetogram: %s", test["path"])
            continue
        angle, date = compute_rotation_angle(test["path"], date_hmi=test["time"])
        logger.info(f"Rotation angle: {angle:.2f} degrees at {date.isoformat()}")
