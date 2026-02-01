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
from coconut_tools.logger_config import setup_logger
from typing import Tuple
from astropy.io import fits
import numpy as np


logger = setup_logger(__name__)

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
    with fits.open(path) as hdul:
        hdr = hdul[0].header

    n = int(hdr["NAXIS1"])
    crpix1 = float(hdr.get("CRPIX1", 1.0))
    crval1 = float(hdr.get("CRVAL1", 0.0))
    cdelt1 = float(hdr.get("CDELT1", 1.0))

    i = np.arange(1, n + 1, dtype=float)
    lon = crval1 + (i - crpix1) * cdelt1
    return lon

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
    logger.info(f"Prefix is: {prefix}")

    # GONG
    if prefix in ['mrbqs', 'mrzqs']:
        logger.info("The magnetogram is GONG (synoptic)")
        date = datetime.strptime(mag_name[5:16], '%y%m%dt%H%M')
        start_lon = int(mag_name[22:25])

        CM_HEEQ = SkyCoord(0 * u.deg, 0 * u.deg,
                           frame=frames.HeliographicStonyhurst,
                           obstime=date, observer='earth')
        CM_CAR = CM_HEEQ.transform_to(frames.HeliographicCarrington(observer='earth', obstime=date))
        CM_CAR_value = CM_CAR.lon.value % 360

        if CM_CAR_value > start_lon:
            angle = CM_CAR_value - start_lon 
        else:
            angle = CM_CAR_value + (360 - start_lon) + 0
        return angle % 360 , date

    # ADAPT
    elif prefix == 'adapt':
        mode = mag_name[5:8]
        date = datetime.strptime(mag_name[18:30], '%Y%m%d%H%M')

        if mode == '403':
            logger.info("The magnetogram is GONG ADAPT in CAR frame")
            CM_HEEQ = SkyCoord(0 * u.deg, 0 * u.deg,
                               frame=frames.HeliographicStonyhurst,
                               obstime=date, observer='earth')
            CM_CAR = CM_HEEQ.transform_to(frames.HeliographicCarrington(observer='earth', obstime=date))
            CM_CAR_value = CM_CAR.lon.value % 360
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

        CM_HEEQ = SkyCoord(0 * u.deg, 0 * u.deg,
                           frame=frames.HeliographicStonyhurst,
                           obstime=date, observer='earth')
        CM_CAR = CM_HEEQ.transform_to(frames.HeliographicCarrington(observer='earth', obstime=date))
        CM_CAR_value = CM_CAR.lon.value % 360
        return (CM_CAR_value + 0) % 360, date
    else:
        logger.error("Magnetogram filename format not recognized: %s", mag_name)
        raise ValueError("Magnetogram filename format not recognized.")

if __name__ == "__main__":
    base_dir = r"C:/Users/luisl/Documents/Travail/processing_scripts"

    test_cases = [
        {
            "desc": "HMI test",
            "path": os.path.join(base_dir, "hmi.Synoptic_Mr_small.2238.fits"),
            "time": "2020-12-07T15:00:00"
        },
        {
            "desc": "GONG test",
            "path": os.path.join(base_dir, "mrzqs220311t1214c2255_247.fits"),
            "time": None
        },
        {
            "desc": "ADAPT test",
            "path": os.path.join(base_dir, "adapt40311_03k012_202308150000_i00025600n1.fts.gz"),
            "time": None
        }
    ]

    for test in test_cases:
        logger.info(f"\n====== {test['desc']} ======")
        angle, date = compute_rotation_angle(test["path"], date_hmi=test["time"])
        logger.info(f"Rotation angle: {angle:.2f} degrees at {date.isoformat()}")

