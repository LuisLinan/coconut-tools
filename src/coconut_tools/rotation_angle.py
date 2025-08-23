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
logger = setup_logger(__name__)

def compute_rotation_angle(mag_name_path: str, date_hmi: str = None) -> float:
    """Compute the rotation angle from magnetogram filename.

    Supports GONG, ADAPT (CM and CAR), and HMI.

    Args:
        mag_name_path (str): Path to the magnetogram file.
        date_hmi (str, optional): ISO datetime string used for HMI (e.g. '2024-07-01T00:00:00')

    Returns:
        float: Rotation angle in degrees to convert from Carrington to Stonyhurst
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
            angle = CM_CAR_value - start_lon + 10
        else:
            angle = CM_CAR_value + (360 - start_lon) + 10
        return angle % 360

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
            return (CM_CAR_value + 10) % 360

        elif mode == '413':
            logger.info("The magnetogram is GONG ADAPT in CM frame")
            return 190.0

    # HMI
    elif 'hmi' in prefix:
        logger.info("The magnetogram is HMI in CAR frame")
        if not date_hmi:
            raise ValueError("You must provide 'date_hmi' for HMI magnetograms.")
        date = datetime.strptime(date_hmi, '%Y-%m-%dT%H:%M:%S')

        CM_HEEQ = SkyCoord(0 * u.deg, 0 * u.deg,
                           frame=frames.HeliographicStonyhurst,
                           obstime=date, observer='earth')
        CM_CAR = CM_HEEQ.transform_to(frames.HeliographicCarrington(observer='earth', obstime=date))
        CM_CAR_value = CM_CAR.lon.value % 360
        return (CM_CAR_value + 10) % 360
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
        angle = compute_rotation_angle(test["path"], date_hmi=test["time"])
        logger.info(f"Rotation angle: {angle:.2f} degrees")
