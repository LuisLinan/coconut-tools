from sunpy.coordinates import sun
from astropy.time import Time


def CarringtonToEarthLong(date_str):
    """
    Return the longitude offset (degrees) to add to a Carrington longitude
    to obtain a Stonyhurst (Earth-referenced) longitude.

    Input format:
        YYYYMMDDHHMMSS

    Example:
        "20250101000000"
    """
    t = Time.strptime(date_str, "%Y%m%d%H%M%S")
    L0 = sun.L0(t).deg

    # Stonyhurst = Carrington - L0
    return -L0


def EarthToCarringtonLong(date_str):
    """
    Return the longitude offset (degrees) to add to a Stonyhurst longitude
    to obtain a Carrington longitude.

    Input format:
        YYYYMMDDHHMMSS

    Example:
        "20250101000000"
    """
    t = Time.strptime(date_str, "%Y%m%d%H%M%S")
    L0 = sun.L0(t).deg

    # Carrington = Stonyhurst + L0
    return L0


def EarthObserverCarringtonLonLat(date_str):
    """
    Return the Earth observer position as Carrington longitude and latitude.

    Input format:
        YYYYMMDDHHMMSS

    The longitude is L0, the Carrington longitude of the central meridian as
    seen from Earth. The latitude is B0, the heliographic latitude of disk
    center as seen from Earth.
    """
    t = Time.strptime(date_str, "%Y%m%d%H%M%S")
    return sun.L0(t).deg, sun.B0(t).deg
