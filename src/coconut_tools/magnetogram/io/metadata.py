"""FITS metadata decoding shared by custom magnetogram workflows."""

from dataclasses import dataclass
from datetime import datetime, timezone
import re
import warnings

import numpy as np
from astropy.io import fits
from astropy.time import Time


_FITS_TIME_KEYS = ("T_REC", "MAPTIME", "T_OBS", "DATE-OBS", "DATE_OBS")
_ASTROPY_TIME_SCALES = {"tai", "utc", "tt", "tdb", "tcg", "tcb", "ut1"}
_DATETIME_PATTERN = re.compile(
    r"^\s*(?P<year>\d{4})[.-](?P<month>\d{2})[.-](?P<day>\d{2})"
    r"[T_ ](?P<clock>\d{2}:\d{2}:\d{2}(?:\.\d+)?)"
    r"(?:_(?P<scale>[A-Za-z0-9]+))?"
    r"(?P<zone>Z|[+-]\d{2}:\d{2})?\s*$"
)


@dataclass(frozen=True)
class FitsEffectiveTime:
    """UTC time selected from one FITS keyword."""

    value: datetime
    keyword: str
    raw_value: str | float
    source_scale: str


@dataclass(frozen=True)
class FitsLongitudeAxis:
    """Normalized longitude geometry decoded from a custom FITS image.

    ``centers_degrees`` follows increasing longitude and starts at the
    smallest wrapped center. ``flip_columns`` and ``roll_columns`` describe
    the operations required to put the image data in that same order.
    ``frame`` is ``"carrington"``, ``"stonyhurst"``, or ``"unknown"``.
    """

    centers_degrees: np.ndarray
    flip_columns: bool
    roll_columns: int
    frame: str
    frame_source: str
    ctype1: str


@dataclass(frozen=True)
class FitsCarringtonCentralMeridian:
    """Carrington longitude of the Stonyhurst central meridian."""

    value_degrees: float
    keyword: str
    assumed_earth_observer: bool = False


def _first_image_header(file_path: str) -> fits.Header:
    """Return a detached header from the first FITS image HDU."""
    with fits.open(file_path) as hdul:
        for hdu in hdul:
            if hdu.data is not None and hdu.data.ndim >= 2:
                return hdu.header.copy()
    raise ValueError(f"No magnetogram image HDU found in FITS file: {file_path}")


def _first_image_shape_and_header(
    file_path: str,
) -> tuple[tuple[int, ...], fits.Header]:
    """Return the first image shape and a detached FITS header."""
    with fits.open(file_path) as hdul:
        for hdu in hdul:
            if hdu.data is not None and hdu.data.ndim >= 2:
                return tuple(hdu.data.shape), hdu.header.copy()
    raise ValueError(f"No magnetogram image HDU found in FITS file: {file_path}")


def infer_known_fits_map_type(file_path: str) -> str | None:
    """Identify a known product exclusively from FITS metadata.

    GONG needs this compatibility path because its historical FITS files omit
    angular CUNIT keywords and use a product-specific longitude origin.
    Original JSOC HMI synoptic maps also need their documented longitude
    convention: a negative CDELT1 does not mean that the stored magnetic-field
    columns must be reversed.  Updated/daily synoptic maps are routed through
    the dynamic HMI reader so their Carrington-zero roll is preserved; complete
    rotation charts use the static HMI convention.
    Filename content is deliberately ignored so renaming a FITS cannot alter
    its physical interpretation.
    """
    try:
        header = _first_image_header(file_path)
    except (OSError, ValueError):
        return None
    identity = " ".join(
        str(header.get(keyword, ""))
        for keyword in ("ORIGIN", "OBS-SITE", "TELESCOP", "INSTRUME")
    ).upper()
    if "GONG" in identity:
        return "GONG"

    model = str(header.get("MODEL", "")).strip().upper()
    map_data = str(header.get("MAPDATA", "")).strip().upper()
    try:
        is_adapt_ensemble = (
            model == "ADAPT"
            and int(header.get("NAXIS", 0)) == 3
            and int(header.get("NREAL", header.get("NAXIS3", 0))) > 0
        )
    except (TypeError, ValueError):
        is_adapt_ensemble = False
    if is_adapt_ensemble:
        return "HMI_fdt" if "HMI" in map_data else "ADAPT"

    origin = str(header.get("ORIGIN", "")).strip().upper()
    telescope = str(header.get("TELESCOP", "")).strip().upper()
    instrument = str(header.get("INSTRUME", "")).strip().upper()
    content = str(header.get("CONTENT", "")).strip().upper()
    ctype1 = str(header.get("CTYPE1", "")).strip().upper()
    ctype2 = str(header.get("CTYPE2", "")).strip().upper()

    is_original_hmi_synoptic = (
        ("HMI" in telescope or "HMI" in instrument)
        and ctype1.startswith("CRLN-")
        and ctype2.startswith("CRLT-")
        and "SYNOPTIC" in content
    )
    if is_original_hmi_synoptic:
        is_dynamic_synoptic_map = any(
            marker in content for marker in ("UPDATE", "DAILY", "SYNFRAME")
        )
        if is_dynamic_synoptic_map:
            return "HMI_hourly"
        if str(header.get("METHOD", "")).strip():
            return "HMI_polfil"
        return "HMI_small"
    return None


def _longitude_frame(header: fits.Header) -> tuple[str, str]:
    """Identify the longitude reference frame without interpreting projection.

    In solar FITS WCS, ``CRLN`` and ``HGLN`` identify Carrington and
    Stonyhurst longitude respectively.  The three-letter projection suffix
    (for example ``-CAR`` or ``-CEA``) is deliberately ignored: ``CAR`` means
    plate carree in that position and is not a Carrington-frame declaration.
    """
    ctype1 = str(header.get("CTYPE1", "")).strip().upper()
    axis_code = ctype1.split("-", 1)[0].strip()
    if "CRLN" in axis_code or "CARRINGTON" in ctype1:
        return "carrington", "CTYPE1"
    if "HGLN" in axis_code or "STONYHURST" in ctype1:
        return "stonyhurst", "CTYPE1"

    # ADAPT documents LNGTYPE=0 as its fixed Carrington-longitude product.
    # It is useful for a local ADAPT FITS whose CTYPE1 is otherwise generic.
    raw_lngtype = header.get("LNGTYPE")
    if raw_lngtype not in (None, ""):
        try:
            if int(raw_lngtype) == 0:
                return "carrington", "LNGTYPE=0"
        except (TypeError, ValueError):
            pass

    for keyword in ("WCSNAME", "COORDSYS", "COORDTYPE"):
        value = str(header.get(keyword, "")).strip().upper()
        if "CARRINGTON" in value:
            return "carrington", keyword
        if "STONYHURST" in value:
            return "stonyhurst", keyword
    return "unknown", ""


def read_fits_longitude_axis(
    file_path: str,
    width: int | None = None,
) -> FitsLongitudeAxis:
    """Decode and normalize a separable full-sphere FITS longitude axis.

    The returned centers are in degrees.  Native decreasing axes are reversed,
    then the complete periodic grid is rolled so its smallest wrapped center
    is first.  Reference-frame identification is kept separate from this
    geometric normalization.
    """
    image_shape, header = _first_image_shape_and_header(file_path)
    if len(image_shape) != 2:
        raise ValueError("A custom magnetogram must contain one 2D FITS image.")
    image_width = int(image_shape[-1])
    width = image_width if width is None else int(width)
    if image_width != width:
        raise ValueError("The requested width does not match the FITS image.")
    if int(header.get("NAXIS1", width)) != width:
        raise ValueError(
            f"FITS NAXIS1 does not match the image longitude dimension: {file_path}"
        )

    ctype1 = str(header.get("CTYPE1", "")).strip().upper()
    if not any(token in ctype1 for token in ("CRLN", "HGLN", "LON")):
        raise ValueError(
            "A custom magnetogram requires CTYPE1 to identify a longitude axis."
        )

    cunit1 = str(header.get("CUNIT1", "")).strip().lower()
    degree_units = {"deg", "degree", "degrees"}
    radian_units = {"rad", "radian", "radians"}
    if cunit1 and cunit1 not in degree_units | radian_units:
        raise ValueError(
            "A custom magnetogram requires an angular CUNIT1 in degrees or radians."
        )

    try:
        crpix1 = float(header["CRPIX1"])
        crval1 = float(header["CRVAL1"])
        if "CD1_1" in header and header["CD1_1"] not in (None, ""):
            longitude_step = float(header["CD1_1"])
        else:
            longitude_step = float(header["CDELT1"]) * float(
                header.get("PC1_1", 1.0)
            )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "A custom magnetogram requires finite CRPIX1, CRVAL1, and "
            "CDELT1 (or CD1_1) longitude metadata."
        ) from exc

    try:
        has_cross_term = any(
            not np.isclose(float(header.get(keyword, 0.0)), 0.0)
            for keyword in ("CD1_2", "PC1_2")
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Invalid custom FITS longitude transformation matrix."
        ) from exc
    if has_cross_term:
        raise ValueError(
            "A custom magnetogram requires a separable longitude axis; "
            "nonzero CD1_2/PC1_2 terms are not supported."
        )

    values = np.array([crpix1, crval1, longitude_step], dtype=float)
    if not np.all(np.isfinite(values)) or np.isclose(longitude_step, 0.0):
        raise ValueError(
            "Custom FITS longitude metadata must be finite and nonzero."
        )

    if not cunit1:
        raw_coverage = abs(longitude_step) * width
        if "CRLN" in ctype1 or "HGLN" in ctype1 or np.isclose(
            raw_coverage,
            360.0,
            atol=1.0e-6,
            rtol=1.0e-9,
        ):
            cunit1 = "deg"
        elif np.isclose(
            raw_coverage,
            2.0 * np.pi,
            atol=1.0e-9,
            rtol=1.0e-9,
        ):
            cunit1 = "rad"
        else:
            raise ValueError(
                "Custom FITS CUNIT1 is missing and its longitude unit cannot "
                "be inferred from a complete 360-degree or 2*pi-radian axis."
            )
        warnings.warn(
            f"FITS CUNIT1 is missing; inferred {cunit1!r} from CTYPE1 and "
            "the full-sphere longitude coverage.",
            RuntimeWarning,
            stacklevel=2,
        )

    pixel = np.arange(1, width + 1, dtype=float)
    longitude = crval1 + (pixel - crpix1) * longitude_step
    if cunit1 in radian_units:
        longitude = np.degrees(longitude)
        longitude_step = np.degrees(longitude_step)

    flip_columns = longitude_step < 0.0
    if flip_columns:
        longitude = longitude[::-1]

    differences = np.diff(longitude)
    if differences.size and (
        not np.all(differences > 0.0)
        or not np.allclose(
            differences,
            np.median(differences),
            atol=1.0e-9,
            rtol=1.0e-9,
        )
    ):
        raise ValueError(
            "The custom FITS longitude centers must form a regular increasing axis."
        )

    step_degrees = abs(float(longitude_step))
    coverage = step_degrees * width
    if not np.isclose(coverage, 360.0, atol=1.0e-6, rtol=1.0e-9):
        raise ValueError(
            "A custom magnetogram must cover one complete 360-degree longitude "
            f"grid; header coverage is {coverage:.12g} degrees."
        )

    wrapped = np.mod(longitude, 360.0)
    wrap_tolerance = max(1.0e-9, step_degrees * 1.0e-8)
    wrapped[np.isclose(wrapped, 360.0, atol=wrap_tolerance, rtol=0.0)] = 0.0
    wrapped[np.isclose(wrapped, 0.0, atol=wrap_tolerance, rtol=0.0)] = 0.0
    zero_column = int(np.argmin(wrapped))
    centers = np.degrees(np.unwrap(np.radians(np.roll(wrapped, -zero_column))))
    if centers.size > 1 and not np.all(np.diff(centers) > 0.0):
        raise ValueError("Could not normalize the custom longitude axis monotonically.")

    frame, frame_source = _longitude_frame(header)
    return FitsLongitudeAxis(
        centers_degrees=centers,
        flip_columns=flip_columns,
        roll_columns=-zero_column,
        frame=frame,
        frame_source=frame_source,
        ctype1=ctype1,
    )


def read_fits_carrington_central_meridian(
    file_path: str,
) -> FitsCarringtonCentralMeridian | None:
    """Read the Carrington longitude of Stonyhurst zero, when declared.

    ``L0`` directly describes the Earth-viewed central meridian. When observer
    coordinates are supplied instead, the frame relation is
    ``L0 = CRLN_OBS - HGLN_OBS``. Solar FITS permits an Earth/low-Earth-orbit
    observer to be assumed when ``HGLN_OBS`` is absent. Values are angular
    degrees by convention and are normalized periodically.
    """
    header = _first_image_header(file_path)
    raw_l0 = header.get("L0")
    if raw_l0 not in (None, ""):
        try:
            value = float(raw_l0)
        except (TypeError, ValueError):
            value = np.nan
        if np.isfinite(value):
            return FitsCarringtonCentralMeridian(
                value_degrees=value % 360.0,
                keyword="L0",
            )

    for keyword in ("CRLN_OBS", "CRLN-OBS"):
        raw_value = header.get(keyword)
        if raw_value in (None, ""):
            continue
        try:
            carrington_observer = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid custom FITS {keyword} value.") from exc
        if not np.isfinite(carrington_observer):
            raise ValueError(f"Invalid custom FITS {keyword} value.")

        raw_stonyhurst_observer = header.get("HGLN_OBS")
        assumed_earth = raw_stonyhurst_observer in (None, "")
        if assumed_earth:
            stonyhurst_observer = 0.0
            source = f"{keyword} (HGLN_OBS assumed 0)"
        else:
            try:
                stonyhurst_observer = float(raw_stonyhurst_observer)
            except (TypeError, ValueError) as exc:
                raise ValueError("Invalid custom FITS HGLN_OBS value.") from exc
            if not np.isfinite(stonyhurst_observer):
                raise ValueError("Invalid custom FITS HGLN_OBS value.")
            source = f"{keyword}-HGLN_OBS"
        return FitsCarringtonCentralMeridian(
            value_degrees=(carrington_observer - stonyhurst_observer) % 360.0,
            keyword=source,
            assumed_earth_observer=assumed_earth,
        )
    return None


def _parse_fits_datetime(
    raw_value,
    *,
    default_scale: str = "utc",
) -> tuple[datetime, str]:
    """Parse a FITS/JSOC datetime and return a naive UTC datetime."""
    text = str(raw_value).strip()
    match = _DATETIME_PATTERN.match(text)
    if match is None:
        raise ValueError(f"Unsupported FITS datetime value: {raw_value!r}")

    date_part = "-".join(
        (match.group("year"), match.group("month"), match.group("day"))
    )
    iso_value = f"{date_part}T{match.group('clock')}"
    zone = match.group("zone")
    explicit_scale = match.group("scale")

    if zone is not None:
        if explicit_scale is not None and explicit_scale.casefold() != "utc":
            raise ValueError(
                "A FITS datetime cannot combine a non-UTC scale suffix with "
                "an ISO timezone offset."
            )
        zone_value = "+00:00" if zone == "Z" else zone
        parsed = datetime.fromisoformat(iso_value + zone_value)
        return parsed.astimezone(timezone.utc).replace(tzinfo=None), "utc"

    scale = (explicit_scale or default_scale or "utc").casefold()
    if scale not in _ASTROPY_TIME_SCALES:
        raise ValueError(f"Unsupported FITS time scale: {scale!r}")
    parsed = Time(iso_value, format="isot", scale=scale)
    return parsed.utc.to_datetime(timezone=None), scale


def _combine_fits_date_and_time(date_value, time_value) -> str:
    """Combine separate legacy FITS date and clock values."""
    date_match = re.match(
        r"^\s*(\d{4})[.-](\d{2})[.-](\d{2})",
        str(date_value),
    )
    if date_match is None:
        raise ValueError("The FITS date value has no recognizable calendar date.")
    date_text = "-".join(date_match.groups())
    time_text = str(time_value).strip()
    if not date_text or not time_text:
        raise ValueError("Both date and time values are required.")
    if len(time_text.split(":")) == 2:
        time_text += ":00"
    return f"{date_text}T{time_text}"


def read_fits_effective_time(file_path: str) -> FitsEffectiveTime:
    """Read the best effective-map time available in a FITS header.

    Priority is given to the nominal record/map time (``T_REC`` or
    ``MAPTIME``), then to the representative observation time (``T_OBS``),
    and finally to FITS observation-date or MJD metadata. Explicit time-scale
    suffixes take precedence over ``TIMESYS``. Returned datetimes are UTC.
    """
    header = _first_image_header(file_path)
    default_scale = str(header.get("TIMESYS", "UTC")).strip().casefold() or "utc"
    errors = []

    for keyword in _FITS_TIME_KEYS:
        raw_value = header.get(keyword)
        if raw_value in (None, ""):
            continue
        parse_value = raw_value
        if keyword == "MAPTIME" and not re.search(r"\d{4}[.-]\d{2}[.-]\d{2}", str(raw_value)):
            map_date = header.get("MAPDATE", header.get("DATE-OBS"))
            try:
                parse_value = _combine_fits_date_and_time(map_date, raw_value)
            except ValueError:
                pass
        elif keyword in {"DATE-OBS", "DATE_OBS"} and not re.search(
            r"[T_ ]\d{2}:\d{2}",
            str(raw_value),
        ):
            time_keyword = "TIME-OBS" if keyword == "DATE-OBS" else "TIME_OBS"
            time_value = header.get(time_keyword)
            if time_value not in (None, ""):
                parse_value = _combine_fits_date_and_time(raw_value, time_value)
        try:
            value, source_scale = _parse_fits_datetime(
                parse_value,
                default_scale=default_scale,
            )
        except ValueError as exc:
            errors.append(f"{keyword}={raw_value!r}: {exc}")
            continue
        return FitsEffectiveTime(
            value=value,
            keyword=keyword,
            raw_value=raw_value,
            source_scale=source_scale,
        )

    mjd_value = header.get("MJD-OBS", header.get("MJD_OBS"))
    if mjd_value not in (None, ""):
        try:
            source_scale = default_scale if default_scale in _ASTROPY_TIME_SCALES else "utc"
            value = Time(float(mjd_value), format="mjd", scale=source_scale)
            return FitsEffectiveTime(
                value=value.utc.to_datetime(timezone=None),
                keyword="MJD-OBS",
                raw_value=mjd_value,
                source_scale=source_scale,
            )
        except (TypeError, ValueError) as exc:
            errors.append(f"MJD-OBS={mjd_value!r}: {exc}")

    detail = "; ".join(errors) if errors else "no supported time keyword is present"
    raise ValueError(f"Could not determine the custom FITS effective time: {detail}.")
