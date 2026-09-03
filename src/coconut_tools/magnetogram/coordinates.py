"""Coordinate helpers shared by the magnetogram readers and diagnostics.

The latitude values stored in a synoptic FITS header are not always expressed
in the same native coordinate.  Some products store latitude itself, others
store sine latitude, and standards-compliant CEA WCS headers store a projected
angular coordinate that must be inverted by WCSLIB.  This module keeps that
interpretation in one place.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from astropy.wcs import WCS


_ANGLE_DEG_UNITS = {"deg", "degree", "degrees"}
_ANGLE_RAD_UNITS = {"rad", "radian", "radians"}


def _header_float(header: Mapping[str, Any], key: str, default: float) -> float:
    """Return a finite floating-point FITS keyword value."""
    try:
        value = float(header.get(key, default))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {key} value in FITS latitude metadata.") from exc
    if not np.isfinite(value):
        raise ValueError(f"Non-finite {key} value in FITS latitude metadata.")
    return value


def _latitude_mode(ctype2: str, cunit2: str, lattype: Any) -> str:
    """Classify the native latitude coordinate described by a FITS header."""
    unit = cunit2.replace("_", " ").replace("-", " ").strip().lower()

    # Explicit sine-latitude declarations take precedence over CRLT-CEA text.
    if "sine" in unit or "sin lat" in unit or "sinlat" in unit:
        return "sine"
    if "CSLT" in ctype2:
        return "sine"

    normalized_lattype = str(lattype).strip().lower()
    if normalized_lattype in {"1", "sin", "sine", "sinlat", "sine latitude"}:
        return "sine"

    if "CEA" in ctype2:
        if unit in _ANGLE_DEG_UNITS or unit in _ANGLE_RAD_UNITS:
            return "cea_wcs"
        # GONG-style CEA maps commonly omit CUNIT2 and store mu=sin(latitude)
        # directly.  The range is checked after constructing the raw axis.
        if not unit:
            return "cea_sine"
        raise ValueError(
            f"Unsupported CEA latitude unit {cunit2!r}; expected an angular "
            "unit or an explicit sine-latitude unit."
        )

    if normalized_lattype in {"0", "lat", "latitude", "linear"}:
        return "latitude"
    if unit in _ANGLE_DEG_UNITS or unit in _ANGLE_RAD_UNITS:
        return "latitude"
    if "CRLT" in ctype2 or "HGLT" in ctype2 or "LAT" in ctype2:
        return "latitude"

    raise KeyError(
        "FITS header does not identify axis 2 as latitude, sine latitude, or CEA."
    )


def _cea_latitude_degrees(header: Mapping[str, Any], ny: int) -> np.ndarray:
    """Invert a standards-compliant angular CEA WCS latitude axis."""
    try:
        celestial_wcs = WCS(header).celestial
        if celestial_wcs.pixel_n_dim != 2 or celestial_wcs.world_n_dim != 2:
            raise ValueError("CEA WCS does not contain two celestial axes.")
        x_reference = _header_float(header, "CRPIX1", 1.0) - 1.0
        x_pixels = np.full(ny, x_reference, dtype=float)
        y_pixels = np.arange(ny, dtype=float)
        _, latitude = celestial_wcs.all_pix2world(x_pixels, y_pixels, 0)
    except Exception as exc:
        raise ValueError("Could not invert the FITS CEA latitude WCS.") from exc
    return np.asarray(latitude, dtype=float)


def validate_latitude_degrees(latitude: np.ndarray) -> None:
    """Reject non-physical or ambiguous latitude-center axes."""
    latitude = np.asarray(latitude, dtype=float)
    if latitude.ndim != 1 or latitude.size == 0:
        raise ValueError("FITS latitude axis must be a non-empty 1D array.")
    if not np.all(np.isfinite(latitude)):
        raise ValueError("FITS latitude axis contains non-finite values.")
    tolerance = 1.0e-8
    if np.any(latitude < -90.0 - tolerance) or np.any(latitude > 90.0 + tolerance):
        raise ValueError("FITS latitude centers lie outside the physical [-90, 90] range.")
    if latitude.size > 1:
        differences = np.diff(latitude)
        if not (np.all(differences > 0.0) or np.all(differences < 0.0)):
            raise ValueError("FITS latitude centers must be strictly monotonic.")


def fits_latitude_axis(
    header: Mapping[str, Any],
    data: np.ndarray | np.ma.MaskedArray | None = None,
    *,
    output: str = "deg",
    one_based_fits: bool = True,
    force_degrees: bool = False,
    force_sine: bool = False,
) -> tuple[np.ndarray, np.ndarray | np.ma.MaskedArray | None, dict[str, Any]]:
    """Decode physical FITS latitude centers and order them south-to-north.

    ``CUNIT2`` values explicitly describing sine latitude and ``CTYPE2=CSLT``
    have priority.  Angular ``*-CEA`` axes are inverted through Astropy/WCSLIB.
    Linear latitude axes use the usual CRPIX/CRVAL/CDELT relation.

    The returned data, when supplied, is flipped on its penultimate axis exactly
    when required to match the returned increasing-latitude coordinate.
    """
    if force_degrees and force_sine:
        raise ValueError("Choose at most one of force_degrees or force_sine.")
    output = output.lower()
    if output not in {"deg", "sin"}:
        raise ValueError('output must be "deg" or "sin".')
    if "NAXIS2" not in header:
        raise KeyError("NAXIS2 missing from FITS header.")
    missing_reference = [
        key
        for key in ("CRPIX2", "CRVAL2")
        if key not in header or header[key] in (None, "")
    ]
    if missing_reference:
        raise KeyError(
            "Missing FITS latitude reference keyword(s): "
            + ", ".join(missing_reference)
        )

    ny = int(header["NAXIS2"])
    if ny < 1:
        raise ValueError("NAXIS2 must be positive for a FITS latitude axis.")
    crpix2 = _header_float(header, "CRPIX2", 1.0)
    crval2 = _header_float(header, "CRVAL2", 0.0)
    if "CDELT2" in header and header["CDELT2"] not in (None, ""):
        cdelt2 = _header_float(header, "CDELT2", 0.0)
    elif "CD2_2" in header and header["CD2_2"] not in (None, ""):
        cdelt2 = _header_float(header, "CD2_2", 0.0)
    else:
        raise KeyError("Missing CDELT2 (or CD2_2) in FITS header.")
    if np.isclose(cdelt2, 0.0):
        raise ValueError("FITS latitude increment must be non-zero.")

    ctype2 = str(header.get("CTYPE2", "")).strip().upper()
    cunit2 = str(header.get("CUNIT2", "")).strip()
    lattype = header.get("LATTYPE", "")

    if force_sine:
        mode = "sine"
    elif force_degrees:
        mode = "latitude"
    else:
        mode = _latitude_mode(ctype2, cunit2, lattype)

    pixel_origin = 1.0 if one_based_fits else 0.0
    pixels = np.arange(ny, dtype=float) + pixel_origin
    cdelt2_used = cdelt2
    crval2_used = crval2
    scale_note = "native"
    unit = cunit2.strip().lower()
    if mode == "latitude" and unit not in _ANGLE_DEG_UNITS | _ANGLE_RAD_UNITS:
        if abs(cdelt2) > 1000.0:
            cdelt2_used = cdelt2 / 1.0e6
            crval2_used = crval2 / 1.0e6
            scale_note = "microdeg_from_cdelt2"

    raw_axis = (pixels - crpix2) * cdelt2_used + crval2_used
    if mode in {"sine", "cea_sine"}:
        if np.any(raw_axis < -1.0 - 1.0e-8) or np.any(raw_axis > 1.0 + 1.0e-8):
            raise ValueError("Sine-latitude FITS centers lie outside [-1, 1].")
        latitude_deg = np.degrees(np.arcsin(np.clip(raw_axis, -1.0, 1.0)))
        detected_mode = "sine_to_degrees"
    elif mode == "cea_wcs":
        latitude_deg = _cea_latitude_degrees(header, ny)
        detected_mode = "cea_wcs_to_degrees"
    else:
        latitude_deg = raw_axis.astype(float)
        if unit in _ANGLE_RAD_UNITS:
            latitude_deg = np.degrees(latitude_deg)
        detected_mode = "degrees"

    validate_latitude_degrees(latitude_deg)
    flipped = bool(latitude_deg.size > 1 and latitude_deg[0] > latitude_deg[-1])
    if flipped:
        latitude_deg = latitude_deg[::-1]
        data_out = None if data is None else data[..., ::-1, :]
    else:
        data_out = data

    latitude_out = (
        latitude_deg
        if output == "deg"
        else np.sin(np.radians(latitude_deg))
    )
    meta = {
        "ny": ny,
        "ctype2": ctype2,
        "cunit2": cunit2,
        "lattype": lattype,
        "detected_mode": detected_mode,
        "native_coordinate": "sine" if mode in {"sine", "cea_sine", "cea_wcs"} else "latitude",
        "flipped": flipped,
        "crpix2": crpix2,
        "cdelt2_used": float(cdelt2_used),
        "crval2_used": float(crval2_used),
        "scale_note": scale_note,
    }
    return latitude_out, data_out, meta
