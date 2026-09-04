"""Coordinate helpers shared by magnetogram readers and diagnostics.

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
    if (
        "sine" in unit
        or "sin lat" in unit
        or "sinlat" in unit
        or unit.startswith("sin(")
    ):
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


def build_theta_phi(theta, phi):
    """Build two-dimensional colatitude and longitude meshgrids."""
    return np.tile(theta, (len(phi), 1)).T, np.tile(phi, (len(theta), 1))


def _validate_theta_axis(theta: np.ndarray) -> None:
    """Validate north-to-south physical colatitude centers."""
    theta = np.asarray(theta, dtype=float)
    if theta.ndim != 1 or theta.size == 0:
        raise ValueError("theta must be a non-empty 1D array.")
    if not np.all(np.isfinite(theta)):
        raise ValueError("theta contains non-finite values.")
    tolerance = 1.0e-12
    if np.any(theta < -tolerance) or np.any(theta > np.pi + tolerance):
        raise ValueError("theta centers must lie in the physical [0, pi] domain.")
    if theta.size > 1 and not np.all(np.diff(theta) > 0.0):
        raise ValueError("theta centers must be strictly increasing north-to-south.")


def _spacing_error(values: np.ndarray) -> float:
    """Return a dimensionless measure of departure from uniform spacing."""
    differences = np.diff(np.asarray(values, dtype=float))
    if differences.size <= 1:
        return 0.0
    reference = float(np.median(differences))
    scale = max(abs(reference), np.finfo(float).eps)
    return float(np.max(np.abs(differences - reference)) / scale)


def _theta_coordinate(theta: np.ndarray) -> str:
    """Infer whether centers are regular in latitude or in sine latitude."""
    theta = np.asarray(theta, dtype=float)
    if theta.size < 3:
        return "latitude"
    theta_error = _spacing_error(theta)
    sine_error = _spacing_error(np.cos(theta))
    uniform_tolerance = 1.0e-7
    if sine_error <= uniform_tolerance and theta_error > uniform_tolerance:
        return "sine"
    return "latitude"


def theta_cell_edges(
    theta: np.ndarray,
    coordinate: str | None = None,
) -> np.ndarray:
    """Construct physical cell edges from colatitude centers.

    Uniform sine-latitude grids are extrapolated in ``mu=cos(theta)``;
    uniform-latitude grids are extrapolated in ``theta``. Extrapolated edges
    are clipped to the physical poles, without manufacturing pole centers.
    """
    theta = np.asarray(theta, dtype=float)
    _validate_theta_axis(theta)
    if coordinate is None:
        coordinate = _theta_coordinate(theta)
    if coordinate not in {"sine", "latitude"}:
        raise ValueError("coordinate must be 'sine' or 'latitude'.")
    if theta.size == 1:
        return np.array([0.0, np.pi])

    if coordinate == "sine":
        mu = np.cos(theta)
        mu_edges = np.empty(mu.size + 1, dtype=float)
        mu_edges[1:-1] = 0.5 * (mu[:-1] + mu[1:])
        mu_edges[0] = mu[0] + 0.5 * (mu[0] - mu[1])
        mu_edges[-1] = mu[-1] + 0.5 * (mu[-1] - mu[-2])
        edges = np.arccos(np.clip(mu_edges, -1.0, 1.0))
    else:
        edges = np.empty(theta.size + 1, dtype=float)
        edges[1:-1] = 0.5 * (theta[:-1] + theta[1:])
        edges[0] = theta[0] - 0.5 * (theta[1] - theta[0])
        edges[-1] = theta[-1] + 0.5 * (theta[-1] - theta[-2])
        edges = np.clip(edges, 0.0, np.pi)

    if not np.all(np.diff(edges) >= 0.0):
        raise ValueError("Computed theta cell edges are not monotonic.")
    return edges


def longitude_cell_widths(phi: np.ndarray) -> np.ndarray:
    """Return periodic longitude-cell widths whose sum is exactly ``2*pi``."""
    phi = np.asarray(phi, dtype=float)
    if phi.ndim != 1 or phi.size == 0 or not np.all(np.isfinite(phi)):
        raise ValueError("phi must be a non-empty finite 1D array.")
    if phi.size == 1:
        return np.array([2.0 * np.pi])
    phi_unwrapped = np.unwrap(phi)
    differences = np.diff(phi_unwrapped)
    if not np.all(differences > 0.0):
        raise ValueError("phi centers must be strictly increasing.")

    span = phi_unwrapped[-1] - phi_unwrapped[0]
    if np.isclose(span, 2.0 * np.pi, atol=1.0e-10, rtol=0.0):
        widths = np.zeros(phi.size, dtype=float)
        widths[:-1] = longitude_cell_widths(phi_unwrapped[:-1])
        return widths
    wrap_gap = 2.0 * np.pi - span
    if wrap_gap <= 0.0:
        raise ValueError("phi centers span more than one periodic revolution.")
    gaps = np.concatenate((differences, [wrap_gap]))
    return 0.5 * (np.roll(gaps, 1) + gaps)


def spherical_pixel_areas(
    theta: np.ndarray,
    phi: np.ndarray,
    shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Compute exact solid angles from latitude and longitude cell edges."""
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    if shape is not None and shape != (theta.size, phi.size):
        raise ValueError("theta and phi lengths must match the Br dimensions.")
    theta_edges = theta_cell_edges(theta)
    latitude_weights = np.abs(
        np.cos(theta_edges[:-1]) - np.cos(theta_edges[1:])
    )
    return latitude_weights[:, None] * longitude_cell_widths(phi)[None, :]
