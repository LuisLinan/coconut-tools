"""Longitude normalization and Carrington-to-Stonyhurst rotation helpers."""

import os
from datetime import datetime

import numpy as np
from astropy.io import fits

from coconut_tools.magnetogram.io.downloads import (
    is_gong_map_type,
    is_gong_temporal_map_type,
    magnetogram_effective_date,
    normalize_map_type,
    parse_iso_datetime,
)
from coconut_tools.magnetogram.io.metadata import (
    infer_known_fits_map_type,
    read_fits_carrington_central_meridian,
    read_fits_effective_time,
    read_fits_longitude_axis,
)
from coconut_tools.tools.logger_config import setup_logger
from coconut_tools.tools.rotation_angle import (
    compute_carrington_central_meridian,
    compute_rotation_angle,
    increasing_longitude_axis,
    is_br_longitude_increasing,
)

logger = setup_logger(__name__)

_HMI_DYNAMIC_MAP_TYPES = {"HMI_SYNC", "HMI_hourly"}
TEMPORAL_LONGITUDE_TOLERANCE_PIXELS = 1.0e-3


def normalize_to_carrington(
    Br: np.ndarray,
    longitude: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sort physical Carrington centers and field columns without resampling.

    Longitudes are in degrees. The complete, uniform, endpoint-free grid is
    validated before returning its exact permutation in ``[0, 360)``.
    """
    if Br.ndim != 2:
        raise ValueError("Br must be a 2D array.")
    longitude = np.asarray(longitude, dtype=float)
    if longitude.ndim != 1 or longitude.size != Br.shape[1]:
        raise ValueError("Longitude centers must match the Br columns.")
    if not longitude.size or not np.all(np.isfinite(longitude)):
        raise ValueError("Longitude centers must be finite and non-empty.")
    wrapped = longitude % 360.0
    order = np.argsort(wrapped, kind="stable")
    centers = wrapped[order]
    gaps = np.diff(np.r_[centers, centers[0] + 360.0])
    if not np.allclose(gaps, 360.0 / centers.size, atol=1e-8, rtol=1e-9):
        raise ValueError("Longitude centers must cover one regular periodic grid.")
    return Br[:, order], centers


def read_carrington_map(
    Br: np.ndarray,
    file_path: str,
    map_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize native FITS columns to their physical Carrington centers."""
    map_type = normalize_map_type(map_type)
    if map_type == "custom":
        map_type = infer_known_fits_map_type(file_path) or map_type
    geometry = read_fits_longitude_axis(
        file_path, Br.shape[1], map_type=map_type, preserve_centers=True
    )
    if geometry.frame == "unknown":
        raise ValueError(
            "Cannot normalize an ambiguous longitude frame to Carrington; "
            "provide CRLN-* or HGLN-* metadata (or ADAPT LNGTYPE=0)."
        )
    if geometry.flip_columns:
        Br = Br[:, ::-1]
    Br = np.roll(Br, geometry.roll_columns, axis=1)
    longitude = geometry.centers_degrees
    if geometry.frame == "stonyhurst":
        central_meridian = read_fits_carrington_central_meridian(file_path)
        if central_meridian is not None:
            offset = central_meridian.value_degrees
        else:
            try:
                source_time = read_fits_effective_time(file_path).value
            except ValueError as exc:
                raise ValueError(
                    "Stonyhurst-to-Carrington conversion requires source L0 "
                    "metadata or a source observation date."
                ) from exc
            offset = compute_carrington_central_meridian(source_time)
        longitude = longitude + offset
    return normalize_to_carrington(Br, longitude)


def rotate_carrington_to_stonyhurst(
    Br: np.ndarray,
    longitude: np.ndarray,
    central_meridian: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll to the nearest central-meridian column and retain its residual.

    Both axes use degrees. Output centers are wrapped into ``[0, 360)``;
    the first center may precede the periodic seam, e.g. 359.7, 0.7, 1.7.
    """
    if not np.isfinite(central_meridian):
        raise ValueError("The Carrington central meridian must be finite.")
    column, _ = closest_longitude_column(longitude, central_meridian)
    return (
        np.roll(Br, -column, axis=1),
        (np.roll(longitude, -column) - central_meridian) % 360.0,
    )


def validate_matching_longitude_axes(
    axes: list[np.ndarray],
    paths: list[str],
) -> list[int]:
    """Return seam-alignment rolls for grids matching within 0.001 pixel.

    Negligible center offsets are accepted against the first map's grid, with
    no subpixel resampling. A center just below zero can sort into the last
    column, so matching must include a periodic roll before comparing centers.
    """
    reference = axes[0]
    step = 360.0 / reference.size
    tolerance = max(1e-8, step * TEMPORAL_LONGITUDE_TOLERANCE_PIXELS)
    shifts = [0]
    for axis, path in zip(axes[1:], paths[1:]):
        if axis.shape != reference.shape or not np.all(np.diff(axis) > 0):
            raise RuntimeError(
                "Temporal interpolation requires compatible physical Carrington "
                f"longitude grids (centers, spacing, orientation and coverage): "
                f"{paths[0]} and {path}."
            )
        column, _ = closest_longitude_column(axis, reference[0])
        aligned = np.roll(axis, -column)
        residual = (aligned - reference + 180.0) % 360.0 - 180.0
        error = float(np.max(np.abs(residual)))
        if not np.isfinite(error) or error > tolerance:
            raise RuntimeError(
                "Temporal interpolation requires compatible physical Carrington "
                f"longitude grids: {paths[0]} and {path}; maximum offset "
                f"{error / step:.6g} pixels exceeds "
                f"{TEMPORAL_LONGITUDE_TOLERANCE_PIXELS:g} pixels."
            )
        shifts.append(-column)
        if error > 1e-8:
            logger.info(
                "Accepting negligible longitude offset %.6g deg (%.6g pixels) "
                "in %s; using the first stencil map's centers after roll %d.",
                error, error / step, path, -column,
            )
    return shifts


def extract_gong_longitude_shift(file_path: str) -> int:
    """Extract the GONG longitude offset encoded in a filename."""
    name = os.path.basename(file_path)
    try:
        return int(name.split("_")[-1].split(".")[0]) - 1
    except (IndexError, ValueError):
        logger.warning("Could not parse GONG longitude shift from %s; using 0.", name)
        return 0


def circular_shift_longitude(Br: np.ndarray, shift: int) -> np.ndarray:
    """Apply a circular longitude shift to a magnetogram."""
    nb_phi = Br.shape[1]
    shift = shift % nb_phi
    if shift == 0:
        return Br
    return np.hstack((Br[:, -shift:], Br[:, :-shift]))


def hmi_dynamic_longitude_shift(file_path: str, width: int) -> int:
    """Return the HMI dynamic-frame roll that places longitude zero first."""
    with fits.open(file_path) as hdul:
        image_hdu = next(
            (hdu for hdu in hdul if hdu.data is not None and hdu.data.ndim >= 2),
            None,
        )
        if image_hdu is None:
            raise ValueError(f"No magnetogram image HDU found in FITS file: {file_path}")
        header = image_hdu.header

    crval1 = float(header.get("CRVAL1", 0.0))
    crpix1 = float(header.get("CRPIX1", width / 2.0 + 0.5))
    cdelt1 = float(header.get("CDELT1", -(360.0 / width)))
    if not np.isfinite(cdelt1) or np.isclose(cdelt1, 0.0):
        raise ValueError(
            f"Invalid HMI dynamic-frame CDELT1 in magnetogram header: {file_path}"
        )

    lon0 = crval1 - (crpix1 - 1.0) * cdelt1
    shift_pixels = int(np.round(lon0 / cdelt1)) % width
    logger.info(
        "Rolling HMI dynamic-frame longitude by %d pixels (native lon0: %.6f deg).",
        shift_pixels,
        lon0,
    )
    return shift_pixels


def hmi_hourly_longitude_shift(file_path: str, width: int) -> int:
    """Backward-compatible alias for the HMI dynamic-frame longitude roll."""
    return hmi_dynamic_longitude_shift(file_path, width)


def ensure_increasing_longitude(
    Br: np.ndarray,
    file_path: str,
    map_type: str,
) -> np.ndarray:
    """Return ``Br`` with columns ordered by increasing native longitude."""
    map_type = normalize_map_type(map_type)
    if map_type.lower() == "wso":
        return Br
    if "hmi" in map_type.lower() and map_type != "HMI_fdt":
        logger.info("HMI maps are assumed to have increasing longitude.")
        return Br
    if is_br_longitude_increasing(file_path):
        return Br
    logger.info("Flipping Br columns to obtain increasing longitude.")
    return np.ascontiguousarray(Br[:, ::-1])


def roll_hmi_dynamic_to_zero_longitude(
    Br: np.ndarray,
    file_path: str,
) -> np.ndarray:
    """Roll an HMI dynamic-frame map so its Carrington-zero column comes first."""
    shift_pixels = hmi_dynamic_longitude_shift(file_path, Br.shape[1])
    return np.roll(Br, shift_pixels, axis=1)


def roll_hmi_hourly_to_zero_longitude(
    Br: np.ndarray,
    file_path: str,
) -> np.ndarray:
    """Backward-compatible alias for rolling an HMI dynamic-frame map."""
    return roll_hmi_dynamic_to_zero_longitude(Br, file_path)


def rotate_longitude_to_stonyhurst(
    Br: np.ndarray,
    angle_degrees: float,
    has_duplicate_endpoint: bool = False,
    zero_column: int | None = None,
) -> np.ndarray:
    """Roll an increasing-longitude map into the requested Stonyhurst frame."""
    unique_longitudes = Br.shape[1] - 1 if has_duplicate_endpoint else Br.shape[1]
    if zero_column is None:
        zero_column = round((angle_degrees % 360.0) / 360.0 * unique_longitudes)
    shift = -zero_column
    logger.info(
        "Rotating Br to Stonyhurst by %.6f degrees (%d longitude cells).",
        angle_degrees,
        shift,
    )
    if not has_duplicate_endpoint:
        return np.roll(Br, shift=shift, axis=1)

    rotated = np.roll(Br[:, :-1], shift=shift, axis=1)
    return np.hstack((rotated, rotated[:, :1]))


def processed_longitude_axis(
    file_path: str,
    map_type: str,
    temporal: bool = False,
) -> np.ndarray:
    """Return the historical processing axis for compatibility callers.

    Dynamic HMI origins were rounded by this API. Physical pipelines instead
    carry the exact centers returned by ``read_carrington_map`` with the field.
    """
    map_type = normalize_map_type(map_type)
    if map_type == "custom":
        inferred_map_type = infer_known_fits_map_type(file_path)
        if inferred_map_type is not None:
            return processed_longitude_axis(
                file_path,
                inferred_map_type,
                temporal=temporal,
            )
        return read_fits_longitude_axis(file_path).centers_degrees
    if map_type.lower() == "wso":
        return np.linspace(0.0, 360.0, 73)
    if map_type in _HMI_DYNAMIC_MAP_TYPES:
        with fits.open(file_path) as hdul:
            image_hdu = next(
                (
                    hdu
                    for hdu in hdul
                    if hdu.data is not None and hdu.data.ndim >= 2
                ),
                None,
            )
            if image_hdu is None:
                raise ValueError(
                    f"No magnetogram image HDU found in FITS file: {file_path}"
                )
            width = image_hdu.data.shape[-1]
            longitude_step = abs(
                float(image_hdu.header.get("CDELT1", 360.0 / width))
            )
            if not np.isfinite(longitude_step) or np.isclose(longitude_step, 0.0):
                raise ValueError(
                    f"Invalid HMI dynamic-frame CDELT1 in magnetogram header: {file_path}"
                )
        return np.arange(width, dtype=float) * longitude_step

    longitude = increasing_longitude_axis(file_path)
    if temporal and is_gong_map_type(map_type):
        longitude = np.roll(
            longitude,
            extract_gong_longitude_shift(file_path),
        )
    return longitude


def resize_processed_longitude_axis(
    longitude_original: np.ndarray,
    nb_phi: int,
    has_duplicate_endpoint: bool = False,
    preserve_cell_edges: bool = False,
) -> np.ndarray:
    """Resize a processed longitude axis for the resized pixel grid."""
    longitude_original = np.asarray(longitude_original, dtype=float)
    if longitude_original.ndim != 1:
        raise ValueError("longitude_original must be a 1D array.")
    if longitude_original.size == 0:
        raise ValueError("longitude_original must not be empty.")
    if longitude_original.size == nb_phi:
        return longitude_original

    unique_longitudes = nb_phi - 1 if has_duplicate_endpoint else nb_phi
    if unique_longitudes < 1:
        raise ValueError("nb_phi must describe at least one unique longitude.")

    output_step = 360.0 / unique_longitudes
    start = longitude_original[0]
    if preserve_cell_edges and longitude_original.size > 1:
        input_step = float(np.median(np.diff(np.unwrap(longitude_original, period=360.0))))
        if not np.isfinite(input_step) or np.isclose(input_step, 0.0):
            raise ValueError("longitude_original must have a finite nonzero spacing.")
        start = start - input_step / 2.0 + output_step / 2.0
    longitude = start + np.arange(unique_longitudes, dtype=float) * output_step
    if has_duplicate_endpoint:
        longitude = np.concatenate((longitude, longitude[:1] + 360.0))
    return longitude


def closest_longitude_column(
    longitude: np.ndarray,
    target_degrees: float,
) -> tuple[int, float]:
    """Find the longitude column closest to a periodic target angle."""
    residuals = (np.asarray(longitude) - target_degrees + 180.0) % 360.0 - 180.0
    index = int(np.nanargmin(np.abs(residuals)))
    return index, float(residuals[index])


def apply_configured_longitude_rotation(
    Br: np.ndarray,
    Br_linear: np.ndarray | None,
    local_file: str | list[str],
    map_type: str,
    target_date: str | datetime,
    use_interpolation: bool,
    rotate_to_stonyhurst: bool,
    effective_date: str | datetime | None = None,
    resize: bool = False,
    *,
    Phi: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, float | None]:
    """Rotate a reader-normalized Carrington map together with its coordinates.

    Physical readers must be requested whenever rotation is enabled. The
    legacy no-rotation path returns the supplied arrays unchanged. WSO retains
    its historical duplicate-endpoint treatment and is outside the physical
    normalization contract.
    """
    if Phi is None:
        raise ValueError(
            "Phi is required: longitude rotation must return field and "
            "coordinates together."
        )
    if not rotate_to_stonyhurst:
        return Br, Br_linear, Phi, None
    map_type = normalize_map_type(map_type)
    source_file = local_file[0] if isinstance(local_file, list) else local_file
    rotation_date = (
        parse_iso_datetime(effective_date)
        if effective_date is not None
        else magnetogram_effective_date(
            source_file, map_type, target_date,
            interpolated=use_interpolation and isinstance(local_file, list),
        )
    )
    angle = compute_carrington_central_meridian(rotation_date)
    if Br_linear is not None and Br_linear.shape != Br.shape:
        raise ValueError("Br_linear must have the same shape as Br before rotation.")
    if map_type.lower() == "wso":
        Br = rotate_longitude_to_stonyhurst(Br, angle, has_duplicate_endpoint=True)
        if Br_linear is not None:
            Br_linear = rotate_longitude_to_stonyhurst(
                Br_linear, angle, has_duplicate_endpoint=True
            )
        return Br, Br_linear, Phi, angle
    if Br.ndim != 2 or Phi.shape != Br.shape:
        raise ValueError("Br and Phi must have the same 2D shape.")
    if not np.allclose(Phi, Phi[:1], atol=1e-14, rtol=0):
        raise ValueError("Phi must describe the same longitude centers in every Br row.")
    longitude = np.degrees(Phi[0])
    Br, output_longitude = rotate_carrington_to_stonyhurst(Br, longitude, angle)
    if Br_linear is not None:
        Br_linear, _ = rotate_carrington_to_stonyhurst(Br_linear, longitude, angle)
    Phi = np.broadcast_to(np.radians(output_longitude), Br.shape).copy()
    return Br, Br_linear, Phi, angle
