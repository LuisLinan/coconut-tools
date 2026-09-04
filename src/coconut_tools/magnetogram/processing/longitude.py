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
    """Return longitude centers matching the normalized magnetic-field columns."""
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
        input_step = float(np.median(np.diff(longitude_original)))
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


def _apply_custom_stonyhurst_rotation(
    Br: np.ndarray,
    Br_linear: np.ndarray | None,
    source_file: str,
    rotation_date: datetime,
    resize: bool,
) -> tuple[np.ndarray, np.ndarray | None, float]:
    """Rotate a header-described custom longitude grid into Stonyhurst."""
    geometry = read_fits_longitude_axis(source_file)
    if geometry.frame == "unknown":
        raise ValueError(
            "Cannot rotate the custom magnetogram to Stonyhurst because its "
            f"longitude frame is ambiguous (CTYPE1={geometry.ctype1!r}). Use "
            "the standard CRLN-* axis code for Carrington longitude or HGLN-* "
            "for Stonyhurst longitude. The projection suffix (for example "
            "'-CAR') does not identify the reference frame."
        )

    longitude = geometry.centers_degrees
    if resize:
        longitude = resize_processed_longitude_axis(
            longitude,
            Br.shape[1],
            preserve_cell_edges=True,
        )
    if longitude.size != Br.shape[1]:
        raise ValueError(
            "The custom longitude axis does not match the processed Br columns."
        )
    if Br_linear is not None and Br_linear.shape != Br.shape:
        raise ValueError("Br_linear must have the same shape as Br before rotation.")

    if geometry.frame == "stonyhurst":
        logger.info(
            "Custom longitude axis is already Stonyhurst (%s=%s); no "
            "Carrington rotation is applied.",
            geometry.frame_source,
            (
                geometry.ctype1
                if geometry.frame_source == "CTYPE1"
                else geometry.frame_source
            ),
        )
        return Br, Br_linear, 0.0

    central_meridian = compute_carrington_central_meridian(rotation_date)
    logger.info(
        "Custom Carrington map uses the configured UTC time %s for "
        "Stonyhurst zero: %.6f deg.",
        rotation_date.isoformat(),
        central_meridian,
    )

    # Phi is not returned by this public API. Preserve its first cell center:
    # output phi[0] must sample source Carrington longitude L0 + phi[0].
    target_longitude = (central_meridian + float(longitude[0])) % 360.0
    zero_column, residual = closest_longitude_column(longitude, target_longitude)
    logger.info(
        "Custom Stonyhurst rotation uses longitude column %d; target source "
        "longitude %.6f deg, grid residual %.6f deg.",
        zero_column,
        target_longitude,
        residual,
    )
    Br = rotate_longitude_to_stonyhurst(
        Br,
        central_meridian,
        zero_column=zero_column,
    )
    if Br_linear is not None:
        Br_linear = rotate_longitude_to_stonyhurst(
            Br_linear,
            central_meridian,
            zero_column=zero_column,
        )
    return Br, Br_linear, central_meridian


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
) -> tuple[np.ndarray, np.ndarray | None, float | None]:
    """Apply the configured Carrington-to-Stonyhurst longitude rotation."""
    map_type = normalize_map_type(map_type)
    is_custom_input = map_type == "custom"
    inferred_map_type = None
    if not rotate_to_stonyhurst:
        return Br, Br_linear, None

    source_file = local_file[0] if isinstance(local_file, list) else local_file
    if is_custom_input:
        inferred_map_type = infer_known_fits_map_type(source_file)
        if inferred_map_type is not None:
            logger.info(
                "Custom FITS identified as %s for longitude rotation.",
                inferred_map_type,
            )
            map_type = inferred_map_type
    interpolated = use_interpolation and (
        is_gong_temporal_map_type(map_type)
        or map_type in {"ADAPT", "HMI_hourly", "HMI_fdt"}
    )
    rotation_date = (
        parse_iso_datetime(effective_date)
        if effective_date is not None
        else magnetogram_effective_date(
            source_file,
            "custom" if is_custom_input else map_type,
            target_date,
            interpolated=interpolated,
        )
    )

    if map_type == "custom":
        return _apply_custom_stonyhurst_rotation(
            Br,
            Br_linear,
            source_file,
            rotation_date,
            resize,
        )

    custom_fixed_carrington_product = (
        is_custom_input
        and inferred_map_type is not None
        and (
            inferred_map_type.startswith("HMI_")
            or inferred_map_type == "ADAPT"
        )
    )
    if interpolated or custom_fixed_carrington_product:
        # Native HMI and fixed-grid ADAPT rotation is simply the Carrington
        # central meridian at the effective map time. For a renamed custom
        # FITS, compute it from the header-derived date directly: the legacy
        # helper identifies these products from their filename.
        rotation_angle = compute_carrington_central_meridian(rotation_date)
    else:
        rotation_angle, rotation_date = compute_rotation_angle(
            source_file,
            date_hmi=parse_iso_datetime(target_date).isoformat(),
            map_type=map_type,
            interpolated=False,
            effective_date=rotation_date,
        )

    has_duplicate_endpoint = map_type.lower() == "wso"
    longitude_original = processed_longitude_axis(
        source_file,
        map_type,
        temporal=interpolated and is_gong_temporal_map_type(map_type),
    )
    if resize:
        longitude = resize_processed_longitude_axis(
            longitude_original,
            Br.shape[1],
            has_duplicate_endpoint=has_duplicate_endpoint,
            preserve_cell_edges=map_type == "HMI_fdt",
        )
    else:
        longitude = longitude_original

    target_longitude = (
        compute_carrington_central_meridian(rotation_date)
        if is_gong_map_type(map_type)
        else rotation_angle
    )
    zero_column, residual = closest_longitude_column(longitude, target_longitude)
    logger.info(
        "Stonyhurst zero uses longitude column %d with residual %.6f degrees.",
        zero_column,
        residual,
    )

    Br = rotate_longitude_to_stonyhurst(
        Br,
        rotation_angle,
        has_duplicate_endpoint=has_duplicate_endpoint,
        zero_column=zero_column,
    )
    if Br_linear is not None:
        Br_linear = rotate_longitude_to_stonyhurst(
            Br_linear,
            rotation_angle,
            has_duplicate_endpoint=has_duplicate_endpoint,
            zero_column=zero_column,
        )
    return Br, Br_linear, rotation_angle
