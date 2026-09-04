"""Read, normalize, resize, and temporally interpolate magnetogram products."""

import numpy as np
from astropy.io import fits
from scipy import interpolate

from coconut_tools.magnetogram.core.config import _as_bool
from coconut_tools.magnetogram.core.coordinates import (
    _validate_theta_axis,
    build_theta_phi,
    fits_latitude_axis,
    theta_cell_edges,
)
from coconut_tools.magnetogram.io.downloads import (
    InterpolationSelection,
    is_gong_map_type,
    normalize_map_type,
)
from coconut_tools.magnetogram.io.metadata import (
    infer_known_fits_map_type,
    read_fits_longitude_axis,
)
from coconut_tools.magnetogram.processing.longitude import (
    circular_shift_longitude,
    ensure_increasing_longitude,
    extract_gong_longitude_shift,
    resize_processed_longitude_axis,
    roll_hmi_dynamic_to_zero_longitude,
)
from coconut_tools.tools.logger_config import setup_logger
from coconut_tools.tools.rotation_angle import increasing_longitude_axis

logger = setup_logger(__name__)

RESIZED_MAGNETOGRAM_SHAPE = (360, 720)
_RESIZED_MAGNETOGRAM_SHAPE = RESIZED_MAGNETOGRAM_SHAPE
_HMI_DYNAMIC_MAP_TYPES = {"HMI_SYNC", "HMI_hourly"}
_ADAPT_ENSEMBLE_MAP_TYPES = {"ADAPT", "HMI_fdt"}


def resize_magnetogram_if_requested(
    Br_data: np.ndarray,
    enabled: bool,
) -> np.ndarray:
    """Resize a Br map to the standard 360x720 grid when requested."""
    if not enabled:
        return Br_data

    from skimage.transform import resize as resize_image

    return resize_image(
        Br_data,
        RESIZED_MAGNETOGRAM_SHAPE,
        preserve_range=True,
        mode="edge",
        clip=False,
        anti_aliasing=True,
    )


def read_first_fits_image(file_path: str) -> np.ndarray:
    """Read the first FITS HDU that contains at least a 2D image array."""
    with fits.open(file_path) as hdul:
        for hdu in hdul:
            if hdu.data is not None and hdu.data.ndim >= 2:
                return np.asarray(hdu.data)
    raise ValueError(f"No magnetogram image HDU found in FITS file: {file_path}")


def _read_first_fits_shape_and_header(
    file_path: str,
) -> tuple[tuple[int, ...], fits.Header]:
    """Read the first image shape and a detached copy of its FITS header."""
    with fits.open(file_path) as hdul:
        for hdu in hdul:
            if hdu.data is not None and hdu.data.ndim >= 2:
                return tuple(hdu.data.shape), hdu.header.copy()
    raise ValueError(f"No magnetogram image HDU found in FITS file: {file_path}")


def validate_hmi_fdt_carrington_frame(file_path: str) -> np.ndarray:
    """Validate and return the fixed Carrington longitude grid of HMI-FDT."""
    with fits.open(file_path) as hdul:
        image_hdu = next(
            (hdu for hdu in hdul if hdu.data is not None and hdu.data.ndim >= 2),
            None,
        )
        if image_hdu is None:
            raise ValueError(f"No magnetogram image HDU found in FITS file: {file_path}")
        header = image_hdu.header
        try:
            longitude_type = int(header["LNGTYPE"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "HMI_fdt requires an explicit FITS LNGTYPE=0 Carrington frame: "
                f"{file_path}"
            ) from exc
        if longitude_type != 0:
            raise ValueError(
                "HMI_fdt interpolation requires the Carrington-fixed adapt40i11 "
                f"product (LNGTYPE=0), not LNGTYPE={longitude_type}: {file_path}"
            )

        width = int(image_hdu.data.shape[-1])
        try:
            longitude_step = abs(float(header["CDELT1"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"HMI_fdt requires a valid FITS CDELT1: {file_path}") from exc
        if (
            not np.isfinite(longitude_step)
            or np.isclose(longitude_step, 0.0)
            or not np.isclose(longitude_step * width, 360.0, atol=1e-6)
        ):
            raise ValueError(
                "HMI_fdt must cover one complete 360-degree Carrington grid: "
                f"{file_path}"
            )

    return increasing_longitude_axis(file_path)


def _fallback_theta_axis(size: int, map_type: str) -> tuple[np.ndarray, str]:
    """Return known cell-centered latitude coordinates for a FITS product."""
    if size < 1:
        raise ValueError("A magnetogram latitude axis must contain at least one row.")
    map_type = normalize_map_type(map_type)
    if map_type in _ADAPT_ENSEMBLE_MAP_TYPES:
        theta = (np.arange(size, dtype=float) + 0.5) * np.pi / size
        return theta, "latitude"
    if is_gong_map_type(map_type) or map_type.startswith("HMI_"):
        mu = 1.0 - (np.arange(size, dtype=float) + 0.5) * 2.0 / size
        return np.arccos(mu), "sine"
    if map_type == "custom":
        raise ValueError(
            "A custom magnetogram requires complete FITS latitude metadata; "
            "no product-specific latitude fallback can be inferred."
        )
    raise ValueError(f"No FITS latitude fallback is defined for {map_type}.")


def _read_custom_longitude_axis(
    file_path: str,
    width: int,
) -> tuple[np.ndarray, bool, int]:
    """Decode and normalize a full-sphere custom FITS longitude axis.

    Returns the endpoint-free longitude centers in radians, whether the source
    columns must be reversed, and the roll that places the smallest wrapped
    longitude first.
    """
    geometry = read_fits_longitude_axis(file_path, width)
    return (
        np.radians(geometry.centers_degrees),
        geometry.flip_columns,
        geometry.roll_columns,
    )


def _resample_theta_centers(
    theta: np.ndarray,
    target_size: int,
    native_coordinate: str,
) -> np.ndarray:
    """Resize latitude centers while preserving their outer physical edges."""
    theta = np.asarray(theta, dtype=float)
    if target_size < 1:
        raise ValueError("target_size must be positive.")
    if target_size == theta.size:
        return theta.copy()

    edges = theta_cell_edges(theta, coordinate=native_coordinate)
    if native_coordinate == "sine":
        north_mu = np.cos(edges[0])
        south_mu = np.cos(edges[-1])
        step = (north_mu - south_mu) / target_size
        mu = north_mu - (np.arange(target_size, dtype=float) + 0.5) * step
        return np.arccos(np.clip(mu, -1.0, 1.0))

    step = (edges[-1] - edges[0]) / target_size
    return edges[0] + (np.arange(target_size, dtype=float) + 0.5) * step


def read_fits_theta_axis(
    file_path: str,
    map_type: str,
    target_size: int | None = None,
) -> tuple[np.ndarray, bool]:
    """Read FITS latitude centers in north-to-south colatitude order.

    The returned boolean indicates whether magnetic-field rows must be
    reversed. Missing coordinate metadata activates a warned product-specific
    fallback. Resizing preserves the source grid's outer physical edges.
    """
    map_type = normalize_map_type(map_type)
    image_shape, header = _read_first_fits_shape_and_header(file_path)
    native_size = int(image_shape[-2])
    if int(header.get("NAXIS2", native_size)) != native_size:
        raise ValueError(
            f"FITS NAXIS2 does not match the image latitude dimension: {file_path}"
        )

    try:
        latitude, _, metadata = fits_latitude_axis(header, output="deg")
        theta = np.pi / 2.0 - np.radians(latitude[::-1])
        flip_rows = not bool(metadata["flipped"])
        native_coordinate = str(metadata["native_coordinate"])
    except KeyError as exc:
        theta, native_coordinate = _fallback_theta_axis(native_size, map_type)
        flip_rows = True
        logger.warning(
            "Incomplete FITS latitude metadata in %s (%s); using the known "
            "cell-centered %s-latitude grid for %s.",
            file_path,
            exc,
            "sine" if native_coordinate == "sine" else "uniform",
            map_type,
        )

    _validate_theta_axis(theta)
    if target_size is not None:
        theta = _resample_theta_centers(theta, int(target_size), native_coordinate)
        _validate_theta_axis(theta)
    return theta, flip_rows


def build_regular_theta_phi(
    Br: np.ndarray,
    map_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the warned product fallback grid when no FITS header is available."""
    map_type = normalize_map_type(map_type)
    nb_th, nb_phi = Br.shape
    theta, _ = _fallback_theta_axis(nb_th, map_type)
    phi = np.linspace(0.0, 2.0 * np.pi, nb_phi, endpoint=False)
    return theta, phi


def _read_temporal_br_map_and_theta(
    file_path: str,
    map_type: str,
    adapt_map: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Read one temporal FITS map together with its normalized latitude axis."""
    map_type = normalize_map_type(map_type)
    input_data = read_first_fits_image(file_path)
    theta, flip_rows = read_fits_theta_axis(file_path, map_type)

    if map_type == "HMI_hourly":
        Br = np.asarray(input_data)
    elif map_type in _ADAPT_ENSEMBLE_MAP_TYPES:
        if map_type == "HMI_fdt":
            validate_hmi_fdt_carrington_frame(file_path)
        Br = np.asarray(input_data[adapt_map, :, :])
    elif is_gong_map_type(map_type):
        Br = np.asarray(input_data)
    else:
        raise ValueError(f"Temporal interpolation is not supported for {map_type}")

    if Br.ndim != 2 or Br.shape[0] != theta.size:
        raise ValueError(
            f"Magnetogram data and FITS latitude axis are incompatible: {file_path}"
        )
    if flip_rows:
        Br = Br[::-1, :]
    Br = np.nan_to_num(Br)

    if map_type == "HMI_hourly":
        Br = roll_hmi_dynamic_to_zero_longitude(Br, file_path)
    else:
        Br = ensure_increasing_longitude(Br, file_path, map_type)
        if is_gong_map_type(map_type):
            Br = circular_shift_longitude(
                Br,
                extract_gong_longitude_shift(file_path),
            )
    return Br, theta


def read_temporal_br_map(
    file_path: str,
    map_type: str,
    adapt_map: int = 0,
) -> np.ndarray:
    """Read and normalize one FITS magnetogram used in interpolation."""
    Br, _ = _read_temporal_br_map_and_theta(file_path, map_type, adapt_map)
    return Br


def interpolate_br_maps(
    Br_maps: list[np.ndarray],
    selection: InterpolationSelection,
    interpolation_order: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate a four-map temporal stencil onto the target time."""
    Br_datam, Br_data, Br_data1, Br_datap = Br_maps
    Br_linear = selection.coef_before * Br_data + selection.coef_after * Br_data1
    if interpolation_order == 1:
        return np.nan_to_num(Br_linear), np.nan_to_num(Br_linear)
    if interpolation_order != 2:
        raise ValueError("interpolation_order must be 1 or 2.")

    time_norm = selection.coef_after
    h00 = 2.0 * time_norm**3.0 - 3.0 * time_norm**2.0 + 1.0
    h10 = time_norm**3.0 - 2.0 * time_norm**2.0 + time_norm
    h01 = -2.0 * time_norm**3.0 + 3.0 * time_norm**2.0
    h11 = time_norm**3.0 - time_norm**2.0

    derivative1 = 0.5 * (
        (Br_data - Br_datam) / selection.previous_interval_seconds
        + (Br_data1 - Br_data) / selection.interval_seconds
    )
    derivative2 = 0.5 * (
        (Br_datap - Br_data1) / selection.next_interval_seconds
        + (Br_data1 - Br_data) / selection.interval_seconds
    )
    Br_interpolated = (
        Br_data * h00
        + derivative1 * selection.interval_seconds * h10
        + Br_data1 * h01
        + derivative2 * selection.interval_seconds * h11
    )
    return np.nan_to_num(Br_interpolated), np.nan_to_num(Br_linear)


def read_interpolated_magnetogram(
    local_files: list[str],
    map_type: str,
    selection: InterpolationSelection,
    adapt_map: int = 0,
    interpolation_order: int = 2,
    resize: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read, normalize, and temporally interpolate a four-map stencil."""
    map_type = normalize_map_type(map_type)
    resize = _as_bool(resize)
    logger.info("Reading interpolation stencil")
    normalized_maps = [
        _read_temporal_br_map_and_theta(path, map_type, adapt_map)
        for path in local_files
    ]
    Br_maps = [item[0] for item in normalized_maps]
    theta_axes = [item[1] for item in normalized_maps]
    shapes = {Br.shape for Br in Br_maps}
    if len(shapes) != 1:
        raise RuntimeError(f"Interpolation stencil has inconsistent shapes: {shapes}")
    reference_theta = theta_axes[0]
    incompatible_latitude_files = [
        path
        for path, theta in zip(local_files[1:], theta_axes[1:])
        if theta.shape != reference_theta.shape
        or not np.allclose(theta, reference_theta, atol=1.0e-12, rtol=0.0)
    ]
    if incompatible_latitude_files:
        files = ", ".join([str(local_files[0]), *map(str, incompatible_latitude_files)])
        raise RuntimeError(
            "Temporal interpolation requires identical physical latitude grids; "
            f"incompatible FITS files: {files}"
        )
    if map_type == "HMI_fdt":
        longitude_axes = [
            validate_hmi_fdt_carrington_frame(path) for path in local_files
        ]
        reference_longitude = longitude_axes[0]
        if any(
            not np.allclose(longitude, reference_longitude, atol=1e-9, rtol=0.0)
            for longitude in longitude_axes[1:]
        ):
            raise RuntimeError(
                "HMI_fdt interpolation stencil does not share one fixed "
                "Carrington longitude grid."
            )
    Br_maps = [resize_magnetogram_if_requested(Br, resize) for Br in Br_maps]
    Br, Br_linear = interpolate_br_maps(Br_maps, selection, interpolation_order)
    theta = (
        read_fits_theta_axis(
            local_files[0],
            map_type,
            target_size=RESIZED_MAGNETOGRAM_SHAPE[0],
        )[0]
        if resize
        else reference_theta
    )
    phi = np.linspace(0.0, 2.0 * np.pi, Br.shape[1], endpoint=False)
    Theta, Phi = build_theta_phi(theta, phi)
    logger.info("End of reading interpolation stencil")
    return Br, Theta, Phi, Br_linear


def read_magnetogram(file_path, map_type="custom", adapt_map=0, resize=False):
    """Read one magnetogram and build its physical spherical grid.

    When ``map_type`` is omitted or ``None``, the file is treated as a generic
    custom 2D FITS magnetogram whose complete latitude and longitude geometry
    must be described by its header.
    """
    map_type = "custom" if map_type is None else normalize_map_type(map_type)
    resize = _as_bool(resize)
    logger.info("Reading file")

    if map_type == "custom":
        inferred_map_type = infer_known_fits_map_type(file_path)
        if inferred_map_type is not None:
            logger.info(
                "Custom FITS identified as %s; using its native reader.",
                inferred_map_type,
            )
            return read_magnetogram(
                file_path,
                inferred_map_type,
                adapt_map=adapt_map,
                resize=resize,
            )

    if map_type == "custom":
        input_data = read_first_fits_image(file_path)
        if input_data.ndim != 2:
            raise ValueError(
                "A custom magnetogram must be a single 2D FITS image; "
                f"received shape {input_data.shape}."
            )
        theta, flip_rows = read_fits_theta_axis(
            file_path,
            map_type,
            target_size=RESIZED_MAGNETOGRAM_SHAPE[0] if resize else None,
        )
        phi, flip_columns, longitude_roll = _read_custom_longitude_axis(
            file_path,
            input_data.shape[1],
        )
        Br_map = np.asarray(input_data)
        if flip_rows:
            Br_map = Br_map[::-1, :]
        if flip_columns:
            Br_map = Br_map[:, ::-1]
        if longitude_roll:
            Br_map = np.roll(Br_map, longitude_roll, axis=1)
        Br_map = resize_magnetogram_if_requested(Br_map, resize)
        if resize:
            phi = np.radians(
                resize_processed_longitude_axis(
                    np.degrees(phi),
                    Br_map.shape[1],
                    preserve_cell_edges=True,
                )
            )
        Br_map = np.nan_to_num(Br_map)
        Theta, Phi = build_theta_phi(theta, phi)

    elif map_type in _ADAPT_ENSEMBLE_MAP_TYPES:
        if map_type == "HMI_fdt":
            validate_hmi_fdt_carrington_frame(file_path)
        input_data = read_first_fits_image(file_path)
        theta, flip_rows = read_fits_theta_axis(
            file_path,
            map_type,
            target_size=RESIZED_MAGNETOGRAM_SHAPE[0] if resize else None,
        )
        Br_map = input_data[adapt_map, :, :]
        if flip_rows:
            Br_map = Br_map[::-1, :]
        Br_map = ensure_increasing_longitude(Br_map, file_path, map_type)
        Br_map = resize_magnetogram_if_requested(Br_map, resize)
        phi = np.linspace(0.0, 2.0 * np.pi, Br_map.shape[1], endpoint=False)
        Theta, Phi = build_theta_phi(theta, phi)

    elif map_type.lower() == "wso":
        with open(file_path, "r") as fwso:
            line = fwso.readline()
            while line.strip() == "":
                line = fwso.readline()
            header = line.split()
            lat_type = "sinlat" if "sine" in header else "lat"
            nb_th = int(header[1])
            nb_phi = 73
            nb_lines = 4 * nb_phi
            nb_thplus = 4
            Br_read = np.zeros((nb_th, nb_phi))
            fwso.readline()
            idx_ph = nb_phi
            for _ in range(nb_lines):
                split_line = fwso.readline().split()
                if split_line[0].startswith("C"):
                    idx_th = 0
                    idx_ph -= 1
                    for value in split_line[1:]:
                        Br_read[idx_th, idx_ph] = float(value)
                        idx_th += 1
                else:
                    for value in split_line:
                        Br_read[idx_th, idx_ph] = float(value)
                        idx_th += 1

        if lat_type == "lat":
            Br_ext = np.pad(Br_read, ((nb_thplus, nb_thplus), (0, 0)), mode="edge")
            Br_map = Br_ext * 0.01
            theta = (np.linspace(-90.0, 90.0, Br_map.shape[0]) + 90.0) * np.pi / 180.0
        else:
            Br_data = Br_read[::-1, :] * 0.01
            sinlat = np.linspace(-14.5 / 15.0, 14.5 / 15.0, nb_th)
            theta_map = np.arcsin(sinlat) + np.pi / 2.0
            theta = np.linspace(0.0, np.pi, nb_th)
            phi = np.linspace(0.0, 360.0, nb_phi) * np.pi / 180.0
            interpolator = interpolate.RectBivariateSpline(theta_map, phi, Br_data)
            Br_map = interpolator(theta, phi)[::-1, :]

        phi = np.linspace(0.0, 360.0, nb_phi) * np.pi / 180.0
        Theta, Phi = build_theta_phi(theta, phi)

    else:
        input_data = read_first_fits_image(file_path)
        theta, flip_rows = read_fits_theta_axis(
            file_path,
            map_type,
            target_size=RESIZED_MAGNETOGRAM_SHAPE[0] if resize else None,
        )
        Br_data = input_data[::-1, :] if flip_rows else input_data
        if map_type in _HMI_DYNAMIC_MAP_TYPES:
            Br_data = roll_hmi_dynamic_to_zero_longitude(Br_data, file_path)
        else:
            Br_data = ensure_increasing_longitude(Br_data, file_path, map_type)
        Br_data = resize_magnetogram_if_requested(Br_data, resize)
        phi = np.linspace(0.0, 2.0 * np.pi, Br_data.shape[1], endpoint=False)
        Theta, Phi = build_theta_phi(theta, phi)
        Br_map = np.nan_to_num(Br_data)

    logger.info("End of reading file")
    return Br_map, Theta, Phi
