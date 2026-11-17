"""Read, normalize, resize, and temporally interpolate magnetogram products."""

import os
import numpy as np
from astropy.io import fits
from scipy import interpolate

from coconut_tools.magnetogram.core.config import _as_bool
from coconut_tools.magnetogram.core.coordinates import (
    _theta_coordinate,
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
    normalize_to_carrington,
    read_carrington_map,
    resize_processed_longitude_axis,
    roll_hmi_dynamic_to_zero_longitude,
    validate_matching_longitude_axes,
)
from coconut_tools.tools.logger_config import setup_logger
from coconut_tools.tools.rotation_angle import increasing_longitude_axis

logger = setup_logger(__name__)

RESIZED_MAGNETOGRAM_SHAPE = (360, 720)
_RESIZED_MAGNETOGRAM_SHAPE = RESIZED_MAGNETOGRAM_SHAPE
_HMI_DYNAMIC_MAP_TYPES = {"HMI_SYNC", "HMI_hourly"}
_ADAPT_ENSEMBLE_MAP_TYPES = {"ADAPT", "HMI_fdt"}


def _overlap_weights(
    input_edges: np.ndarray,
    output_edges: np.ndarray,
    measure=None,
) -> np.ndarray:
    """Return the row-normalized overlap matrix between two 1D cell partitions.

    Args:
        input_edges: Increasing input cell edges, length ``n_in + 1``.
        output_edges: Increasing output cell edges, length ``n_out + 1``,
            spanning the same interval as ``input_edges``.
        measure: Optional increasing function mapping the edge coordinate to
            the measure used for weighting (e.g. ``-cos(theta)`` for solid
            angle). ``None`` weights by overlap length in the edge coordinate.

    Returns:
        ``(n_out, n_in)`` matrix whose rows sum to one: entry ``(i, k)`` is the
        fraction of output cell ``i`` covered by input cell ``k``.
    """
    lower = np.maximum(output_edges[:-1, None], input_edges[None, :-1])
    upper = np.minimum(output_edges[1:, None], input_edges[None, 1:])
    overlaps = upper > lower
    if measure is None:
        weights = np.where(overlaps, upper - lower, 0.0)
    else:
        weights = np.where(overlaps, measure(upper) - measure(lower), 0.0)
    row_sums = weights.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("Resized cells must overlap the input grid.")
    return weights / row_sums


def _resized_regular_longitude(
    n_phi_native: int,
    n_phi_out: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the column order and longitude centers of a resized regular map.

    On the non-Carrington paths the native columns are labelled with centers
    ``linspace(0, 2*pi, n_phi_native, endpoint=False)``. The resize maps the
    first native cell edge to the first output edge, so the output centers
    are shifted by half an output pixel minus half a native pixel (e.g.
    +0.2 deg for HMI 0.1 deg -> 0.5 deg, -0.25 deg for GONG 1 deg -> 0.5 deg),
    exactly as ``resize_processed_longitude_axis(..., preserve_cell_edges=True)``
    does on the Carrington path. Centers are wrapped to ``[0, 2*pi)``; the
    returned order sorts the resized columns accordingly (a pure relabelling,
    no resampling).

    Returns:
        ``(order, phi)``: column permutation to apply to the resized field and
        the sorted longitude centers in radians.
    """
    if n_phi_native == n_phi_out:
        return (
            np.arange(n_phi_out),
            np.linspace(0.0, 2.0 * np.pi, n_phi_out, endpoint=False),
        )
    native = np.linspace(0.0, 360.0, n_phi_native, endpoint=False)
    longitude = resize_processed_longitude_axis(
        native, n_phi_out, preserve_cell_edges=True
    ) % 360.0
    order = np.argsort(longitude, kind="stable")
    return order, np.radians(longitude[order])


def resize_magnetogram_if_requested(
    Br_data: np.ndarray,
    enabled: bool,
    theta: np.ndarray | None = None,
    theta_coordinate: str | None = None,
) -> np.ndarray:
    """Resize a Br map to the standard 360x720 grid when requested.

    The resize is a first-order conservative remap: every output pixel is the
    solid-angle-weighted mean of the input pixels it overlaps (the field is
    assumed constant inside each input pixel). The surface integral of ``Br``
    (net and signed flux per region) is therefore preserved and no extra
    smoothing is applied, unlike image interpolation with anti-aliasing.

    The geometry is unchanged with respect to the former ``skimage`` resize:
    both axes map the first input edge to the first output edge and the last
    to the last. In colatitude, output cells are uniform in the native
    coordinate (sine latitude or latitude) between the native outer edges,
    exactly as ``_resample_theta_centers`` builds the returned ``Theta``; in
    longitude, output cells are uniform between the input edges, as assumed by
    ``resize_processed_longitude_axis``. Column order (Carrington, Stonyhurst,
    rolls, flips) is preserved because the remap is applied afterwards.

    Missing values (NaN) are counted as zero field, which is how the pipeline
    already treats them at native resolution (``np.nan_to_num``).

    Args:
        Br_data: 2D field with rows ordered north to south (increasing
            colatitude), as produced by the readers.
        enabled: When false, ``Br_data`` is returned unchanged.
        theta: Native colatitude centers (radians) of the rows of
            ``Br_data``. When omitted, rows are assumed to be equal-area
            (uniform in sine latitude) and a warning is logged.
        theta_coordinate: ``"sine"`` or ``"latitude"``, the coordinate in
            which ``theta`` is uniform. Inferred from ``theta`` when omitted.

    Returns:
        The resized field with shape ``RESIZED_MAGNETOGRAM_SHAPE``.
    """
    if not enabled:
        return Br_data

    Br = np.nan_to_num(np.asarray(Br_data, dtype=float))
    if Br.ndim != 2:
        raise ValueError("Only 2D magnetograms can be resized.")
    n_theta_in, n_phi_in = Br.shape
    n_theta_out, n_phi_out = RESIZED_MAGNETOGRAM_SHAPE

    if theta is None:
        logger.warning(
            "No colatitude axis given for the resize; assuming equal-area "
            "(uniform sine-latitude) rows."
        )
        theta_weights = _overlap_weights(
            np.arange(n_theta_in + 1, dtype=float),
            np.linspace(0.0, n_theta_in, n_theta_out + 1),
        )
    else:
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (n_theta_in,):
            raise ValueError("theta must match the number of magnetogram rows.")
        if theta_coordinate is None:
            theta_coordinate = _theta_coordinate(theta)
        edges = theta_cell_edges(theta, coordinate=theta_coordinate)
        if theta_coordinate == "sine":
            # Edge coordinate u = -cos(theta): uniform on sine grids and
            # directly proportional to solid angle.
            input_edges = -np.cos(edges)
            measure = None
        else:
            # Edge coordinate u = theta; solid angle measured by -cos(theta).
            input_edges = edges
            measure = lambda u: -np.cos(u)  # noqa: E731
        output_edges = np.linspace(input_edges[0], input_edges[-1], n_theta_out + 1)
        theta_weights = _overlap_weights(input_edges, output_edges, measure)

    # Longitude pixels are uniform, so solid angle is proportional to the
    # overlap length in pixel units.
    phi_weights = _overlap_weights(
        np.arange(n_phi_in + 1, dtype=float),
        np.linspace(0.0, n_phi_in, n_phi_out + 1),
    )
    return theta_weights @ Br @ phi_weights.T


def warn_nan_pixels(Br: np.ndarray, file_path, map_type: str) -> int:
    """Log a warning when a field contains NaN (missing) pixels.

    The readers count NaN pixels as 0 G (``np.nan_to_num``), as EUHFORIA does
    (``euhforia.magnetogram.hmi.read``): HMI maps without polar filling have
    NaN polar caps that would otherwise propagate to the whole solution. This
    function only reports them; it does not change the data.

    Returns:
        The number of NaN pixels.
    """
    count = int(np.count_nonzero(np.isnan(Br)))
    if count:
        logger.warning(
            "%s: %d NaN pixels (%.2f %% of the map) in %s set to 0 G "
            "(missing data, e.g. unfilled polar caps; a polar-filled product "
            "is preferable).",
            map_type,
            count,
            100.0 * count / Br.size,
            os.path.basename(str(file_path)),
        )
    return count


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
    theta, flip_rows, native_coordinate = _read_native_theta_axis(file_path, map_type)
    return _output_theta_axis(theta, native_coordinate, target_size), flip_rows


def _output_theta_axis(
    theta: np.ndarray,
    native_coordinate: str,
    target_size: int | None,
) -> np.ndarray:
    """Return native centers, or centers resized in the native coordinate."""
    if target_size is None:
        return theta
    theta = _resample_theta_centers(theta, int(target_size), native_coordinate)
    _validate_theta_axis(theta)
    return theta


def _read_native_theta_axis(
    file_path: str,
    map_type: str,
) -> tuple[np.ndarray, bool, str]:
    """Read native colatitude centers, the row flip, and the native coordinate.

    Returns:
        ``(theta, flip_rows, native_coordinate)`` where ``native_coordinate`` is
        ``"sine"`` or ``"latitude"``.
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
    return theta, flip_rows, native_coordinate


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
    carrington: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Read one temporal field, latitude axis, and optional Carrington centers."""
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
    warn_nan_pixels(Br, file_path, map_type)
    Br = np.nan_to_num(Br)

    if carrington:
        Br_carrington, longitude = read_carrington_map(Br, file_path, map_type)
        return Br_carrington, theta, longitude

    if map_type == "HMI_hourly":
        Br = roll_hmi_dynamic_to_zero_longitude(Br, file_path)
    else:
        Br = ensure_increasing_longitude(Br, file_path, map_type)
        if is_gong_map_type(map_type):
            Br = circular_shift_longitude(
                Br,
                extract_gong_longitude_shift(file_path),
            )
    return Br, theta, None


def read_temporal_br_map(
    file_path: str,
    map_type: str,
    adapt_map: int = 0,
) -> np.ndarray:
    """Read and normalize one FITS magnetogram used in interpolation."""
    Br, _, _ = _read_temporal_br_map_and_theta(file_path, map_type, adapt_map)
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
    carrington: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read, normalize, and temporally interpolate a four-map stencil."""
    map_type = normalize_map_type(map_type)
    resize = _as_bool(resize)
    carrington = _as_bool(carrington)
    logger.info("Reading interpolation stencil")
    normalized_maps = [
        _read_temporal_br_map_and_theta(path, map_type, adapt_map, carrington=True)
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
    longitude_axes = [item[2] for item in normalized_maps]
    longitude_shifts = validate_matching_longitude_axes(longitude_axes, local_files)
    Br_maps = [
        np.roll(Br, shift, axis=1)
        for Br, shift in zip(Br_maps, longitude_shifts)
    ]
    if not carrington:
        # Preserve the historical resampling seam and arithmetic exactly,
        # after establishing that every source samples the same physical grid.
        Br_maps = [
            _read_temporal_br_map_and_theta(path, map_type, adapt_map)[0]
            for path in local_files
        ]
    theta_coordinate = (
        _read_native_theta_axis(local_files[0], map_type)[2] if resize else None
    )
    n_phi_native = Br_maps[0].shape[1]
    Br_maps = [
        resize_magnetogram_if_requested(
            Br, resize, theta=reference_theta, theta_coordinate=theta_coordinate
        )
        for Br in Br_maps
    ]
    Br, Br_linear = interpolate_br_maps(Br_maps, selection, interpolation_order)
    theta = _output_theta_axis(
        reference_theta,
        theta_coordinate,
        RESIZED_MAGNETOGRAM_SHAPE[0] if resize else None,
    )
    phi = np.linspace(0.0, 2.0 * np.pi, Br.shape[1], endpoint=False)
    if resize and not carrington:
        order, phi = _resized_regular_longitude(n_phi_native, Br.shape[1])
        Br = Br[:, order]
        Br_linear = Br_linear[:, order]
    if carrington:
        longitude = resize_processed_longitude_axis(
            longitude_axes[0], Br.shape[1], preserve_cell_edges=True
        )
        # Resizing can move the first center just across the periodic seam.
        order = np.argsort(longitude % 360.0)
        Br, longitude = normalize_to_carrington(Br, longitude)
        Br_linear = Br_linear[:, order]
        phi = np.radians(longitude)
    Theta, Phi = build_theta_phi(theta, phi)
    logger.info("End of reading interpolation stencil")
    return Br, Theta, Phi, Br_linear


def read_magnetogram(
    file_path, map_type="custom", adapt_map=0, resize=False, carrington=False,
):
    """Read one magnetogram and build its physical spherical grid.

    When ``map_type`` is omitted or ``None``, the file is treated as a generic
    custom 2D FITS magnetogram whose complete latitude and longitude geometry
    must be described by its header.
    """
    map_type = "custom" if map_type is None else normalize_map_type(map_type)
    resize = _as_bool(resize)
    carrington = _as_bool(carrington)
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
                carrington=carrington,
            )

    if carrington:
        if map_type.lower() == "wso":
            raise ValueError("Physical Carrington normalization is not supported for WSO.")
        if map_type == "HMI_fdt":
            validate_hmi_fdt_carrington_frame(file_path)
        input_data = read_first_fits_image(file_path)
        Br_map = (
            input_data[adapt_map] if map_type in _ADAPT_ENSEMBLE_MAP_TYPES else input_data
        )
        if Br_map.ndim != 2:
            raise ValueError("A magnetogram must provide a single 2D field.")
        warn_nan_pixels(Br_map, file_path, map_type)
        theta_native, flip_rows, theta_coordinate = _read_native_theta_axis(
            file_path, map_type
        )
        theta = _output_theta_axis(
            theta_native,
            theta_coordinate,
            RESIZED_MAGNETOGRAM_SHAPE[0] if resize else None,
        )
        if flip_rows:
            Br_map = Br_map[::-1, :]
        Br_map, longitude = read_carrington_map(Br_map, file_path, map_type)
        Br_map = resize_magnetogram_if_requested(
            Br_map, resize, theta=theta_native, theta_coordinate=theta_coordinate
        )
        longitude = resize_processed_longitude_axis(
            longitude, Br_map.shape[1], preserve_cell_edges=True
        )
        Br_map, longitude = normalize_to_carrington(Br_map, longitude)
        Theta, Phi = build_theta_phi(theta, np.radians(longitude))
        return np.nan_to_num(Br_map), Theta, Phi

    if map_type == "custom":
        input_data = read_first_fits_image(file_path)
        if input_data.ndim != 2:
            raise ValueError(
                "A custom magnetogram must be a single 2D FITS image; "
                f"received shape {input_data.shape}."
            )
        warn_nan_pixels(input_data, file_path, map_type)
        theta_native, flip_rows, theta_coordinate = _read_native_theta_axis(
            file_path, map_type
        )
        theta = _output_theta_axis(
            theta_native,
            theta_coordinate,
            RESIZED_MAGNETOGRAM_SHAPE[0] if resize else None,
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
        Br_map = resize_magnetogram_if_requested(
            Br_map, resize, theta=theta_native, theta_coordinate=theta_coordinate
        )
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
        theta_native, flip_rows, theta_coordinate = _read_native_theta_axis(
            file_path, map_type
        )
        theta = _output_theta_axis(
            theta_native,
            theta_coordinate,
            RESIZED_MAGNETOGRAM_SHAPE[0] if resize else None,
        )
        Br_map = input_data[adapt_map, :, :]
        if flip_rows:
            Br_map = Br_map[::-1, :]
        Br_map = ensure_increasing_longitude(Br_map, file_path, map_type)
        n_phi_native = Br_map.shape[1]
        Br_map = resize_magnetogram_if_requested(
            Br_map, resize, theta=theta_native, theta_coordinate=theta_coordinate
        )
        phi = np.linspace(0.0, 2.0 * np.pi, Br_map.shape[1], endpoint=False)
        if resize:
            order, phi = _resized_regular_longitude(n_phi_native, Br_map.shape[1])
            Br_map = Br_map[:, order]
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
        theta_native, flip_rows, theta_coordinate = _read_native_theta_axis(
            file_path, map_type
        )
        theta = _output_theta_axis(
            theta_native,
            theta_coordinate,
            RESIZED_MAGNETOGRAM_SHAPE[0] if resize else None,
        )
        warn_nan_pixels(input_data, file_path, map_type)
        Br_data = input_data[::-1, :] if flip_rows else input_data
        if map_type in _HMI_DYNAMIC_MAP_TYPES:
            Br_data = roll_hmi_dynamic_to_zero_longitude(Br_data, file_path)
        else:
            Br_data = ensure_increasing_longitude(Br_data, file_path, map_type)
        n_phi_native = Br_data.shape[1]
        Br_data = resize_magnetogram_if_requested(
            Br_data, resize, theta=theta_native, theta_coordinate=theta_coordinate
        )
        phi = np.linspace(0.0, 2.0 * np.pi, Br_data.shape[1], endpoint=False)
        if resize:
            order, phi = _resized_regular_longitude(n_phi_native, Br_data.shape[1])
            Br_data = Br_data[:, order]
        Theta, Phi = build_theta_phi(theta, phi)
        Br_map = np.nan_to_num(Br_data)

    logger.info("End of reading file")
    return Br_map, Theta, Phi
