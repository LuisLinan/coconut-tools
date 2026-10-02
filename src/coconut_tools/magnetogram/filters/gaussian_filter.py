"""Surface-area-weighted Gaussian smoothing for global magnetograms.

Author: Haopeng Wang (original Fortran implementation)
Converted to Python with spherical boundary handling.
"""

from __future__ import annotations

import operator

import numpy as np

from coconut_tools.magnetogram.core.coordinates import spherical_pixel_areas


def _validate_inputs(
    Br: np.ndarray,
    phi: np.ndarray,
    theta: np.ndarray,
    template_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return validated float64 filter inputs."""
    field = np.asarray(Br, dtype=np.float64)
    longitude = np.asarray(phi, dtype=np.float64)
    colatitude = np.asarray(theta, dtype=np.float64)

    if field.ndim != 2:
        raise ValueError("Br must be a 2D array.")
    if longitude.ndim != 1 or colatitude.ndim != 1:
        raise ValueError("phi and theta must be 1D arrays.")
    if field.shape != (colatitude.size, longitude.size):
        raise ValueError(
            f"Br shape {field.shape} does not match theta/phi sizes "
            f"({colatitude.size}, {longitude.size})."
        )
    if not (
        np.all(np.isfinite(field))
        and np.all(np.isfinite(longitude))
        and np.all(np.isfinite(colatitude))
    ):
        raise ValueError("Br, theta, and phi must contain only finite values.")

    try:
        size = operator.index(template_size)
    except TypeError as exc:
        raise TypeError("template_size must be an integer.") from exc
    if isinstance(template_size, (bool, np.bool_)):
        raise TypeError("template_size must be an integer.")
    if size < 1 or size % 2 == 0:
        raise ValueError("template_size must be a positive odd integer.")
    unique_longitude_count = longitude.size
    if longitude.size > 1 and np.isclose(
        longitude[-1] - longitude[0],
        2.0 * np.pi,
        atol=1.0e-10,
        rtol=0.0,
    ):
        unique_longitude_count -= 1
    if size > unique_longitude_count:
        raise ValueError(
            "template_size cannot exceed the number of unique longitude cells."
        )

    return field, longitude, colatitude, size


def gaussian_kernel(template_size: int) -> np.ndarray:
    """Build the unnormalised Fortran Gaussian template in float64.

    The original implementation defines ``sigma = n / 6`` and omits the
    constant Gaussian prefactor because the final, area-weighted template is
    normalised separately.
    """
    try:
        size = operator.index(template_size)
    except TypeError as exc:
        raise TypeError("template_size must be an integer.") from exc
    if isinstance(template_size, (bool, np.bool_)):
        raise TypeError("template_size must be an integer.")
    if size < 1 or size % 2 == 0:
        raise ValueError("template_size must be a positive odd integer.")

    half = size // 2
    sigma = np.float64(size) / np.float64(6.0)
    offsets = np.arange(-half, half + 1, dtype=np.float64)
    row_offset, column_offset = np.meshgrid(offsets, offsets, indexing="ij")
    return np.exp(
        -(row_offset * row_offset + column_offset * column_offset)
        / (np.float64(2.0) * sigma * sigma)
    )


def gaussian_filtering(
    Br: np.ndarray,
    phi: np.ndarray,
    theta: np.ndarray,
    template_size: int = 9,
) -> np.ndarray:
    """Smooth a global radial-field map with spherical-area-weighted weights.

    ``theta`` is increasing colatitude (north to south), and ``phi`` is an
    increasing periodic longitude axis. For every output cell,
    the effective weight is the Fortran Gaussian template multiplied by the
    exact solid angle of the source cell. The radius-squared factor from the
    physical cell area cancels during normalisation and is therefore omitted.

    Longitude indices wrap periodically. An explicit duplicate endpoint at
    ``phi[0] + 2*pi`` is removed for the calculation and restored from the
    filtered first column. At the non-periodic polar boundary,
    only physical cells present on the sphere are used and the local weights
    are renormalised. This intentionally replaces the historical Fortran
    behaviour, which left a border of cells unfiltered.

    Args:
        Br: Radial magnetic field with shape ``(theta.size, phi.size)``.
        phi: Cell-centre longitudes in radians, with an optional duplicated
            endpoint at one full revolution.
        theta: Cell-centre colatitudes in radians, increasing from north to
            south. Both latitude-regular and sine-latitude-regular grids are
            supported by the shared geometry helpers.
        template_size: Odd side length ``n`` of the Gaussian template.

    Returns:
        A float64 array with the same shape as ``Br``.
    """
    field, longitude, colatitude, size = _validate_inputs(
        Br, phi, theta, template_size
    )
    duplicate_endpoint = longitude.size > 1 and np.isclose(
        longitude[-1] - longitude[0],
        2.0 * np.pi,
        atol=1.0e-10,
        rtol=0.0,
    )
    if duplicate_endpoint:
        field = field[:, :-1]
        longitude = longitude[:-1]

    if size == 1:
        filtered = field.copy()
        if duplicate_endpoint:
            filtered = np.concatenate((filtered, filtered[:, :1]), axis=1)
        return filtered

    areas = spherical_pixel_areas(colatitude, longitude, field.shape)
    kernel = gaussian_kernel(size)
    half = size // 2

    numerator = np.zeros_like(field, dtype=np.float64)
    denominator = np.zeros_like(field, dtype=np.float64)

    # Accumulate source cell (i + di, j + dj) into output cell (i, j).
    # Longitude is periodic; latitude is deliberately clipped at the poles.
    for kernel_row, row_offset in enumerate(range(-half, half + 1)):
        if row_offset < 0:
            output_rows = slice(-row_offset, None)
            source_rows = slice(None, row_offset)
        elif row_offset > 0:
            output_rows = slice(None, -row_offset)
            source_rows = slice(row_offset, None)
        else:
            output_rows = slice(None)
            source_rows = slice(None)

        for kernel_column, column_offset in enumerate(range(-half, half + 1)):
            spatial_weight = kernel[kernel_row, kernel_column]
            source_area = np.roll(
                areas[source_rows, :], -column_offset, axis=1
            )
            effective_weight = spatial_weight * source_area
            source_field = np.roll(
                field[source_rows, :], -column_offset, axis=1
            )
            numerator[output_rows, :] += effective_weight * source_field
            denominator[output_rows, :] += effective_weight

    north_pole = np.isclose(colatitude[0], 0.0, atol=1.0e-12, rtol=0.0)
    south_pole = np.isclose(colatitude[-1], np.pi, atol=1.0e-12, rtol=0.0)
    positive_weight = denominator > 0.0
    if north_pole:
        positive_weight[0, :] = True
    if south_pole:
        positive_weight[-1, :] = True
    if not np.all(positive_weight):
        raise ValueError(
            "The supplied coordinates produce a zero-area filter window."
        )

    filtered = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0.0,
    )

    # Every longitude at an exact pole denotes the same physical point. The
    # adjacent rings are therefore averaged over all longitudes, using only
    # the Gaussian's meridional factor and the exact cell areas.
    if north_pole:
        source_rows = slice(0, min(half + 1, field.shape[0]))
        meridional_weight = kernel[
            half : half + field[source_rows, :].shape[0], half
        ]
        weights = meridional_weight[:, None] * areas[source_rows, :]
        filtered[0, :] = np.sum(weights * field[source_rows, :]) / np.sum(weights)
    if south_pole:
        row_count = min(half + 1, field.shape[0])
        source_rows = slice(field.shape[0] - row_count, field.shape[0])
        meridional_weight = kernel[half - row_count + 1 : half + 1, half]
        weights = meridional_weight[:, None] * areas[source_rows, :]
        filtered[-1, :] = np.sum(weights * field[source_rows, :]) / np.sum(weights)

    if duplicate_endpoint:
        filtered = np.concatenate((filtered, filtered[:, :1]), axis=1)
    return filtered
