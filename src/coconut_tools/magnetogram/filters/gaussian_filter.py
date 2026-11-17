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


GEODESIC_TRUNCATE_SIGMA = 3.0
"""Kernel support radius in units of sigma (the legacy template spans +-3 sigma)."""


def geodesic_gaussian_filtering(
    Br: np.ndarray,
    phi: np.ndarray,
    theta: np.ndarray,
    sigma_deg: float,
    truncate: float = GEODESIC_TRUNCATE_SIGMA,
) -> np.ndarray:
    """Smooth a global radial field with a Gaussian of great-circle distance.

    The weight between two cell centres is ``exp(-d**2 / (2 sigma**2))`` where
    ``d`` is their great-circle angle, truncated at ``d <= truncate * sigma``.
    The physical width is therefore the same for every product, resolution and
    latitude, the kernel is isotropic, periodic in longitude and continuous
    across the poles.

    Normalisation is on the source side: every source cell redistributes
    exactly its own flux ``Br * area`` over its neighbours, proportionally to
    ``weight * area`` of the receiving cells. The surface integral of ``Br``
    is conserved to round-off (a uniform field is preserved only up to
    discretisation error).

    Implementation: ``phi`` must be uniform, so the weight between rows ``i``
    and ``k`` depends only on the longitude offset; each output row is a sum of
    circular FFT convolutions over the rows ``k`` with
    ``|theta_i - theta_k| <= truncate * sigma``.

    Args:
        Br: Radial field with shape ``(theta.size, phi.size)``.
        phi: Uniform, increasing cell-centre longitudes in radians, with an
            optional duplicated endpoint at one full revolution.
        theta: Increasing cell-centre colatitudes in radians (north to south);
            any spacing (sine-latitude or latitude grids).
        sigma_deg: Gaussian standard deviation in heliographic degrees
            (1 deg ~ 12.15 Mm on the photosphere). Must be positive.
        truncate: Support radius in units of ``sigma``.

    Returns:
        A float64 array with the same shape as ``Br``.

    Raises:
        ValueError: For invalid inputs or a non-uniform longitude axis.
    """
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
    sigma_deg = float(sigma_deg)
    if not np.isfinite(sigma_deg) or sigma_deg <= 0.0:
        raise ValueError("sigma_deg must be a finite positive number.")
    if not np.isfinite(truncate) or truncate <= 0.0:
        raise ValueError("truncate must be a finite positive number.")

    duplicate_endpoint = longitude.size > 1 and np.isclose(
        longitude[-1] - longitude[0], 2.0 * np.pi, atol=1.0e-10, rtol=0.0
    )
    if duplicate_endpoint:
        field = field[:, :-1]
        longitude = longitude[:-1]

    n_theta, n_phi = field.shape
    if n_phi > 1:
        steps = np.diff(np.unwrap(longitude))
        if not np.allclose(steps, 2.0 * np.pi / n_phi, atol=1.0e-9, rtol=0.0):
            raise ValueError(
                "geodesic Gaussian filtering requires a uniform, complete, "
                "increasing longitude axis."
            )

    sigma = np.radians(sigma_deg)
    cutoff = truncate * sigma
    cos_cutoff = np.cos(min(cutoff, np.pi))

    row_area = spherical_pixel_areas(colatitude, longitude, field.shape)[:, 0]
    sin_theta = np.sin(colatitude)
    cos_theta = np.cos(colatitude)
    # Longitude offsets of every column with respect to column 0.
    cos_dphi = np.cos(np.arange(n_phi) * (2.0 * np.pi / n_phi))

    def kernel_rows(i: int, rows: np.ndarray) -> np.ndarray:
        """Weights between row ``i`` and ``rows`` for every longitude offset."""
        cos_d = (
            cos_theta[i] * cos_theta[rows, None]
            + sin_theta[i] * sin_theta[rows, None] * cos_dphi[None, :]
        )
        cos_d = np.clip(cos_d, -1.0, 1.0)
        distance = np.arccos(cos_d)
        weights = np.exp(-0.5 * (distance / sigma) ** 2)
        weights[cos_d < cos_cutoff] = 0.0
        return weights

    # Rows that can interact: the closest points of two rings are on the same
    # meridian, at distance |theta_i - theta_k|. Pairs across the pole are
    # included automatically (offset ~ pi gives distance theta_i + theta_k).
    neighbours = [
        np.nonzero(np.abs(colatitude - colatitude[i]) <= cutoff)[0]
        for i in range(n_theta)
    ]

    # Pass 1: source normalisation N_k = sum_i area_i * sum_j w_ik(j).
    # The kernel is symmetric, so row i contributes area_i * sum_j w_ik to N_k.
    normalisation = np.zeros(n_theta)
    for i in range(n_theta):
        rows = neighbours[i]
        normalisation[rows] += row_area[i] * kernel_rows(i, rows).sum(axis=1)
    if np.any(normalisation <= 0.0):
        raise ValueError("The supplied coordinates produce a zero-weight kernel.")

    # Pass 2: B_out[i] = sum_k (area_k / N_k) * (B[k] circularly convolved
    # with w_ik). Flux is conserved: sum_i area_i B_out[i] = sum_k area_k B[k].
    weighted_spectrum = np.fft.rfft(
        field * (row_area / normalisation)[:, None], axis=1
    )
    filtered = np.empty_like(field)
    for i in range(n_theta):
        rows = neighbours[i]
        kernel_spectrum = np.fft.rfft(kernel_rows(i, rows), axis=1)
        filtered[i] = np.fft.irfft(
            np.sum(kernel_spectrum * weighted_spectrum[rows], axis=0),
            n=n_phi,
        )

    if duplicate_endpoint:
        filtered = np.concatenate((filtered, filtered[:, :1]), axis=1)
    return filtered
