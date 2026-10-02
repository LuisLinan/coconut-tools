import numpy as np
import pytest

from coconut_tools.magnetogram.core.coordinates import spherical_pixel_areas
from coconut_tools.magnetogram.filters.gaussian_filter import (
    gaussian_filtering,
    gaussian_kernel,
)
from coconut_tools.magnetogram.gaussian_smoothing import filter_radial_field


def _centered_grid(n_theta=7, n_phi=12):
    theta = (np.arange(n_theta, dtype=float) + 0.5) * np.pi / n_theta
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    return theta, phi


def _fortran_reference_cell(field, theta, phi, row, column, size):
    """Direct transcription of the Fortran operation for one interior cell."""
    half = size // 2
    sigma = size / 6.0
    numerator = 0.0
    denominator = 0.0
    for di in range(-half, half + 1):
        source_row = row + di
        theta_n = 0.5 * (theta[source_row + 1] + theta[source_row])
        theta_p = 0.5 * (theta[source_row - 1] + theta[source_row])
        for dj in range(-half, half + 1):
            source_column = column + dj
            phi_n = 0.5 * (phi[source_column - 1] + phi[source_column])
            phi_p = 0.5 * (phi[source_column + 1] + phi[source_column])
            area = (phi_p - phi_n) * (np.cos(theta_n) - np.cos(theta_p))
            weight = np.exp(-(di * di + dj * dj) / (2.0 * sigma * sigma)) * area
            numerator += weight * field[source_row, source_column]
            denominator += weight
    return numerator / denominator


def test_constant_map_and_normalized_weights_remain_constant():
    theta, phi = _centered_grid()
    field = np.full((theta.size, phi.size), 7.25, dtype=np.float32)

    filtered = gaussian_filtering(field, phi, theta, template_size=5)

    assert filtered.dtype == np.float64
    np.testing.assert_allclose(filtered, 7.25, atol=4.0e-15, rtol=0.0)


def test_default_template_size_is_nine():
    theta, phi = _centered_grid(n_theta=11, n_phi=12)
    field = np.arange(theta.size * phi.size, dtype=float).reshape(
        theta.size, phi.size
    )

    default_result = gaussian_filtering(field, phi, theta)
    explicit_result = gaussian_filtering(field, phi, theta, template_size=9)

    np.testing.assert_array_equal(default_result, explicit_result)


def test_periodic_longitude_is_shift_equivariant_at_seam():
    theta, phi = _centered_grid()
    field = np.arange(theta.size * phi.size, dtype=float).reshape(
        theta.size, phi.size
    )

    filtered = gaussian_filtering(field, phi, theta, template_size=3)
    shifted = gaussian_filtering(np.roll(field, 1, axis=1), phi, theta, 3)

    np.testing.assert_allclose(shifted, np.roll(filtered, 1, axis=1), atol=1.0e-14)


def test_duplicate_zero_two_pi_endpoint_is_filtered_once():
    theta, unique_phi = _centered_grid()
    phi = np.concatenate((unique_phi, [2.0 * np.pi]))
    unique_field = np.arange(theta.size * unique_phi.size, dtype=float).reshape(
        theta.size, unique_phi.size
    )
    field = np.concatenate((unique_field, unique_field[:, :1]), axis=1)

    expected = gaussian_filtering(unique_field, unique_phi, theta, 3)
    filtered = gaussian_filtering(field, phi, theta, 3)

    np.testing.assert_allclose(filtered[:, :-1], expected)
    np.testing.assert_allclose(filtered[:, -1], filtered[:, 0])


def test_surface_area_weighting_matches_direct_calculation():
    theta, phi = _centered_grid(n_theta=5, n_phi=8)
    field = np.arange(40, dtype=float).reshape(5, 8) ** 2
    row, column, size = 2, 3, 3
    half = size // 2
    areas = spherical_pixel_areas(theta, phi)
    kernel = gaussian_kernel(size)
    window = field[row - half : row + half + 1, column - half : column + half + 1]
    area_window = areas[
        row - half : row + half + 1, column - half : column + half + 1
    ]
    weights = kernel * area_window
    weights /= weights.sum()

    filtered = gaussian_filtering(field, phi, theta, size)

    assert weights.sum() == pytest.approx(1.0, abs=2.0e-16)
    assert filtered[row, column] == pytest.approx(np.sum(weights * window), abs=1.0e-13)


def test_regular_theta_interior_matches_fortran_reference():
    theta, phi = _centered_grid(n_theta=9, n_phi=14)
    field = np.sin(theta[:, None]) + np.cos(2.0 * phi[None, :])
    row, column = 4, 7

    filtered = gaussian_filtering(field, phi, theta, template_size=3)
    reference = _fortran_reference_cell(field, theta, phi, row, column, 3)

    assert filtered[row, column] == pytest.approx(reference, abs=5.0e-16)


def test_sine_latitude_grid_uses_equal_area_cells():
    mu = 1.0 - (np.arange(6, dtype=float) + 0.5) * 2.0 / 6.0
    theta = np.arccos(mu)
    phi = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    areas = spherical_pixel_areas(theta, phi)

    np.testing.assert_allclose(areas, np.full_like(areas, 4.0 * np.pi / 48.0))


def test_polar_boundary_is_clipped_and_renormalized():
    theta, phi = _centered_grid(n_theta=5, n_phi=8)
    field = np.broadcast_to(np.arange(theta.size, dtype=float)[:, None], (5, 8))
    areas = spherical_pixel_areas(theta, phi)
    kernel = gaussian_kernel(3)
    expected_weights = kernel[1:, :] * areas[:2, :3]
    expected = np.sum(expected_weights * field[:2, :3]) / expected_weights.sum()

    filtered = gaussian_filtering(field, phi, theta, template_size=3)

    assert np.all(np.isfinite(filtered))
    np.testing.assert_allclose(filtered[0, :], expected, atol=2.0e-16)


def test_exact_pole_is_single_physical_point():
    theta = np.array([0.0, np.pi / 2.0, np.pi])
    phi = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    field = np.arange(24, dtype=float).reshape(3, 8)

    filtered = gaussian_filtering(field, phi, theta, template_size=3)

    np.testing.assert_allclose(filtered[0, :], filtered[0, 0])
    np.testing.assert_allclose(filtered[-1, :], filtered[-1, 0])
    assert np.all(np.isfinite(filtered))


@pytest.mark.parametrize("template_size", [1, 3, 5])
def test_odd_template_sizes_are_accepted(template_size):
    theta, phi = _centered_grid(n_theta=5, n_phi=8)
    field = np.ones((5, 8))

    result = filter_radial_field(field, phi, theta, template_size)

    np.testing.assert_allclose(result, field)


@pytest.mark.parametrize("template_size", [0, 2, 4, -1])
def test_invalid_template_sizes_are_rejected(template_size):
    theta, phi = _centered_grid(n_theta=5, n_phi=8)

    with pytest.raises(ValueError, match="positive odd integer"):
        gaussian_filtering(np.ones((5, 8)), phi, theta, template_size)
